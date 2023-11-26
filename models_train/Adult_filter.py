import os
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
from torchvision import datasets, transforms
from torchsummary import summary
from utils.helpers import Progressbar
import wandb
import utils.utils as utils
import copy

from dataloader.Adult import AdultDataset
from models_train.module import *
from models_train.module.MINE.model import M
from models_train.module.MINE.utils import mi_criterion


class Generator(nn.Module):
    def __init__(self, in_dim, encode_dims, decode_dims, out_dim):
        super().__init__()
        enc_layers = []
        for dim in encode_dims:
            enc_layers.append(nn.Linear(in_dim, dim))
            enc_layers.append(nn.ReLU())
            in_dim = dim
        self.enc_layers = nn.Sequential(*enc_layers)
        dec_layers = []
        for dim in decode_dims:
            dec_layers.append(nn.Linear(in_dim + 1, dim))
            dec_layers.append(nn.ReLU())
            in_dim = dim
        self.dec_layers = nn.Sequential(*dec_layers)
        self.out = nn.Linear(in_dim, out_dim)

    def encode(self, x):
        z = x
        zs = []
        for layer in self.enc_layers:
            z = layer(z)
            zs.append(z)
        return zs

    def decode(self, zs, a):
        z = torch.cat([zs[-1], a], dim=-1)
        for i, layer in enumerate(self.dec_layers):
            z = layer(z)
        z = self.out(z)
        return z

    def forward(self, x, a=None, mode="enc-dec"):
        if mode == "enc-dec":
            assert a is not None, "No given attribute."
            return self.decode(self.encode(x), a)
        if mode == "enc":
            return self.encode(x)
        if mode == "dec":
            assert a is not None, "No given attribute."
            return self.decode(x, a)
        raise Exception("Unrecognized mode: " + mode)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim):
        super().__init__()
        layers = []
        for dim in hidden_dims:
            layers.append(nn.Linear(in_dim, dim))
            layers.append(nn.ReLU())
            in_dim = dim
        self.layers = nn.Sequential(*layers)
        self.out = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        if len(x.size()) > 2:
            x = x.view(x.size(0), -1)
        x = self.layers(x)
        return self.out(x)


class Model:
    def __init__(self, args):
        self.set_data(args)
        self.mode = args.mode
        self.gpu = args.gpu
        self.multi_gpu = args.multi_gpu if "multi_gpu" in args else False
        self.gr = args.gr
        self.gc = args.gc
        self.dc = args.dc
        self.gp = args.gp
        self.mi = args.mi
        self.num_iter_MI = args.num_iter_MI
        self.dim_per_attr = args.dim_per_attr
        self.dim_attrs = args.dim_per_attr * args.n_attrs
        self.hyperparameter = args.hyperparameter
        self.epoch = 0
        self.it = 0

        # G
        in_dim = 108
        enc_dims = [64, 10]
        dec_dims = [64]
        out_dim = 108
        self.G = Generator(in_dim, enc_dims, dec_dims, out_dim)
        self.G.train()
        if self.gpu:
            self.G.cuda()
        summary(self.G, [(1, in_dim), (1, args.n_attrs)], batch_size=4, device="cuda" if args.gpu else "cpu")

        # D
        in_dim = 108
        hidden_dims = [64]
        out_dim = 1
        self.D = MLP(in_dim, hidden_dims, out_dim)
        self.D.train()
        if self.gpu:
            self.D.cuda()
        summary(self.D, (1, in_dim), batch_size=4, device="cuda" if args.gpu else "cpu")

        # P
        in_dim = 108
        hidden_dims = [64]
        out_dim = 1
        self.P = MLP(in_dim, hidden_dims, out_dim)
        self.P.train()
        if self.gpu:
            self.P.cuda()
        summary(self.P, (1, in_dim), batch_size=4, device="cuda" if args.gpu else "cpu")

        # MI
        hidden_dim = [10]
        self.mine = M(input_dim=10 + 1, hidden_dim=hidden_dim)
        self.mine.train()
        if self.gpu:
            self.mine.cuda()
        summary(self.mine, (1, 10 + 1), batch_size=4, device="cuda" if args.gpu else "cpu")

        if torch.cuda.device_count() > 1:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)
            self.P = nn.DataParallel(self.P)
            self.mine = nn.DataParallel(self.mine)

        self.optim_G = optim.Adam(self.G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        self.optim_D = optim.Adam(self.D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        self.optim_P = optim.Adam(self.P.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        self.optim_mine = optim.Adam(self.mine.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    def set_data(self, args):
        train_dataset = AdultDataset(path=args.data_path, mode=args.Adult_train_mode, quick_load=True, bias_name=args.Adult_attrs[0])
        valid_dataset = AdultDataset(path=args.data_path, mode="all", quick_load=True, bias_name=args.Adult_attrs[0])

        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        self.valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=args.n_samples,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        print(
            "Training images:",
            len(train_dataset),
            "/",
            "Validating images:",
            len(valid_dataset),
        )
        args.it_per_epoch = len(train_dataset) // args.batch_size

    def train_epoch(self, args):
        progressbar = Progressbar()
        # train with base lr in the first 100 epochs and half the lr in the last 100 epochs
        lr = args.lr / (10 ** (self.epoch // 20))
        self.set_lr(lr)

        # pretrain MINE at the beginning of every epoch
        mine_loader = copy.deepcopy(self.train_dataloader)
        batch_iter = iter(mine_loader)
        self.pretrainMI(mine_loader, args)

        # start iteration
        errG, errD = None, None
        for img_a, _, att_a in progressbar(self.train_dataloader):
            # prepare data
            img_a, att_a, att_a_, att_b, att_b_ = self.prepare_data(img_a, att_a, args)

            # train model
            self.train()

            if (self.it + 1) % (args.n_d + 1) != 0:
                errD = self.trainD(img_a, att_a, att_a_, att_b, att_b_)
            else:
                errG = self.trainG(img_a, att_a, att_a_, att_b, att_b_)
                batch_iter = self.trainMI(batch_iter, mine_loader, args)
            if errD and errG:
                progressbar.say(epoch=self.epoch, iter=self.it + 1, d_loss=errD["d_loss"], g_loss=errG["g_loss"])

            if (self.it + 1) % args.save_interval == 0:
                self.save_model(args)
            self.it += 1
        self.epoch += 1
        # renew mine for new epoch
        self.initMINE()

    def prepare_data(self, img_a, att_a, args):
        att_a = torch.unsqueeze(att_a, 1) if len(list(att_a.size())) == 1 else att_a
        img_a = img_a.cuda() if args.gpu else img_a
        att_a = att_a.cuda() if args.gpu else att_a
        att_b = 1 - att_a
        att_a = att_a.type(torch.float)
        att_b = att_b.type(torch.float)
        att_a_ = (att_a * 2 - 1) * args.thres_int  # -1/2, 1/2 for all
        att_b_ = (att_b * 2 - 1) * args.thres_int  # -1/2, 1/2 for all
        return img_a, att_a, att_a_, att_b, att_b_

    def pretrainMI(self, mine_loader, args):
        progressbar = Progressbar()
        loss_mine = 0
        pretrain_it = 0
        print("Warmup mutual information estimator")
        for img_a, _, att_a in progressbar(mine_loader):
            # prepare data
            img_a, att_a, att_a_, att_b, att_b_ = self.prepare_data(img_a, att_a, args)
            self.optim_mine.zero_grad()

            z = self.G(img_a, mode="enc")[-1].detach()
            mine_loss = mi_criterion(z.view(z.size(0), -1), att_a.view(att_a.size(0), -1), self.mine)

            mine_loss.backward()
            self.optim_mine.step()
            loss_mine += mine_loss.item()
            wandb.log({"pretrain/batch mine loss": mine_loss.item()})
            pretrain_it += 1
            progressbar.say(iter=pretrain_it + 1, mine_loss=mine_loss.item())

    def trainMI(self, batch_iter, mine_loader, args):
        for i in range(self.num_iter_MI):
            img_a, att_a, batch_iter = utils.nextbatch_with_dummy(batch_iter, mine_loader)
            img_a, att_a, att_a_, att_b, att_b_ = self.prepare_data(img_a, att_a, args)
            self.optim_mine.zero_grad()
            z = self.G(img_a, mode="enc")[-1].detach()
            mine_loss = mi_criterion(z.view(z.size(0), -1), att_a.view(att_a.size(0), -1), self.mine)
            mine_loss.backward()
            self.optim_mine.step()
            wandb.log({"train/batch mine loss": mine_loss.item()})
        return batch_iter

    def initMINE(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.mine.apply(init_weights)

    def _criterion_pred(self, output, target):
        L = nn.BCEWithLogitsLoss()
        loss = L(output, target)
        return loss

    def trainG(self, img_a, att_a, att_a_, att_b, att_b_):
        self.D.eval()
        self.P.eval()
        self.mine.eval()
        for p in self.D.parameters():
            p.requires_grad = False

        zs_a = self.G(img_a, mode="enc")
        img_fake = self.G(zs_a, att_b, mode="dec")
        img_recon = self.G(zs_a, att_a, mode="dec")
        d_fake, dc_fake = self.D(img_fake), self.P(img_fake)

        if self.mode == "wgan":
            gf_loss = -d_fake.mean()
        if self.mode == "lsgan":  # mean_squared_error
            gf_loss = F.mse_loss(d_fake, torch.ones_like(d_fake))
        if self.mode == "dcgan":  # sigmoid_cross_entropy
            gf_loss = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
        gc_loss = self._criterion_pred(dc_fake, att_b)
        gr_loss = F.l1_loss(img_recon, img_a)
        z = zs_a[-1]
        mine_loss = -mi_criterion(z.view(z.size(0), -1), att_a.detach().view(att_a.size(0), -1), self.mine)
        g_loss = gf_loss + self.gc * gc_loss + self.gr * gr_loss + self.mi * mine_loss

        self.optim_G.zero_grad()
        g_loss.backward()
        self.optim_G.step()

        wandb.log(
            {
                "g/total_loss": g_loss.item(),
                "g/fake_loss": gf_loss.item(),
                "g/classifier_loss": gc_loss.item(),
                "g/reconstuct_loss": gr_loss.item(),
            }
        )
        errG = {"g_loss": g_loss.item(), "gf_loss": gf_loss.item(), "gc_loss": gc_loss.item(), "gr_loss": gr_loss.item()}
        return errG

    def trainD(self, img_a, att_a, att_a_, att_b, att_b_):
        for p in self.D.parameters():
            p.requires_grad = True

        img_fake = self.G(img_a, att_b).detach()
        d_real, dc_real = self.D(img_a), self.P(img_a)
        d_fake, dc_fake = self.D(img_fake), self.P(img_fake)

        def gradient_penalty(f, real, fake=None):
            def interpolate(a, b=None):
                if b is None:  # interpolation in DRAGAN
                    beta = torch.rand_like(a)
                    b = a + 0.5 * a.var().sqrt() * beta
                alpha = torch.rand(a.size(0), 1)
                alpha = alpha.cuda() if self.gpu else alpha
                inter = a + alpha * (b - a)
                return inter

            x = interpolate(real, fake).requires_grad_(True)
            pred = f(x)
            if isinstance(pred, tuple):
                pred = pred[0]
            grad = autograd.grad(outputs=pred, inputs=x, grad_outputs=torch.ones_like(pred), create_graph=True, retain_graph=True, only_inputs=True)[0]
            grad = grad.view(grad.size(0), -1)
            norm = grad.norm(2, dim=1)
            gp = ((norm - 1.0) ** 2).mean()
            return gp

        if self.mode == "wgan":  # discriminator becomes critic
            wd = d_real.mean() - d_fake.mean()
            df_loss = -wd
            df_gp = gradient_penalty(self.D, img_a, img_fake)
        if self.mode == "lsgan":  # mean_squared_error
            df_loss = F.mse_loss(d_real, torch.ones_like(d_fake)) + F.mse_loss(d_fake, torch.zeros_like(d_fake))
            df_gp = gradient_penalty(self.D, img_a)
        if self.mode == "dcgan":  # Deep Convolutional gan  # sigmoid_cross_entropy
            df_loss = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real)) + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
            df_gp = gradient_penalty(self.D, img_a)
        dc_loss = self._criterion_pred(dc_real, att_a)
        d_loss = df_loss + self.gp * df_gp + self.dc * dc_loss

        self.optim_D.zero_grad()
        self.optim_P.zero_grad()
        d_loss.backward()
        self.optim_D.step()
        self.optim_P.step()

        errD = {"d_loss": d_loss.item(), "df_loss": df_loss.item(), "df_gp": df_gp.item(), "dc_loss": dc_loss.item()}
        wandb.log(
            {
                "d/total_loss": d_loss.item(),
                "d/fake_loss": df_loss.item(),
                "d/classifier_loss": dc_loss.item(),
                "d/df_gp_loss": df_gp.item(),
            }
        )
        return errD

    def set_lr(self, lr):
        for g in self.optim_G.param_groups:
            g["lr"] = lr
        for g in self.optim_D.param_groups:
            g["lr"] = lr
        for g in self.optim_P.param_groups:
            g["lr"] = lr
        for g in self.optim_mine.param_groups:
            g["lr"] = lr

    def train(self):
        self.G.train()
        self.D.train()
        self.P.train()
        self.mine.train()

    def eval(self):
        self.G.eval()
        self.D.eval()
        self.P.eval()
        self.mine.eval()

    def save(self, path):
        states = {
            "G": self.G.state_dict(),
            "D": self.D.state_dict(),
            "P": self.P.state_dict(),
            "mine": self.mine.state_dict(),
            "optim_G": self.optim_G.state_dict(),
            "optim_D": self.optim_D.state_dict(),
            "optim_P": self.optim_P.state_dict(),
            "optim_mine": self.optim_mine.state_dict(),
        }
        torch.save(states, path)

    def load(self, path):
        states = torch.load(path, map_location=lambda storage, loc: storage)
        if "G" in states:
            self.G.load_state_dict(states["G"])
        if "D" in states:
            self.D.load_state_dict(states["D"])
        if "P" in states:
            self.P.load_state_dict(states["P"])
        if "mine" in states:
            self.mine.load_state_dict(states["mine"])
        if "optim_G" in states:
            self.optim_G.load_state_dict(states["optim_G"])
        if "optim_D" in states:
            self.optim_D.load_state_dict(states["optim_D"])
        if "optim_P" in states:
            self.optim_P.load_state_dict(states["optim_P"])
        if "optim_mine" in states:
            self.optim_mine.load_state_dict(states["optim_mine"])

    def saveG(self, path):
        states = {"G": self.G.state_dict()}
        torch.save(states, path)

    def load(self, path):
        states = torch.load(path, map_location=lambda storage, loc: storage)
        if "G" in states:
            self.G.load_state_dict(states["G"])
        if "D" in states:
            self.D.load_state_dict(states["D"])
        if "P" in states:
            self.P.load_state_dict(states["P"])
        if "mine" in states:
            self.mine.load_state_dict(states["mine"])
        if "optim_G" in states:
            self.optim_G.load_state_dict(states["optim_G"])
        if "optim_D" in states:
            self.optim_D.load_state_dict(states["optim_D"])
        if "optim_P" in states:
            self.optim_P.load_state_dict(states["optim_P"])
        if "optim_mine" in states:
            self.optim_mine.load_state_dict(states["optim_mine"])

    def save_model(self, args):
        if not args.save_all:
            self.saveG(
                os.path.join(
                    args.ssd_path,
                    args.experiment,
                    args.name,
                    args.Adult_train_mode,
                    self.hyperparameter,
                    "checkpoint",
                    "weights.{:d}.pth".format(self.epoch),
                )
            )
        else:
            self.save(
                os.path.join(
                    args.ssd_path,
                    args.experiment,
                    args.name,
                    args.Adult_train_mode,
                    self.hyperparameter,
                    "checkpoint",
                    "weights.{:d}.pth".format(self.epoch),
                )
            )