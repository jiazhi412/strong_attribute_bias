import torch
from models_eval.set_CMNIST import Model as CMNIST
from models_eval.module import *
import utils
import wandb
import os
import glob
import datetime


class Model(CMNIST):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.epoch = 0
        self.pretrain_epoch = 0
        self.device = opt["device"]
        self.save_folder = opt["save_folder"]
        self.save_file = opt["save_file"]
        self.print_freq = opt["print_freq"]
        self.init_lr = opt["lr"]

        self.best_dev_overall_performance = 0.0
        self.best_test_overall_performance = 0.0
        self.set_network(opt, opt["filter_parameters"])
        self.set_data(opt)
        self.set_optimizer(opt)
        # load filter
        self.load_filter_weights(self.opt["filter_path"])

    def set_network(self, opt, args):
        # ========= create models ===========
        self.filter = Generator(args.enc_dim, args.enc_layers, args.enc_norm, args.enc_acti, args.dec_dim, args.dec_layers, args.dec_norm, args.dec_acti, args.n_attrs, args.shortcut_layers, args.inject_layers, args.img_size, args.dim_per_attr).to(self.device)
        self.filter.eval()
        self.encoder = CNN(in_channels=3).to(self.device)
        self.predictor = Predictor2(r_dim=400, out_dim=10).to(self.device)

    def set_optimizer(self, opt):
        self.optimizer = utils.choose_optimizer(opt["optimizer"])
        self.optim_pred = self.optimizer(params=filter(lambda p: p.requires_grad, self.pred_parameters()), lr=opt["lr"], weight_decay=opt["wd"])
        self.lr_scheduler_pred = torch.optim.lr_scheduler.StepLR(self.optim_pred, step_size=10, gamma=0.1)

    def pred_parameters(self):
        res = list(self.encoder.parameters()) + list(self.predictor.parameters())
        return res

    def _criterion(self, output, target):
        loss = nn.CrossEntropyLoss()
        return loss(output, target)

    def forward(self, x):
        r = self.encoder(x)
        pred = self.predictor(r)
        return pred, r

    def save_weights(self, file_path, optim_pred=None):
        torch.save(
            {
                "epoch": self.epoch,
                "encoder": self.encoder.state_dict(),
                "predictor": self.predictor.state_dict(),
                "optim_pred": optim_pred.state_dict() if optim_pred is not None else None,
            },
            file_path,
        )

    def load_weights(self, file_path, optim_pred=None):
        ckpt = torch.load(file_path)
        self.epoch = ckpt["epoch"]
        self.encoder.load_state_dict(ckpt["encoder"])
        self.predictor.load_state_dict(ckpt["predictor"])
        if optim_pred is not None:
            optim_pred_state_dict = ckpt["optim_pred"]
            if optim_pred_state_dict is None:
                print("WARNING: No optim_pred state dict found")
            else:
                optim_pred.load_state_dict(optim_pred_state_dict)

    def load_filter_weights(self, file_path):
        states = torch.load(file_path, map_location=lambda storage, loc: storage)
        if "module" in list(states["G"].keys())[0]:
            new_state_dict = {"G": {}}
            for key, value in states["G"].items():
                new_key = key.replace("module.", "")  # Remove 'module.' from the key
                new_state_dict["G"][new_key] = value
            states = new_state_dict
        if "G" in states:
            self.filter.load_state_dict(states["G"])

    def forward_filter(self, x, a):
        x_filtered = self.filter(x, a)
        return x_filtered

    def _train(self, loader):
        """Train the model for one epoch"""

        self.encoder.train()
        self.predictor.train()

        train_loss = 0
        for i, (images, targets, a) in enumerate(loader):
            images, targets, a = images.to(self.device), targets.to(self.device), a.to(self.device)
            origin_images = images

            a_assigned = torch.ones_like(a)
            images = self.forward_filter(images, a_assigned)

            self.optim_pred.zero_grad()
            outputs, _ = self.forward(images.detach())
            loss = self._criterion(outputs, targets)
            loss.backward()
            self.optim_pred.step()
            train_loss += loss.item()
            wandb.log({"train/batch prediction loss": loss.item()})
            if self.print_freq and (i % self.print_freq == 0):
                print("Training epoch {}: [{}|{}], batch loss: {}".format(self.epoch + 1, i + 1, len(loader), loss.item()))
        if self.epoch % 10 == 0:
            utils.log_image("train/origin images", origin_images)
            utils.log_image("train/filtered images", images)
        self.epoch += 1

    def _test(self, loader, forward_fn):
        """Compute output on loader"""

        self.predictor.eval()

        test_loss = 0
        output_list = []
        feature_list = []
        target_list = []
        with torch.no_grad():
            for i, (x, y, a) in enumerate(loader):
                x, y, a = x.to(self.device), y.to(self.device), a.to(self.device)
                origin_images = x

                a_assigned = torch.ones_like(a)
                x = self.forward_filter(x, a_assigned)

                outputs, z = forward_fn(x)
                loss = self._criterion(outputs, y)
                test_loss += loss.item()
                output_list.append(outputs)
                feature_list.append(z)
                target_list.append(y)
        if self.epoch % 10 == 0:
            utils.log_image("test/origin images", origin_images)
            utils.log_image("test/filtered images", x)
        return test_loss / len(loader), torch.cat(output_list), torch.cat(feature_list), torch.cat(target_list)

    @utils.timer
    def train(self):
        """Train the model for one epoch, evaluate on validation set and save the best model"""
        # resume (maybe)
        if self.opt["resume"]:
            assert os.path.exists(os.path.join(self.save_folder, "_".join((self.save_file, "ckpt.pth")))), "no resume model found!"
            print("Resume from pre-trained model...")
            self.load_weights(os.path.join(self.save_folder, "_".join((self.save_file, "ckpt.pth"))))

        # load filter
        self.load_filter_weights(self.opt["filter_path"])

        # train the downstream predictor
        self._train(self.train_loader)

        # evaluation
        train_loss, train_output, train_feature, train_target = self._test(self.train_loader, forward_fn=self.forward)
        train_acc = utils.compute_Acc_withlogits_nonbinary(train_output, train_target)
        dev_loss, dev_output, dev_feature, dev_target = self._test(self.dev_loader, forward_fn=self.forward)
        dev_acc = utils.compute_Acc_withlogits_nonbinary(dev_output, dev_target)

        wandb.log(
            {
                "train/epoch": self.epoch,
                "train/train loss": train_loss,
                "train/train acc": train_acc,
                "train/dev loss": dev_loss,
                "train/dev acc": dev_acc,
            }
        )
        print("Finish training epoch {}, {}: {}, {}: {}".format(self.epoch, "Train Loss", train_loss, "Dev Loss", dev_loss))

        # save model after one single epoch (maybe not)
        if not self.opt["no_sm"]:
            self.save_weights(os.path.join(self.save_folder, "_".join((self.save_file, "ckpt.pth"))))

        # save best model (maybe not)
        if dev_acc > self.best_dev_overall_performance and not self.opt["no_sm"]:
            self.best_dev_overall_performance = dev_acc
            self.save_weights(os.path.join(self.save_folder, "_".join((self.save_file, "best.pth"))))

    def test(self):
        """Test"""
        # only test mode to load best model
        if not self.opt["debug"] and (self.opt["test"] or self.epoch == self.opt["epochs"]):
            print("loading best model...")
            best_model_files = glob.glob(os.path.join(self.save_folder, "*best.pth"))
            assert best_model_files, "No best model found!"
            self.load_weights(best_model_files[0])
            self.epoch = self.opt["epochs"]
        elif os.path.exists(os.path.join(self.save_folder, "_".join((self.save_file, "ckpt.pth")))):
            self.load_weights(os.path.join(self.save_folder, "_".join((self.save_file, "ckpt.pth"))))

        # calculate overall Acc for dev
        dev_loss, dev_output, dev_feature, dev_target = self._test(self.dev_loader, self.forward)
        dev_acc = utils.compute_Acc_withlogits_nonbinary(dev_output, dev_target)

        # calculate overall Acc for test
        test_loss, test_output, test_feature, test_target = self._test(self.test_loader, self.forward)
        test_acc = utils.compute_Acc_withlogits_nonbinary(test_output, test_target)

        # Output the mean AP for the best model on dev and test set
        wandb.log(
            {
                "test/Dev loss": dev_loss,
                "test/Test loss": test_loss,
                "test/Dev acc": dev_acc,
                "test/Test acc": test_acc,
            }
        )

        if self.opt["test"] or self.epoch == self.opt["epochs"]:
            ft = self.opt["filter_train_mode"]
            data = {"Time": [datetime.datetime.now()], "Var": [self.opt["biased_var"]], f"Our ({ft})": [test_acc * 100]}
            utils.append_data_to_csv(data, os.path.join("result", self.opt["experiment"], self.opt["name"], f"CMNIST_Our ({ft})_trials.csv"))
