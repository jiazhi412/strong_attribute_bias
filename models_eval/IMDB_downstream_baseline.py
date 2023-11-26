import torch
from models_eval.set_IMDB import Model as IMDB
import models_eval.module.basenet as basenet
from models_eval.module import *
import utils
import wandb
import os
import datetime


class Model(IMDB):
    def __init__(self, opt):
        self.opt = opt
        self.epoch = 0
        self.device = opt["device"]
        self.save_folder = opt["save_folder"]
        self.save_file = opt["save_file"]
        self.print_freq = opt["print_freq"]

        self.best_dev_overall_performance = 0.0
        self.set_network(opt)
        self.set_data(opt)
        self.set_optimizer(opt)

    def set_network(self, opt):
        self.hidden_size = 128
        # ========= create models ===========
        if opt["backbone"] == "ResNet50":
            self.predictor = basenet.ResNet50(n_classes=1, hidden_size=self.hidden_size, dropout=opt["dropout"]).to(self.device)
        elif opt["backbone"] == "ResNet18":
            self.predictor = basenet.ResNet18(n_classes=1, hidden_size=self.hidden_size, dropout=opt["dropout"]).to(self.device)
        elif opt["backbone"] == "VGG16":
            self.predictor = basenet.Vgg16(n_classes=1, dropout=opt["dropout"]).to(self.device)

    def _train(self, loader):
        """Train the model for one epoch"""

        self.predictor.train()

        train_loss = 0
        for i, (x, y, a) in enumerate(loader):
            x, y, a = x.to(self.device), y.to(self.device), a.to(self.device)
            self.optim_pred.zero_grad()
            _, outputs = self.forward_pred(x.detach())
            loss = self._criterion_pred(outputs, y)
            loss.backward()
            self.optim_pred.step()
            train_loss += loss.item()
            wandb.log({"train/batch prediction loss": loss.item()})
            if self.print_freq and (i % self.print_freq == 0):
                print("Training epoch {}: [{}|{}], batch loss: {}".format(self.epoch + 1, i + 1, len(loader), loss.item()))
        if self.epoch % 10 == 0:
            utils.log_image("train/origin images", x)
        self.lr_scheduler_pred.step()
        self.epoch += 1

    def _test(self, loader, forward_fn):
        """Compute output on loader"""

        self.predictor.eval()

        test_loss = 0
        output_list = []
        feature_list = []
        target_list = []
        a_list = []
        with torch.no_grad():
            for i, (x, y, a) in enumerate(loader):
                x, y, a = x.to(self.device), y.to(self.device), a.to(self.device)
                z, outputs = forward_fn(x)
                loss = self._criterion_pred(outputs, y)
                test_loss += loss.item()
                output_list.append(outputs)
                feature_list.append(z)
                target_list.append(y)
                a_list.append(a)
        if self.epoch % 10 == 0:
            utils.log_image("test/origin images", x)
        return test_loss / len(loader), torch.cat(output_list), torch.cat(feature_list), torch.cat(target_list), torch.cat(a_list)

    @utils.timer
    def train(self):
        """Train the model for one epoch, evaluate on validation set and save the best model"""
        # resume (maybe)
        if self.opt["resume"]:
            assert os.path.exists(os.path.join(self.save_folder, "_".join((self.save_file, "ckpt.pth")))), "no resume model found!"
            print("Resume from pre-trained model...")
            self.load_weights(os.path.join(self.save_folder, "_".join((self.save_file, "ckpt.pth"))))

        # train the downstream predictor
        self._train(self.train_loader)

        # evaluation
        train_loss, train_output, train_feature, train_target, train_a = self._test(self.train_loader, forward_fn=self.forward_pred)
        train_acc = utils.compute_Acc_withlogits_binary(train_output, train_target)
        dev_loss, dev_output, dev_feature, dev_target, dev_a = self._test(self.dev_loader, forward_fn=self.forward_pred)
        dev_acc = utils.compute_Acc_withlogits_binary(dev_output, dev_target)

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
        if self.opt["test"]:
            self.load_weights(os.path.join(self.save_folder, "_".join((self.save_file, "best.pth"))))
        elif os.path.exists(os.path.join(self.save_folder, "_".join((self.save_file, "ckpt.pth")))):
            self.load_weights(os.path.join(self.save_folder, "_".join((self.save_file, "ckpt.pth"))))

        # calculate overall Acc for dev
        dev_loss, dev_output, dev_feature, dev_target, dev_a = self._test(self.dev_loader, self.forward_pred)
        dev_acc = utils.compute_Acc_withlogits_binary(dev_output, dev_target)

        # calculate overall Acc for test
        test_loss, test_output, test_feature, test_target, test_a = self._test(self.test_loader, self.forward_pred)
        test_acc = utils.compute_Acc_withlogits_binary(test_output, test_target)

        # Output the mean AP for the best model on dev and test set
        wandb.log(
            {
                "test/Dev loss": dev_loss,
                "test/Test loss": test_loss,
                "test/Dev acc": dev_acc,
                "test/Test acc": test_acc,
            }
        )

        if self.epoch == self.opt["epochs"]:
            data = {
                "Time": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                "Train": [self.opt["IMDB_train_mode"]],
                "Test": [self.opt["IMDB_test_mode"]],
                "Test Acc": [test_acc * 100],
            }
            utils.append_data_to_csv(data, os.path.join("result", self.opt["experiment"], self.opt["name"], "IMDB_Baseline_trials.csv"))

    def set_optimizer(self, opt):
        self.optimizer = utils.choose_optimizer(opt["optimizer"])
        self.optim_pred = self.optimizer(params=filter(lambda p: p.requires_grad, self.pred_parameters()), lr=opt["lr"], weight_decay=opt["wd"])
        self.lr_scheduler_pred = torch.optim.lr_scheduler.StepLR(self.optim_pred, step_size=10, gamma=0.1)

    def pred_parameters(self):
        return list(self.predictor.parameters())

    def inference(self, output):
        predict_prob = torch.sigmoid(output)
        return predict_prob.cpu().detach().numpy()

    def _criterion_pred(self, output, target):
        return F.binary_cross_entropy_with_logits(output, target)

    def forward_pred(self, x):
        out, feature = self.predictor(x)
        return feature, out

    def save_weights(self, file_path):
        torch.save(
            {
                "epoch": self.epoch,
                "predictor": self.predictor.state_dict(),
                "optim_pred": self.optim_pred.state_dict(),
            },
            file_path,
        )

    def load_weights(self, file_path):
        ckpt = torch.load(file_path)
        self.epoch = ckpt["epoch"]
        self.predictor.load_state_dict(ckpt["predictor"])
        self.optim_pred.load_state_dict(ckpt["optim_pred"])
