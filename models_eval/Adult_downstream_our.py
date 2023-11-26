import os
import torch
import utils
import wandb
from models_eval.Adult_downstream_baseline import Model as Adult
from models_eval.module import *
import datetime
import glob


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
        z = torch.cat([zs[-1], a], dim=1)
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


class Model(Adult):
    def __init__(self, opt):
        super(Model, self).__init__(opt)
        # load filter
        self.load_filter_weights(self.opt["filter_path"])

    def set_network(self, opt):
        # Filter
        in_dim = 108
        enc_dims = [64, 10]
        dec_dims = [64]
        out_dim = 108
        self.filter = Generator(in_dim, enc_dims, dec_dims, out_dim).to(self.device)

        # Predictor
        in_dim = 108
        hidden_dims = [32, 10]
        out_dim = 1
        self.predictor = MLP(in_dim, hidden_dims, out_dim).to(self.device)

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

        self.predictor.train()

        train_loss = 0
        for i, (x, y, a) in enumerate(loader):
            x, y, a = x.to(self.device), y.to(self.device), a.to(self.device)

            if self.opt["filter_mode"] == "reverse":
                a_reverse = utils.reverse_sex(a, 0)
                reverse_images = self.forward_filter(x, a_reverse)
                x_filtered = (x + reverse_images) / 2
            elif self.opt["filter_mode"] == "neutral":
                a_neutral = utils.neutral_sex(a, 0)
                x_filtered = self.forward_filter(x, a_neutral)
            elif self.opt["filter_mode"] == "male":
                a_male = utils.to_male(a, 0)
                x_filtered = self.forward_filter(x, a_male)
            elif self.opt["filter_mode"] == "female":
                a_female = utils.to_female(a, 0)
                x_filtered = self.forward_filter(x, a_female)

            self.optim_pred.zero_grad()
            _, outputs = self.forward_pred(x_filtered.detach())
            loss = self._criterion_pred(outputs, y.float())
            loss.backward()
            self.optim_pred.step()
            train_loss += loss.item()
            wandb.log({"train/batch prediction loss": loss.item()})
            if self.print_freq and (i % self.print_freq == 0):
                print("Training epoch {}: [{}|{}], batch loss: {}".format(self.epoch + 1, i + 1, len(loader), loss.item()))
        self.lr_scheduler_pred.step()
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

                if self.opt["filter_mode"] == "reverse":
                    a_reverse = utils.reverse_sex(a, 0)
                    reverse_images = self.forward_filter(x, a_reverse)
                    x = (origin_images + reverse_images) / 2
                elif self.opt["filter_mode"] == "neutral":
                    a_neutral = utils.neutral_sex(a, 0)
                    x = self.forward_filter(x, a_neutral)
                elif self.opt["filter_mode"] == "male":
                    a_male = utils.to_male(a, 0)
                    x = self.forward_filter(x, a_male)
                elif self.opt["filter_mode"] == "female":
                    a_female = utils.to_female(a, 0)
                    x = self.forward_filter(x, a_female)

                _, outputs = forward_fn(x)
                loss = self._criterion_pred(outputs, y.float())
                test_loss += loss.item()
                output_list.append(outputs)
                target_list.append(y)
        return test_loss / len(loader), torch.cat(output_list), torch.cat(target_list)

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
        train_loss, train_output, train_target = self._test(self.train_loader, forward_fn=self.forward_pred)
        train_acc = utils.compute_Acc_withlogits_binary(train_output, train_target)
        dev_loss, dev_output, dev_target = self._test(self.dev_loader, forward_fn=self.forward_pred)
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
        if not self.opt['debug'] and (self.opt["test"] or self.epoch == self.opt["epochs"]):
            print("loading best model...")
            best_model_files = glob.glob(os.path.join(self.save_folder, '*best.pth'))
            assert best_model_files, "No best model found!"
            self.load_weights(best_model_files[0])
            self.epoch = self.opt["epochs"]
        elif os.path.exists(os.path.join(self.save_folder, "_".join((self.save_file, "ckpt.pth")))):
            self.load_weights(os.path.join(self.save_folder, "_".join((self.save_file, "ckpt.pth"))))

        # calculate overall Acc for dev
        dev_loss, dev_output, dev_target = self._test(self.dev_loader, self.forward_pred)
        dev_acc = utils.compute_Acc_withlogits_binary(dev_output, dev_target)

        # calculate overall Acc for test
        test_loss, test_output, test_target = self._test(self.test_loader, self.forward_pred)
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

        if self.opt["test"] or self.epoch == self.opt["epochs"]:
            ft = self.opt["filter_train_mode"]
            data = {"Time": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")], "Train": [self.opt["Adult_train_mode"]], "Test": [self.opt["Adult_test_mode"]], "H(Y|A)": [self.ce], "p_bc": [self.opt["p_bc"]], "Test Acc": [test_acc * 100]}
            utils.append_data_to_csv(data, os.path.join("result", self.opt["experiment"], self.opt["name"], f"Adult_Ours ({ft})_trials.csv"))
