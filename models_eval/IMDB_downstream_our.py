import torch
from models_eval.IMDB_downstream_baseline import Model as IMDB
import models_eval.module.basenet as basenet
from models_eval.module import *
import utils
import wandb
import os
import datetime
import glob


class Model(IMDB):

    def __init__(self, opt):
        self.opt = opt
        self.epoch = 0
        self.device = torch.device(opt['device'])
        self.save_folder = opt["save_folder"]
        self.save_file = opt["save_file"]
        self.print_freq = opt["print_freq"]
        self.IMDB_train_mode = opt['IMDB_train_mode']

        self.best_dev_overall_performance = 0.0
        self.best_test_overall_performance = 0.0
        self.set_network(opt)
        self.set_data(opt)
        self.set_optimizer(opt)
        # load filter
        self.load_filter_weights(self.opt["filter_path"])

    def set_network(self, opt):
        self.hidden_size = 128
        args = opt['filter_parameters']
        self.filter = Generator(
            args.enc_dim, args.enc_layers, args.enc_norm, args.enc_acti,
            args.dec_dim, args.dec_layers, args.dec_norm, args.dec_acti,
            args.n_attrs, args.shortcut_layers, args.inject_layers, args.img_size,
            args.dim_per_attr
        ).to(self.device)
        self.filter.eval()
        if opt['backbone'] == 'ResNet50':
            self.predictor = basenet.ResNet50(n_classes=1, hidden_size=self.hidden_size, dropout=opt['dropout']).to(self.device)
        elif opt['backbone'] == 'ResNet18':
            self.predictor = basenet.ResNet18(n_classes=1, hidden_size=self.hidden_size, dropout=opt['dropout']).to(self.device)
        elif opt['backbone'] == 'VGG16':
            self.predictor = basenet.Vgg16(n_classes=1, dropout=opt['dropout']).to(self.device)

    def load_filter_weights(self, file_path):
        states = torch.load(file_path, map_location=lambda storage, loc: storage)
        if 'module' in list(states['G'].keys())[0]:
            new_state_dict = {'G': {}}
            for key, value in states['G'].items():
                new_key = key.replace('module.', '')  # Remove 'module.' from the key
                new_state_dict['G'][new_key] = value
            states = new_state_dict
        if 'G' in states:
            self.filter.load_state_dict(states['G'])
    
    def forward_filter(self, x, a):
        x_filtered = self.filter(x, a)
        return x_filtered

    def _train(self, loader):
        """Train the model for one epoch"""
        
        self.predictor.train()
        
        train_loss = 0
        for i, (x, y, a) in enumerate(loader):
            x, y, a = x.to(self.device), y.to(self.device), a.to(self.device)

            origin_images = x 
            if self.IMDB_train_mode.endswith('ex'):
                if self.opt['filter_mode'] == 'old':
                    a_old = torch.ones_like(a)
                    x = self.forward_filter(x, a_old)
                elif self.opt['filter_mode'] == 'middle':
                    a_middle = torch.ones_like(a) / 2
                    x = self.forward_filter(x, a_middle)
                elif self.opt['filter_mode'] == 'young':
                    a_young = torch.ones_like(a) * 0
                    x = self.forward_filter(x, a_young)
            else:
                if self.opt['filter_mode'] == 'old':
                    a_old = torch.ones_like(a) * 11
                    x = self.forward_filter(x, a_old)
                elif self.opt['filter_mode'] == 'middle':
                    a = torch.ones_like(a) * 6
                    x = self.forward_filter(x, a)
                elif self.opt['filter_mode'] == 'young':
                    a_young = torch.ones_like(a) * 0
                    x = self.forward_filter(x, a_young)
            
            self.optim_pred.zero_grad()
            _, outputs = self.forward_pred(x.detach())
            loss = self._criterion_pred(outputs, y)
            loss.backward()
            self.optim_pred.step()
            train_loss += loss.item()
            wandb.log({'train/batch prediction loss': loss.item()})
            if self.print_freq and (i % self.print_freq == 0):
                print('Training epoch {}: [{}|{}], batch loss: {}'.format(self.epoch+1, i+1, len(loader), loss.item()))
        if self.epoch % 10 == 0:
            utils.log_image('train/origin images', origin_images)
            utils.log_image('train/filtered images', x)
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
                origin_images = x 

                if self.IMDB_train_mode.endswith('ex'):
                    if self.opt['filter_mode'] == 'old':
                        a_old = torch.ones_like(a)
                        x = self.forward_filter(x, a_old)
                    elif self.opt['filter_mode'] == 'middle':
                        a_middle = torch.ones_like(a) / 2
                        x = self.forward_filter(x, a_middle)
                    elif self.opt['filter_mode'] == 'young':
                        a_young = torch.ones_like(a) * 0
                        x = self.forward_filter(x, a_young)
                else:
                    if self.opt['filter_mode'] == 'old':
                        a_old = torch.ones_like(a) * 11
                        x = self.forward_filter(x, a_old)
                    elif self.opt['filter_mode'] == 'middle':
                        a = torch.ones_like(a) * 6
                        x = self.forward_filter(x, a)
                    elif self.opt['filter_mode'] == 'young':
                        a_young = torch.ones_like(a) * 0
                        x = self.forward_filter(x, a_young)
            
                z, outputs = forward_fn(x)
                loss = self._criterion_pred(outputs, y)
                test_loss += loss.item()
                output_list.append(outputs)
                feature_list.append(z)
                target_list.append(y)
                a_list.append(a)
        if self.epoch % 10 == 0:
            utils.log_image('test/origin images', origin_images)
            utils.log_image('test/filtered images', x)
        return test_loss / len(loader), torch.cat(output_list), torch.cat(feature_list), torch.cat(target_list), torch.cat(a_list)

    @utils.timer
    def train(self):
        """Train the model for one epoch, evaluate on validation set and save the best model"""
        # resume (maybe)
        if self.opt['resume']:
            assert os.path.exists(os.path.join(self.save_folder, '_'.join((self.save_file, "ckpt.pth")))), "no resume model found!"
            print("Resume from pre-trained model...")
            self.load_weights(os.path.join(self.save_folder, '_'.join((self.save_file, "ckpt.pth"))))

        # train the downstream predictor
        self._train(self.train_loader)

        # evaluation
        train_loss, train_output, train_feature, train_target, train_a = self._test(self.train_loader, forward_fn=self.forward_pred)
        train_acc = utils.compute_Acc_withlogits_binary(train_output, train_target)
        dev_loss, dev_output, dev_feature, dev_target, dev_a = self._test(self.dev_loader, forward_fn=self.forward_pred)
        dev_acc = utils.compute_Acc_withlogits_binary(dev_output, dev_target)

        wandb.log({'train/epoch': self.epoch,
                   'train/train loss': train_loss,
                   'train/train acc': train_acc,
                   'train/dev loss': dev_loss,
                   'train/dev acc': dev_acc,
                   })
        print("Finish training epoch {}, {}: {}, {}: {}".format(self.epoch, "Train Loss", train_loss, "Dev Loss", dev_loss))

        # save model after one single epoch (maybe not)
        if not self.opt['no_sm']:
            self.save_weights(os.path.join(self.save_folder, '_'.join((self.save_file, "ckpt.pth"))))
        
        # save best model (maybe not)
        if dev_acc > self.best_dev_overall_performance and not self.opt['no_sm']:
            self.best_dev_overall_performance = dev_acc
            self.save_weights(os.path.join(self.save_folder, '_'.join((self.save_file, "best.pth"))))
    
    def test(self):
        """Test"""
        # only test mode to load best model
        if not self.opt['debug'] and (self.opt["test"] or self.epoch == self.opt["epochs"]):
            print("loading best model...")
            best_model_files = glob.glob(os.path.join(self.save_folder, '*best.pth'))
            assert best_model_files, "No best model found!"
            self.load_weights(best_model_files[0])
            self.epoch = self.opt["epochs"]
        elif os.path.exists(os.path.join(self.save_folder, '_'.join((self.save_file, "ckpt.pth")))):
            self.load_weights(os.path.join(self.save_folder, '_'.join((self.save_file, "ckpt.pth"))))

        # calculate overall Acc for dev
        dev_loss, dev_output, dev_feature, dev_target, dev_a = self._test(self.dev_loader, self.forward_pred)
        dev_acc = utils.compute_Acc_withlogits_binary(dev_output, dev_target)

        # calculate overall Acc for test
        test_loss, test_output, test_feature, test_target, test_a = self._test(self.test_loader, self.forward_pred)
        test_acc = utils.compute_Acc_withlogits_binary(test_output, test_target)

        # Output the mean AP for the best model on dev and test set
        wandb.log({
                   'test/Dev loss': dev_loss,
                   'test/Test loss': test_loss,
                   'test/Dev acc': dev_acc,
                   'test/Test acc': test_acc,
                   }) 

        if self.opt['test'] or self.epoch == self.opt['epochs']:
            ft = self.opt['filter_train_mode'] 
            data = {
                'Time': [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                'Train': [self.opt['IMDB_train_mode']],
                'Test': [self.opt['IMDB_test_mode']],
                'Test Acc': [test_acc * 100],
                }
            utils.append_data_to_csv(data, os.path.join("result", self.opt["experiment"], self.opt["name"], f'IMDB_Our ({ft})_trials.csv'))
            