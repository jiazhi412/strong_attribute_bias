import h5py
import torch
import torch.nn as nn
from dataloader.CelebA_eval import CelebADataset
import torchvision.transforms as transforms
import utils
import os
from dataloader.FFHQ import FFHQ


class Model:
    def _transform(self, opt):
        if opt["backbone"].startswith("AdaFace"):
            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        else:
            # normalize according to ImageNet
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        transform_train = transforms.Compose(
            [
                transforms.CenterCrop(148),
                transforms.Resize(opt["image_size"]),
                transforms.RandomHorizontalFlip() if ["augment"] else nn.Identity(),
                transforms.ToTensor(),
                normalize if opt["normalize"] else nn.Identity(),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.CenterCrop(148),
                transforms.Resize(opt["image_size"]),
                transforms.ToTensor(),
                normalize if opt["normalize"] else nn.Identity(),
            ]
        )
        return transform_train, transform_test

    def set_data(self, opt):
        """Set up the dataloaders"""

        data_folder = {
            "origin_image_feature_path": os.path.join(opt["data_folder"], "CelebA.h5py"),
            "origin_target_dict_path": os.path.join(opt["data_folder"], "labels_dict"),
            "origin_sex_dict_path": os.path.join(opt["data_folder"], "sex_dict"),
            "origin_train_key_list_path": os.path.join(opt["data_folder"], "train_key_list"),
            "origin_dev_key_list_path": os.path.join(opt["data_folder"], "dev_key_list"),
            "origin_test_key_list_path": os.path.join(opt["data_folder"], "test_key_list"),
        }

        image_feature = h5py.File(data_folder["origin_image_feature_path"], "r")
        target_dict = utils.load_pkl(data_folder["origin_target_dict_path"])
        sex_dict = utils.load_pkl(data_folder["origin_sex_dict_path"])
        train_key_list = utils.load_pkl(data_folder["origin_train_key_list_path"])
        dev_key_list = utils.load_pkl(data_folder["origin_dev_key_list_path"])
        test_key_list = utils.load_pkl(data_folder["origin_test_key_list_path"])
        attribute_list = utils.get_attr_index(opt["attrs_pred"])
        target_dict = utils.transfer_origin_for_testing_only(target_dict, attribute_list)

        # modify dev and test to unbiased and bias conflict
        train_key_list = utils.CelebA_eval_mode(train_key_list, target_dict, sex_dict, mode=opt["CelebA_test_mode"], train_or_test="train", n_bc=opt["n_bc"], p_bc=opt["p_bc"], balance=opt["balance"])
        self.ce = utils.conditional_entropy(train_key_list, target_dict, sex_dict)
        utils.entropy(train_key_list, target_dict, sex_dict)
        dev_key_list = utils.CelebA_eval_mode(dev_key_list, target_dict, sex_dict, mode=opt["CelebA_test_mode"], train_or_test="dev")
        test_key_list = utils.CelebA_eval_mode(test_key_list, target_dict, sex_dict, mode=opt["CelebA_test_mode"], train_or_test="test")

        transform_train, transform_test = self._transform(opt)
        train_set = CelebADataset(train_key_list, image_feature, target_dict, sex_dict, transform_train)
        dev_set = CelebADataset(dev_key_list, image_feature, target_dict, sex_dict, transform_test)
        test_set = CelebADataset(test_key_list, image_feature, target_dict, sex_dict, transform_test)

        # small piece for debug
        if opt["debug"]:
            num_train = 200
            num_test = 20
            train_set, dev_set, test_set, _ = torch.utils.data.random_split(train_set, [num_train, num_test, num_test, len(train_key_list) - num_train - num_test * 2])

        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=opt["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
        self.dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=opt["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=opt["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
