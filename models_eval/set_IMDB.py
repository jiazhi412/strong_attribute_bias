import h5py
import torch
import torch.nn as nn
from dataloader.IMDB import IMDBDataset
import utils
import torchvision.transforms as transforms
import os


class Model:
    def _transform(self, opt):
        # normalize according to ImageNet
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)

        transform_train = transforms.Compose(
            [
                transforms.Resize(opt["image_size"]),
                transforms.CenterCrop(180),
                transforms.Resize(opt["image_size"]),
                transforms.RandomHorizontalFlip() if ["augment"] else nn.Identity(),
                transforms.ToTensor(),
                normalize if opt["normalize"] else nn.Identity(),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize(opt["image_size"]),
                transforms.CenterCrop(180),
                transforms.Resize(opt["image_size"]),
                transforms.ToTensor(),
                normalize if opt["normalize"] else nn.Identity(),
            ]
        )
        return transform_train, transform_test

    def set_data(self, opt):
        """Set up the dataloaders"""

        data_folder = {
            "origin_image_feature_path": os.path.join(opt["data_folder"], "IMDB.h5py"),
            "origin_target_dict_path": os.path.join(opt["data_folder"], "age_dict"),
            "origin_sex_dict_path": os.path.join(opt["data_folder"], "sex_dict"),
            "origin_eb1_key_list_path": os.path.join(opt["data_folder"], "eb1_img_list"),
            "origin_eb2_key_list_path": os.path.join(opt["data_folder"], "eb2_img_list"),
            "origin_unbiased_key_list_path": os.path.join(opt["data_folder"], "test_img_list"),
        }

        image_feature_path = data_folder["origin_image_feature_path"]
        image_feature = h5py.File(data_folder["origin_image_feature_path"], "r")
        target_dict = utils.load_pkl(data_folder["origin_target_dict_path"])
        sex_dict = utils.load_pkl(data_folder["origin_sex_dict_path"])
        eb1_key_list = utils.load_pkl(data_folder["origin_eb1_key_list_path"])
        eb2_key_list = utils.load_pkl(data_folder["origin_eb2_key_list_path"])
        unbiased_key_list = utils.load_pkl(data_folder["origin_unbiased_key_list_path"])

        transform_train, transform_test = self._transform(opt)
        train_set = IMDBDataset(image_feature_path, target_dict, sex_dict, opt["IMDB_train_mode"], eb1_key_list, eb2_key_list, unbiased_key_list, "train", transform_train)
        dev_set = IMDBDataset(image_feature_path, target_dict, sex_dict, opt["IMDB_test_mode"], eb1_key_list, eb2_key_list, unbiased_key_list, "dev", transform_test)
        test_set = IMDBDataset(image_feature_path, target_dict, sex_dict, opt["IMDB_test_mode"], eb1_key_list, eb2_key_list, unbiased_key_list, "dev_test", transform_test)

        # small piece for debug
        if opt["debug"]:
            num_train = 200
            num_test = 20
            train_set, dev_set, test_set, _ = torch.utils.data.random_split(train_set, [num_train, num_test, num_test, len(train_set) - num_train - num_test * 2])

        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=opt["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
        self.dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=opt["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=opt["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
