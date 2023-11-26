import numpy as np
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


class CelebA(data.Dataset):
    def __init__(self, data_path, attr_path, image_size, mode, selected_attrs, attrs_d=None):
        super(CelebA, self).__init__()
        self.data_path = data_path
        att_list = open(attr_path, "r", encoding="utf-8").readlines()[1].split()
        atts = [att_list.index(att) + 1 for att in selected_attrs]
        images = np.loadtxt(attr_path, skiprows=2, usecols=[0], dtype=np.str)
        labels = np.loadtxt(attr_path, skiprows=2, usecols=atts, dtype=np.int)
        if attrs_d != None:
            atts_d = [att_list.index(att) + 1 for att in attrs_d]
            labels_d = np.loadtxt(attr_path, skiprows=2, usecols=atts_d, dtype=np.int)

        if mode == "train":
            self.images = images[:182000]
            self.labels = labels[:182000]
        if mode == "valid":
            self.images = images[182000:182637]
            self.labels = labels[182000:182637]
        if mode == "test":
            self.images = images[182637:]
            self.labels = labels[182637:]
        if mode == "all":
            self.images = images
            self.labels = labels
        if mode == "ex":
            pp, pn, npl, nn = split_by_attr(labels, labels_d)
            self.images = images[pn + nn]
            self.labels = labels[pn + nn]

        self.tf = transforms.Compose(
            [
                transforms.CenterCrop(148),
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ]
        )
        self.length = len(self.images)

    def __getitem__(self, index):
        img = self.tf(Image.open(os.path.join(self.data_path, self.images[index])))
        att = torch.tensor((self.labels[index] + 1) // 2)
        return img, att

    def __len__(self):
        return self.length


def split_by_attr(labels, labels_d):
    pp, pn, npl, nn = [], [], [], []
    for k, (i, j) in enumerate(zip(labels, labels_d)):
        if i == 1 and j == 1:
            pp.append(k)
        elif i == 1 and j == -1:
            pn.append(k)
        elif i == -1 and j == 1:
            npl.append(k)
        elif i == -1 and j == -1:
            nn.append(k)
    return pp, pn, npl, nn


def check_attribute_conflict(att_batch, att_name, att_names):
    def _get(att, att_name):
        if att_name in att_names:
            return att[att_names.index(att_name)]
        return None

    def _set(att, value, att_name):
        if att_name in att_names:
            att[att_names.index(att_name)] = value

    att_id = att_names.index(att_name)
    for att in att_batch:
        if att_name in ["Bald", "Receding_Hairline"] and att[att_id] != 0:
            if _get(att, "Bangs") != 0:
                _set(att, 1 - att[att_id], "Bangs")
        elif att_name == "Bangs" and att[att_id] != 0:
            for n in ["Bald", "Receding_Hairline"]:
                if _get(att, n) != 0:
                    _set(att, 1 - att[att_id], n)
                    _set(att, 1 - att[att_id], n)
        elif att_name in ["Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair"] and att[att_id] != 0:
            for n in ["Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair"]:
                if n != att_name and _get(att, n) != 0:
                    _set(att, 1 - att[att_id], n)
        elif att_name in ["Straight_Hair", "Wavy_Hair"] and att[att_id] != 0:
            for n in ["Straight_Hair", "Wavy_Hair"]:
                if n != att_name and _get(att, n) != 0:
                    _set(att, 1 - att[att_id], n)
        elif att_name in ["Mustache", "No_Beard"] and att[att_id] != 0:
            for n in ["Mustache", "No_Beard"]:
                if n != att_name and _get(att, n) != 0:
                    _set(att, 1 - att[att_id], n)
    return att_batch
