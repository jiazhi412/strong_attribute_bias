import numpy as np
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import utils
import h5py


class CelebA_synthetic(data.Dataset):
    def __init__(self, data_path, attr_path, syn_data_path, image_size, mode, selected_attrs, attrs_d):
        super(CelebA_synthetic, self).__init__()
        self.data_path = data_path
        att_list = open(attr_path, "r", encoding="utf-8").readlines()[1].split()
        atts = [att_list.index(att) + 1 for att in selected_attrs]
        images = np.loadtxt(attr_path, skiprows=2, usecols=[0], dtype=np.str)
        labels = np.loadtxt(attr_path, skiprows=2, usecols=atts, dtype=np.int)
        atts_d = [att_list.index(att) + 1 for att in attrs_d]
        labels_d = np.loadtxt(attr_path, skiprows=2, usecols=atts_d, dtype=np.int)

        if mode == "train":
            self.origin_images = images[:182000]
            self.origin_labels = labels[:182000]
            self.origin_labels_d = labels_d[:182000]
        if mode == "valid":
            self.origin_images = images[182000:182637]
            self.origin_labels = labels[182000:182637]
            self.origin_labels_d = labels_d[182000:182637]
        if mode == "test":
            self.origin_images = images[182637:]
            self.origin_labels = labels[182637:]
            self.origin_labels_d = labels_d[182637:]
        if mode == "all":
            self.origin_images = images
            self.origin_labels = labels
            self.origin_labels_d = labels_d

        # get statistics of origin dataset
        s = statistics(self.origin_labels, self.origin_labels_d)

        # Load synthetic data
        self.syn_data_path = syn_data_path
        self.syn_target_dict = utils.load_pkl(os.path.join(syn_data_path, "target_dict"))

        # supplement with synthetic dataset
        self.key_list = use_syn(s, self.syn_target_dict)

        s_syn = statistics_syn(self.key_list, self.syn_target_dict)
        print(f"Origin dataset: {s}")
        print(f"size of synthetic to supplement: {len(self.key_list)}")
        print(f"Sythetic dataset: {s_syn}")

        self.tf = transforms.Compose(
            [
                transforms.CenterCrop(148),
                transforms.Resize(image_size),
                transforms.ToTensor(),
                # normalize,
            ]
        )

    def __getitem__(self, index):
        # origin
        if index < self.origin_images.shape[0]:
            img = self.tf(Image.open(os.path.join(self.data_path, self.origin_images[index])))
            att = torch.tensor((self.origin_labels[index] + 1) // 2)
        else:  # synthetic
            syn_images = h5py.File(os.path.join(self.syn_data_path, "CelebA.h5py"), "r")
            key = self.key_list[index - self.origin_images.shape[0]]
            img = self.tf(Image.fromarray(syn_images[key][()]))
            att = torch.tensor(self.syn_target_dict[key][0])
        return img, att

    def __len__(self):
        return self.origin_images.shape[0] + len(self.key_list)


def statistics(pa, y):
    pp, pn, np, nn = 0, 0, 0, 0
    for i in range(y.shape[0]):
        if pa[i] == 1 and y[i] == 1:
            pp += 1
        elif pa[i] == 1 and y[i] == -1:
            pn += 1
        elif pa[i] == -1 and y[i] == 1:
            np += 1
        elif pa[i] == -1 and y[i] == -1:
            nn += 1
    return [pp, pn, np, nn]


def statistics_syn(key_list, target_dict):
    pp, pn, np, nn = 0, 0, 0, 0
    for k in key_list:
        if target_dict[k][0] == 1 and target_dict[k][1] == 1:
            pp += 1
        elif target_dict[k][0] == 1 and target_dict[k][1] == 0:
            pn += 1
        elif target_dict[k][0] == 0 and target_dict[k][1] == 1:
            np += 1
        elif target_dict[k][0] == 0 and target_dict[k][1] == 0:
            nn += 1
    return [pp, pn, np, nn]


def use_syn(s, target_dict):
    m = max(s)
    pp, pn, np, nn = s.copy()
    key_list = []
    for key, value in target_dict.items():
        # label sample: [1,0,1] male, black_hair, wavy_hair
        if pp < m and value[0] == 1 and value[1] == 1:
            pp += 1
            key_list.append(key)
        elif pn < m and value[0] == 1 and value[1] == 0:
            pn += 1
            key_list.append(key)
        elif np < m and value[0] == 0 and value[1] == 1:
            np += 1
            key_list.append(key)
        elif nn < m and value[0] == 0 and value[1] == 0:
            nn += 1
            key_list.append(key)
    return key_list
