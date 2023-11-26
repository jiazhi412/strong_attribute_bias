import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np_
import dataloader.Adult_data_utils as utils
import random
import os


class AdultDataset(Dataset):
    def __init__(self, path, mode, quick_load, bias_name=None, training_number=40000, p_bc=-1, n_bc=-1, balance=True, pseudo_label_path=None):
        self.bias_name = bias_name
        self.features, self.labels_onehotv, self.bias_onehotv = self.load_data(path, quick_load)
        self.labels = self.onehotvector_to_label(self.labels_onehotv)
        self.bias = self.onehotvector_to_label(self.bias_onehotv)

        pp, pn, np, nn = utils.split_by_attr(self.features, self.labels, self.bias)
        minimum = min(len(pp), len(pn), len(np), len(nn))
        print("Before", len(pp), len(pn), len(np), len(nn))

        if mode == "train":
            self.features = self.features[:training_number]
            self.labels = self.labels[:training_number]
            self.bias = self.bias[:training_number]
        elif mode == "validation":
            self.features = self.features[training_number:]
            self.labels = self.labels[training_number:]
            self.bias = self.bias[training_number:]
        elif mode == "all":
            pass
        else:
            if mode == "eb1":
                key_list = pp + nn
            elif mode == "eb2":
                key_list = np + pn
            elif mode == "eb1_balanced":
                key_list = np_.array(pp)[self._select(len(pp), minimum)].tolist() + np_.array(nn)[self._select(len(nn), minimum)].tolist()
                if p_bc != -1:
                    key_list = np_.array(pp)[self._select(len(pp), minimum)].tolist() + np_.array(nn)[self._select(len(nn), minimum)].tolist() + np_.array(pn)[self._select(len(pn), int(minimum * p_bc))].tolist() + np_.array(np)[self._select(len(np), int(minimum * p_bc))].tolist()
            elif mode == "eb2_balanced":
                key_list = np_.array(pn)[self._select(len(pn), minimum)].tolist() + np_.array(np)[self._select(len(np), minimum)].tolist()
                if p_bc != -1:
                    key_list = np_.array(pn)[self._select(len(pn), minimum)].tolist() + np_.array(np)[self._select(len(np), minimum)].tolist() + np_.array(pp)[self._select(len(pp), int(minimum * p_bc))].tolist() + np_.array(nn)[self._select(len(nn), int(minimum * p_bc))].tolist()
            elif mode == "balanced":
                key_list = np_.array(pn)[self._select(len(pn), minimum)].tolist() + np_.array(np)[self._select(len(np), minimum)].tolist() + np_.array(pp)[self._select(len(pp), minimum)].tolist() + np_.array(nn)[self._select(len(nn), minimum)].tolist()
            self._sift(key_list)

        pp, pn, np, nn = utils.split_by_attr(self.features, self.labels, self.bias)
        self.ce = utils.conditional_entropy(self.features, self.labels, self.bias)
        print("After", len(pp), len(pn), len(np), len(nn))

    def _select(self, max, n):
        return random.sample(range(0, max), n)

    def get_ce(self):
        return self.ce

    def _sift(self, key_list):
        self.features = self.features[key_list]
        self.labels = self.labels[key_list]
        self.bias = self.bias[key_list]

    def __getitem__(self, i):
        feature = self.features[i]
        label = self.labels[i]
        label = np_.array(label)
        bias = self.bias[i]
        return feature, label, bias

    def __len__(self):
        data_len = self.features.shape[0]
        return data_len

    def load_data(self, path, quick_load):
        if quick_load:
            data = utils.quick_load(path)
        else:
            data = utils.data_processing(path)
        data = self.data_preproccess(data)
        features = data[:, : data.shape[1] - 2]
        labels = data[:, data.shape[1] - 2 :]
        bias = utils.get_bias(data, self.bias_name)
        return features, labels, bias

    def data_preproccess(self, data):
        data = torch.tensor(data, dtype=torch.float)
        return data

    def onehotvector_to_label(self, onehotvector):
        labels = []
        for v in onehotvector:
            for i in range(v.shape[0]):
                if v[i] != 0:
                    label = i
                    labels.append(label)
                    break
        labels = torch.tensor(labels).unsqueeze_(1)
        return labels
