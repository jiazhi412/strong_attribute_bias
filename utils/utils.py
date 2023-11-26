# Copyright (C) 2019 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>

"""Helper functions"""

import os
from glob import glob
import pickle
import torch


def find_model(path, epoch="latest"):
    if epoch == "latest":
        files = glob(os.path.join(path, "*.pth"))
        file = sorted(files, key=lambda x: int(x.rsplit(".", 2)[1]))[-1]
    else:
        file = os.path.join(path, "weights.{:d}.pth".format(int(epoch)))
    assert os.path.exists(file), "File not found: " + file
    print("Find model of {} epoch: {}".format(epoch, file))
    return file


def load_pkl(load_path):
    with open(load_path, "rb") as f:
        pkl_data = pickle.load(f)
    return pkl_data


def nextbatch_with_dummy(it, loader):
    try:
        x, _, target = next(it)
    except StopIteration:
        it = iter(loader)
        x, _, target = next(it)
    return x, target, it


def nextbatch(it, loader):
    try:
        x, target = next(it)
    except StopIteration:
        it = iter(loader)
        x, target = next(it)
    return x, target, it


def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
    labels: (LongTensor) class labels, sized [N,].
    num_classes: (int) number of classes.

    Returns:
    (tensor) encoded labels, sized [N, #classes].
    """
    labels = labels.type(torch.LongTensor)
    y = torch.eye(num_classes)
    res = y[labels].squeeze_()
    return res
