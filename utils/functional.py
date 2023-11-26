# Copyright (C) 2023 Jiazhi Li <jiazhil412@gmail.com>
#
# This work is licensed under the MIT License. To view a copy of this license,
# visit https://opensource.org/licenses/MIT.

import time
import torch
import torch.nn as nn
import itertools
import wandb
import torchvision
import pandas as pd
import os
from sklearn.metrics import average_precision_score
import numpy as np
from sklearn import metrics


def append_data_to_csv(data, csv_name):
    df = pd.DataFrame(data)
    if os.path.exists(csv_name):
        df.to_csv(csv_name, mode="a", index=False, header=False)
    else:
        df.to_csv(csv_name, index=False)


def timer(f):
    def wrapper(*args, **kwargs):
        start = time.time()
        rv = f(*args, **kwargs)
        total = time.time() - start
        total = convert_to_preferred_format(total)
        print(f"Time: {total}")
        return rv

    return wrapper


def convert_to_preferred_format(sec):
    sec = sec % (24 * 3600)
    hour = sec // 3600
    sec %= 3600
    min = sec // 60
    sec %= 60
    return "%02d:%02d:%02d" % (hour, min, sec)


def choose_optimizer(name):
    if name == "Adam":
        res = torch.optim.Adam
    return res


def training_strategy(num_adv, num_main):
    return itertools.cycle(["main" if i == 0 else "adv" for i in [0] * num_main + [1] * num_adv])


def nextbatch_withz(it, loader):
    try:
        x, target, z = next(it)
    except StopIteration:
        it = iter(loader)
        x, target, z = next(it)
    return x, target, z, it


def log_image(name, data):
    wandb.log({name: wandb.Image(torchvision.utils.make_grid(data, nrow=10, padding=0, normalize=False))})


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


def compute_Acc_onehot(predict_prob, target):
    predict_prob = predict_prob.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    assert predict_prob.shape == target.shape, "Acc, shape are not matched!"
    Acc_per_class = (predict_prob.round() == target).mean(axis=0)
    Acc = Acc_per_class.mean()
    return Acc_per_class, Acc


def compute_Acc_withlogits_binary(logits, target):
    assert logits.shape == target.shape, f"Acc, output {logits.shape} and target {target.shape} are not matched!"
    # output is logits and predict_prob is probability
    predict_prob = torch.sigmoid(logits)
    predict_prob = predict_prob.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    Acc_per_class = (predict_prob.round() == target).mean(axis=0)
    Acc = Acc_per_class.mean()
    return Acc


def compute_MI_withlogits(logits, a):
    assert logits.shape == a.shape, f"MI, output {logits.shape} and target {a.shape} are not matched!"
    # output is logits and predict_prob is probability
    predict_prob = torch.sigmoid(logits)
    predict_prob = predict_prob.cpu().detach().numpy()
    a = a.cpu().detach().numpy()
    mi = metrics.mutual_info_score(predict_prob.round().squeeze(), a.squeeze())
    return mi


def compute_MI(y, a):
    y = y.cpu().detach().numpy()
    a = a.cpu().detach().numpy()
    mi = metrics.mutual_info_score(y.squeeze(), a.squeeze())
    return mi


def compute_Acc_withlogits_nonbinary(logits, target):
    # output is logits and predict_prob is log probability
    logsoftmax = torch.nn.LogSoftmax(dim=1)
    predict_prob = logsoftmax(logits)
    n_correct = 0
    pred = predict_prob.max(1)[1]
    n_correct = int(pred.eq(target.view_as(pred)).sum().cpu().detach())
    return n_correct / target.shape[0]


def compute_Acc_withlogits_binary_onehot(logits, target):
    assert logits.shape == target.shape, f"Acc, output {logits.shape} and target {target.shape} are not matched!"
    # output is logits and predict_prob is probability
    logsoftmax = torch.nn.LogSoftmax(dim=1)
    predict_prob = logsoftmax(logits)

    n_correct = 0
    n_wrong = 0
    for i in range(predict_prob.size()[0]):
        if torch.argmax(predict_prob[i]) == torch.argmax(target[i]):
            n_correct += 1
        else:
            n_wrong += 1
    Acc = (n_correct * 1.0) / (n_correct + n_wrong)
    return Acc


def compute_class_weight(target):
    domain_label = target[:, -1]
    per_class_weight = []
    for i in range(target.shape[1]):
        class_label = target[:, i]
        cp = class_label.sum()  # class is positive
        cn = target.shape[0] - cp  # class is negative
        cn_dn = ((class_label + domain_label) == 0).sum()  # class is negative, domain is negative
        cn_dp = ((class_label - domain_label) == -1).sum()
        cp_dn = ((class_label - domain_label) == 1).sum()
        cp_dp = ((class_label + domain_label) == 2).sum()
        per_class_weight.append((class_label * cp + (1 - class_label) * cn) / (2 * ((1 - class_label) * (1 - domain_label) * cn_dn + (1 - class_label) * domain_label * cn_dp + class_label * (1 - domain_label) * cp_dn + class_label * domain_label * cp_dp)))
    return per_class_weight


def compute_weighted_AP(target, predict_prob, class_weight_list):
    per_class_AP = []
    for i in range(target.shape[1]):
        class_weight = target[:, i] * class_weight_list[i] + (1 - target[:, i]) * np.ones(class_weight_list[i].shape)
        per_class_AP.append(average_precision_score(target[:, i], predict_prob[:, i], sample_weight=class_weight))
    return per_class_AP


def compute_AP(target, predict_prob):
    per_class_AP = []
    for i in range(target.shape[1]):
        per_class_AP.append(average_precision_score(target[:, i], predict_prob[:, i]))
    return per_class_AP


def compute_Acc(predict_prob, target):
    assert predict_prob.shape == target.shape, "Acc, shape are not matched!"
    Acc_per_class = (predict_prob.round() == target).mean(axis=0)
    Acc = Acc_per_class.mean()
    return Acc_per_class, Acc


def compute_mAP(per_class_AP, subclass_idx):
    return np.mean([per_class_AP[idx] for idx in subclass_idx])
