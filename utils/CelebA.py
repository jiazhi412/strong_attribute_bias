# Copyright (C) 2023 Jiazhi Li <jiazhil412@gmail.com>
#
# This work is licensed under the MIT License. To view a copy of this license,
# visit https://opensource.org/licenses/MIT.

import numpy as np
from collections import Counter
import torch
import utils


def split_by_attr(key_list, target_dict, sex_dict):
    pp, pn, np, nn = [], [], [], []
    for k in key_list:
        if sex_dict[k] == 1 and target_dict[k] == 1:
            pp.append(k)
        elif sex_dict[k] == 1 and target_dict[k] == 0:
            pn.append(k)
        elif sex_dict[k] == 0 and target_dict[k] == 1:
            np.append(k)
        elif sex_dict[k] == 0 and target_dict[k] == 0:
            nn.append(k)
    return pp, pn, np, nn


def sub_accuracy_by_attr(outputs, targets, pa):
    outputs_pp, outputs_pn, outputs_np, outputs_nn = [], [], [], []
    targets_pp, targets_pn, targets_np, targets_nn = [], [], [], []
    for k in range(outputs.size(0)):
        if pa[k] == 1 and targets[k] == 1:
            outputs_pp.append(outputs[k])
            targets_pp.append(targets[k])
        elif pa[k] == 1 and targets[k] == 0:
            outputs_pn.append(outputs[k])
            targets_pn.append(targets[k])
        elif pa[k] == 0 and targets[k] == 1:
            outputs_np.append(outputs[k])
            targets_np.append(targets[k])
        elif pa[k] == 0 and targets[k] == 0:
            outputs_nn.append(outputs[k])
            targets_nn.append(targets[k])
    # statistics
    print(f"(Male-BlondHair) pp = {len(outputs_pp)}, pn = {len(outputs_pn)}, np = {len(outputs_np)}, nn = {len(outputs_nn)}")
    # accuracy
    acc_pp = utils.compute_Acc_withlogits_binary(torch.cat(outputs_pp), torch.cat(targets_pp))
    acc_pn = utils.compute_Acc_withlogits_binary(torch.cat(outputs_pn), torch.cat(targets_pn))
    acc_np = utils.compute_Acc_withlogits_binary(torch.cat(outputs_np), torch.cat(targets_np))
    acc_nn = utils.compute_Acc_withlogits_binary(torch.cat(outputs_nn), torch.cat(targets_nn))
    print(f"Accuracy pp = {acc_pp}, pn = {acc_pn}, np = {acc_np}, nn = {acc_nn}")


def CelebA_eval_mode(key_list, target_dict, sex_dict, mode, train_or_test, n_bc=-1, p_bc=-1, balance=False):
    pp, pn, np, nn = split_by_attr(key_list, target_dict, sex_dict)
    print(f"[{mode}] {train_or_test}: (Male-BlondHair) pp = {len(pp)}, pn = {len(pn)}, np = {len(np)}, nn = {len(nn)}")
    m = min(len(pp), len(pn), len(np), len(nn))
    if train_or_test == "test" or train_or_test == "dev":
        if mode.startswith("unbiased"):
            r_key_list = pp[:m] + pn[:m] + np[:m] + nn[:m]
        elif mode.startswith("conflict") and not mode.startswith("conflict_pp"):
            r_key_list = pp[:m] + nn[:m]
        elif mode.startswith("conflict_pp"):
            r_key_list = pp
    elif train_or_test == "train":
        if n_bc != -1:
            r_key_list = pn + np + pp[: n_bc // 2] + nn[: n_bc // 2]
        elif p_bc != -1:
            if balance:
                r_key_list = pn + np + pp[: int(m * p_bc)] + nn[: int(m * p_bc)]
            else:
                r_key_list = pn + np + pp[: int(len(pp) * p_bc)] + nn[: int(len(nn) * p_bc)]
        else:
            if mode == "unbiased_ex" or mode == "conflict_ex" or mode == "conflict_pp_ex":
                r_key_list = pn + np
            else:
                r_key_list = key_list
    pp, pn, np, nn = split_by_attr(r_key_list, target_dict, sex_dict)
    print(f"[{mode}] {train_or_test}: (Male-BlondHair) pp = {len(pp)}, pn = {len(pn)}, np = {len(np)}, nn = {len(nn)}")
    return r_key_list


def conditional_entropy(r_key_list, target_dict, sex_dict):
    pp, pn, np, nn = split_by_attr(r_key_list, target_dict, sex_dict)
    pp, pn, np, nn = len(pp), len(pn), len(np), len(nn)
    py = pp + pn
    ny = np + nn
    ap = np + pp
    an = nn + pn
    n = py + ny
    Pya = torch.tensor([[pp, pn], [np, nn]]) / n
    H_YA = -(Pya * torch.log(Pya)).sum()
    Pyca = torch.stack([torch.tensor([pp, pn]) / py, torch.tensor([np, nn]) / ny], dim=0)
    res = -(Pya * torch.log(Pyca)).sum()
    res = torch.nan_to_num(res)
    print(f"H(Y|A)={res}, H(Y,A)={H_YA}")
    return res.item()


def entropy(r_key_list, target_dict, sex_dict):
    pp, pn, np, nn = split_by_attr(r_key_list, target_dict, sex_dict)
    pp, pn, np, nn = len(pp), len(pn), len(np), len(nn)
    py = pp + pn
    ny = np + nn
    ap = np + pp
    an = nn + pn
    n = py + ny

    Py = torch.tensor([ap, an]) / n
    Pa = torch.tensor([py, ny]) / n
    H_Y = -(Py * torch.log(Py)).sum()
    H_A = -(Pa * torch.log(Pa)).sum()
    print(f"H(Y)={H_Y}, H(A)={H_A}")
    return H_Y.item(), H_A.item()


def reverse_sex(b, sex_idx):
    a = b.clone()
    a[:, sex_idx] = 1 - b[:, sex_idx]
    a = (a * 2 - 1) / 2
    a[:, sex_idx] = a[:, sex_idx] * 2
    return a


def neutral_sex(b, sex_idx):
    a = b.clone()
    a = (a * 2 - 1) / 2
    a[:, sex_idx] = torch.zeros_like(a[:, sex_idx])
    return a


def to_female(b, sex_idx):
    a = b.clone()
    for i in range(a.size()[0]):
        a[i, sex_idx] = 1 - b[i, sex_idx] if a[i, sex_idx] != 0 else b[i, sex_idx]
    a = (a * 2 - 1) / 2
    a[:, sex_idx] = a[:, sex_idx] * 2
    return a


def to_male(b, sex_idx):
    a = b.clone()
    for i in range(a.size()[0]):
        a[i, sex_idx] = 1 - b[i, sex_idx] if a[i, sex_idx] == 0 else b[i, sex_idx]
    a = (a * 2 - 1) / 2
    a[:, sex_idx] = a[:, sex_idx] * 2
    return a


def get_attr_index(attributes):
    all_attr = ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags", "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbone", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young", "Male"]
    res = []
    for attribute in attributes:
        res.append(all_attr.index(attribute))
    return res


def transfer_origin_for_testing_only(testing_dev_target_dict, attribute_list):
    res = dict()
    for k, v in testing_dev_target_dict.items():
        res[k] = v[attribute_list].reshape(len(attribute_list))
    return res


def imbalance_training(train_target_dict, dev_test_target_dict, train_key_list, dev_key_list, test_key_list, train_percentage_of_female, dev_test_percentage_of_female, **kwargs):
    print("\n")
    print("## before ## train, dev, test")
    CelebA_gender_distribution(train_target_dict, train_key_list)
    CelebA_gender_distribution(dev_test_target_dict, dev_key_list)
    CelebA_gender_distribution(dev_test_target_dict, test_key_list)

    print("\n")
    if train_percentage_of_female != -1:
        print(f"change percentage of female in training dataset to {train_percentage_of_female} ...\n")
        train_key_list = change_CelebA_gender_distribution(train_target_dict, train_key_list, train_percentage_of_female, seed=kwargs["seed"])
    if dev_test_percentage_of_female != -1:
        print(f"change percentage of female in testing dataset to {dev_test_percentage_of_female} ...\n")
        dev_key_list = change_CelebA_gender_distribution(dev_test_target_dict, dev_key_list, dev_test_percentage_of_female, seed=kwargs["seed"])
        test_key_list = change_CelebA_gender_distribution(dev_test_target_dict, test_key_list, dev_test_percentage_of_female, seed=kwargs["seed"])

    print("## after ## train, dev, test")
    CelebA_gender_distribution(train_target_dict, train_key_list)
    CelebA_gender_distribution(dev_test_target_dict, dev_key_list)
    CelebA_gender_distribution(dev_test_target_dict, test_key_list)
    print("\n")
    return train_key_list, dev_key_list, test_key_list


def CelebA_gender_distribution(target_dict, key_list):
    if not key_list:
        print(f"Female: 0 (0.00); Male: 0 (0.00); Sum: 0")
        return [], 0, 0, 0
    gender_statistic = []
    for key in key_list:
        # last attribute is about sex
        gender = target_dict[key][-1]
        gender_statistic.append(gender)
    whole_number = len(key_list)
    statistics_list = list(Counter(gender_statistic).values())
    if len(statistics_list) == 2:
        number_of_female, number_of_male = statistics_list
    elif list(Counter(gender_statistic).keys())[0] == 0:
        number_of_female, number_of_male = statistics_list[0], 0
    else:
        number_of_female, number_of_male = 0, statistics_list[0]
    percentage_of_female = number_of_female / (number_of_female + number_of_male)
    percentage_of_male = number_of_male / (number_of_female + number_of_male)
    print(f"Female: {number_of_female} ({percentage_of_female:.2f}); Male: {number_of_male} ({percentage_of_male:.2f}); Sum: {whole_number}")
    return gender_statistic, whole_number, number_of_female, number_of_male


def change_CelebA_gender_distribution(target_dict, key_list, percentage_of_female, **kwargs):
    female_key_list = []
    male_key_list = []
    for key in key_list:
        # last attribute is about sex
        gender = target_dict[key][-1]
        if gender == 0:
            female_key_list.append(key)
        else:
            male_key_list.append(key)
    max_female_male_ratio = len(female_key_list) / len(male_key_list)
    female_male_ratio = percentage_of_female / (1 - percentage_of_female) if percentage_of_female != 1 else float("inf")
    rng = np.random.default_rng(kwargs["seed"])
    if percentage_of_female <= 0.5:
        female_number = rng.choice(len(female_key_list), size=int(np.ceil(len(male_key_list) * female_male_ratio)), replace=False)
        female_key_list = [female_key_list[i] for i in female_number]
    elif percentage_of_female > 0.5 and percentage_of_female <= max_female_male_ratio / (1 + max_female_male_ratio):
        female_number = rng.choice(len(female_key_list), size=int(np.ceil(len(male_key_list) * female_male_ratio)), replace=False)
        female_key_list = [female_key_list[i] for i in female_number]
    else:
        male_number = rng.choice(len(male_key_list), size=int(np.ceil(len(female_key_list) / female_male_ratio)), replace=False)
        male_key_list = [male_key_list[i] for i in male_number]
    res_key_list = female_key_list + male_key_list
    return res_key_list


def change_CelebA_attribute_distribution(target_dict, key_list, percentage_of_female, **kwargs):
    female_key_list = []
    male_key_list = []
    for key in key_list:
        # last attribute is about sex
        gender = target_dict[key][-1]
        if gender == 0:
            female_key_list.append(key)
        else:
            male_key_list.append(key)
    max_female_male_ratio = len(female_key_list) / len(male_key_list)
    female_male_ratio = percentage_of_female / (1 - percentage_of_female) if percentage_of_female != 1 else float("inf")
    rng = np.random.default_rng(kwargs["seed"])
    if percentage_of_female <= 0.5:
        female_number = rng.choice(len(female_key_list), size=int(np.ceil(len(male_key_list) * female_male_ratio)), replace=False)
        female_key_list = [female_key_list[i] for i in female_number]
    elif percentage_of_female > 0.5 and percentage_of_female <= max_female_male_ratio / (1 + max_female_male_ratio):
        female_number = rng.choice(len(female_key_list), size=int(np.ceil(len(male_key_list) * female_male_ratio)), replace=False)
        female_key_list = [female_key_list[i] for i in female_number]
    else:
        male_number = rng.choice(len(male_key_list), size=int(np.ceil(len(female_key_list) / female_male_ratio)), replace=False)
        male_key_list = [male_key_list[i] for i in male_number]
    res_key_list = female_key_list + male_key_list
    return res_key_list
