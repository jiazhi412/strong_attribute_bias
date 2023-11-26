# Copyright (C) 2023 Jiazhi Li <jiazhil412@gmail.com>
#
# This work is licensed under the MIT License. To view a copy of this license,
# visit https://opensource.org/licenses/MIT.

import os
import re
import json
import pickle
import torch
import numpy as np
import copy


# 1. running
def set_random_seed(seed_number):
    torch.manual_seed(seed_number)
    torch.cuda.manual_seed(seed_number)
    np.random.seed(seed_number)


def str_list(s):
    if type(s) is type([]):
        return s
    range_re = re.compile(r"^(\d+)-(\d+)$")
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    vals = s.split(",")
    return [x for x in vals]


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_pkl(pkl_data, save_path):
    with open(save_path, "wb") as f:
        pickle.dump(pkl_data, f)


def load_pkl(load_path):
    with open(load_path, "rb") as f:
        pkl_data = pickle.load(f)
    return pkl_data


def save_dict(dict, save_path, file_name):
    with open(os.path.join(save_path, file_name), "w") as f:
        for key, value in dict.items():
            f.write("   " + str(key) + ": " + str(value) + "\n")
    f.close()


def save_list(list, save_path, file_name):
    with open((save_path + file_name), "w") as f:
        for item in list:
            f.write("%s\n" % item)
    f.close()


def save_json(json_data, save_path):
    save_data = copy.copy(json_data)
    save_data["filter_parameters"] = vars(save_data["filter_parameters"])
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=4, separators=(",", ":"))


def load_json(load_path):
    with open(load_path, "r") as f:
        json_data = json.load(f)
    return json_data


def save_state_dict(state_dict, save_path):
    torch.save(state_dict, save_path)


def write_dict(f, dict):
    for k, v in dict.items():
        f.write("   " + str(k) + ": " + str(v) + "\n")


def write_by_line(save_path, list):
    with open(save_path, "w") as f:
        for item in list:
            f.write("%s\n" % item)
    f.close()


def save_settings(args, train_mark):
    # save samples in local
    os.makedirs(
        os.path.join(
            args.save_path,
            args.experiment,
            args.name,
            train_mark,
            args.hyperparameter,
            "visualization",
        ),
        exist_ok=True,
    )
    with open(
        os.path.join(
            args.save_path,
            args.experiment,
            args.name,
            train_mark,
            args.hyperparameter,
            "setting.txt",
        ),
        "w",
    ) as f:
        f.write(json.dumps(vars(args), indent=4, separators=(",", ":")))
    # save model in ssd
    os.makedirs(
        os.path.join(
            args.ssd_path,
            args.experiment,
            args.name,
            train_mark,
            args.hyperparameter,
            "checkpoint",
        ),
        exist_ok=True,
    )
    with open(
        os.path.join(
            args.ssd_path,
            args.experiment,
            args.name,
            train_mark,
            args.hyperparameter,
            "setting.txt",
        ),
        "w",
    ) as f:
        f.write(json.dumps(vars(args), indent=4, separators=(",", ":")))
