import os
import argparse
from argparse import Namespace
from datetime import datetime
import torch
import utils


def collect_args():
    parser = argparse.ArgumentParser()
    # experiment choices
    parser.add_argument("--experiment", metavar="")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--filter_path", type=str, default=None)
    parser.add_argument("--ssd_path", type=str, default=None)
    parser.add_argument("--data_prefix", type=str, default="./data")

    ## Colored MNIST
    parser.add_argument("--color_dataset", type=str, default="generated")
    parser.add_argument("--biased_var", type=float, default=-1, help="color variance for training classifier")  # -1 uniformly

    ## CelebA
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--backbone", type=str, default="ResNet18")
    parser.add_argument("--attributes", dest="attrs_pred", type=utils.str_list, default=["Blond_Hair"])
    parser.add_argument("--CelebA_train_mode", type=str, default=None, choices=["CelebA_train", "CelebA_train_ex", "CelebA_FFHQ", "CelebA_synthetic", "CelebA_all", "FFHQ"])
    parser.add_argument("--CelebA_test_mode", type=str, default="unbiased", choices=["unbiased", "unbiased_ex", "conflict", "conflict_ex"])
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--n_bc", default=-1, type=int, help="number of bias-conflicting samples")
    parser.add_argument("--p_bc", default=-1, type=float, help="percentage of bias-conflicting samples")
    parser.add_argument("--balance", action="store_true")

    ## Adult
    parser.add_argument("--Adult_train_mode", type=str, default="eb1_balanced")
    parser.add_argument("--Adult_test_mode", type=str, default="eb2_balanced", choices=["eb1", "eb2", "eb1_balanced", "eb2_balanced", "balanced", "all"])

    ## IMDB
    parser.add_argument("--IMDB_train_mode", type=str, default="eb1")
    parser.add_argument("--IMDB_test_mode", type=str, default="eb2", choices=["eb1", "eb2", "unbiased", "all"])

    # TODO Custom
    parser.add_argument(
        "--Custom_train_mode",
        type=str,
    )
    parser.add_argument(
        "--Custom_test_mode",
        type=str,
    )

    ## filter choices
    parser.add_argument("--filter_name", default=None, type=str)
    parser.add_argument("--filter_hp", default=None, type=str)
    parser.add_argument("--filter_idx", default=None, type=int)
    parser.add_argument("--filter_train_mode", type=str, default="universal", choices=["biased", "universal"])
    parser.add_argument("--filter_mode", type=str, default="neutral")

    # hyper-parameter
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--augment", action="store_true")

    # running settings
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reproduce", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--random_seed", type=int, default=26)
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--no_sm", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--wandb_mode", type=str, default='disabled', choices = ['online', 'offline', 'disabled'])

    return create_experiment_setting(vars(parser.parse_args()))


def create_experiment_setting(opt):
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        opt["batch_size"] *= torch.cuda.device_count() // 2
        opt["lr"] *= torch.cuda.device_count() // 2

    # debug corner case
    if opt["debug"]:
        opt["name"] = "debug"
        opt["epochs"] = 1
        opt["batch_size"] = 32
        opt["num_iter_MI"] = 1
        opt["no_sm"] = True
    # sweep corner case
    if opt["name"] == "sweep":
        opt["no_sm"] = True
    # reproduce corner case, set a unified random seed
    if opt["reproduce"]:
        utils.set_random_seed(opt["random_seed"])
    # All experiment setting
    opt["device"] = "cuda" if not opt["no_cuda"] and torch.cuda.is_available() else "cpu"
    opt["name"] = str(datetime.now()) if not opt["name"] else opt["name"]
    opt["save_folder"] = os.path.join("result", opt["experiment"], opt["name"])
    opt["save_file"] = "_".join(
        (
            str(opt["batch_size"]),
            str(opt["lr"]),
        )
    )

    # Colored MNIST
    if opt["experiment"].startswith("CMNIST"):
        opt["data_folder"] = os.path.join(opt["data_prefix"], "MNIST")
        middle_description = (str(opt["biased_var"]),)
        ### baseline
        if opt["experiment"] == "CMNIST_downstream_baseline":
            from models_eval.CMNIST_downstream_baseline import Model
        ### ours
        if opt["experiment"].startswith("CMNIST_downstream_our"):
            filter_experiment = "CMNIST_filter"
            filter_name = opt["filter_name"] if opt["filter_name"] else "NAME"
            filter_hyperparameter = opt["filter_hp"] if opt["filter_hp"] else "mi10.0_gc100.0_dc100.0_gr100.0"
            filter_number = f"weights.{opt['filter_idx']}.pth" if opt["filter_idx"] else "weights.23.pth"
            if opt["filter_train_mode"] == "biased":  # filter trained under the corresponding biased dataset
                opt["f_color_var"] = opt["biased_var"]
            elif opt["filter_train_mode"] == "universal":  # filter trained under universal distribution
                opt["f_color_var"] = -1
            if opt["filter_path"]:
                components = opt["filter_path"].rsplit("/", 1)  # Split from the right, max 1 split
                filter_dir = "/".join(components[:-1])  # Join all components except the last one
                opt["filter_parameters"] = Namespace(**utils.load_json(os.path.join(filter_dir, "setting.txt")))
            else:
                filter_path = os.path.join(opt["ssd_path"], filter_experiment, filter_name, str(float(opt["f_color_var"])), filter_hyperparameter)
                opt["filter_parameters"] = Namespace(**utils.load_json(os.path.join(filter_path, "setting.txt")))
                opt["filter_path"] = os.path.join(filter_path, "checkpoint", filter_number)
            from models_eval.CMNIST_downstream_our import Model

    # CelebA
    if opt["experiment"].startswith("CelebA_downstream"):
        opt["data_folder"] = os.path.join(opt["data_prefix"], "CelebA/processed_data")
        middle_description = (opt["attrs_pred"][0] if len(opt["attrs_pred"]) == 1 else "_".join(opt["attrs_pred"]), opt["CelebA_test_mode"])
        ### baseline
        if opt["experiment"].startswith("CelebA_downstream_baseline"):
            from models_eval.CelebA_downstream_baseline import Model
        ### ours
        elif opt["experiment"].startswith("CelebA_downstream_our"):
            filter_experiment = "CelebA_filter"
            filter_name = opt["filter_name"] if opt["filter_name"] else "NAME"
            filter_hyperparameter = opt["filter_hp"] if opt["filter_hp"] else "mi50.0_gc50.0_dc50.0_gr100.0"
            filter_number = f"weights.{opt['filter_idx']}.pth" if opt["filter_idx"] else "weights.49.pth"
            if opt["filter_train_mode"] == "biased":  # filter trained under the corresponding biased dataset
                if opt["CelebA_train_mode"]:
                    f_train_mode = opt["CelebA_train_mode"]
                elif opt["CelebA_test_mode"] in ["unbiased_ex", "conflict_ex"]:
                    f_train_mode = "CelebA_train_ex"
                elif opt["CelebA_test_mode"] in ["unbiased", "conflict"]:
                    f_train_mode = "CelebA_train"
            elif opt["filter_train_mode"] == "universal":  # filter trained under true distribution
                f_train_mode = "CelebA_FFHQ"
            if opt["filter_path"]:
                components = opt["filter_path"].rsplit("/", 1)  # Split from the right, max 1 split
                filter_dir = "/".join(components[:-1])  # Join all components except the last one
                opt["filter_parameters"] = Namespace(**utils.load_json(os.path.join(filter_dir, "setting.txt")))
            else:
                filter_path = opt["filter_path"] if opt["filter_path"] else os.path.join(opt["ssd_path"], filter_experiment, filter_name, f_train_mode, filter_hyperparameter)
                opt["filter_parameters"] = Namespace(**utils.load_json(os.path.join(filter_path, "setting.txt")))
                opt["filter_path"] = os.path.join(filter_path, "checkpoint", filter_number)
            from models_eval.CelebA_downstream_our import Model

    # Adult
    if opt["experiment"].startswith("Adult_downstream"):
        opt["data_path"] = os.path.join(opt["data_prefix"], "Adult/processed_data/adult_newData.csv")
        middle_description = (opt["Adult_train_mode"], opt["Adult_test_mode"])
        if opt["experiment"] == "Adult_downstream_baseline":
            from models_eval.Adult_downstream_baseline import Model
        elif opt["experiment"] == "Adult_downstream_our":
            filter_experiment = "Adult_filter"
            filter_name = opt["filter_name"] if opt["filter_name"] else "NAME"
            filter_hyperparameter = opt["filter_hp"] if opt["filter_hp"] else "mi5.0_gc5.0_dc5.0_gr10.0"
            filter_number = f"weights.{opt['filter_idx']}.pth" if opt["filter_idx"] else "weights.99.pth"
            if opt["filter_train_mode"] == "biased":  # filter trained under the corresponding biased dataset
                f_train_mode = opt["Adult_train_mode"]
            elif opt["filter_train_mode"] == "universal":  # filter trained under true distribution
                f_train_mode = "all"
            if opt["filter_path"]:
                components = opt["filter_path"].rsplit("/", 1)  # Split from the right, max 1 split
                filter_dir = "/".join(components[:-1])  # Join all components except the last one
                opt["filter_parameters"] = Namespace(**utils.load_json(os.path.join(filter_dir, "setting.txt")))
            else:
                filter_path = opt["filter_path"] if opt["filter_path"] else os.path.join(opt["ssd_path"], filter_experiment, filter_name, f_train_mode, filter_hyperparameter)
                opt["filter_parameters"] = Namespace(**utils.load_json(os.path.join(filter_path, "setting.txt")))
                opt["filter_path"] = os.path.join(filter_path, "checkpoint", filter_number)
            from models_eval.Adult_downstream_our import Model

    # IMDB
    if opt["experiment"].startswith("IMDB_downstream"):
        opt["data_folder"] = os.path.join(opt["data_prefix"], "IMDB/processed_data")
        middle_description = (opt["IMDB_train_mode"], opt["IMDB_test_mode"])
        if opt["experiment"] == "IMDB_downstream_baseline":
            from models_eval.IMDB_downstream_baseline import Model
        elif opt["experiment"] == "IMDB_downstream_our":
            filter_experiment = "IMDB_filter"
            filter_name = opt["filter_name"] if opt["filter_name"] else "NAME"
            filter_hyperparameter = opt["filter_hp"] if opt["filter_hp"] else "mi50.0_gc50.0_dc50.0_gr100.0"
            filter_number = f"weights.{opt['filter_idx']}.pth" if opt["filter_idx"] else "weights.26.pth"
            if opt["filter_train_mode"] == "biased":  # filter trained under the corresponding biased dataset
                f_train_mode = opt["IMDB_train_mode"]
            elif opt["filter_train_mode"] == "universal":  # filter trained under true distribution
                f_train_mode = "all"
            if opt["filter_path"]:
                components = opt["filter_path"].rsplit("/", 1)  # Split from the right, max 1 split
                filter_dir = "/".join(components[:-1])  # Join all components except the last one
                opt["filter_parameters"] = Namespace(**utils.load_json(os.path.join(filter_dir, "setting.txt")))
            else:
                filter_path = opt["filter_path"] if opt["filter_path"] else os.path.join(opt["ssd_path"], filter_experiment, filter_name, f_train_mode, filter_hyperparameter)
                opt["filter_parameters"] = Namespace(**utils.load_json(os.path.join(filter_path, "setting.txt")))
                opt["filter_path"] = os.path.join(filter_path, "checkpoint", filter_number)
            from models_eval.IMDB_downstream_our import Model

    # TODO Custom
    if opt["experiment"].startswith("Custom_downstream"):
        opt["data_folder"] = os.path.join(opt["data_prefix"], "Custom/raw_data")
        middle_description = (opt["Custom_train_mode"], opt["Custom_test_mode"])
        if opt["experiment"] == "Custom_downstream_baseline":
            from models_eval.Custom_downstream_baseline import Model
        elif opt["experiment"] == "Custom_downstream_our":
            filter_experiment = "Custom_filter"
            filter_name = opt["filter_name"] if opt["filter_name"] else "NAME"
            filter_hyperparameter = opt["filter_hp"] if opt["filter_hp"] else "mi50.0_gc50.0_dc50.0_gr100.0"
            filter_number = f"weights.{opt['filter_idx']}.pth" if opt["filter_idx"] else "weights.26.pth"
            if opt["filter_train_mode"] == "biased":  # filter trained under the corresponding biased dataset
                f_train_mode = opt["Custom_train_mode"]
            elif opt["filter_train_mode"] == "universal":  # filter trained under true distribution
                f_train_mode = "all"
            if opt["filter_path"]:
                components = opt["filter_path"].rsplit("/", 1)  # Split from the right, max 1 split
                filter_dir = "/".join(components[:-1])  # Join all components except the last one
                opt["filter_parameters"] = Namespace(**utils.load_json(os.path.join(filter_dir, "setting.txt")))
            else:
                filter_path = opt["filter_path"] if opt["filter_path"] else os.path.join(opt["ssd_path"], filter_experiment, filter_name, f_train_mode, filter_hyperparameter)
                opt["filter_parameters"] = Namespace(**utils.load_json(os.path.join(filter_path, "setting.txt")))
                opt["filter_path"] = os.path.join(filter_path, "checkpoint", filter_number)
            from models_eval.Custom_downstream_our import Model

    opt["save_folder"] = os.path.join(opt["save_folder"], *middle_description)
    opt["wandb"] = "-".join((*middle_description, opt["save_file"]))
    utils.create_folder(opt["save_folder"])
    utils.save_json(opt, os.path.join(opt["save_folder"], "setting.json"))
    utils.save_dict(opt, opt["save_folder"], "setting.txt")
    print(opt)
    return Model(opt), opt
