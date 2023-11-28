import argparse
import datetime
import torch
import wandb
import utils
import os

torch.cuda.empty_cache()


def collect_args():
    parser = argparse.ArgumentParser()

    # experiment choices
    parser.add_argument("--experiment", metavar="")
    parser.add_argument("--name", default=datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))
    parser.add_argument("--save_path", type=str, default="result")
    parser.add_argument("--ssd_path", type=str, default=None)
    parser.add_argument("--data_prefix", type=str, default="./data")

    # CMNIST
    parser.add_argument("--color_dataset", type=str, default="generated")
    parser.add_argument("--biased_var", type=float, default=-1, help="color variance for training filter")  # -1 uniformly

    # CelebA
    parser.add_argument("--CelebA_attrs", default=["Male"], nargs="+", help="attributes to eliminate")
    parser.add_argument("--CelebA_attrs_d", default=["Blond_Hair"], nargs="+", help="attributes for downstream")
    parser.add_argument("--CelebA_train_mode", type=str, default="CelebA_FFHQ", choices=["CelebA_train", "CelebA_train_ex", "CelebA_FFHQ", "CelebA_synthetic", "CelebA_all", "FFHQ"])

    # Adult
    parser.add_argument("--Adult_attrs", default=["sex"], nargs="+", help="attributes to eliminate")
    parser.add_argument("--Adult_train_mode", type=str, choices=["eb1", "eb2", "eb1_balanced", "eb2_balanced", "balanced", "all"], default="all")

    # IMDB
    parser.add_argument("--IMDB_attrs", default=["age"], nargs="+", help="attributes to eliminate")
    parser.add_argument("--IMDB_train_mode", type=str, choices=["eb1", "eb2", "unbiased", "all"], default="all")

    # TODO Custom
    parser.add_argument("--Custom_attrs", nargs="+", help="attributes to eliminate")
    parser.add_argument(
        "--Custom_train_mode",
        type=str,
    )

    # hyper-parameter
    parser.add_argument("--gr", dest="gr", type=float, default=100.0)
    parser.add_argument("--gc", dest="gc", type=float, default=100.0)
    parser.add_argument("--dc", dest="dc", type=float, default=100.0)
    parser.add_argument("--mi", dest="mi", type=float, default=10.0)
    parser.add_argument("--gp", dest="gp", type=float, default=10.0)  # default in WGAN-GP
    parser.add_argument("--num_iter_MI", type=int, default=20)
    parser.add_argument("--dim_per_attr", type=int, default=5)
    parser.add_argument("--shortcut_layers", dest="shortcut_layers", type=int, default=0)
    parser.add_argument("--inject_layers", dest="inject_layers", type=int, default=0)
    parser.add_argument("--enc_dim", dest="enc_dim", type=int, default=64)
    parser.add_argument("--dec_dim", dest="dec_dim", type=int, default=64)
    parser.add_argument("--dis_dim", dest="dis_dim", type=int, default=64)
    parser.add_argument("--dis_fc_dim", dest="dis_fc_dim", type=int, default=256)
    parser.add_argument("--enc_layers", dest="enc_layers", type=int, default=3)
    parser.add_argument("--dec_layers", dest="dec_layers", type=int, default=3)
    parser.add_argument("--dis_layers", dest="dis_layers", type=int, default=3)
    parser.add_argument("--enc_norm", dest="enc_norm", type=str, default="batchnorm")
    parser.add_argument("--dec_norm", dest="dec_norm", type=str, default="batchnorm")
    parser.add_argument("--dis_norm", dest="dis_norm", type=str, default="instancenorm")
    parser.add_argument("--dis_fc_norm", dest="dis_fc_norm", type=str, default="none")
    parser.add_argument("--enc_acti", dest="enc_acti", type=str, default="lrelu")
    parser.add_argument("--dec_acti", dest="dec_acti", type=str, default="relu")
    parser.add_argument("--dis_acti", dest="dis_acti", type=str, default="lrelu")
    parser.add_argument("--dis_fc_acti", dest="dis_fc_acti", type=str, default="relu")
    parser.add_argument("--mode", dest="mode", default="wgan", choices=["wgan", "lsgan", "dcgan"])
    parser.add_argument("--beta1", dest="beta1", type=float, default=0)  # default in WGAN-GP
    parser.add_argument("--beta2", dest="beta2", type=float, default=0.9)  # default in WGAN-GP
    parser.add_argument("--thres_int", dest="thres_int", type=float, default=0.5)
    parser.add_argument("--test_int", dest="test_int", type=float, default=1.0)
    parser.add_argument("--epochs", dest="epochs", type=int, default=50, help="# of epochs")
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=32)
    parser.add_argument("--num_workers", dest="num_workers", type=int, default=4)
    parser.add_argument("--lr", dest="lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--n_d", dest="n_d", type=int, default=5, help="# of d updates per g update")

    # running settings
    parser.add_argument("--no_gpu", dest="gpu", action="store_false")
    parser.add_argument("--save_all", action="store_true", help="save all models otherwise save generator only")
    parser.add_argument("--save_interval", dest="save_interval", type=int, default=1000)
    parser.add_argument("--sample_interval", dest="sample_interval", type=int, default=1000)
    parser.add_argument("--n_samples", dest="n_samples", type=int, default=16, help="# of sample images")

    args = parser.parse_args()
    args.hyperparameter = f"mi{args.mi}_gc{args.gc}_dc{args.dc}_gr{args.gr}"

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        args.batch_size *= torch.cuda.device_count()
        args.lr *= torch.cuda.device_count()

    return create_experiment_setting(args)


def create_experiment_setting(args):
    if args.experiment == "CMNIST_filter":
        args.n_attrs = 3  # RGB
        args.img_size = 32
        train_mark = str(float(args.biased_var))
        args.data_path = os.path.join(args.data_prefix, "MNIST")
        from models_train.CMNIST_filter import Model
    if args.experiment == "CelebA_filter":
        args.n_attrs = len(args.CelebA_attrs)
        args.img_size = 224
        args.beta1 = 0.5
        args.beta2 = 0.999
        args.num_iter_mi = 40
        train_mark = args.CelebA_train_mode
        args.data_path = os.path.join(args.data_prefix, "CelebA/raw_data/img_align_celeba")
        args.attr_path = os.path.join(args.data_prefix, "CelebA/raw_data/list_attr_celeba.txt")
        args.ffhq_path = os.path.join(args.data_prefix, "ffhq")
        CelebA_attr_d = args.CelebA_attrs_d[0].lower()
        args.syn_data_path = os.path.join(args.data_prefix, f"synthetic_datasets/CelebA/processed_data/{CelebA_attr_d}")
        from models_train.CelebA_filter import Model
    if args.experiment == "Adult_filter":
        args.n_attrs = len(args.Adult_attrs)
        train_mark = args.Adult_train_mode
        args.data_path = os.path.join(args.data_prefix, "Adult/processed_data/adult_newData.csv")
        from models_train.Adult_filter import Model
    if args.experiment == "IMDB_filter":
        args.n_attrs = len(args.IMDB_attrs)
        args.img_size = 224
        train_mark = args.IMDB_train_mode
        args.data_path = os.path.join(args.data_prefix, "IMDB/processed_data")
        from models_train.IMDB_filter import Model
    # TODO Custom
    if args.experiment == "Custom_filter":
        args.n_attrs = len(args.Custom_attrs)
        args.img_size = 224
        train_mark = "universal"
        args.data_path = os.path.join(args.data_prefix, "Custom/raw_data")
        from models_train.Custom_filter import Model
    print(args)
    utils.save_settings(args, train_mark)
    return Model(args), args


@utils.timer
def main(model, args):
    for _ in range(args.epochs):
        model.train_epoch(args)


if __name__ == "__main__":
    # 1. settings and choose model
    model, args = collect_args()

    wandb.init(
        project="strong_attribute_bias",
        config=args,
        group=args.experiment,
        job_type=args.name,
        name=args.hyperparameter,
    )

    # 2. start epoch
    main(model, args)
