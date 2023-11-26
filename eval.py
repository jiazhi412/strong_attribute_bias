import torch
import parse_args
import utils
import wandb
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.empty_cache()


@utils.timer
def main(model, opt):
    if opt["test"]:
        model.test()
    else:
        for _ in range(opt["epochs"]):
            model.train()
            model.test()


if __name__ == "__main__":
    model, opt = parse_args.collect_args()

    wandb.init(
        project="strong_attribute_bias",
        config=opt,
        group=opt["experiment"],
        job_type=opt["name"],
        name=opt["wandb"],
        mode=opt["wandb_mode"],
    )

    main(model, opt)
