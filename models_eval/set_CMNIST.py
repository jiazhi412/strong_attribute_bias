import torch
from torchvision import datasets, transforms
from dataloader.ColoredMNIST import ColoredDataset_generated, ColoredDataset_given


class Model:
    def set_data(self, opt):
        print(
            "Training dataset of Colored MNIST with variance = {}".format(
                opt["biased_var"]
            )
        )
        if opt["color_dataset"] == "given":
            train_set = ColoredDataset_given(var=opt["biased_var"], mode="train")
            dev_set = ColoredDataset_given(var=opt["biased_var"], mode="dev")
            test_set = ColoredDataset_given(var=opt["biased_var"], mode="test")
        elif opt["color_dataset"] == "generated":
            # load grey scale data to generate dataset
            transform = transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                ]
            )
            train_set_grey = datasets.MNIST(
                root=opt["data_folder"], train=True, download=True, transform=transform
            )
            test_set_grey = datasets.MNIST(
                root=opt["data_folder"],
                train=False,
                download=True,
                transform=transform,
            )
            train_set_grey, dev_set_grey = torch.utils.data.random_split(
                train_set_grey, [50000, 10000]
            )

            train_set = ColoredDataset_generated(train_set_grey, var=opt["biased_var"])
            dev_set = ColoredDataset_generated(dev_set_grey, var=-1)
            test_set = ColoredDataset_generated(test_set_grey, var=-1)

        # small piece for debug
        if opt["debug"]:
            num_train = 200
            num_test = 20
            train_set, dev_set, test_set, _ = torch.utils.data.random_split(
                train_set,
                [num_train, num_test, num_test, 50000 - num_train - num_test * 2],
            )

        self.opt["num_train"] = len(train_set)
        self.train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=opt["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        self.dev_loader = torch.utils.data.DataLoader(
            dev_set,
            batch_size=opt["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=opt["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
