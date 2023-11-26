import os
import torch
import numpy as np
from torch.utils.data import Dataset


class ColoredDataset_generated(Dataset):
    def __init__(self, dataset, classes=10, colors=[0, 1], var=0, num_color=10):
        self.dataset = dataset
        self.colors = colors
        # assign uniformly random color to all images
        if var == -1:
            self.colors = torch.zeros(classes, 3, 1, 1)
            self.perturb = torch.rand(len(self.dataset), 3, 1, 1)
        # assign normally random color to different groups of digits
        else:
            std = np.sqrt(var) if var != 0 else 0
            if isinstance(colors, torch.Tensor):
                assert colors.shape == torch.Size([10, 3, 1, 1])
                self.colors = colors
            elif isinstance(colors, list):
                if num_color == 2:
                    half1 = torch.zeros(classes // 2, 3, 1, 1) + torch.rand(1, 3, 1, 1)
                    half2 = torch.zeros(classes // 2, 3, 1, 1) + torch.rand(1, 3, 1, 1)
                    self.colors = torch.cat((half1, half2))
                elif num_color == 10:
                    g = torch.tensor([float(i) / 20 for i in range(1, 20, 2)]).view((-1, 1))
                    gg = torch.cat((g, g[torch.randperm(10)]), dim=1)
                    ggg = torch.cat((gg, g[torch.randperm(10)]), dim=1)
                    ggg = ggg.view((-1, 3, 1, 1))
                    self.colors = torch.Tensor(classes, 3, 1, 1) + ggg
            else:
                raise ValueError("Unsupported colors!")
            self.perturb = std * torch.randn(len(self.dataset), 3, 1, 1)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        color = (self.colors[label] + self.perturb[idx]).clamp(0, 1)
        img_dup = img.repeat(3, 1, 1)
        color_map = color.repeat(1, 32, 32)
        zeros_map = torch.zeros((3, 32, 32))
        color_img = torch.where(img_dup < 0.3, zeros_map, color_map)
        return color_img, label, color.squeeze()


class ColoredDataset_given(Dataset):
    def __init__(self, dir="../../dataset/CMNIST/", var=0.02, mode="train"):
        filename = "mnist_10color_jitter_var_{}.npy".format(var)
        filepath = os.path.join(dir, filename)

        data = np.load(filepath, encoding="latin1", allow_pickle=True).item()

        if mode == "train":
            x = data["train_image"][:50000] / 255
            x = x.astype(np.float32)
            y = data["train_label"][:50000]
            y = y.astype(np.int64)
        elif mode == "dev":
            x = data["train_image"][50000:] / 255
            x = x.astype(np.float32)
            y = data["train_label"][50000:]
            y = y.astype(np.int64)
        elif mode == "test":
            x = data["test_image"] / 255
            x = x.astype(np.float32)
            y = data["test_label"]
            y = y.astype(np.int64)

        self.x = np.moveaxis(x, 3, 1)
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        color_img = self.x[idx]
        label = self.y[idx]
        color_fn = lambda x: x.reshape(x.shape[0], -1).max(1).squeeze()
        color = color_fn(color_img)
        return color_img, label, color


def analyzer(x, y, label=2):
    print(x.size())
    print(y.size())
    idx1 = y == label
    x1 = x[idx1]
    y1 = y[idx1]
    print(f"Label: {y1}")

    color_fn = lambda x: x.view(x.size(0), x.size(1), -1).max(2)[0]
    colors = color_fn(x1)
    print(colors)
    print(f"Size: {colors.size()}")
    print(f"Mean: {colors.mean(0)}")
    print(f"STD: {colors.std(0)}")


def get_colors(x, y):
    res = []
    for i in range(10):
        idx = y == i
        x1 = x[idx]
        y1 = y[idx]
        print(x1.size())
        print(f"Label: {y1}")

        color_fn = lambda x: x.view(x.size(0), x.size(1), -1).max(2)[0]
        colors = color_fn(x1)
        print(colors)
        print(f"Size: {colors.size()}")
        print(f"Mean: {colors.mean(0)}")
        print(f"STD: {colors.std(0)}")
        res.append(colors.mean(0))
        print("\n")
    res = torch.stack(res)
    res = res.unsqueeze(2).unsqueeze(3)
    print(res.size())
    return res
