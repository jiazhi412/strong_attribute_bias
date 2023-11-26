import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import pandas as pd


class FFHQ(Dataset):
    def __init__(self, data_dir) -> None:
        super().__init__()
        self.img_dir = os.path.join(data_dir, "images/thumbnails128x128_together")
        self.label_dir = os.path.join(data_dir, "labels/FFHQ-Aging-Dataset/ffhq_aging_labels.csv")
        self.labels = pd.read_csv(self.label_dir)

        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_dir, str(idx).zfill(5) + ".png"))
        sex = 1 if self.labels.iloc[idx, 3] == "male" else 0
        return self.transform(img), torch.tensor(sex)

    def __show__(self, idx, transform):
        if transform:
            img = transform(self.__getitem__(idx))
        else:
            img = self.__getitem__(idx)
        img

        