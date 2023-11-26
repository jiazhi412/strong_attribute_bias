import torch
from PIL import Image
import numpy as np


class CelebADataset(torch.utils.data.Dataset):
    """CelebA dataloader, output image and target"""

    def __init__(self, key_list, image_feature, target_dict, sex_dict, transform=None):
        self.key_list = key_list
        self.image_feature = image_feature
        self.target_dict = target_dict
        self.sex_dict = sex_dict
        self.transform = transform

    def __getitem__(self, index):
        key = self.key_list[index]
        img, target = Image.fromarray(self.image_feature[key][()]), self.target_dict[key]
        sex = np.array(self.sex_dict[key])[np.newaxis]

        if self.transform is not None:
            img = self.transform(img)
        return img, torch.FloatTensor(target), torch.FloatTensor(sex)

    def __len__(self):
        return len(self.key_list)
