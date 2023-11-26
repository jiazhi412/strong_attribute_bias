import torch
from PIL import Image
import numpy as np
import h5py


class IMDBDataset(torch.utils.data.Dataset):
    """IMDB dataloader, output image and target"""

    def __init__(self, image_feature_path, target_dict, sex_dict, mode, eb1, eb2, unbiased, dev_or_test, transform=None):
        self.image_feature_path = image_feature_path
        self.target_dict = target_dict
        self.sex_dict = sex_dict
        self.transform = transform
        self.mode = mode

        if mode.startswith("eb1"):
            self.key_list = eb1
        elif mode.startswith("eb2"):
            self.key_list = eb2
        elif mode.startswith("unbiased"):
            self.key_list = unbiased
        elif mode.startswith("all"):
            self.key_list = eb1 + eb2 + unbiased

        if dev_or_test == "train":
            pass
        elif dev_or_test == "dev":
            self.key_list = self.key_list[: len(self.key_list) // 10]
        elif dev_or_test == "test":
            self.key_list = self.key_list[len(self.key_list) // 10 :]
        elif dev_or_test == "dev_test":
            pass

    def __getitem__(self, index):
        key = self.key_list[index]
        image_feature = h5py.File(self.image_feature_path, "r")
        img = Image.fromarray(image_feature[key][()])
        age = np.array(self.target_dict[key.encode("ascii")])[np.newaxis]
        sex = np.array(self.sex_dict[key.encode("ascii")])[np.newaxis]

        if self.mode.endswith("ex"):  # ex binary class of age -> bce
            self.bins = np.array([0, 30, 120])
            age = np.digitize(age, self.bins) - 1
            if age <= -1:
                age = np.zeros_like(age)
            age = torch.FloatTensor(age)
        else:  # non-ex 12 class of age -> ce
            self.bins = np.array([0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 120])
            age = np.digitize(age, self.bins) - 1
            if age <= -1:
                age = np.zeros_like(age)
            age = torch.tensor(age)

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.FloatTensor(sex), age

    def __len__(self):
        return len(self.key_list)


def quantize_age(age_vec, bins):
    """
    Get binary vector associated with age value
    """
    n_samples = age_vec.shape[0]
    n_bins = bins.shape[0] - 1

    age_lb = np.zeros((n_samples, n_bins))
    hh = np.digitize(age_vec, bins) - 1

    for i in range(n_samples):
        age_lb[i, hh[i]] = 1

    return age_lb
