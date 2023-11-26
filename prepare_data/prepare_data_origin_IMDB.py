import os
import pickle
import h5py
import numpy as np
from PIL import Image
import utils


def crop2subset(load_path):
    eb1 = np.load(os.path.join(load_path, 'auxiliary', "eb1_img_list.npy"), allow_pickle=True, encoding="latin1")
    eb2 = np.load(os.path.join(load_path, 'auxiliary', "eb2_img_list.npy"), allow_pickle=True, encoding="latin1")

    split = "test_list"
    sp = np.load(os.path.join(load_path, 'auxiliary', "imdb_split.npy"), allow_pickle=True, encoding="latin1").item()
    test = sp[split]  # depends on whether train or test (train_list/test_list)

    for name in eb1:
        img = Image.open(os.path.join(load_path, "imdb_crop", name))
        sole_img_name = os.path.split(name)[1]
        utils.create_folder(os.path.join(load_path, 'subset', "eb1"))
        img.save(os.path.join(load_path, 'subset', "eb1", sole_img_name))

    for name in eb2:
        img = Image.open(os.path.join(load_path, "imdb_crop", name))
        sole_img_name = os.path.split(name)[1]
        utils.create_folder(os.path.join(load_path, 'subset', "eb2"))
        img.save(os.path.join(load_path, 'subset', "eb2", sole_img_name))

    for name in test:
        img = Image.open(os.path.join(load_path, "imdb_crop", name))
        sole_img_name = os.path.split(name)[1]
        utils.create_folder(os.path.join(load_path, 'subset', "test"))
        img.save(os.path.join(load_path, 'subset', "test", sole_img_name))


def create_IMDB_data(load_path, save_path):
    """Create dataset for CelebA experiments"""
    crop2subset(load_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # # load data
    file_list = []
    feature_file = h5py.File(os.path.join(save_path, "IMDB.h5py"), "w")
    for sub_dir in ["eb1", "eb2", "test"]:
        for filename in os.listdir(os.path.join(load_path, sub_dir)):
            print(filename.encode("utf-8"))
            feature_file.create_dataset(filename, data=np.asarray(Image.open(os.path.join(load_path, sub_dir, filename)).convert("RGB")))
            file_list.append(filename.encode("utf-8"))
    feature_file.close()

    # load label
    labels = np.load(os.path.join(load_path, 'auxiliary', "imdb_age_gender.npy"), allow_pickle=True, encoding="latin1").item()
    new_labels = dict()
    for k, v in labels.items():
        new_labels[os.path.split(k)[1]] = v

    age_dict = dict()
    sex_dict = dict()
    for n, tr_img in enumerate(file_list):
        age_dict[tr_img] = new_labels[tr_img]["age"]
        sex_dict[tr_img] = new_labels[tr_img]["gender"]

    with open(os.path.join(save_path, "age_dict"), "wb") as f:
        pickle.dump(age_dict, f)
    with open(os.path.join(save_path, "sex_dict"), "wb") as f:
        pickle.dump(sex_dict, f)

    eb1 = np.load(os.path.join(load_path, 'auxiliary', "eb1_img_list.npy"), allow_pickle=True, encoding="latin1")
    eb2 = np.load(os.path.join(load_path, 'auxiliary', "eb2_img_list.npy"), allow_pickle=True, encoding="latin1")
    test = np.load(os.path.join(load_path, 'auxiliary', "imdb_split.npy"), allow_pickle=True, encoding="latin1").item()["test_list"]

    def cut_upper_dir(list, save_path):
        res = []
        for n, name in enumerate(list):
            res.append(os.path.split(name)[1])
        import pickle

        with open(save_path, "wb") as fp:  # Pickling
            pickle.dump(res, fp)

    cut_upper_dir(eb1, os.path.join(save_path, "eb1_img_list"))
    cut_upper_dir(eb2, os.path.join(save_path, "eb2_img_list"))
    cut_upper_dir(test, os.path.join(save_path, "test_img_list"))


if __name__ == "__main__":
    print("Preparing IMDB experiment data")
    load_path = "../data/IMDB/raw_data/"
    save_path = "../data/IMDB/processed_data/"
    create_IMDB_data(load_path, save_path)
    print("Finshed")
