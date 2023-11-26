import os
import pickle
import h5py
import numpy as np
from PIL import Image

def create_CelebA_data(load_path, save_path):
    """Create dataset for CelebA experiments"""
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    feature_file = h5py.File(os.path.join(save_path, 'CelebA.h5py'), "w")
    for filename in os.listdir(os.path.join(load_path,'img_align_celeba')):
        feature_file.create_dataset(filename, 
            data=np.asarray(Image.open(os.path.join(load_path, 'img_align_celeba', filename)).convert('RGB')))
    feature_file.close()

    with open(os.path.join(load_path, 'identity_celeba.txt'), 'r') as f:
        split_lines = f.readlines()

    identity_dict = {}
    for i, line in enumerate(split_lines):
        line = line.strip().split()
        key = line[0]
        attr = int(line[1])
        identity_dict[key] = attr

    with open(os.path.join(save_path, 'identity_dict'), 'wb') as f:
        pickle.dump(identity_dict, f)
    
    with open(os.path.join(load_path, 'list_attr_celeba.txt'), 'r') as f:
        lines = f.readlines()
        
    attr_list = lines[1].strip().split()
    attr_idx_dict = {attr: i for i, attr in enumerate(attr_list)}
    labels_dict = {}
    gender_dict = {}
    for line in lines[2:]:
        line = line.strip().split()
        key = line[0]
        attr = line[1:]
        gender = attr.pop(attr_idx_dict['Male'])
        attr.append(gender) # male attribute will be the last one
        attr = np.array(attr).astype(int)
        attr = (attr + 1) / 2
        labels_dict[key] = attr.copy()
        gender_dict[key] = attr[-1].copy()
        
    with open(os.path.join(save_path, 'labels_dict'), 'wb') as f:
        pickle.dump(labels_dict, f)

    with open(os.path.join(save_path, 'sex_dict'), 'wb') as f:
        pickle.dump(gender_dict, f)
    
    with open(os.path.join(load_path, 'list_eval_partition.txt'), 'r') as f:
        split_lines = f.readlines()
        
    train_list = []
    dev_list = []
    test_list = []
    for i, line in enumerate(split_lines):
        line = line.strip().split()
        if line[1] == '0':
            train_list.append(line[0])
        elif line[1] == '1':
            dev_list.append(line[0])
        elif line[1] == '2':
            test_list.append(line[0])
        else:
            print('error')
            break
            
    with open(os.path.join(save_path, 'train_key_list'), 'wb') as f:
        pickle.dump(train_list, f)
    with open(os.path.join(save_path, 'dev_key_list'), 'wb') as f:
        pickle.dump(dev_list, f)
    with open(os.path.join(save_path, 'test_key_list'), 'wb') as f:
        pickle.dump(test_list, f)
    
    subclass_idx = list(set(range(39)) - {0,16,21,29,37})
    with open(os.path.join(save_path, 'subclass_idx'), 'wb') as f:
        pickle.dump(subclass_idx, f)

if __name__ == '__main__':
    print('Preparing CelebA experiment data')
    load_path = '../data/CelebA/raw_data'
    save_path = '../data/CelebA/processed_data'
    create_CelebA_data(load_path, save_path)
    print('Finshed')