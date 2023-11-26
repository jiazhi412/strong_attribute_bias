import numpy as np
from dataloader.Adult_data_utils import data_processing, quick_load, get_bias

def create_Adult_data(load_path, save_path):
    newData = data_processing(load_path)
    np.savetxt(save_path, newData, delimiter=",")
    

if __name__ == "__main__":
    # read data and save data
    load_path = "../data/Adult/raw_data/adult.csv"
    newData = data_processing(load_path)

    # Save to csv file
    save_path = "../data/Adult/processed_data/newData.csv"
    np.savetxt(save_path, newData, delimiter=",")

    # read save data directly to save time
    save_path = "../data/Adult/processed_data/newData.csv"
    data = quick_load(save_path)
    print(data.shape)

    # test get bias
    bias_data = get_bias(data, "sex")
    print(bias_data.shape)