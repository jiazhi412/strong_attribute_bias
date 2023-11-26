import argparse
from prepare_data.prepare_data_origin_CelebA import create_CelebA_data
from prepare_data.prepare_data_origin_Adult import create_Adult_data
from prepare_data.prepare_data_origin_IMDB import create_IMDB_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for training")
    parser.add_argument("--dataset", type=str, default="CelebA", choices=["CelebA", "Adult", "IMDB"], help="dataset name")
    args = parser.parse_args()

    print(f"Preparing {args.dataset} experiment data")
    if args.dataset == "CelebA":
        load_path = "../data/CelebA/raw_data"
        save_path = "../data/CelebA/processed_data"
        create_CelebA_data(load_path, save_path)
    elif args.dataset == "Adult":
        load_path = "../data/Adult/raw_data/adult.csv"
        save_path = "../data/Adult/processed_data/newData.csv"
        create_Adult_data(load_path, save_path)
    elif args.dataset == "IMDB":
        load_path = "../data/IMDB/raw_data/"
        save_path = "../data/IMDB/processed_data/"
        create_IMDB_data(load_path, save_path)

    print("Finshed")
