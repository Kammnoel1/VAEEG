# split_data.py
import os
import json
import random
import numpy as np
from tqdm import tqdm
import argparse

def split_channel_data(channel_folder, train_ratio=0.9):
    """
    For a given channel folder, load each frequency band .npy file,
    shuffle and split the data into training and test sets.
    
    Args:
        channel_folder: str, path to the channel folder (e.g., new_data/single_channel_segments/channel_1)
        train_ratio: float, proportion of data to assign to training.
        
    Returns:
        A dictionary with keys as band names and values as dicts with keys "train" and "test" 
        containing the split data arrays.
    """
    # Expected frequency bands
    band_names = ["whole", "delta", "theta", "alpha", "low_beta", "high_beta"]
    split_dict = {}
    
    for band in band_names:
        file_path = os.path.join(channel_folder, f"{band}.npy")
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping band '{band}'.")
            continue
        
        data = np.load(file_path)
        num_segments = data.shape[0]
        indices = np.arange(num_segments)
        np.random.shuffle(indices)
        n_train = int(train_ratio * num_segments)
        train_data = data[indices[:n_train]]
        test_data = data[indices[n_train:]]
        
        split_dict[band] = {"train": train_data, "test": test_data}
        
    return split_dict

def save_split_data(channel_name, split_dict, base_out_dir):
    """
    Save the split data for a single channel into the appropriate train/test subfolders.
    
    Args:
        channel_name: str (e.g., "channel_1").
        split_dict: dict, output from split_channel_data.
        base_out_dir: str, base directory for output (e.g., new_data).
    """
    # Create channel-specific folders under train and test.
    train_channel_dir = os.path.join(base_out_dir, "train", channel_name)
    test_channel_dir = os.path.join(base_out_dir, "test", channel_name)
    os.makedirs(train_channel_dir, exist_ok=True)
    os.makedirs(test_channel_dir, exist_ok=True)
    
    for band, data_dict in split_dict.items():
        train_path = os.path.join(train_channel_dir, f"{band}.npy")
        test_path = os.path.join(test_channel_dir, f"{band}.npy")
        np.save(train_path, data_dict["train"])
        np.save(test_path, data_dict["test"])

def process_all_channels(input_dir, output_dir, train_ratio=0.9):
    """
    Process all channel folders under input_dir: for each channel, perform the split
    and save the results under output_dir. Also, return a dictionary with the saved paths.
    
    Args:
        input_dir: str, path to the input directory containing channel subfolders.
        output_dir: str, base directory where 'train' and 'test' folders will be created.
        train_ratio: float, the ratio for training split.
    
    Returns:
        dataset_split: dict, keys are channel names and each value is a dict with keys for 
                       each frequency band containing the train and test file paths.
    """
    dataset_split = {}
    # List channel folders in the input directory.
    channel_folders = [os.path.join(input_dir, d) for d in os.listdir(input_dir)
                       if os.path.isdir(os.path.join(input_dir, d))]
    
    for ch_folder in tqdm(channel_folders, desc="Processing channels"):
        channel_name = os.path.basename(ch_folder)
        split_dict = split_channel_data(ch_folder, train_ratio=train_ratio)
        save_split_data(channel_name, split_dict, output_dir)
        # Record the relative paths for later reference.
        dataset_split[channel_name] = {band: {"train": os.path.join("train", channel_name, f"{band}.npy"),
                                               "test": os.path.join("test", channel_name, f"{band}.npy")}
                                        for band in split_dict.keys()}
    return dataset_split

def main():
    parser = argparse.ArgumentParser(description="Split EEG data into training and testing data sets for each channel.")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to input directory containing frequency-partitioned data (one folder per channel).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory where train and test sets will be saved.")
    parser.add_argument("--train_ratio", type=float, required=True, 
                        help="Train-Test split ratio (e.g., 0.9 means 90% training and 10% testing).")
    args = parser.parse_args()

    dataset_split = process_all_channels(args.input_dir, args.output_dir, train_ratio=args.train_ratio)
    
    # Optionally, save the dataset split info to a JSON file.
    out_json_path = os.path.join(args.output_dir, "dataset_paths.json")
    with open(out_json_path, "w") as fo:
        json.dump(dataset_split, fo, indent=2)
    print("Dataset split info saved to:", out_json_path)

if __name__ == "__main__":
    main()
