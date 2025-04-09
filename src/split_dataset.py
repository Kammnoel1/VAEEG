import os
import json
import random
import numpy as np
from tqdm import tqdm

def split_channel_data(channel_folder, train_ratio=0.9):
    """
    For a given channel folder, load each frequency band .npy file,
    shuffle and split the data into training and test sets.
    
    Args:
        channel_folder: str, path to the channel folder (e.g., new_data/single_channel_segments/channel_1)
        train_ratio: float, proportion of data to assign to training.
        
    Returns:
        A dictionary with keys as band names and values as dicts with keys "train" and "test" holding the file paths.
    """
    # List of expected frequency band file names (without extension)
    band_names = ["whole", "delta", "theta", "alpha", "low_beta", "high_beta"]
    split_paths = {}
    
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
        
        split_paths[band] = {"train": train_data, "test": test_data}
        print(f"Channel '{os.path.basename(channel_folder)}', band '{band}': Total: {num_segments}, "
              f"Train: {train_data.shape[0]}, Test: {test_data.shape[0]}")
        
    return split_paths

def save_split_data(channel_name, split_dict, base_out_dir):
    """
    Save the split data for a single channel into the appropriate train/test folders.
    
    Args:
        channel_name: str, e.g., "channel_1"
        split_dict: dict, output from split_channel_data.
        base_out_dir: str, base directory for output (e.g., new_data)
    """
    # Create channel-specific folders inside both train and test directories.
    train_channel_dir = os.path.join(base_out_dir, "train", channel_name)
    test_channel_dir = os.path.join(base_out_dir, "test", channel_name)
    os.makedirs(train_channel_dir, exist_ok=True)
    os.makedirs(test_channel_dir, exist_ok=True)
    
    for band, data_dict in split_dict.items():
        train_path = os.path.join(train_channel_dir, f"{band}.npy")
        test_path = os.path.join(test_channel_dir, f"{band}.npy")
        np.save(train_path, data_dict["train"])
        np.save(test_path, data_dict["test"])
        print(f"Saved {band}: train -> {train_path}, test -> {test_path}")

if __name__ == "__main__":
    # Input directory containing channel folders (each channel folder contains 6 .npy files)
    input_base_dir = "./new_data/single_channel_segments_downsampled"
    # Output base directory where train/ and test/ folders will be created
    out_base_dir = "./new_data_downsampled"
    train_ratio = 0.9
    
    # Get list of channel directories inside the input base directory.
    channel_folders = [os.path.join(input_base_dir, d) for d in os.listdir(input_base_dir)
                       if os.path.isdir(os.path.join(input_base_dir, d))]
    
    dataset_split = {}
    
    for ch_folder in tqdm(channel_folders, desc="Processing channels"):
        channel_name = os.path.basename(ch_folder)
        split_dict = split_channel_data(ch_folder, train_ratio=train_ratio)
        save_split_data(channel_name, split_dict, out_base_dir)
        dataset_split[channel_name] = {band: {"train": os.path.join("train", channel_name, f"{band}.npy"),
                                               "test": os.path.join("test", channel_name, f"{band}.npy")}
                                        for band in split_dict.keys()}
    
    # Optionally, save a JSON log of the dataset split paths
    json_path = os.path.join(out_base_dir, "dataset_paths.json")
    with open(json_path, "w") as f:
        json.dump(dataset_split, f, indent=2)
    print("Dataset split info saved to:", json_path)
