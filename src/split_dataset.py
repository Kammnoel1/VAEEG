import json
import os
import random
import numpy as np
from tqdm import tqdm

# Assuming get_files is defined in src/utils/io.py; update the import if needed.
from src.utils.io import get_files

def merge_data_from_single_file(in_file, out_dir, train_ratio=0.9):
    """
    Load a single .npy file, split it into training and test portions,
    and save each frequency band separately.
    
    Args:
        in_file: str, path to the single .npy file.
        out_dir: str, base directory where "train" and "test" subfolders will be created.
        train_ratio: float, proportion of data to use for training.
    """
    # Load the single merged file (assumed shape: (N, 6, L))
    full_data = np.load(in_file)
    num_segments = full_data.shape[0]
    n_train = int(train_ratio * num_segments)
    train_data = full_data[:n_train]
    test_data = full_data[n_train:]
    
    print(f"Total segments: {num_segments}, Training segments: {n_train}, Test segments: {num_segments - n_train}")

    # Define band names corresponding to axis=1
    band_names = ["whole", "delta", "theta", "alpha", "low_beta", "high_beta"]
    
    # Create output directories for train and test if they don't exist
    train_dir = os.path.join(out_dir, "train")
    test_dir = os.path.join(out_dir, "test")
    if not os.path.isdir(train_dir):
        os.makedirs(train_dir)
    if not os.path.isdir(test_dir):
        os.makedirs(test_dir)
    
    # For each band, extract that slice and save separately for train and test
    for i, name in enumerate(band_names):
        train_file = os.path.join(train_dir, name + ".npy")
        test_file = os.path.join(test_dir, name + ".npy")
        np.save(train_file, train_data[:, i, :])
        np.save(test_file, test_data[:, i, :])
        print(f"Saved band '{name}' to train: {train_file} and test: {test_file}")

if __name__ == "__main__":
    base_dir = "./new_data/"
    # We assume that the single merged file is in the "clips" subfolder.
    merged_file = os.path.join(base_dir, "clips", "VAEEG.npy")
    path_file = os.path.join(base_dir, "dataset_paths.json")
    
    # Check if there is only one file in the clips folder
    files = get_files(os.path.join(base_dir, "clips"), [".npy"])
    
    if len(files) == 0:
        raise ValueError("No .npy files found in ./new_data/clips")
    elif len(files) == 1:
        print("Only one file found. Splitting the single merged file into train and test.")
        merge_data_from_single_file(files[0], base_dir, train_ratio=0.9)
        # Write a simple JSON log indicating that train and test were split from the single file.
        split_info = {"train": "split from single file", "test": "split from single file"}
        with open(path_file, "w") as fo:
            json.dump(split_info, fp=fo, indent=1)
    else:
        # If there are multiple files, use the original approach.
        random.shuffle(files)
        ratio = 0.1
        n_train = int((1.0 - ratio) * len(files))
        train_paths = files[:n_train]
        test_paths = files[n_train:]
        with open(path_file, "w") as fo:
            json.dump({"train": train_paths, "test": test_paths}, fp=fo, indent=1)
        # merge_data is assumed to merge multiple files; call it on train and test lists.
        def merge_data(input_paths, out_dir):
            band_names = ["whole", "delta", "theta", "alpha",  "low_beta", "high_beta"]
            # Load all files and concatenate along axis=0
            data = [np.load(f) for f in tqdm(input_paths)]
            data = np.concatenate(data, axis=0)
            np.random.shuffle(data)
            for i, name in enumerate(band_names):
                out_file = os.path.join(out_dir, name + ".npy")
                sx = data[:, i, :]
                np.save(out_file, sx)
        test_dir = os.path.join(base_dir, "test")
        if not os.path.isdir(test_dir):
            os.makedirs(test_dir)
        merge_data(test_paths, test_dir)
        train_dir = os.path.join(base_dir, "train")
        if not os.path.isdir(train_dir):
            os.makedirs(train_dir)
        merge_data(train_paths, train_dir)
