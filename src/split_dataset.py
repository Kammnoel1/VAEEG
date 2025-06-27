# -*- coding: utf-8 -*-
import os
import json
import random
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
from src.lighten.utils.io import get_files


def load_labels_csv(base_dir):
    """
    Load the labels.csv file created by gen_data.py
    
    Args:
        base_dir: Base directory containing labels.csv
        
    Returns:
        dict: Mapping from .npy filename to label
    """
    labels_csv_path = os.path.join(base_dir, "labels.csv")
    if not os.path.exists(labels_csv_path):
        print(f"Warning: No labels.csv found in {base_dir}")
        return {}
    
    df = pd.read_csv(labels_csv_path)
    # Create mapping from filename to label
    label_map = {}
    for _, row in df.iterrows():
        filename = os.path.basename(row['npy_path'])  # Get just the filename
        label_map[filename] = row['label']
    
    return label_map


def merge_data_with_labels(input_paths, out_dir, label_map, n_jobs=10):
    """
    Load .npy clip files in input_paths, concatenate with label tracking,
    then save one .npy per frequency band and corresponding labels under out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    band_names = ["whole", "delta", "theta", "alpha", "low_beta", "high_beta"]

    # Load all data and collect corresponding labels
    print("Loading data files...")
    data_list = Parallel(n_jobs=n_jobs)(delayed(np.load)(f) for f in tqdm(input_paths))
    
    # Collect labels for each clip
    all_labels = []
    for file_path in input_paths:
        filename = os.path.basename(file_path)
        file_data = np.load(file_path)
        n_clips = file_data.shape[0]  # Number of clips in this file
        
        # Get label for this file (default to 0 if not found)
        file_label = label_map.get(filename, 0)
        
        # All clips from this file have the same label
        all_labels.extend([file_label] * n_clips)
    
    # Concatenate all data
    data = np.concatenate(data_list, axis=0)
    labels = np.array(all_labels)
    
    # Create shuffling indices to maintain data-label correspondence
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    
    # Apply shuffling to both data and labels
    data = data[indices]
    labels = labels[indices]

    # Split by band and save along with labels
    for i, name in enumerate(band_names):
        out_file = os.path.join(out_dir, f"{name}.npy")
        labels_file = os.path.join(out_dir, f"{name}_labels.npy")
        
        band_data = data[:, i, :]
        np.save(out_file, band_data)
        np.save(labels_file, labels)  # Same labels for all bands
        
        print(f"Saved {name}: {band_data.shape}, Labels: {labels.shape}")
        print(f"  - Seizure samples: {np.sum(labels == 1)}")
        print(f"  - Background samples: {np.sum(labels == 0)}")


def merge_data(input_paths, out_dir, n_jobs=10):
    """
    Load .npy clip files in input_paths, concatenate and shuffle,
    then save one .npy per frequency band under out_dir.
    
    NOTE: This function is kept for backward compatibility but loses label information.
    Use merge_data_with_labels() instead for label tracking.
    """
    os.makedirs(out_dir, exist_ok=True)
    band_names = ["whole", "delta", "theta", "alpha", "low_beta", "high_beta"]

    # load and concatenate
    data_list = Parallel(n_jobs=n_jobs)(delayed(np.load)(f) for f in tqdm(input_paths))
    data = np.concatenate(data_list, axis=0)
    np.random.shuffle(data)

    # split by band and save
    for i, name in enumerate(band_names):
        out_file = os.path.join(out_dir, f"{name}.npy")
        sx = data[:, i, :]
        np.save(out_file, sx)


def split_dataset(
    base_dir: str,
    ratio: float = 0.1,
    n_jobs: int = 10,
    save_json: bool = True,
    track_labels: bool = True,
):
    """
    Discover clip .npy files under base_dir/clips,
    split into train/test by ratio,
    save paths JSON to base_dir/dataset_paths.json if desired,
    and merge per-split datasets.

    Args:
        base_dir: Base directory containing 'clips' subfolder and optionally 'labels.csv'
        ratio: Fraction of data to reserve for test set
        n_jobs: Number of parallel jobs for merging data
        save_json: Whether to save dataset_paths.json
        track_labels: Whether to track labels from labels.csv (default: True)

    Returns:
        dict: {'train': <train_dir>, 'test': <test_dir>}
    """
    clips_dir = os.path.join(base_dir, "clips")
    files = get_files(clips_dir, [".npy"])
    random.shuffle(files)

    n_total = len(files)
    n_train = int((1.0 - ratio) * n_total)
    train_paths = files[:n_train]
    test_paths = files[n_train:]

    os.makedirs(base_dir, exist_ok=True)

    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    if save_json:
        paths_json = {"train": train_paths, "test": test_paths}
        json_file = os.path.join(base_dir, "dataset_paths.json")
        with open(json_file, "w") as fo:
            json.dump(paths_json, fo, indent=2)

    if track_labels:
        # Load label mapping from labels.csv
        label_map = load_labels_csv(base_dir)
        
        if label_map:
            print("Using label tracking mode")
            print(f"Found labels for {len(label_map)} files")
            
            # Merge data with label tracking
            print("Processing test set...")
            merge_data_with_labels(test_paths, test_dir, label_map, n_jobs=n_jobs)
            print("Processing train set...")
            merge_data_with_labels(train_paths, train_dir, label_map, n_jobs=n_jobs)
        else:
            print("No labels found, falling back to standard mode")
            merge_data(test_paths, test_dir, n_jobs=n_jobs)
            merge_data(train_paths, train_dir, n_jobs=n_jobs)
    else:
        print("Label tracking disabled, using standard mode")
        merge_data(test_paths, test_dir, n_jobs=n_jobs)
        merge_data(train_paths, train_dir, n_jobs=n_jobs)

    return {"train": train_dir, "test": test_dir}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Split clip .npy files and merge per-band datasets"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base directory containing 'clips' subfolder with .npy files",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.1,
        help="Fraction of data to reserve for test set",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=10,
        help="Number of parallel jobs for merging data",
    )
    parser.add_argument(
        "--no_json", action="store_true", help="Disable saving dataset_paths.json"
    )
    parser.add_argument(
        "--no_labels", action="store_true", help="Disable label tracking (for backward compatibility)"
    )
    args = parser.parse_args()

    result = split_dataset(
        base_dir=args.base_dir,
        ratio=args.ratio,
        n_jobs=args.n_jobs,
        save_json=not args.no_json,
        track_labels=not args.no_labels,
    )
    print("Output directories:", result)
