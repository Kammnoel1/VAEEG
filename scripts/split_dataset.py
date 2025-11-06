# -*- coding: utf-8 -*-
import os
import json
from tqdm import tqdm
import numpy as np
import time 
from lighten.utils.io import get_files
from utils.labels import load_labels_csv


def merge_data_with_labels(input_paths, out_dir, labels_dir, label_map, n_jobs):
    """
    Merge data with label tracking using parallel loading and in-memory operations.
    Similar to original approach but with 4D arrays and label tracking.
    
    Args:
        input_paths: List of .npy file paths to process
        out_dir: Output directory for frequency band data
        labels_dir: Output directory for labels
        label_map: Dictionary mapping filenames to labels
        n_jobs: Number of parallel jobs for loading files 
    """
    band_names = ["whole", "delta", "theta", "alpha", "low_beta", "high_beta"]
    print(f"Loading data in parallel with {n_jobs} jobs...")
    from joblib import Parallel, delayed
    
    def load_file_with_label(file_path):
        """Load file and return data with corresponding labels"""
        filename = os.path.basename(file_path)
        file_data = np.load(file_path)
        file_label = label_map.get(filename, 0)
        labels_array = np.full(file_data.shape[0], file_label, dtype=np.int32)
        return file_data, labels_array
        
    results = Parallel(n_jobs=n_jobs)(
        delayed(load_file_with_label)(f) for f in tqdm(input_paths, desc="Loading files")
    )
    
    # Filter out failed loads and separate data and labels
    data_list, labels_list = zip(*results)
    # Concatenate all data and labels
    print("Concatenating data...")
    data = np.concatenate(data_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    
    # Shuffle data and labels together
    print("Shuffling data...")
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data[:] = data[indices]
    labels[:] = labels[indices]
    
    # Save labels
    split_name = os.path.basename(out_dir)  # 'train' or 'test'
    labels_file = os.path.join(labels_dir, f"{split_name}.npy")
    np.save(labels_file, labels)
    print(f"Saved labels: {labels_file} - Shape: {labels.shape}")
    print(f"  - Seizure samples: {np.sum(labels == 1)}")
    print(f"  - Background samples: {np.sum(labels == 0)}")

    # Save each frequency band
    for i, name in enumerate(band_names):
        print(f"Saving {name} to {out_dir}")
        out_file = os.path.join(out_dir, f"{name}.npy")
        band_data = data[:, i, :, :]
        np.save(out_file, band_data)
        print(f"Saved {name}: {band_data.shape}")


def split_dataset(
    base_dir: str,
    labels_dir: str = None,
    labels_csv_path: str = None,
    ratio: float = 0.1,
    n_jobs: int = 1,
):
    """
    Discover clip .npy files under base_dir/clips,
    split into train/test by ratio,
    save paths JSON to base_dir/dataset_paths.json if desired,
    and merge per-split datasets.

    Args:
        base_dir: Base directory containing 'clips' subfolder
        labels_dir: Directory to save label files (if None, defaults to base_dir/labels)
        labels_csv_path: Path to the labels.csv file (if None, defaults to base_dir/labels.csv)
        ratio: Fraction of data to reserve for test set
        n_jobs: Number of parallel jobs for loading files (default: 1)

    Returns:
        dict: {'train': <train_dir>, 'test': <test_dir>, 'labels': <labels_dir>}
    """
    clips_dir = os.path.join(base_dir, "clips")
    files = get_files(clips_dir, [".npy"])
    np.random.shuffle(files)

    n_total = len(files)
    n_train = int((1.0 - ratio) * n_total)
    train_paths = files[:n_train]
    test_paths = files[n_train:]

    os.makedirs(base_dir, exist_ok=True)
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    paths_json = {"train": train_paths, "test": test_paths}
    json_file = os.path.join(base_dir, "dataset_paths.json")
    with open(json_file, "w") as fo:
        json.dump(paths_json, fo, indent=2)

    
    # Load label mapping from labels.csv
    label_map = load_labels_csv(labels_csv_path)
    
    if label_map:
        print("Using label tracking mode")
        print(f"Found labels for {len(label_map)} files")
        
        # Merge data with label tracking
        print("Processing test set...")
        merge_data_with_labels(test_paths, test_dir, labels_dir, label_map, n_jobs)
        print("Processing train set...")
        merge_data_with_labels(train_paths, train_dir, labels_dir, label_map, n_jobs)
    else:
        raise ValueError("Labels are required but not found")

    return {"train": train_dir, "test": test_dir, "labels": labels_dir}


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
        "--labels_dir",
        type=str,
        default=None,
        help="Directory to save label files (default: <base_dir>/labels)",
    )
    parser.add_argument(
        "--labels_csv_path",
        type=str,
        default=None,
        help="Path to the labels.csv file (default: <base_dir>/labels.csv)",
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
        default=1,
        help="Number of parallel jobs for loading files (default: 1)",
    )
    args = parser.parse_args()
    
    start_time = time.time()
    result = split_dataset(
        base_dir=args.base_dir,
        labels_dir=args.labels_dir,
        labels_csv_path=args.labels_csv_path,
        ratio=args.ratio,
        n_jobs=args.n_jobs,
    )
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")
    print(f"Results: {result}")