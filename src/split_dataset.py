# -*- coding: utf-8 -*-
import os
import json
import random
import shutil
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


def get_data_shape_and_count(input_paths, chunk_size=100):
    """
    Determine the total data shape and count without loading all data into memory.
    """
    print("Analyzing data structure...")
    total_samples = 0
    n_bands = None
    n_features = None
    
    # Process files in chunks to avoid memory issues
    for i in tqdm(range(0, len(input_paths), chunk_size), desc="Analyzing files"):
        chunk_paths = input_paths[i:i+chunk_size]
        for file_path in chunk_paths:
            try:
                # Use memmap to avoid loading into memory
                data = np.load(file_path, mmap_mode='r')
                if n_bands is None:
                    n_bands = data.shape[1]
                    n_features = data.shape[2]
                total_samples += data.shape[0]
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
                continue
    
    return total_samples, n_bands, n_features


def merge_data_with_labels(input_paths, out_dir, label_map, n_jobs=10, chunk_size=100):
    """
    Memory-efficient merge using memmap and chunked processing.
    Load .npy clip files in input_paths, concatenate with label tracking,
    then save one .npy per frequency band and corresponding labels under out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    band_names = ["whole", "delta", "theta", "alpha", "low_beta", "high_beta"]

    # First pass: determine total data size
    total_samples, n_bands, n_features = get_data_shape_and_count(input_paths, chunk_size)
    print(f"Total samples: {total_samples}, Bands: {n_bands}, Features: {n_features}")
    
    # Create temporary memmap files for each band
    temp_dir = os.path.join(out_dir, "temp_memmap")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create memmap arrays for each band
    band_memmaps = {}
    for i, name in enumerate(band_names):
        temp_file = os.path.join(temp_dir, f"{name}_temp.dat")
        band_memmaps[name] = np.memmap(
            temp_file, dtype=np.float32, mode='w+', 
            shape=(total_samples, n_features)
        )
    
    # Create memmap for labels
    labels_temp_file = os.path.join(temp_dir, "labels_temp.dat")
    labels_memmap = np.memmap(
        labels_temp_file, dtype=np.int32, mode='w+', shape=(total_samples,)
    )
    
    # Second pass: populate memmap arrays in chunks
    print("Populating memmap arrays...")
    current_idx = 0
    
    for i in tqdm(range(0, len(input_paths), chunk_size), desc="Processing chunks"):
        chunk_paths = input_paths[i:i+chunk_size]
        chunk_data_list = []
        chunk_labels = []
        
        # Load current chunk
        for file_path in chunk_paths:
            try:
                filename = os.path.basename(file_path)
                file_data = np.load(file_path)
                n_clips = file_data.shape[0]
                
                # Get label for this file
                file_label = label_map.get(filename, 0)
                
                chunk_data_list.append(file_data)
                chunk_labels.extend([file_label] * n_clips)
            except Exception as e:
                print(f"Warning: Could not process {file_path}: {e}")
                continue
        
        if not chunk_data_list:
            continue
            
        # Concatenate chunk data
        chunk_data = np.concatenate(chunk_data_list, axis=0)
        chunk_labels = np.array(chunk_labels)
        chunk_size_actual = chunk_data.shape[0]
        
        # Copy to memmap arrays
        for j, name in enumerate(band_names):
            band_memmaps[name][current_idx:current_idx+chunk_size_actual] = chunk_data[:, j, :]
        
        labels_memmap[current_idx:current_idx+chunk_size_actual] = chunk_labels
        current_idx += chunk_size_actual
        
        # Force sync to disk and cleanup
        for memmap_array in band_memmaps.values():
            memmap_array.flush()
        labels_memmap.flush()
    
    # Create shuffling indices
    print("Creating shuffling indices...")
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    
    # Apply shuffling and save final arrays
    print("Shuffling and saving final arrays...")
    labels_shuffled = labels_memmap[indices]
    
    for name in band_names:
        print(f"Processing band: {name}")
        out_file = os.path.join(out_dir, f"{name}.npy")
        labels_file = os.path.join(out_dir, f"{name}_labels.npy")
        
        # Apply shuffling using memmap
        band_shuffled = band_memmaps[name][indices]
        
        # Save to final files
        np.save(out_file, band_shuffled)
        np.save(labels_file, labels_shuffled)
        
        print(f"Saved {name}: {band_shuffled.shape}, Labels: {labels_shuffled.shape}")
        print(f"  - Seizure samples: {np.sum(labels_shuffled == 1)}")
        print(f"  - Background samples: {np.sum(labels_shuffled == 0)}")
    
    # Cleanup temporary memmap files
    print("Cleaning up temporary files...")
    shutil.rmtree(temp_dir)


def merge_data(input_paths, out_dir, n_jobs=10, chunk_size=100):
    """
    Memory-efficient merge using memmap and chunked processing.
    Load .npy clip files in input_paths, concatenate and shuffle,
    then save one .npy per frequency band under out_dir.
    
    NOTE: This function is kept for backward compatibility but loses label information.
    Use merge_data_with_labels() instead for label tracking.
    """
    os.makedirs(out_dir, exist_ok=True)
    band_names = ["whole", "delta", "theta", "alpha", "low_beta", "high_beta"]

    # First pass: determine total data size
    total_samples, n_bands, n_features = get_data_shape_and_count(input_paths, chunk_size)
    print(f"Total samples: {total_samples}, Bands: {n_bands}, Features: {n_features}")
    
    # Create temporary memmap files for each band
    temp_dir = os.path.join(out_dir, "temp_memmap")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create memmap arrays for each band
    band_memmaps = {}
    for i, name in enumerate(band_names):
        temp_file = os.path.join(temp_dir, f"{name}_temp.dat")
        band_memmaps[name] = np.memmap(
            temp_file, dtype=np.float32, mode='w+', 
            shape=(total_samples, n_features)
        )
    
    # Second pass: populate memmap arrays in chunks
    print("Populating memmap arrays...")
    current_idx = 0
    
    for i in tqdm(range(0, len(input_paths), chunk_size), desc="Processing chunks"):
        chunk_paths = input_paths[i:i+chunk_size]
        chunk_data_list = []
        
        # Load current chunk
        for file_path in chunk_paths:
            try:
                file_data = np.load(file_path)
                chunk_data_list.append(file_data)
            except Exception as e:
                print(f"Warning: Could not process {file_path}: {e}")
                continue
        
        if not chunk_data_list:
            continue
            
        # Concatenate chunk data
        chunk_data = np.concatenate(chunk_data_list, axis=0)
        chunk_size_actual = chunk_data.shape[0]
        
        # Copy to memmap arrays
        for j, name in enumerate(band_names):
            band_memmaps[name][current_idx:current_idx+chunk_size_actual] = chunk_data[:, j, :]
        
        current_idx += chunk_size_actual
        
        # Force sync to disk
        for memmap_array in band_memmaps.values():
            memmap_array.flush()
    
    # Create shuffling indices
    print("Creating shuffling indices...")
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    
    # Apply shuffling and save final arrays
    print("Shuffling and saving final arrays...")
    for name in band_names:
        print(f"Processing band: {name}")
        out_file = os.path.join(out_dir, f"{name}.npy")
        
        # Apply shuffling using memmap
        band_shuffled = band_memmaps[name][indices]
        
        # Save to final file
        np.save(out_file, band_shuffled)
        print(f"Saved {name}: {band_shuffled.shape}")
    
    # Cleanup temporary memmap files
    print("Cleaning up temporary files...")
    shutil.rmtree(temp_dir)


def split_dataset(
    base_dir: str,
    ratio: float = 0.1,
    n_jobs: int = 10,
    save_json: bool = True,
    track_labels: bool = True,
    chunk_size: int = 50,
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
        chunk_size: Number of files to process at once (default: 50)

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
            merge_data_with_labels(test_paths, test_dir, label_map, n_jobs=n_jobs, chunk_size=chunk_size)
            print("Processing train set...")
            merge_data_with_labels(train_paths, train_dir, label_map, n_jobs=n_jobs, chunk_size=chunk_size)
        else:
            print("No labels found, falling back to standard mode")
            merge_data(test_paths, test_dir, n_jobs=n_jobs, chunk_size=chunk_size)
            merge_data(train_paths, train_dir, n_jobs=n_jobs, chunk_size=chunk_size)
    else:
        print("Label tracking disabled, using standard mode")
        merge_data(test_paths, test_dir, n_jobs=n_jobs, chunk_size=chunk_size)
        merge_data(train_paths, train_dir, n_jobs=n_jobs, chunk_size=chunk_size)

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
        "--chunk_size",
        type=int,
        default=50,
        help="Number of files to process at once (lower values use less memory)",
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
        chunk_size=args.chunk_size,
    )
    print("Output directories:", result)
