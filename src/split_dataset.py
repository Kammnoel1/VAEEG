# -*- coding: utf-8 -*-
import os
import json
import random
import shutil
import pandas as pd
from tqdm import tqdm
import numpy as np
from src.lighten.utils.io import get_files


def load_labels_csv(labels_csv_path):
    """
    Load the labels.csv file created by gen_data.py
    
    Args:
        labels_csv_path: Full path to the labels.csv file
        
    Returns:
        dict: Mapping from .npy filename to label
    """
    if not os.path.exists(labels_csv_path):
        print(f"Warning: No labels.csv found at {labels_csv_path}")
        return {}
    
    df = pd.read_csv(labels_csv_path)
    # Create mapping from filename to label
    label_map = {}
    for _, row in df.iterrows():
        filename = os.path.basename(row['npy_path'])  # Get just the filename
        label_map[filename] = row['label']
    
    return label_map


def get_data_shape_and_count(input_paths):
    """
    Determine the total data shape and count without loading all data into memory.
    """
    print("Analyzing data structure...")
    total_samples = 0
    n_bands = None
    n_features = None
    
    # Process files one by one using memmap
    for file_path in tqdm(input_paths, desc="Analyzing files"):
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


def merge_data_with_labels(input_paths, out_dir, labels_dir, label_map):
    """
    Memory-efficient merge using memmap and direct file-by-file processing.
    Load .npy clip files in input_paths one at a time, copy directly to memmap arrays
    with label tracking, then save one .npy per frequency band under out_dir and labels under labels_dir.
    
    Args:
        input_paths: List of .npy file paths to process
        out_dir: Output directory for frequency band data
        labels_dir: Output directory for labels
        label_map: Dictionary mapping filenames to labels
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    band_names = ["whole", "delta", "theta", "alpha", "low_beta", "high_beta"]

    # First pass: determine total data size
    total_samples, n_bands, n_features = get_data_shape_and_count(input_paths)
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
    
    # Second pass: populate memmap arrays file by file (no chunking needed)
    print("Populating memmap arrays...")
    current_idx = 0
    
    for file_path in tqdm(input_paths, desc="Processing files"):
        try:
            filename = os.path.basename(file_path)
            # Load file data directly
            file_data = np.load(file_path)
            n_clips = file_data.shape[0]
            
            # Get label for this file
            file_label = label_map.get(filename, 0)
            
            # Copy directly to memmap arrays (no intermediate concatenation)
            end_idx = current_idx + n_clips
            for j, name in enumerate(band_names):
                band_memmaps[name][current_idx:end_idx] = file_data[:, j, :]
            
            # Set labels for all clips in this file
            labels_memmap[current_idx:end_idx] = file_label
            
            current_idx = end_idx
            
            # Periodic flush to ensure data is written to disk
            if current_idx % 50000 == 0:  # Flush every 50k samples
                for memmap_array in band_memmaps.values():
                    memmap_array.flush()
                labels_memmap.flush()
                
        except Exception as e:
            print(f"Warning: Could not process {file_path}: {e}")
            continue
    
    # Final flush
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
    
    # Save labels only once (not per frequency band)
    # Since labels are identical across all frequency bands, we only save:
    # - train_labels.npy in the labels directory
    # - test_labels.npy in the labels directory
    split_name = os.path.basename(out_dir)  # 'train' or 'test'
    labels_file = os.path.join(labels_dir, f"{split_name}_labels.npy")
    np.save(labels_file, labels_shuffled)
    print(f"Saved labels: {labels_file} - Shape: {labels_shuffled.shape}")
    print(f"  - Seizure samples: {np.sum(labels_shuffled == 1)}")
    print(f"  - Background samples: {np.sum(labels_shuffled == 0)}")
    
    for name in band_names:
        print(f"Processing band: {name}")
        out_file = os.path.join(out_dir, f"{name}.npy")
        
        # Apply shuffling using memmap
        band_shuffled = band_memmaps[name][indices]
        
        # Save to final files (frequency band data only, no labels)
        np.save(out_file, band_shuffled)
        
        print(f"Saved {name}: {band_shuffled.shape}")
    
    # Cleanup temporary memmap files
    print("Cleaning up temporary files...")
    shutil.rmtree(temp_dir)


def merge_data(input_paths, out_dir):
    """
    Memory-efficient merge using memmap and direct file-by-file processing.
    Load .npy clip files in input_paths one at a time, copy directly to memmap arrays,
    then save one .npy per frequency band under out_dir.
    
    NOTE: This function is kept for backward compatibility but loses label information.
    Use merge_data_with_labels() instead for label tracking.
    
    Args:
        input_paths: List of .npy file paths to process
        out_dir: Output directory for frequency band data
    """
    os.makedirs(out_dir, exist_ok=True)
    band_names = ["whole", "delta", "theta", "alpha", "low_beta", "high_beta"]

    # First pass: determine total data size
    total_samples, n_bands, n_features = get_data_shape_and_count(input_paths)
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
    
    # Second pass: populate memmap arrays file by file (no chunking needed)
    print("Populating memmap arrays...")
    current_idx = 0
    
    for file_path in tqdm(input_paths, desc="Processing files"):
        try:
            # Load file data directly
            file_data = np.load(file_path)
            n_clips = file_data.shape[0]
            
            # Copy directly to memmap arrays (no intermediate concatenation)
            end_idx = current_idx + n_clips
            for j, name in enumerate(band_names):
                band_memmaps[name][current_idx:end_idx] = file_data[:, j, :]
            
            current_idx = end_idx
            
            # Periodic flush to ensure data is written to disk
            if current_idx % 50000 == 0:  # Flush every 50k samples
                for memmap_array in band_memmaps.values():
                    memmap_array.flush()
                    
        except Exception as e:
            print(f"Warning: Could not process {file_path}: {e}")
            continue
    
    # Final flush
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
    labels_dir: str = None,
    labels_csv_path: str = None,
    ratio: float = 0.1,
    save_json: bool = True,
    track_labels: bool = True,
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
        save_json: Whether to save dataset_paths.json
        track_labels: Whether to track labels from labels.csv (default: True)

    Returns:
        dict: {'train': <train_dir>, 'test': <test_dir>, 'labels': <labels_dir>}
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
    
    # Set default labels directory if not provided
    if labels_dir is None:
        labels_dir = os.path.join(base_dir, "labels")
    
    # Set default labels.csv path if not provided
    if labels_csv_path is None:
        labels_csv_path = os.path.join(base_dir, "labels.csv")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    if save_json:
        paths_json = {"train": train_paths, "test": test_paths}
        json_file = os.path.join(base_dir, "dataset_paths.json")
        with open(json_file, "w") as fo:
            json.dump(paths_json, fo, indent=2)

    if track_labels:
        # Load label mapping from labels.csv
        label_map = load_labels_csv(labels_csv_path)
        
        if label_map:
            print("Using label tracking mode")
            print(f"Found labels for {len(label_map)} files")
            
            # Merge data with label tracking
            print("Processing test set...")
            merge_data_with_labels(test_paths, test_dir, labels_dir, label_map)
            print("Processing train set...")
            merge_data_with_labels(train_paths, train_dir, labels_dir, label_map)
        else:
            print("No labels found, falling back to standard mode")
            merge_data(test_paths, test_dir)
            merge_data(train_paths, train_dir)
    else:
        print("Label tracking disabled, using standard mode")
        merge_data(test_paths, test_dir)
        merge_data(train_paths, train_dir)

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
        "--no_json", action="store_true", help="Disable saving dataset_paths.json"
    )
    parser.add_argument(
        "--no_labels", action="store_true", help="Disable label tracking (for backward compatibility)"
    )
    args = parser.parse_args()

    result = split_dataset(
        base_dir=args.base_dir,
        labels_dir=args.labels_dir,
        labels_csv_path=args.labels_csv_path,
        ratio=args.ratio,
        save_json=not args.no_json,
        track_labels=not args.no_labels,
    )
    print("Output directories:", result)
