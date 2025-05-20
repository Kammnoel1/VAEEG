# -*- coding: utf-8 -*-
import os
import json
import random
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
from src.lighten.utils.io import get_files


def merge_data(input_paths, out_dir, n_jobs=10):
    """
    Load .npy clip files in input_paths, concatenate and shuffle,
    then save one .npy per frequency band under out_dir.
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
):
    """
    Discover clip .npy files under base_dir/clips,
    split into train/test by ratio,
    save paths JSON to base_dir/dataset_paths.json if desired,
    and merge per-split datasets.

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
    args = parser.parse_args()

    result = split_dataset(
        base_dir=args.base_dir,
        ratio=args.ratio,
        n_jobs=args.n_jobs,
        save_json=not args.no_json,
    )
    print("Output directories:", result)
