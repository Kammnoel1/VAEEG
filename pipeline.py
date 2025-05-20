# pipeline.py
import os
import glob
import argparse
import yaml

from src.gen_data import generate_clips
from src.split_dataset import split_dataset
from src.train_vae import run_training


def main():
    parser = argparse.ArgumentParser(
        description="Full EEG VAE pipeline: clip generation, dataset split, and training"
    )
    parser.add_argument(
        "--raw_base",
        required=True,
        help="Root directory containing EDFs organized by subfolders",
    )
    parser.add_argument(
        "--out_base",
        required=True,
        help="Output base directory for generated clips and splits",
    )
    parser.add_argument(
        "--yaml_file",
        required=True,
        help="Path to the YAML file containing training configuration",
    )
    parser.add_argument(
        "--z_dim", type=int, default=50, help="Latent space dimension for VAE"
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=10,
        help="Number of parallel jobs for preprocessing",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.1,
        help="Fraction of data reserved for test set",
    )
    args = parser.parse_args()

    # 1. Generate all EEG clips into one folder
    clips_dir = os.path.join(args.out_base, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    print("Generating EEG clips...")
    generate_clips(base_dir=args.raw_base, out_base=clips_dir, n_jobs=args.n_jobs)

    # 2. Split into train/test and merge per-band
    print("Splitting dataset and merging per band...")
    dirs = split_dataset(
        base_dir=args.out_base,
        ratio=args.ratio,
        n_jobs=args.n_jobs,
    )
    train_dir = dirs["train"]
    print(f"Train data directory: {train_dir}")

    # 3. Load and update configuration
    with open(args.yaml_file, "r") as f:
        configs = yaml.safe_load(f)

    configs["DataSet"]["data_dir"] = train_dir

    # 4. Train VAEs for each band
    band_names = ["alpha", "theta", "low_beta", "high_beta", "delta"]
    for band in band_names:
        print(f"=== Starting training for band: {band} ===")
        # update model name in config
        configs["Model"]["name"] = band

        # write a temporary YAML for this band
        tmp_yaml = os.path.join(args.out_base, f"config_{band}.yaml")
        with open(tmp_yaml, "w") as yf:
            yaml.safe_dump(configs, yf)

        # run training
        run_training(yaml_file=tmp_yaml, z_dim=args.z_dim, band_name=band)


if __name__ == "__main__":
    main()
