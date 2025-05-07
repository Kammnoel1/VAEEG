import argparse
import yaml
import os
import itertools

# Import the pipeline functions from your modules.
from src.gen_data import generate_full
from src.split_data import split_memmap
import src.gen_data
import src.train_vae

def main():
    parser = argparse.ArgumentParser(description="End-to-End Pipeline for VAEEG")
    parser.add_argument("--yaml_file", type=str, required=True,
                        help="Path to the YAML configuration file")
    parser.add_argument("--z_dim", type=int, required=True,
                        help="Latent space dimension for training")
    args = parser.parse_args()

    # Load the configuration from YAML.
    with open(args.yaml_file, "r") as f:
        config = yaml.safe_load(f)

    #############################################
    # Step 1: Data Segmentation and Partitioning
    #############################################
    
    raw_config = config.get("RawData", {})
    raw_in_file = raw_config.get("in_file")
    sfreq = raw_config.get("sfreq")
    out_dir = raw_config.get("out_dir")
    num_workers = raw_config.get("num_workers")
    ratio = raw_config.get("ratio")
    
    print("=== Step 1: Generating data ===")
    data_path = generate_full(raw_in_file, sfreq, out_dir, num_workers)

    #############################################
    # Step 3: Data Splitting
    #############################################

    print("=== Step 1: Splitting data ===")
    paths = split_memmap(data_path, ratio, out_dir)

    #############################################
    # Step 2: Train Models for Each Frequency Band
    #############################################
    
    print("=== Step 2: Training models for each frequency band ===")
    # Define the grid of hyperparameter values.
    # n_epoch_options = [50]
    # lr_options = [0.01]
    # z_dim_options = [50]
    # negative_slope_options = [0.2]
    # decoder_last_lstm_options = [True, False]
    # batch_size_options = [64]
    # # You can add more hyperparameters as needed.

    # # Generate all combinations.
    # for n_epoch, lr, z_dim, negative_slope, decoder_last_lstm, batch_size in itertools.product(
    #         n_epoch_options,
    #         lr_options,
    #         z_dim_options,
    #         negative_slope_options,
    #         decoder_last_lstm_options,
    #         batch_size_options
    # ):
    #     # Update your config (loaded from YAML) with these values.
    #     config["Train"]["n_epoch"] = n_epoch
    #     config["Train"]["lr"] = lr
    #     config["Model"]["z_dim"] = z_dim
    #     config["Model"]["negative_slope"] = negative_slope
    #     config["Model"]["decoder_last_lstm"] = decoder_last_lstm
    #     config["DataSet"]["batch_size"] = batch_size 

    #     # Optionally, log or print the current hyperparameter combination.
    #     print(
    #         f"Training with n_epoch={n_epoch}, lr={lr}, z_dim={z_dim}, "
    #         f"negative_slope={negative_slope}, decoder_last_lstm={decoder_last_lstm}, "
    #         f"batch_size={batch_size}"
    #     )
    z_dim = args.z_dim     
    for band in ["delta", "theta", "alpha", "low_beta", "high_beta"]:
        src.train_vae.train_model_for_band(band, config, z_dim, paths)

    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()

