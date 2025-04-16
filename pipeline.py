import argparse
import yaml
import os
import itertools

# Import the pipeline functions from your modules.
from src.gen_data import generate_data
from src.partition_frequency_bands import save_partitioned_data
from src.split_dataset import process_all_channels
from src.train_vae import train_model_for_band

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
    # Step 1: Data Generation (Segmentation)
    #############################################
    # Expected keys under "RawData" in the YAML.
    raw_config = config.get("RawData", {})
    raw_in_file = raw_config.get("in_file")
    sfreq = raw_config.get("sfreq")
    seg_out_dir = raw_config.get("out_dir")
    seg_out_prefix = raw_config.get("out_prefix")
    
    print("=== Step 1: Generating segmented data ===")
    segments = generate_data(raw_in_file, sfreq, seg_out_dir, seg_out_prefix)

    #############################################
    # Step 2: Partition Data into Frequency Bands
    #############################################
    # Expected keys under "Partition" in the YAML.
    partition_config = config.get("Partition", {})
    partition_out_dir = partition_config.get("output_dir")
    merged_seg_file = os.path.join(seg_out_dir, f"{seg_out_prefix}.npy")
    
    print("=== Step 2: Partitioning data into frequency bands per channel ===")
    save_partitioned_data(merged_seg_file, partition_out_dir, sfreq)

    #############################################
    # Step 3: Split Data into Train/Test Sets
    #############################################
    # Expected keys under "Split" in the YAML.
    split_config = config.get("Split", {})
    split_input_dir = split_config.get("input_dir")
    split_output_dir = split_config.get("output_dir")
    train_ratio = split_config.get("train_ratio", 0.9)
    
    print("=== Step 3: Splitting data into train and test sets ===")
    dataset_split = process_all_channels(split_input_dir, split_output_dir, train_ratio=train_ratio)
    
    #############################################
    # Step 4: Train Models for Each Frequency Band
    #############################################
    # The "Train", "Model", and "DataSet" sections in the YAML are used for training.
    print("=== Step 4: Training models for each frequency band ===")
    # Define the grid of hyperparameter values.
    n_epoch_options = [30, 50, 70, 90, 100]
    lr_options = [0.1, 0.01, 0.001, 0.0005]
    z_dim_options = [3, 50, 100, 256]
    negative_slope_options = [0.01, 0.1, 0.2, 0.3]
    decoder_last_lstm_options = [True, False]
    batch_size_options = [8, 16, 32, 64, 128]
    # You can add more hyperparameters as needed.

    # Generate all combinations.
    for n_epoch, lr, z_dim, negative_slope, decoder_last_lstm, batch_size in itertools.product(
            n_epoch_options,
            lr_options,
            z_dim_options,
            negative_slope_options,
            decoder_last_lstm_options,
            batch_size_options
    ):
        # Update your config (loaded from YAML) with these values.
        config["Train"]["n_epoch"] = n_epoch
        config["Train"]["lr"] = lr
        config["Model"]["z_dim"] = z_dim
        config["Model"]["negative_slope"] = negative_slope
        config["Model"]["decoder_last_lstm"] = decoder_last_lstm
        config["DataSet"]["batch_size"] = batch_size 

        # Optionally, log or print the current hyperparameter combination.
        print(
            f"Training with n_epoch={n_epoch}, lr={lr}, z_dim={z_dim}, "
            f"negative_slope={negative_slope}, decoder_last_lstm={decoder_last_lstm}, "
            f"batch_size={batch_size}"
        )
            
        # Loop over the frequency bands (if that’s part of your pipeline).
        for band in ["delta", "theta", "alpha", "low_beta", "high_beta"]:
            config["Model"]["name"] = band
            model_dir = os.path.join(
                        config["Train"]["model_dir"],
                        f"{band}_z{z_dim}_ep{n_epoch}_lr{lr}_bs{batch_size}_lstm{decoder_last_lstm}_ns{negative_slope}"
                        )
            os.makedirs(model_dir, exist_ok=True)
            train_model_for_band(band, config, z_dim)

    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()

