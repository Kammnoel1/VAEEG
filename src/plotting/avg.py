import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def plot_average_frequency_band(data_path, labels_path, output_dir):
    """
    Plot the average of the first dimension of a frequency band.
    
    Args:
        data_path (str): Path to the frequency band data (.npy file)
        labels_path (str): Path to the corresponding labels (.npy file)
        output_dir (str): Directory to save the output figure
    """
    # Extract frequency band name from data path
    frequency_band = os.path.basename(data_path).replace('.npy', '').title()
    
    # Load data and labels
    data = np.load(data_path)  # Shape: (n_segments, seg_len)
    labels = np.load(labels_path)  # Shape: (n_segments,)
    
    print(f"Data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Frequency band: {frequency_band}")
    
    # Separate seizure and background data
    seizure_mask = labels == 1
    background_mask = labels == 0
    
    seizure_data = data[seizure_mask]
    background_data = data[background_mask]
    
    print(f"Seizure segments: {np.sum(seizure_mask)}")
    print(f"Background segments: {np.sum(background_mask)}")
    
    # Calculate averages across the first dimension (average across segments)
    seizure_avg = np.mean(seizure_data, axis=0)
    background_avg = np.mean(background_data, axis=0)
    
    # Create time axis
    time_points = np.arange(data.shape[1])
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(time_points, seizure_avg, label='Seizure', color='red', linewidth=2)
    plt.plot(time_points, background_avg, label='Background', color='blue', linewidth=2)
    
    plt.xlabel('Time Points')
    plt.ylabel('Average Amplitude')
    plt.title(f'{frequency_band} Band Average: Seizure vs Background')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the figure with frequency band in filename
    output_path = os.path.join(output_dir, f'{frequency_band.lower()}_band_average.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")
    
    plt.show()


def main():
    """
    Main function to parse arguments and generate the plot.
    """
    parser = argparse.ArgumentParser(
        description='Plot average of first dimension of frequency band data'
    )
    
    parser.add_argument(
        "--data_path",
        required=True,
        help='Path to the frequency band data (.npy file)'
    )
    
    parser.add_argument(
        "--labels_path",
        required=True,
        help='Path to the corresponding labels (.npy file)'
    )
    
    parser.add_argument(
        '--output_dir',
        default="/u/noka/VAEEG/figs",
        type=str,
        help='Directory to save the output figure'
    )
    
    args = parser.parse_args()
    
    # Validate input files exist
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")
    
    if not os.path.exists(args.labels_path):
        raise FileNotFoundError(f"Labels file not found: {args.labels_path}")
    
    # Generate the plot
    plot_average_frequency_band(args.data_path, args.labels_path, args.output_dir)


if __name__ == "__main__":
    main()