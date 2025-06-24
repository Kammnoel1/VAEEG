#!/usr/bin/env python3
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_segment_norm_histograms(norms, labels, band_name, output_dir, bins=50):
    """
    Plot histograms of segment norms separated by seizure/background labels.
    
    Args:
        norms (np.ndarray): Array of segment L2 norms
        labels (np.ndarray): Array of corresponding labels (0=background, 1=seizure)
        band_name (str): Name of the frequency band
        output_dir (str): Directory to save the plot
        bins (int): Number of histogram bins
    """
    # Separate norms by label
    seizure_norms = norms[labels == 1]
    background_norms = norms[labels == 0]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot overlapping histograms
    plt.hist(background_norms, bins=bins, alpha=0.7, color='blue', 
             label=f'Background (n={len(background_norms)})', density=True)
    plt.hist(seizure_norms, bins=bins, alpha=0.7, color='red', 
             label=f'Seizure (n={len(seizure_norms)})', density=True)
    
    # Add vertical lines for means
    plt.axvline(np.mean(background_norms), color='blue', linestyle='--', 
                label=f'Background mean: {np.mean(background_norms):.3f}')
    plt.axvline(np.mean(seizure_norms), color='red', linestyle='--',
                label=f'Seizure mean: {np.mean(seizure_norms):.3f}')
    
    # Formatting
    plt.xlabel('Euclidean Norm (L2 magnitude)')
    plt.ylabel('Density')
    plt.title(f'{band_name.title()} Band: Segment Norm Distribution\nSeizure vs Background')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{band_name}_segment_norms_histogram.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved histogram plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot histograms of segment L2 norms for seizure vs background data"
    )
    parser.add_argument(
        "--input_npz",
        type=str,
        required=True,
        help="Path to .npz file containing 'norms' and 'labels' arrays"
    )
    parser.add_argument(
        "--band_name",
        type=str,
        required=True,
        help="Name of the frequency band for plot titles"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/u/noka/VAEEG/figs",
        help="Directory to save histogram plots (default: /u/noka/VAEEG/figs)"
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of histogram bins (default: 50)"
    )
    
    args = parser.parse_args()
    
    # Load data
    if not os.path.exists(args.input_npz):
        raise FileNotFoundError(f"Input file not found: {args.input_npz}")
    
    data = np.load(args.input_npz)
    norms = data['norms']
    labels = data['labels']
    
    print(f"Loaded {len(norms)} segment norms with labels")
    
    # Generate plots
    plot_segment_norm_histograms(norms, labels, args.band_name, args.output_dir, args.bins)


if __name__ == "__main__":
    main()
