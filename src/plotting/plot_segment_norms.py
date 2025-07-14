#!/usr/bin/env python3
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_segment_norm_histograms(norms, labels, band_name, output_dir, bins=50):
    """
    Plot histograms of segment norms separated by seizure/background labels.
    Creates only separate subplot visualizations without overlapping histograms.
    
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
    
    # Calculate percentiles to set reasonable x-axis limits
    p99 = np.percentile(norms, 99)
    
    # Define bins based on data range, focusing on where most data lies
    bin_range = (0, min(p99 * 1.1, np.max(norms)))
    
    # Calculate means and standard deviations
    bg_mean = np.mean(background_norms)
    sz_mean = np.mean(seizure_norms)
    bg_std = np.std(background_norms)
    sz_std = np.std(seizure_norms)
    
    # Create plot with separate subplots for clearer comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    
    # Background histogram
    ax1.hist(background_norms, bins=bins, alpha=0.8, color='blue', density=True,
             range=bin_range, edgecolor='darkblue', linewidth=0.5)
    ax1.axvline(bg_mean, color='darkblue', linestyle='--', linewidth=2)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title(f'{band_name.title()} Band - Background Segments (n={len(background_norms):,})', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.text(0.98, 0.98, f'μ={bg_mean:.2f}, σ={bg_std:.2f}',
             transform=ax1.transAxes, fontsize=11, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Seizure histogram
    ax2.hist(seizure_norms, bins=bins, alpha=0.8, color='red', density=True,
             range=bin_range, edgecolor='darkred', linewidth=0.5)
    ax2.axvline(sz_mean, color='darkred', linestyle='--', linewidth=2)
    ax2.set_xlabel('Euclidean Norm (L2 magnitude)', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title(f'{band_name.title()} Band - Seizure Segments (n={len(seizure_norms):,})', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.text(0.98, 0.98, f'μ={sz_mean:.2f}, σ={sz_std:.2f}',
             transform=ax2.transAxes, fontsize=11, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the separate subplots
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{band_name}_segment_norms_histogram.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


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
        default="/u/noka/VAEEG/figs/norm/test",
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
    
    # Generate plots
    plot_segment_norm_histograms(norms, labels, args.band_name, args.output_dir, args.bins)


if __name__ == "__main__":
    main()
