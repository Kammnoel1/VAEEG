#!/usr/bin/env python3
"""
Script to plot a 2x2 grid of randomly sampled EEG clips:
- Left column: Seizure clips
- Right column: Background clips
- Each plot shows low beta and high beta frequency bands overlaid
"""

import argparse
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def load_labels_csv(base_dir):
    """
    Load the labels.csv file to get seizure/background labels for clips.
    
    Args:
        base_dir: Base directory containing labels.csv
        
    Returns:
        dict: Mapping from .npy filename to label (0=background, 1=seizure)
    """
    labels_csv_path = os.path.join(base_dir, "labels.csv")
    if not os.path.exists(labels_csv_path):
        print(f"Warning: No labels.csv found in {base_dir}")
        return {}
    
    df = pd.read_csv(labels_csv_path)
    label_map = {}
    for _, row in df.iterrows():
        filename = os.path.basename(row['npy_path'])
        label_map[filename] = row['label']
    
    return label_map


def get_clip_files_by_label(clips_dir, label_map):
    """
    Get lists of clip files separated by seizure/background labels.
    
    Args:
        clips_dir: Directory containing .npy clip files
        label_map: Dictionary mapping filenames to labels
        
    Returns:
        tuple: (seizure_files, background_files)
    """
    seizure_files = []
    background_files = []
    
    for filename in os.listdir(clips_dir):
        if filename.endswith('.npy'):
            label = label_map.get(filename, 0)  # Default to background if not found
            filepath = os.path.join(clips_dir, filename)
            
            if label == 1:
                seizure_files.append(filepath)
            else:
                background_files.append(filepath)
    
    return seizure_files, background_files


def plot_clip_segment(ax, data, title, band_names):
    """
    Plot a single EEG segment with low beta and high beta bands overlaid.
    
    Args:
        ax: Matplotlib axis object
        data: EEG data array of shape (6, 1280) - 6 frequency bands, 1280 time points
        title: Title for the subplot
        band_names: List of band names corresponding to the 6 bands
    """
    # Extract low beta (index 4) and high beta (index 5) bands
    low_beta_idx = band_names.index('low_beta')
    high_beta_idx = band_names.index('high_beta')
    
    low_beta = data[low_beta_idx, :]
    high_beta = data[high_beta_idx, :]
    
    # Create time axis (assuming some sampling rate - adjust if needed)
    time_points = np.arange(len(low_beta))
    
    # Plot both bands
    ax.plot(time_points, low_beta, label='Low Beta', color='blue', alpha=0.7, linewidth=1)
    ax.plot(time_points, high_beta, label='High Beta', color='red', alpha=0.7, linewidth=1)
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Time Points')
    ax.set_ylabel('Amplitude')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Set reasonable y-limits based on data range
    y_min = min(np.min(low_beta), np.min(high_beta))
    y_max = max(np.max(low_beta), np.max(high_beta))
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)


def plot_sample_clips_grid(seizure_files, background_files, output_dir, n_samples=2):
    """
    Create a 2x2 grid plot with seizure clips on left, background on right.
    
    Args:
        seizure_files: List of seizure clip file paths
        background_files: List of background clip file paths
        output_dir: Directory to save the plot
        n_samples: Number of samples per category (default: 2 for 2x2 grid)
    """
    band_names = ["whole", "delta", "theta", "alpha", "low_beta", "high_beta"]
    
    # Randomly sample files
    seizure_samples = random.sample(seizure_files, min(n_samples, len(seizure_files)))
    background_samples = random.sample(background_files, min(n_samples, len(background_files)))
    
    # Create figure with 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('EEG Clip Samples: Seizure (Left) vs Background (Right)\nLow Beta and High Beta Frequency Bands', 
                 fontsize=16, fontweight='bold')
    
    # Plot seizure clips on the left column
    for i in range(n_samples):
        if i < len(seizure_samples):
            # Load clip data
            clip_data = np.load(seizure_samples[i])
            
            # Randomly select one segment from the clip
            n_segments = clip_data.shape[0]
            segment_idx = random.randint(0, n_segments - 1)
            segment_data = clip_data[segment_idx]  # Shape: (6, 1280)
            
            # Get filename for title
            filename = os.path.basename(seizure_samples[i])
            title = f'Seizure Clip\n{filename}\nSegment {segment_idx+1}/{n_segments}'
            
            plot_clip_segment(axes[i, 0], segment_data, title, band_names)
    
    # Plot background clips on the right column  
    for i in range(n_samples):
        if i < len(background_samples):
            # Load clip data
            clip_data = np.load(background_samples[i])
            
            # Randomly select one segment from the clip
            n_segments = clip_data.shape[0]
            segment_idx = random.randint(0, n_segments - 1)
            segment_data = clip_data[segment_idx]  # Shape: (6, 1280)
            
            # Get filename for title
            filename = os.path.basename(background_samples[i])
            title = f'Background Clip\n{filename}\nSegment {segment_idx+1}/{n_segments}'
            
            plot_clip_segment(axes[i, 1], segment_data, title, band_names)
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "sample_clips_2x2_grid.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot 2x2 grid of randomly sampled EEG clips (seizure vs background)"
    )
    parser.add_argument(
        "--clips_dir",
        type=str,
        default="/ptmp/noka/new_data/tusz/clips/",
        help="Directory containing .npy clip files"
    )
    parser.add_argument(
        "--labels_csv",
        type=str,
        default="/ptmp/noka/new_data/tusz2/labels.csv",
        help="Path to labels.csv file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str,
        default="/u/noka/VAEEG/figs",
        help="Directory to save the plot"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Load labels
    base_dir = os.path.dirname(args.labels_csv)
    label_map = load_labels_csv(base_dir)
    
    if not label_map:
        print("Error: Could not load labels. Please check the labels.csv path.")
        return
    
    # Get clip files by label
    seizure_files, background_files = get_clip_files_by_label(args.clips_dir, label_map)
    
    print(f"Found {len(seizure_files)} seizure clips and {len(background_files)} background clips")
    
    if len(seizure_files) < 2:
        print("Error: Need at least 2 seizure clips for 2x2 grid")
        return
    
    if len(background_files) < 2:
        print("Error: Need at least 2 background clips for 2x2 grid")
        return
    
    # Generate plot
    output_path = plot_sample_clips_grid(seizure_files, background_files, args.output_dir)
    print(f"Successfully created plot at: {output_path}")


if __name__ == "__main__":
    main()
