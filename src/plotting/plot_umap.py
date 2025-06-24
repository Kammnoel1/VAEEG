#!/usr/bin/env python3
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_umap(umap_array, labels, out_folder, out_name="umap_plot.png"):
    """
    Creates a 2D scatter plot of a UMAP embedding, colored by labels, and saves it to disk.

    Parameters
    ----------
    umap_array : np.ndarray of shape (n_samples, 2)
    labels     : np.ndarray of shape (n_samples,) with integer class labels
    out_folder : str
    out_name   : str
    """
    os.makedirs(out_folder, exist_ok=True)

    if umap_array.ndim != 2 or umap_array.shape[1] != 2:
        raise ValueError(f"Expected array of shape (n_samples, 2), got {umap_array.shape}")
    if labels.shape[0] != umap_array.shape[0]:
        raise ValueError("Labels length must match number of points")

    x = umap_array[:, 0]
    y = umap_array[:, 1]

    plt.figure(figsize=(8, 8))
    cmap = ListedColormap(['blue', 'red'])

    sc = plt.scatter(
        x, y,
        c=labels,
        s=5,
        alpha=0.6,
        cmap=cmap,
        edgecolors="none"
    )
    plt.title("UMAP Embedding (colored by label)")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.grid(True)
    # if you want a legend:
    handles, _ = sc.legend_elements()
    plt.legend(handles, ["background", "seizure"], title="label", loc="best")

    plt.tight_layout()
    out_path = os.path.join(out_folder, out_name)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved UMAP plot to: {out_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Plot a labeled 2D UMAP embedding from an .npz bundle"
    )
    parser.add_argument(
        "--input_npz",
        required=True,
        type=str,
        help="Path to the .npz file containing 'embedding' (n×2) and 'labels' (n,)"
    )
    parser.add_argument(
        "--out_folder",
        default="/u/noka/VAEEG/figs",
        type=str,
        help="Directory where the plot image will be saved"
    )
    parser.add_argument(
        "--band",
        type=str,
        required=True,
        help="Frequency band name to include in the plot title and filename"
    )
    args = parser.parse_args()

    # Load both arrays from the .npz bundle
    data = np.load(args.input_npz)
    umap_array = data["embedding"]
    labels     = data["labels"]

    args.out_name = f"{args.band}_umap.png"

    plot_umap(umap_array, labels, args.out_folder, args.out_name)

if __name__ == "__main__":
    main()
