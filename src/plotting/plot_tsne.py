#!/usr/bin/env python3
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_tsne(tsne_array, labels, out_folder, out_name="tsne_plot.png"):
    """
    Creates a 2D scatter plot of a t-SNE embedding, colored by labels, and saves it to disk.

    Parameters
    ----------
    tsne_array : np.ndarray of shape (n_samples, 2)
    labels     : np.ndarray of shape (n_samples,) with integer class labels
    out_folder : str
    out_name   : str
    """
    os.makedirs(out_folder, exist_ok=True)

    if tsne_array.ndim != 2 or tsne_array.shape[1] != 2:
        raise ValueError(f"Expected array of shape (n_samples, 2), got {tsne_array.shape}")
    if labels.shape[0] != tsne_array.shape[0]:
        raise ValueError("Labels length must match number of points")

    x = tsne_array[:, 0]
    y = tsne_array[:, 1]

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
    plt.title("t-SNE Embedding (colored by label)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)
    # if you want a legend:
    handles, _ = sc.legend_elements()
    plt.legend(handles, ["background", "seizure"], title="label", loc="best")

    plt.tight_layout()
    out_path = os.path.join(out_folder, out_name)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved t-SNE plot to: {out_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Plot a labeled 2D t-SNE embedding from an .npz bundle"
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
        "--out_name",
        default="tsne_32d_plot.png",
        help="Filename for the saved plot (default: tsne_plot.png)"
    )
    args = parser.parse_args()

    # Load both arrays from the .npz bundle
    data = np.load(args.input_npz)
    tsne_array = data["embedding"]
    labels     = data["labels"]

    plot_tsne(tsne_array, labels, args.out_folder, args.out_name)

if __name__ == "__main__":
    main()
