#!/usr/bin/env python3
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_2d(pts, labels, out_folder, band):
    os.makedirs(out_folder, exist_ok=True)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected (n,2), got {pts.shape}")
    if labels.shape[0] != pts.shape[0]:
        raise ValueError("Labels length must match number of points")

    cmap = ListedColormap(["blue", "red"])  # 0→blue, 1→red
    sc = plt.scatter(
        pts[:,0], pts[:,1],
        c=labels,
        cmap=cmap,
        s=5,
        alpha=0.6,
        edgecolors="none"
    )
    handles, _ = sc.legend_elements()
    plt.legend(handles, ["background", "seizure"], title="label", loc="best")

    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.tight_layout()

    out_path = os.path.join(out_folder, band)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved plot to: {out_path}")

def main():
    p = argparse.ArgumentParser(description="Plot 2D embedding from an .npz bundle")
    p.add_argument(
        "--input_npz",
        required=True,
        help="Path to .npz with 'embedding' (n×2) and 'labels' (n,)"
    )
    p.add_argument(
        "--out_folder",
        default="/u/noka/VAEEG/figs/pca/test",
        help="Directory to save the plot"
    )
    p.add_argument(
        "--band",
        required=True,
        help="Filename for the saved plot"
    )
    args = p.parse_args()

    data = np.load(args.input_npz)
    embedding = data["embedding"]
    labels    = data["labels"]
    plot_2d(embedding, labels, args.out_folder, args.band)

if __name__ == "__main__":
    main()
