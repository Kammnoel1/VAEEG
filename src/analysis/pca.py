#!/usr/bin/env python3
import os
import argparse
import numpy as np
from sklearn.decomposition import PCA

def main():
    p = argparse.ArgumentParser(description="CPU-based PCA with scikit-learn")
    p.add_argument(
        "--input_data",
        type=str,
        required=True,
        help="Path to the NumPy file containing array (shape: N×D)."
    )
    p.add_argument(
        "--input_labels",
        type=str,
        required=True,
        help="Path to the NumPy file containing labels for the array (shape: N,)."
    )
    p.add_argument(
        "--band",
        type=str,
        required=True,
        help="Band name (e.g., 'alpha', 'theta'). Used for naming output files."
    )
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where to save PCA embeddings (.npz)."
    )
    p.add_argument(
        "--n_components",
        type=int,
        default=2,
        help="Number of principal components to compute (default: 2)."
    )
    args = p.parse_args()

    # load the vectors and labels
    Z = np.load(args.input_data)          # shape: (N, D)
    labels = np.load(args.input_labels)     # shape: (N,)
    N, D = Z.shape
    if labels.shape[0] != N:
        raise ValueError(f"Labels length {labels.shape[0]} ≠ samples {N}")
    print(f"Loaded array: {N} samples × {D} dims")

    # run PCA
    pca = PCA(n_components=args.n_components, random_state=42)
    Xp = pca.fit_transform(Z)               # shape: (N, n_components)
    var_pct = pca.explained_variance_ratio_ * 100
    for i, pct in enumerate(var_pct, start=1):
        print(f"PC{i} explains {pct:.1f}% of variance")

    # save
    os.makedirs(args.output_dir, exist_ok=True)
    bundle_path = os.path.join(
        args.output_dir,
        f"{args.band}_pca{args.n_components}d_with_labels.npz"
    )
    np.savez(
        bundle_path,
        embedding=Xp,
        labels=labels
    )
    print(f"Saved PCA bundle to: {bundle_path}")

if __name__ == "__main__":
    main()
