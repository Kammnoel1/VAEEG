#!/usr/bin/env python3
import os
import argparse
import numpy as np
from sklearn.decomposition import KernelPCA

def main():
    p = argparse.ArgumentParser(description="CPU-based Kernel PCA with scikit-learn")
    p.add_argument(
        "--input_latent",
        type=str,
        required=True,
        help="Path to the NumPy file containing latent array (shape: N×D)."
    )
    p.add_argument(
        "--input_labels",
        type=str,
        required=True,
        help="Path to the NumPy file containing labels for the latent array (shape: N,)."
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
        help="Directory where to save KPCA embeddings (.npz)."
    )
    p.add_argument(
        "--n_components",
        type=int,
        default=2,
        help="Number of KPCA components to compute (default: 2)."
    )
    p.add_argument(
        "--kernel",
        type=str,
        default="rbf",
        choices=["linear","poly","rbf","sigmoid","cosine"],
        help="Kernel to use for KPCA (default: rbf)."
    )
    p.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Kernel coefficient for rbf, poly, and sigmoid. If None, uses 1/n_features."
    )
    p.add_argument(
        "--degree",
        type=int,
        default=3,
        help="Degree for poly kernel (default: 3)."
    )
    args = p.parse_args()

    # load the latent vectors and labels
    Z = np.load(args.input_latent)          # shape: (N, D)
    labels = np.load(args.input_labels)     # shape: (N,)
    N, D = Z.shape
    if labels.shape[0] != N:
        raise ValueError(f"Labels length {labels.shape[0]} ≠ samples {N}")
    print(f"Loaded latent array: {N} samples × {D} dims")

    # run Kernel PCA
    kpca = KernelPCA(
        n_components=args.n_components,
        kernel=args.kernel,
        gamma=args.gamma,
        degree=args.degree,
        fit_inverse_transform=False,
        random_state=42
    )
    Xk = kpca.fit_transform(Z)              # shape: (N, n_components)
    print(f"Computed KernelPCA ({args.kernel}) → shape {Xk.shape}")

    # save bundle
    os.makedirs(args.output_dir, exist_ok=True)
    bundle_path = os.path.join(
        args.output_dir,
        f"{args.band}_kpca{args.n_components}d_{args.kernel}.npz"
    )
    np.savez(
        bundle_path,
        embedding=Xk,
        labels=labels
    )
    print(f"Saved KPCA bundle to: {bundle_path}")

if __name__ == "__main__":
    main()
