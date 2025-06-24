import argparse
import os
import numpy as np
import umap


def main():
    p = argparse.ArgumentParser(description="UMAP dimensionality reduction for latent embeddings")
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
        help="Path to the NumPy file containing labels for the latent array."
    )
    p.add_argument(
        "--band",
        type=str,
        required=True,
        help="Band name for which to compute the UMAP embedding (e.g., 'alpha', 'theta')."
    )
    p.add_argument(
        "--output_emb",
        type=str,
        required=True,
        help="Directory where to save the 2D embeddings (as <band>_umap2d.npy)."
    )
    p.add_argument(
        "--n_neighbors",
        type=int,
        default=15,
        help="Number of nearest neighbors for UMAP (default: 15)."
    )
    p.add_argument(
        "--min_dist",
        type=float,
        default=0.1,
        help="Minimum distance for UMAP embedding (default: 0.1)."
    )
    p.add_argument(
        "--n_components",
        type=int,
        default=2,
        help="Number of dimensions for the embedding (default: 2)."
    )
    p.add_argument(
        "--metric",
        type=str,
        default="euclidean",
        help="Distance metric for UMAP (default: euclidean)."
    )
    p.add_argument(
        "--spread",
        type=float,
        default=1.0,
        help="Spread parameter for UMAP (default: 1.0)."
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel jobs for UMAP computation."
    )
    args = p.parse_args()

    # Load data
    if not os.path.isfile(args.input_latent):
        raise FileNotFoundError(f"Could not find file: {args.input_latent}")
    if not os.path.isfile(args.input_labels):
        raise FileNotFoundError(f"Could not find file: {args.input_labels}")
        
    Z = np.load(args.input_latent)
    labels = np.load(args.input_labels)
    N, D = Z.shape
    print(f"Loaded latent array: {N} samples × {D} dims")
    print(f"Loaded labels: {labels.shape[0]} labels")

    # Initialize UMAP
    print(f"Running UMAP with parameters:")
    print(f"  n_neighbors: {args.n_neighbors}")
    print(f"  min_dist: {args.min_dist}")
    print(f"  n_components: {args.n_components}")
    print(f"  metric: {args.metric}")
    print(f"  spread: {args.spread}")
    
    umap_reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        n_components=args.n_components,
        metric=args.metric,
        spread=args.spread,
        random_state=42,
        n_jobs=args.num_workers,
        verbose=True
    )
    
    # Fit and transform
    print("Fitting UMAP...")
    Y = umap_reducer.fit_transform(Z)  # shape: (N, n_components)
    print(f"UMAP embedding shape: {Y.shape}")

    # Save results
    os.makedirs(args.output_emb, exist_ok=True)
    
    # Save embedding only
    save_path = os.path.join(
        args.output_emb,
        f"{args.band}_umap{args.n_components}d.npy"
    )
    np.save(save_path, Y)
    print(f"Saved UMAP embedding to: {save_path}")
    
    # Save embedding with labels
    bundle_path = os.path.join(
        args.output_emb, 
        f"{args.band}_umap{args.n_components}d_with_labels.npz"
    )
    np.savez(
        bundle_path,
        embedding=Y,     # shape (N, n_components)
        labels=labels    # shape (N,)
    )
    print(f"Saved UMAP embedding with labels to: {bundle_path}")


if __name__ == "__main__":
    main()
