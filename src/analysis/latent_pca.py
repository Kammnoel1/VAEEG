import argparse
import os
import numpy as np
from sklearn.decomposition import PCA


def main():
    p = argparse.ArgumentParser(description="PCA dimensionality reduction for latent embeddings")
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
        help="Band name for which to compute the PCA embedding (e.g., 'alpha', 'theta')."
    )
    p.add_argument(
        "--output_emb",
        type=str,
        required=True,
        help="Directory where to save the PCA embeddings."
    )
    p.add_argument(
        "--n_components",
        type=int,
        default=2,
        help="Number of principal components to compute (default: 2)."
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

    # Initialize PCA
    print(f"Running PCA with parameters:")
    print(f"  n_components: {args.n_components}")
    
    pca = PCA(
        n_components=args.n_components,
        random_state=42
    )
    
    # Fit and transform
    print("Fitting PCA...")
    Y = pca.fit_transform(Z)  # shape: (N, n_components)
    print(f"PCA embedding shape: {Y.shape}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.4f}")

    # Save results
    os.makedirs(args.output_emb, exist_ok=True)
    
    save_path = os.path.join(
        args.output_emb,
        f"{args.band}.npy"
    )
    np.save(save_path, Y)
    print(f"Saved PCA embedding to: {save_path}")
    
    # Save PCA components for visual inspection
    components_path = os.path.join(
        args.output_emb,
        f"{args.band}_components.npy"
    )
    np.save(components_path, pca.components_)
    print(f"Saved PCA components to: {components_path}")
    
    # Save embedding with labels and additional PCA information
    bundle_path = os.path.join(
        args.output_emb, 
        f"{args.band}_with_labels.npz"
    )
    np.savez(
        bundle_path,
        embedding=Y,                                    # shape (N, n_components)
        labels=labels,                                  # shape (N,)
        components=pca.components_,                     # shape (n_components, D)
        explained_variance_ratio=pca.explained_variance_ratio_,  # shape (n_components,)
        explained_variance=pca.explained_variance_,     # shape (n_components,)
        mean=pca.mean_                                  # shape (D,)
    )
    print(f"Saved PCA embedding with labels and components to: {bundle_path}")


if __name__ == "__main__":
    main()
