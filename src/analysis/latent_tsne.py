import argparse
import os
import numpy as np
from sklearn.manifold import TSNE


def main():
    p = argparse.ArgumentParser(description="CPU-based t-SNE with scikit-learn")
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
        help="Band name for which to compute the t-SNE embedding (e.g., 'alpha', 'theta')."
    )
    p.add_argument(
        "--output_emb",
        type=str,
        required=True,
        help="Directory where to save the 2D embeddings (as <band>_tsne2d.npy)."
    )
    p.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="Perplexity for t-SNE (default: 30)."
    )
    p.add_argument(
        "--learning_rate",
        type=float,
        default=200.0,
        help="Learning rate for t-SNE (default: 200)."
    )
    p.add_argument(
        "--n_iter",
        type=int,
        default=1000,
        help="Number of iterations for t-SNE (default: 1000)."
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel jobs for nearest‐neighbor search (n_jobs in TSNE)."
    )
    args = p.parse_args()


    if not os.path.isfile(args.input_latent):
        raise FileNotFoundError(f"Could not find file: {args.input_latent}")
    Z = np.load(args.input_latent)
    labels = np.load(args.input_labels)
    N, D = Z.shape
    print(f"Loaded latent array: {N} samples × {D} dims")


    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        learning_rate=args.learning_rate,
        n_iter=args.n_iter,
        random_state=42,
        n_jobs=args.num_workers,
        verbose=1
    )
    
    Y = tsne.fit_transform(Z)  # shape: (N, 2)


    os.makedirs(args.output_emb, exist_ok=True)
    save_path = os.path.join(
        args.output_emb,
        f"{args.band}_tsne2d.npy"
    )
    bundle_path = os.path.join(args.output_emb, f"{args.band}_tsne2d_with_labels.npz")
    np.savez(
        bundle_path,
        embedding=Y,     # shape (N,2)
        labels=labels    # shape (N,)
    )



if __name__ == "__main__":
    main()
