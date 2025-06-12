import numpy as np
import pandas as pd
import os
from tqdm import tqdm

def build_label_array(latent_path, mapping_csv, out_label_npy):
    Z = np.load(latent_path)
    N, D = Z.shape
    print(f"Loaded latent (N={N}, D={D}) from {latent_path}")
    df = pd.read_csv(mapping_csv)
    clip_paths = df["npy_path"].tolist()
    labels_small = df["label"].values   
    lengths = []
    for p in tqdm(clip_paths):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing clip file: {p}")
        arr = np.load(p, mmap_mode='r')
        lengths.append(arr.shape[0])

    labels_big = np.repeat(labels_small, lengths)

    # sanity check
    if labels_big.shape[0] != N:
        raise ValueError(
            f"Label array length {labels_big.shape[0]} ≠ latent row count {N}"
        )

    np.save(out_label_npy, labels_big)
    print(f"Saved labels → {out_label_npy}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--latent",      required=True,
                   help="Path to your merged latent .npy (shape: N×D)")
    p.add_argument("--mapping_csv", required=True,
                   help="CSV with columns ['npy_path','label']")
    p.add_argument("--out_labels",  required=True,
                   help="Where to write the label array (as .npy)")
    args = p.parse_args()

    build_label_array(
        latent_path=args.latent,
        mapping_csv=args.mapping_csv,
        out_label_npy=args.out_labels
    )
