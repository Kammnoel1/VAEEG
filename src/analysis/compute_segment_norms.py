import argparse
import os
import numpy as np


def compute_segment_norms(data_path, labels_path, band_name, output_dir):
    """
    Compute Euclidean norms (L2 vector magnitudes) for all segments in a frequency band.
    
    Args:
        data_path (str): Path to the frequency band data file (.npy)
        labels_path (str): Path to the corresponding labels file (.npy)
        band_name (str): Name of the frequency band (e.g., 'alpha', 'theta')
        output_dir (str): Directory to save the computed norms
    """
    print(f"Computing segment norms for {band_name} band...")
    
    data = np.load(data_path)  # Shape: (n_segments, segment_len)
    
    labels = np.load(labels_path)  # Shape: (n_segments,)
    
    
    # Validate dimensions match
    if data.shape[0] != labels.shape[0]:
        raise ValueError(f"Mismatch: data has {data.shape[0]} segments but labels has {labels.shape[0]} entries")
    
    # Compute L2 norm for each segment
    segment_norms = np.linalg.norm(data, axis=1)  # L2 norm along segment dimension
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine split from labels path
    split = "train" if "train" in labels_path else "test"
    
    # Save norms and labels with split information
    norms_path = os.path.join(output_dir, f"{band_name}_segment_norms_{split}.npy")
    labels_out_path = os.path.join(output_dir, f"{band_name}_segment_norm_labels_{split}.npy")
    
    np.save(norms_path, segment_norms)
    np.save(labels_out_path, labels)
    
    
    # Save as bundle for convenience
    bundle_path = os.path.join(output_dir, f"{band_name}_segment_norms_{split}_with_labels.npz")
    np.savez(bundle_path, norms=segment_norms, labels=labels)
    
    # Print basic statistics
    seizure_norms = segment_norms[labels == 1]
    background_norms = segment_norms[labels == 0]
    
    print(f"\nStatistics for {band_name} band ({split} set):")
    print(f"Seizure segments - Mean norm: {np.mean(seizure_norms):.4f}, Std: {np.std(seizure_norms):.4f}")
    print(f"Background segments - Mean norm: {np.mean(background_norms):.4f}, Std: {np.std(background_norms):.4f}")
    print(f"Ratio (Seizure/Background): {np.mean(seizure_norms)/np.mean(background_norms):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute Euclidean norms (L2 magnitudes) for EEG segments in a frequency band"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the frequency band data file (.npy)"
    )
    parser.add_argument(
        "--labels_path",
        type=str,
        required=True,
        help="Path to the corresponding labels file (.npy)"
    )
    parser.add_argument(
        "--band_name",
        type=str,
        required=True,
        help="Name of the frequency band (e.g., 'alpha', 'theta')"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/ptmp/noka/analysis/tusz/norms",
        help="Directory to save computed norms (default: /ptmp/noka/analysis/tusz/norms)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")
    
    if not os.path.exists(args.labels_path):
        raise FileNotFoundError(f"Labels file not found: {args.labels_path}")
    
    compute_segment_norms(args.data_path, args.labels_path, args.band_name, args.output_dir)


if __name__ == "__main__":
    main()
