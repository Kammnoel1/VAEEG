import os
import argparse
import numpy as np

def split_memmap(data_path: str, ratio: float, output_dir: str) -> dict:
    """Split the memmap at data_path into train and test memmaps.

    Returns:
        dict: {'train': train_path, 'test': test_path}
    """
    # Load input memmap
    data_mm = np.lib.format.open_memmap(data_path, mode='r')
    total_samples = data_mm.shape[0]

    # Compute split index
    split_idx = int(total_samples * ratio)
    if split_idx <= 0 or split_idx >= total_samples:
        raise ValueError(f"Split ratio {ratio} results in invalid split index {split_idx}")

    # Prepare output directory
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, 'train.npy')
    test_path = os.path.join(output_dir, 'test.npy')

    # Create train memmap and write
    train_shape = (split_idx,) + data_mm.shape[1:]
    train_mm = np.lib.format.open_memmap(
        train_path, mode='w+', dtype=data_mm.dtype, shape=train_shape
    )
    train_mm[:] = data_mm[:split_idx]
    del train_mm

    # Create test memmap and write
    test_shape = (total_samples - split_idx,) + data_mm.shape[1:]
    test_mm = np.lib.format.open_memmap(
        test_path, mode='w+', dtype=data_mm.dtype, shape=test_shape
    )
    test_mm[:] = data_mm[split_idx:]
    del test_mm

    # Return paths
    return {'train': train_path, 'test': test_path}


def main():
    parser = argparse.ArgumentParser(
        description='Split a 4D data memmap into train and test sets.'
    )
    parser.add_argument(
        '--data_path', required=True,
        help='Path to the combined data.npy file.'
    )
    parser.add_argument(
        '--ratio', type=float, required=True,
        help='Fraction of data for training (e.g., 0.8 for 80% train).' 
    )
    parser.add_argument(
        '--output_dir', default='.',
        help='Directory to write train.npy and test.npy.'
    )
    args = parser.parse_args()

    split_memmap(args.data_path, args.ratio, args.output_dir)

if __name__ == '__main__':
    main()