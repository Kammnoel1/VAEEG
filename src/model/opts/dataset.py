import numpy as np
import torch.utils.data

class ClipDataset(torch.utils.data.Dataset):
    """
    Dataset for model training, with on-the-fly normalization.
    Each sample is normalized by subtracting its mean and dividing by its standard deviation.
    """

    def __init__(self, data_file, band_idx, channel):
        """
        Args:
            data_file (str): Path to the numpy file with EEG data (shape: N, C, B, T).
            band_name (str): The frequency band to extract (e.g., 'alpha').
            clip_len (int): The length of the clip to be used for training.
            channel (int): The EEG channel index to be used (0-based).
        """
        # Use memory-mapping to directly load the relevant slice (for specific channel and band)
        self.data = np.lib.format.open_memmap(data_file, mode='r')[:, channel, band_idx, :]  # Shape: (N, T)

        # Get the number of samples (N) and timepoints (T)
        self.n_item, self.n_time = self.data.shape

    def __getitem__(self, index):
        """
        Get a slice of the data, normalize it, and return the clip.
        """
        # Extract the slice from the memory-mapped data directly
        x = self.data[index:index+1,:]

        # On-the-fly normalization: z-score normalization for each clip.
        mean = np.mean(x, axis=1, keepdims=True)
        std = np.std(x, axis=1, keepdims=True) + 1e-8
        x = (x - mean) / std
        return x

    def __len__(self):
        return self.n_item