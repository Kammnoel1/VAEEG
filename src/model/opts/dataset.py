# -*- coding: utf-8 -*-

import numpy as np
import torch.utils.data
import os

class ClipDataset(torch.utils.data.Dataset):
    """
    Dataset for model training, with on-the-fly normalization.
    Each sample (clip) is normalized by subtracting its mean and dividing by its standard deviation.
    """

    def __init__(self, data_dir, band_name, clip_len=256):
        in_file = os.path.join(data_dir, "%s.npy" % band_name)
        self.data = np.load(in_file)
        self.band = band_name
        self.n_item, self.n_len = self.data.shape
        self.clip_len = clip_len

    def __getitem__(self, index):
        # If the available length equals the clip length, take the clip starting at 0.
        if self.n_len - self.clip_len <= 0:
            idx = 0
        else:
            idx = np.random.randint(0, self.n_len - self.clip_len)
        x = self.data[index:index+1, idx: idx + self.clip_len]
        # On-the-fly normalization: z-score normalization for each clip.
        # Compute mean and standard deviation over the time dimension (axis=1).
        mean = np.mean(x, axis=1, keepdims=True)
        std = np.std(x, axis=1, keepdims=True) + 1e-8
        x = (x - mean) / std
        return x

    def __len__(self):
        return self.n_item
