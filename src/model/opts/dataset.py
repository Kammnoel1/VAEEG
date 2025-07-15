# -*- coding: utf-8 -*-

import numpy as np
import torch.utils.data
import os


class ClipDataset(torch.utils.data.Dataset):
    """
    Dataset for model training.
    """

    def __init__(self, data_dir, band_name, clip_len):
        in_file = os.path.join(data_dir, "%s.npy" % band_name)
        data = np.load(in_file)
        self.data = data.astype(np.float32)

        self.band = band_name
        self.clip_len = clip_len
        
        # Handle both 2D and 3D data
        if len(self.data.shape) == 2:
            # 2D data: (n_item, n_len)
            self.n_item, self.n_len = self.data.shape
            self.n_channels = 1
            self.is_3d = False
        elif len(self.data.shape) == 3:
            # 3D data: (n_item, n_channels, n_len)
            self.n_item, self.n_channels, self.n_len = self.data.shape
            self.is_3d = True
        else:
            raise ValueError(f"Data must be 2D or 3D, got shape: {self.data.shape}")

    def __getitem__(self, index):
        idx = np.random.randint(0, self.n_len - self.clip_len)
        
        if self.is_3d:
            # For 3D data: extract (n_channels, clip_len)
            x = self.data[index, :, idx: idx + self.clip_len]
        else:
            # For 2D data: extract (1, clip_len) - add channel dimension
            x = self.data[index:index+1, idx: idx + self.clip_len]
            
        return x

    def __len__(self):
        return self.n_item