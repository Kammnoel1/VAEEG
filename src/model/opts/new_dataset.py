# dataset.py
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.utils.data as tud
import os

class ClipDataset(tud.Dataset):
    """
    Dataset for model training (labeled, supports 2D/3D).
    Returns (x, y). x is (C, clip_len) float32, y is int64.
    """

    def __init__(self, data_dir, band_name, clip_len, labels,
                 indices=None, split="train", rng=None):
        """
        Args:
            data_dir (str): folder with {band_name}.npy and labels aligned to items.
            band_name (str): base filename for data .npy
            clip_len (int): length of the clip (in samples)
            labels (array-like): shape (n_item,), integer labels per item
            indices (array-like|None): subset indices for split; if None, use all
            split (str): 'train' -> random crop, else deterministic center crop
            rng (np.random.Generator|None): optional RNG for reproducible crops
        """
        in_file = os.path.join(data_dir, f"{band_name}.npy")
        data = np.load(in_file)
        self.data = data.astype(np.float32)
        self.labels = labels.astype(np.int64)

        self.band = band_name
        self.clip_len = int(clip_len)
        self.split = split.lower()

        # handle shape: (n_item, n_len) OR (n_item, n_channels, n_len)
        if self.data.ndim == 2:
            self.n_item, self.n_len = self.data.shape
            self.n_channels = 1
            self.is_3d = False
        elif self.data.ndim == 3:
            self.n_item, self.n_channels, self.n_len = self.data.shape
            self.is_3d = True
        else:
            raise ValueError(f"Data must be 2D or 3D, got shape: {self.data.shape}")

        # subset indices for split
        if indices is None:
            self.indices = np.arange(self.n_item)
        else:
            self.indices = np.asarray(indices)

        # RNG
        self.rng = rng if rng is not None else np.random.default_rng()

    def __len__(self):
        return len(self.indices)

    def _crop(self, sig):
        """sig: (C, n_len) float32 ndarray -> (C, clip_len)"""
        n_len = sig.shape[-1]
        if self.split == "train":
            start = self.rng.integers(0, n_len - self.clip_len + 1)
        else:
            # deterministic center crop for val/test
            start = (n_len - self.clip_len) // 2
        return sig[..., start:start + self.clip_len]

    def __getitem__(self, i):
        item_idx = int(self.indices[i])

        if self.is_3d:
            # (C, L)
            sig = self.data[item_idx, :, :]
        else:
            # (1, L) — add channel dim
            sig = self.data[item_idx:item_idx+1, :]

        x = self._crop(sig)                        # (C, clip_len)
        y = self.labels[item_idx]                  # scalar

        # return tensors
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)
