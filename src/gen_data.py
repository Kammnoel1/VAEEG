# -*- coding: utf-8 -*-
import os
import gc
import time
import random
import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

class DataGen(object):
    _SFREQ = 2048.0  # Sampling rate in Hz

    def __init__(self, in_file, out_dir, out_prefix, ch_mapper=None):
        self.in_file = in_file
        self.out_dir = out_dir
        self.out_prefix = out_prefix
        self.ch_mapper = ch_mapper
        self.raw = self.load_data()  # Load raw EEG data
        self.data = self.raw.get_data()  # Expect shape (channels, total_samples)
        self.ch_names = self.raw.ch_names

    def load_data(self):
        """
        Load raw EEG data from an EEGLAB file using MNE.
        """
        raw = mne.io.read_raw_eeglab(self.in_file, preload=True, verbose=False)
        montage = mne.channels.read_custom_montage('raw_data/standard_waveguard256_duke.txt')
        raw.set_montage(montage)
        return raw

    def segment_data(self, seg_len=1.0):
        """
        Split the clean EEG data into fixed-length segments.
        
        Args:
            seg_len: Segment length in seconds (default: 1.0 second)
            
        Returns:
            segments: A NumPy array of shape (num_segments, channels, seg_samples)
        """
        seg_samples = int(seg_len * self._SFREQ)
        total_samples = self.data.shape[1]
        num_segments = total_samples // seg_samples
        if num_segments == 0:
            raise ValueError("Not enough data to form a single segment.")
        segments = []
        for i in range(num_segments):
            start = i * seg_samples
            stop = start + seg_samples
            segments.append(self.data[:, start:stop])
        segments = np.stack(segments, axis=0)
        return segments

    def run(self, seg_len=1.0):
        segments = self.segment_data(seg_len=seg_len)
        out_file = os.path.join(self.out_dir, f"{self.out_prefix}.npy")
        np.save(out_file, segments)
        print(f"Saved {segments.shape[0]} segments to {out_file}")

def worker(in_file, out_dir, out_prefix):
    try:
        dg = DataGen(in_file, out_dir, out_prefix)
        dg.run(seg_len=1.0)
    except Exception as e:
        print("Error processing", in_file, ":", e)
        time.sleep(3.0)
        return in_file, False
    return in_file, True

if __name__ == "__main__":
    # Path to your raw EEGLAB file (.set file should reference its .fdt file)
    in_file = "./raw_data/sep_uwgr_prepro.set"
    out_dir = "./new_data/clips"
    log_file = "./new_data/log.csv"

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # Process the single raw file directly
    out_prefix = "VAEEG"  # This will be the prefix for your output .npy file
    worker(in_file, out_dir, out_prefix)

    # Optionally, write a simple log file
    with open(log_file, "w") as fo:
        fo.write(f"{in_file},True\n")

    print("Data generation complete. Processed clips are saved in:", out_dir)
    gc.collect()