# -*- coding: utf-8 -*-
import os
import gc
import time
import mne
import numpy as np
import argparse

class DataGen(object):
    """
    This class loads a raw EEG file (in EEGLAB .set format),
    segments the continuous EEG into fixed 1-second intervals 
    (using the provided sampling frequency), and saves the result
    as a single NumPy array. The expected output shape is
    (num_segments, num_channels, sampling_frequency).
    """
    def __init__(self, in_file, sfreq, out_dir, out_prefix):
        """
        Args:
            in_file: str
                Path to the input .set file (EEGLAB format).
            sfreq: float or int
                Sampling frequency.
            out_dir: str
                Directory where the output .npy file will be stored.
            out_prefix: str
                Prefix/name for the output file.
        """
        self.in_file = in_file
        self.sfreq = sfreq 
        self.out_dir = out_dir
        self.out_prefix = out_prefix

    def load_data(self):
        """
        Loads raw EEG data from the specified .set file using MNE.
        
        Returns:
            raw: MNE Raw object.
        """
        # Load raw EEG data. The associated .fdt file is automatically read.
        raw = mne.io.read_raw_eeglab(self.in_file, preload=True, verbose=False)
        return raw

    def segment_data(self):
        """
        Segment the continuous EEG data into fixed-length 1-second intervals.
        
        Returns:
            segments: A NumPy array of shape (num_segments, num_channels, seg_samples)
                      where seg_samples is equal to sfreq.
        """
        data = self.raw.get_data()  # Shape: (num_channels, total_samples)
        num_channels, total_samples = data.shape
        seg_samples = int(self.sfreq)  # Each segment is 1 second long
        num_segments = total_samples // seg_samples
        
        # Create segments by splitting the time dimension into chunks of seg_samples.
        segments = []
        for i in range(num_segments):
            start = i * seg_samples
            stop = start + seg_samples
            segments.append(data[:, start:stop])
        # Stack along a new axis so that the shape becomes (num_segments, num_channels, seg_samples)
        segments = np.stack(segments, axis=0)
        return segments

    def run(self):
        """
        Loads the raw EEG data, segments it into 1-second intervals,
        and saves the resulting NumPy array. Also returns the segments.
        """
        self.raw = self.load_data()
        segments = self.segment_data()  # Expected shape: (num_segments, num_channels, sfreq)
        
        # Ensure the output directory exists.
        if not os.path.isdir(self.out_dir):
            os.makedirs(self.out_dir)
        
        out_file = os.path.join(self.out_dir, f"{self.out_prefix}.npy")
        np.save(out_file, segments)
        print(f"Saved segmented EEG data with shape {segments.shape} to {out_file}")
        return segments

def generate_data(in_file, sfreq, out_dir, out_prefix):
    """
    Generates segmented EEG data from a raw .set file.
    
    Args:
        in_file: str - Path to the input .set file.
        sfreq: int/float - Sampling frequency.
        out_dir: str - Output directory.
        out_prefix: str - Prefix for the output file.
    
    Returns:
        segments: NumPy array containing the segmented EEG data.
    """
    dg = DataGen(in_file, sfreq, out_dir, out_prefix)
    segments = dg.run()
    # Optionally call garbage collection.
    gc.collect()
    return segments

def main(): 
    parser = argparse.ArgumentParser(
        description="Process a .set file: read raw EEG, segment it into 1-second intervals, and store as a NumPy array.")
    parser.add_argument("--in_file", type=str, required=True,
                        help="Path to the input .set file (EEGLAB format).")
    parser.add_argument("--sfreq", type=int, required=True,
                        help="Sampling frequency.")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory where the output .npy file will be saved.")
    parser.add_argument("--out_prefix", type=str, required=True,
                        help="Prefix (or name) for the output file (e.g., 'VAEEG').")
    args = parser.parse_args()
    
    generate_data(args.in_file, args.sfreq, args.out_dir, args.out_prefix)


# Standalone execution remains available.
if __name__ == "__main__":
    main()