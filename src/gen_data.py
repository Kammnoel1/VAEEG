# -*- coding: utf-8 -*-
import gc
import os
import random
import glob
from joblib import Parallel, delayed

import mne
import numpy as np
import pandas as pd
import lighten.utils.interval as lui
from mne.filter import create_filter

from lighten.utils.io import load_raw_
from tqdm import tqdm


def parse_label_file(edf_path):
    """
    Parse the corresponding .csv_bi file for an EDF file to determine if seizure is present.
    
    Args:
        edf_path: Path to the EDF file
        
    Returns:
        int: 1 if seizure is present, 0 if only background
    """
    # Construct the corresponding .csv_bi file path
    csv_bi_path = edf_path.replace('.edf', '.csv_bi')
    
    if not os.path.exists(csv_bi_path):
        print(f"Warning: Label file not found for {edf_path}")
        return 0  # Default to background if no label file
    
    try:
        # Read the CSV file, skipping comment lines that start with #
        df = pd.read_csv(csv_bi_path, comment='#')
        
        # Check if any row contains 'seiz' in the label column
        has_seizure = (df['label'] == 'seiz').any()
        
        return 1 if has_seizure else 0
        
    except Exception as e:
        print(f"Error parsing label file {csv_bi_path}: {e}")
        return 0  # Default to background if parsing fails


class DataGen:
    _SFREQ = 256.0
    max_clips = 500
    _l_trans_bandwidth = 0.1
    _h_trans_bandwidth = 0.1

    _BANDS = {
        "whole": (1.0, 30.0),
        "delta": (1.0, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "low_beta": (13, 20),
        "high_beta": (20.0, 30.0),
    }

    def __init__(self, in_file, out_dir, out_prefix, ch_mapper=None):
        self.in_file = in_file
        self.ch_mapper = ch_mapper
        self.out_dir = out_dir
        self.out_prefix = out_prefix
        self._filter_lens = {
            band: len(create_filter(
                data=None, sfreq=self._SFREQ,
                l_freq=lf, h_freq=hf,
                l_trans_bandwidth=self._l_trans_bandwidth,
                h_trans_bandwidth=self._h_trans_bandwidth,
                fir_design='firwin', 
                verbose=False
            ))
            for band, (lf, hf) in self._BANDS.items()
        }
        self.data, self.ch_names = self.get_filter_data(verbose=False)

    @staticmethod
    def _check_channel_names(raw_obj, ch_mapper, verbose):
        rc_1 = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1',
                'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ',
                'CZ', 'PZ', 'A1', 'A2']
        rc_2 = rc_1[:-2]

        if ch_mapper is None:
            mapper = {c: c.upper() for c in raw_obj.ch_names}
        elif isinstance(ch_mapper, dict):
            mapper = ch_mapper
        else:
            raise RuntimeError("ch_mapper must be None or dict.")

        raw_obj.rename_channels(mapper, verbose=verbose)
        names = set(raw_obj.ch_names)
        if set(rc_1).issubset(names):
            raw_obj.pick(picks=rc_1)
        elif set(rc_2).issubset(names):
            raw_obj.pick(picks=rc_2)
        else:
            raise RuntimeError("Channel Error")

    def get_filter_data(self, verbose=False):
        raw = load_raw_(self.in_file)
        variant = os.path.basename(os.path.dirname(self.in_file))
        suffix = "-LE" if variant == "02_tcp_le" else "-REF"
        self.ch_mapper = {
            ch: ch.removeprefix("EEG ")
                 .removesuffix(suffix)
                 .removesuffix("-0")
            for ch in raw.ch_names
        }
        self._check_channel_names(raw, self.ch_mapper, verbose)

        raw.load_data(verbose=verbose)
        raw.resample(self._SFREQ, verbose=verbose)
        raw.set_eeg_reference(ref_channels="average", verbose=verbose)

        data0 = raw.get_data()
        max_len = max(self._filter_lens.values())
        if data0.shape[1] < max_len:
            raise RuntimeError(f"Data too short ({data0.shape[1]}) for longest filter ({max_len})")

        filtered = {
            key: mne.filter.filter_data(
                data0, self._SFREQ, l_freq=lf, h_freq=hf,
                l_trans_bandwidth=self._l_trans_bandwidth,
                h_trans_bandwidth=self._h_trans_bandwidth,
                verbose=verbose
            ).astype(np.float32)
            for key, (lf, hf) in self._BANDS.items()
        }
        ch_names = raw.ch_names
        del raw, data0
        gc.collect()
        return filtered, ch_names

    @staticmethod
    def _filter_intervalset(input_set, threshold):
        return [iv for iv in input_set if iv.upper_bound - iv.lower_bound > threshold]

    def get_regions(self, seg_len=5.0, amp_threshold=400, merge_len=1.0, drop=60.0):
        """
        :param seg_len: unit s, keep > seg_len
        :param amp_threshold: unit uV, keep < amp_threshold
        :param merge_len: unit s
        :param drop: unit s, drop regions at the first and in the end.
        :return:
        """
        data = self.data["whole"]
        m_threshold = int(merge_len * self._SFREQ)
        seg_threshold = int(seg_len * self._SFREQ)
        start = int(drop * self._SFREQ)
        end = data.shape[1] - int(drop * self._SFREQ)

        whole_r = lui.IntervalSet([lui.Interval(lower_bound=start, upper_bound=end)])

        flag = np.abs(data) * 1e6 > amp_threshold
        art = lui.find_continuous_area_2d(flag)
        c_art = [lui.merge_continuous_area(s, threshold=m_threshold) for s in art]
        c_clean = [whole_r.difference(s) for s in c_art]

        keep_clean = [self._filter_intervalset(s, seg_threshold) for s in c_clean]

        out = []
        for idx, item_set in enumerate(keep_clean, 0):
            for item in item_set:
                tmp = np.arange(item.lower_bound, item.upper_bound + 1, seg_threshold)
                if len(tmp) > 1:
                    for idj in range(len(tmp) - 1):
                        out.append([idx, self.ch_names[idx], tmp[idj], tmp[idj + 1]])

        df_clean = pd.DataFrame(out, columns=["idx", "ch_names", "clip_start", "clip_stop"])
      

        udata = np.stack([self.data["whole"],
                          self.data["delta"],
                          self.data["theta"],
                          self.data["alpha"],
                          self.data["low_beta"],
                          self.data["high_beta"]], axis=0) * 1.0e6

        ts = list(zip(df_clean.idx, df_clean.clip_start, df_clean.clip_stop))

        # Only keep signal segments of which there are at least 200.
        if len(ts) >= 200:
            if len(ts) > self.max_clips:
                ts = random.sample(ts, self.max_clips)

            outputs = []
            for p in ts:
                idx, start, stop = p
                tmp = udata[:, idx, start:stop]
                outputs.append(tmp)

            outputs = np.stack(outputs, axis=0)
            out_file = os.path.join(self.out_dir, "%s.npy" % self.out_prefix)
            np.save(out_file, outputs)

def worker(in_file, out_dir, out_prefix, ch_mapper=None):
    """
    Process one EDF file to generate its clip .npy in out_dir with name out_prefix.npy.
    Returns (output_path, label) on success, or (None, None) on failure.
    """
    os.makedirs(out_dir, exist_ok=True)
    try:
        # Get the label for this EDF file
        label = parse_label_file(in_file)
        
        dg = DataGen(in_file, out_dir, out_prefix, ch_mapper)
        out_file = dg.get_regions()
        
        # Only return the result if the .npy file was actually created
        npy_path = os.path.join(out_dir, f"{out_prefix}.npy")
        if os.path.exists(npy_path):
            return (npy_path, label)
        else:
            return (None, None)
    except Exception as e:
        print(f"Error processing {in_file}: {e}")
        return (None, None)


def generate_clips(base_dir, out_base, n_jobs):
    """
    Discover all EDFs under each subdirectory of base_dir,
    and generate clips for each into a single out_base directory.

    All .npy files (one per EDF) will be placed directly in out_base.
    Also creates a CSV file mapping each .npy file to its label.
    """
    # find all EDF files under base_dir/<any_split>/...
    splits = [d for d in os.listdir(base_dir)
              if os.path.isdir(os.path.join(base_dir, d))]
    tasks = []
    for split in splits:
        split_dir = os.path.join(base_dir, split)
        pattern = os.path.join(split_dir, "**", "*.edf")
        for edf in glob.glob(pattern, recursive=True):
            prefix = os.path.splitext(os.path.basename(edf))[0]
            tasks.append((edf, out_base, prefix))

    os.makedirs(out_base, exist_ok=True)

    results = Parallel(n_jobs=n_jobs)(
        delayed(worker)(edf_path, out_base, prefix)
        for edf_path, out_base, prefix in tqdm(tasks)
    )

    # Filter out failed results and create label mapping
    successful_results = [(npy_path, label) for npy_path, label in results if npy_path is not None]
    
    if successful_results:
        # Create DataFrame with file paths and labels
        label_data = []
        for npy_path, label in successful_results:
            # Use relative path from out_base for cleaner CSV
            rel_path = os.path.relpath(npy_path, out_base)
            label_data.append({
                'npy_path': rel_path,
                'label': label
            })
        
        # Save to CSV
        label_df = pd.DataFrame(label_data)
        label_csv_path = os.path.join(out_base, 'labels.csv')
        label_df.to_csv(label_csv_path, index=False)
        print(f"Label mapping saved to {label_csv_path}")
        print(f"Total files processed: {len(successful_results)}")
        print(f"Seizure files: {sum(1 for _, label in successful_results if label == 1)}")
        print(f"Background files: {sum(1 for _, label in successful_results if label == 0)}")
    
    return successful_results




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_base", required=True,
                        help="Root directory containing EDFs organized by split")
    parser.add_argument("--out_base", required=True,
                        help="Output base directory for clips")
    parser.add_argument("--n_jobs", type=int, default=10)
    args = parser.parse_args()
    results = generate_clips(args.raw_base, args.out_base, args.n_jobs)
