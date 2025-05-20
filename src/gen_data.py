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

    def get_regions(self, seg_len=5.0, amp_threshold=400,
                    merge_len=1.0, drop=60.0):
        data = self.data["whole"]
        seg_thr = int(seg_len * self._SFREQ)
        m_thr = int(merge_len * self._SFREQ)
        start = int(drop * self._SFREQ)
        end = data.shape[1] - start

        whole = lui.IntervalSet([lui.Interval(start, end)])
        art = lui.find_continuous_area_2d(np.abs(data) * 1e6 > amp_threshold)
        cleaned = [whole.difference(lui.merge_continuous_area(s, threshold=m_thr)) for s in art]
        intervals = [self._filter_intervalset(s, seg_thr) for s in cleaned]

        rows = []
        for ch_idx, ivs in enumerate(intervals):
            for iv in ivs:
                points = np.arange(iv.lower_bound, iv.upper_bound+1, seg_thr)
                for s, e in zip(points[:-1], points[1:]):
                    rows.append((ch_idx, self.ch_names[ch_idx], s, e))
        df = pd.DataFrame(rows, columns=["idx", "ch_name", "start", "stop"])

        udata = np.stack([
            self.data[b] for b in self._BANDS.keys()
        ], axis=0) * 1e6

        recs = list(zip(df.idx, df.start, df.stop))
        if len(recs) > self.max_clips:
            recs = random.sample(recs, self.max_clips)

        clips = [udata[:, i, s:e] for i, s, e in recs]
        out = np.stack(clips, axis=0).astype(np.float32)

        out_file = os.path.join(self.out_dir, f"{self.out_prefix}.npy")
        os.makedirs(self.out_dir, exist_ok=True)
        np.save(out_file, out)
        return out_file

def worker(in_file, out_dir, out_prefix, ch_mapper=None):
    """
    Process one EDF file to generate its clip .npy in out_dir with name out_prefix.npy.
    Returns output_path on success, or None on failure.
    """
    os.makedirs(out_dir, exist_ok=True)
    try:
        dg = DataGen(in_file, out_dir, out_prefix, ch_mapper)
        out_file = dg.get_regions()
        return out_file
    except Exception as e:
        print(f"Error processing {in_file}: {e}")
        return None


def generate_clips(base_dir, out_base, n_jobs=10):
    """
    Discover all EDFs under each subdirectory of base_dir,
    and generate clips for each into a single out_base directory.

    All .npy files (one per EDF) will be placed directly in out_base.
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




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_base", required=True,
                        help="Root directory containing EDFs organized by split")
    parser.add_argument("--out_base", required=True,
                        help="Output base directory for clips")
    parser.add_argument("--n_jobs", type=int, default=10)
    args = parser.parse_args()
    res = generate_clips(args.raw_base, args.out_base, args.n_jobs)
