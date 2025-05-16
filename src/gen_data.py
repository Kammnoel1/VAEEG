# -*- coding: utf-8 -*-
import gc
import os
import random
import time
import glob

import lighten.utils.interval as lui
import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from utils.io import load_raw_
from mne.filter import create_filter

  
class DataGen(object):
    _SFREQ = 256.0

    max_clips = 500
    _l_trans_bandwidth = 0.1
    _h_trans_bandwidth = 0.1

    _BANDS = {"whole": (1.0, 30.0),
              "delta": (1.0, 4.0),
              "theta": (4.0, 8.0),
              "alpha": (8.0, 13.0),
              "low_beta": (13, 20),
              "high_beta": (20, 30.0)}

    def __init__(self, in_file, out_dir, out_prefix, ch_mapper=None):
        self.in_file = in_file
        self.ch_mapper = ch_mapper
        self.out_dir = out_dir
        self.out_prefix = out_prefix
        self._filter_lens = {}
        for band, (lf, hf) in self._BANDS.items():
            kernel = create_filter(
                data=None, sfreq=self._SFREQ,
                l_freq=lf, h_freq=hf,
                l_trans_bandwidth=self._l_trans_bandwidth,
                h_trans_bandwidth=self._h_trans_bandwidth,
                fir_design='firwin'
            )
            self._filter_lens[band] = len(kernel)
        self.data, self.ch_names = self.get_filter_data(verbose=False)


    @staticmethod
    def _check_channel_names(raw_obj, ch_mapper, verbose):
        rc_1 = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1',
                'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ',
                'CZ', 'PZ', 'A1', 'A2']
        rc_2 = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1',
                'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ',
                'CZ', 'PZ']
        

        # rename channel
        if ch_mapper is None:
            mapper = {cname: cname.upper() for cname in raw_obj.ch_names}
        elif isinstance(ch_mapper, dict):
            mapper = ch_mapper
        else:
            raise RuntimeError("ch_mapper must be None or dict.")

        raw_obj.rename_channels(mapper, verbose=verbose)

        ch_names = set(raw_obj.ch_names)

        if set(rc_1).issubset(ch_names):
            raw_obj.pick(picks=rc_1)
        elif set(rc_2).issubset(ch_names):
            raw_obj.pick(picks=rc_2)
        else:
            raise RuntimeError("Channel Error")

    def get_filter_data(self, verbose=None):
        raw = load_raw_(self.in_file)
        # Determine montage variant based on parent directory name
        variant = os.path.basename(os.path.dirname(self.in_file))
        suffix = "-LE" if variant == "02_tcp_le" else "-REF"
        self.ch_mapper = {
            ch: (ch
                .removeprefix("EEG ")
                .removesuffix(suffix)
                .removesuffix("-0")    
            )
            for ch in raw.ch_names
        }
        self._check_channel_names(raw_obj = raw,
                                  ch_mapper=self.ch_mapper, 
                                  verbose=verbose) 

        raw.load_data(verbose=verbose)
        raw.resample(self._SFREQ, verbose=verbose)

        raw.set_eeg_reference(ref_channels="average", verbose=verbose)

        data0 = raw.get_data()
        max_len = max(self._filter_lens.values())
        if data0.shape[1] < max_len:
            raise RuntimeError(
                f"Data too short ({data0.shape[1]} samples) for "
                f"longest filter ({max_len} taps)"
            )
        filter_results = {}
         
        for key, (lf, hf) in self._BANDS.items():
            
            filter_results[key] = mne.filter.filter_data(data0, self._SFREQ, l_freq=lf, h_freq=hf,
                                                        l_trans_bandwidth=self._l_trans_bandwidth,
                                                        h_trans_bandwidth=self._h_trans_bandwidth,
                                                        verbose=verbose).astype(np.float32)
        ch_names = raw.ch_names
        del raw, data0
        gc.collect()
        return filter_results, ch_names

    @staticmethod
    def _filter_intervalset(input_intervalset, threshold):
        out = []
        for item in input_intervalset:
            if item.upper_bound - item.lower_bound > threshold:
                out.append(item)
        return out

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
    mne.set_log_level('WARNING') 
    try:
        dg = DataGen(in_file, out_dir, out_prefix, ch_mapper)
        dg.get_regions()
    except Exception as e:
        print(f"Error processing {in_file}: {e}")
        import traceback
        traceback.print_exc()
        flag = False
    else:
        flag = True
    return in_file, flag


if __name__=="__main__":

    BASE    = "/Users/noelkamm/VAEEG/raw_data/tusz_mini/edf"
    out_dir = "./new_data/clips"
    n_jobs  = 10

    # build task list
    tasks = []
    for split in ("train","dev","eval"):
        for edf in glob.glob(f"{BASE}/{split}/**/*.edf", recursive=True):
            prefix = os.path.splitext(os.path.basename(edf))[0]
            tasks.append((edf, split, prefix))

    for _, split, _ in tasks:
        split_out = os.path.join(out_dir, split)
        os.makedirs(split_out, exist_ok=True)

    res = Parallel(n_jobs=n_jobs)(
        delayed(worker)(
            in_file=edf,
            out_dir=os.path.join(out_dir, split),
            out_prefix=prefix,
        )
        for edf, split, prefix in tasks
    )