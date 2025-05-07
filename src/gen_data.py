import os
import argparse
import warnings

from tqdm import tqdm
import numpy as np
import mne
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings('ignore', 'Estimated head radius', category=RuntimeWarning)

# Frequency bands
BANDS = {
    "delta":     (1.0,  4.0),
    "theta":     (4.0,  8.0),
    "alpha":     (8.0, 13.0),
    "low_beta":  (13.0,20.0),
    "high_beta": (20.0,30.0),
}


def partition_channel(data_st: np.ndarray, sfreq: float):
    """FFT‐based partition of (S,T) → dict of (S,T)."""
    S, T = data_st.shape
    fft_vals = np.fft.rfft(data_st, axis=1)
    freqs    = np.fft.rfftfreq(T, d=1.0/sfreq)
    out = {}
    for name, (f_lo, f_hi) in BANDS.items():
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        out[name] = np.fft.irfft(fft_vals * mask[None, :], n=T, axis=1)
    return out


def worker_channel_file_memmap(
    memmap_path, C, total_samples,
    sfreq, S, B, T,
    dtype, ch_idx, output_dir
):
    # Open memmap
    full_data = np.lib.format.open_memmap(
        memmap_path, mode="r", dtype=dtype, shape=(C, total_samples)
    )

    # Slice & reshape
    ch_data = full_data[ch_idx, :S*T]
    data_st = ch_data.reshape(S, T)

    # FFT‐band partition
    parts = partition_channel(data_st, sfreq)

    # Write per‐channel file
    out_path = os.path.join(output_dir, f"channel_{ch_idx:03d}.npy")
    mm = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=dtype,
        shape=(S, 1, B, T)
    )
    for b, name in enumerate(BANDS):
        mm[:, 0, b, :] = parts[name].astype(dtype)
    del mm


def combine_parallel(
    channel_dir, output_path,
    num_channels, num_bands,
    segment_len, num_segments,
    num_workers=8
):
    """Combine per-channel .npy files into one memmap."""
    shape = (num_segments, num_channels, num_bands, segment_len)
    # create output memmap
    np.lib.format.open_memmap(output_path, mode='w+', dtype=np.float32, shape=shape)

    args = [
        (ch, channel_dir, output_path, shape)
        for ch in range(num_channels)
    ]

    with ProcessPoolExecutor(max_workers=num_workers) as exe:
        futures = [exe.submit(_write_one_channel, a) for a in args]
        for _ in tqdm(as_completed(futures), total=num_channels, desc="Merging (parallel)"):
            pass


def _write_one_channel(args):
    ch, channel_dir, out_path, shape = args
    S, C, B, T = shape
    ch_mm = np.lib.format.open_memmap(
        os.path.join(channel_dir, f"channel_{ch:03d}.npy"), mode='r', dtype=np.float32
    )
    out_mm = np.lib.format.open_memmap(out_path, mode='r+', dtype=np.float32, shape=shape)
    out_mm[:, ch, :, :] = ch_mm[:, 0, :, :]
    del ch_mm, out_mm


def generate_full(input_file, sfreq, output_dir, num_workers):
    os.makedirs(output_dir, exist_ok=True)

    # Read raw data into memmap
    raw = mne.io.read_raw_eeglab(input_file, preload=False, verbose=False)
    data = raw.get_data()
    C, total_samples = data.shape
    memmap_path = os.path.join(output_dir, "all_channels_memmap.npy")
    mm = np.lib.format.open_memmap(
        memmap_path, mode="w+", dtype=np.float32, shape=data.shape
    )
    mm[:] = data
    del raw, data, mm

    # Partition parameters
    T = int(sfreq)
    S = int(total_samples // T)
    B = len(BANDS)

    # Parallel partitioning
    with ProcessPoolExecutor(max_workers=num_workers) as exe:
        futures = [
            exe.submit(
                worker_channel_file_memmap,
                memmap_path, C, total_samples,
                sfreq, S, B, T,
                np.float32, ch_idx, output_dir
            )
            for ch_idx in range(C)
        ]
        with tqdm(total=C, desc="Partitioning channels") as pbar:
            for fut in as_completed(futures):
                fut.result()
                pbar.update(1)

    # Combine into one memmap
    combined_path = os.path.join(output_dir, "data.npy")
    combine_parallel(
        channel_dir=output_dir,
        output_path=combined_path,
        num_channels=C,
        num_bands=B,
        segment_len=T,
        num_segments=S,
        num_workers=num_workers
    )
    return combined_path


def main():
    p = argparse.ArgumentParser(
        description="Parallel FFT‐band partition, per‐channel files + combine"
    )
    p.add_argument("--input_file",  required=True)
    p.add_argument("--sfreq",       type=float, required=True)
    p.add_argument("--output_dir",  required=True)
    p.add_argument("--num_workers", type=int, default=os.cpu_count(),
                   help="Max parallel workers")
    args = p.parse_args()

    path = generate_full(
        args.input_file, args.sfreq,
        args.output_dir, args.num_workers
    )
    print(f"Full data written to {path}")

if __name__ == "__main__":
    main()