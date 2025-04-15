import os
import numpy as np
import argparse

def fft_bandpass(segment, sfreq, f_low, f_high):
    """
    Apply an FFT-based bandpass filter on a 1D segment.
    
    Args:
        segment: 1D NumPy array of the signal.
        sfreq: Sampling frequency in Hz.
        f_low: Lower frequency cutoff.
        f_high: Upper frequency cutoff.
        
    Returns:
        filtered: 1D NumPy array of the filtered signal.
    """
    n = segment.shape[0]
    fft_vals = np.fft.rfft(segment)
    freqs = np.fft.rfftfreq(n, d=1/sfreq)
    mask = (freqs >= f_low) & (freqs <= f_high)
    fft_filtered = fft_vals * mask
    filtered = np.fft.irfft(fft_filtered, n=n)
    return filtered

def partition_channel(data_channel, sfreq):
    """
    Partition the data for one channel into frequency bands using FFT.
    
    Args:
        data_channel: 2D NumPy array with shape (num_segments, segment_length)
                      representing one channel across segments.
        sfreq: Sampling frequency.
    
    Returns:
        A dictionary with keys: 'whole', 'delta', 'theta', 'alpha', 'low_beta', 'high_beta'
        Each value is a 2D NumPy array of shape (num_segments, segment_length)
    """
    bands = {
        "delta": (1.0, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "low_beta": (13.0, 20.0),
        "high_beta": (20.0, 30.0)
    }
    num_segments, seg_length = data_channel.shape
    out = {}
    # 'whole' is the unfiltered signal.
    out["whole"] = data_channel.copy()
    for band_name, (f_low, f_high) in bands.items():
        filtered_segments = []
        for seg in data_channel:
            filtered = fft_bandpass(seg, sfreq, f_low, f_high)
            filtered_segments.append(filtered)
        out[band_name] = np.stack(filtered_segments, axis=0)
    return out

def save_partitioned_data(input_file, output_dir, sfreq):
    """
    Load the merged data, partition each channel into frequency bands,
    and save the results in separate folders for each channel.
    
    Args:
        input_file: Path to the input NumPy file 
                    (expected shape: (num_segments, num_channels, seg_length)).
        output_dir: Directory where the channel folders will be created.
        sfreq: Sampling frequency in Hz.
    
    Returns:
        None. Files are saved to disk.
    """
    data = np.load(input_file)
    num_segments, num_channels, seg_length = data.shape
    print(f"Loaded data with shape: {data.shape}")
    
    # Iterate over each channel.
    for ch in range(num_channels):
        channel_data = data[:, ch, :]  # Shape: (num_segments, seg_length)
        partitioned = partition_channel(channel_data, sfreq)
        ch_folder = os.path.join(output_dir, f"channel_{ch+1}")
        os.makedirs(ch_folder, exist_ok=True)
        for band_name, band_data in partitioned.items():
            out_path = os.path.join(ch_folder, f"{band_name}.npy")
            np.save(out_path, band_data)

def main(): 
    parser = argparse.ArgumentParser(description="Partition EEG data into frequency bands using FFT.")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to input NumPy file (merged data).")
    parser.add_argument("--sfreq", type=int,
                        help="Sampling frequency (Hz).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory where partitioned data will be saved (one folder per channel).")
    args = parser.parse_args()
    
    save_partitioned_data(args.input_file, args.output_dir, args.sfreq)
    

if __name__ == "__main__":
    main()
