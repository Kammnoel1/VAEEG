import os
import numpy as np
import matplotlib.pyplot as plt

# Folder where the partitioned data for each channel is saved.
channel_folder = "./new_data/train/channel_32"

# Frequency band names and their corresponding colors.
bands = ["whole", "delta", "theta", "alpha", "low_beta", "high_beta"]
colors = ["black", "blue", "green", "red", "orange", "purple"]

# Choose the 20th segment (index 19)
segment_index = 19

# Sampling frequency (assumed to be 2048 Hz)
sfreq = 2048

plt.figure(figsize=(10, 6))

# Loop through each band, load the corresponding data and plot the 20th segment.
for band, color in zip(bands, colors):
    file_path = os.path.join(channel_folder, f"{band}.npy")
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        continue
    data = np.load(file_path)  # Expected shape: (num_segments, segment_length)
    if segment_index >= data.shape[0]:
        raise ValueError(f"Segment index {segment_index} out of range; data has only {data.shape[0]} segments.")
    segment = data[segment_index]  # 1D array of length seg_length (should be 2048)
    # Create a time axis (1-second segment)
    t = np.linspace(0, 1, len(segment), endpoint=False)
    plt.plot(t, segment, label=band, color=color)

plt.xlabel("Time (s)", fontsize=14)
plt.ylabel("Amplitude", fontsize=14)
plt.title("Channel 32, 20th 1-second Segment: Frequency Bands", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()