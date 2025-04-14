import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from src.model.net.modelA import VAEEG
from src.model.opts.ckpt import load_model

# === Settings ===
channel_folder = "./new_data_downsampled/train/channel_1"  # Folder for channel 32 data
segment_index = 43          # The segment index to test (e.g., 20th segment)
z_dim = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Select the frequency band to test (e.g., "theta")
band = "theta"
color_orig = "blue"
color_recon = "red"

# === Load the original segment ===
# Since you're training on normalized data, we ensure that the segment is normalized.
file_path = os.path.join(channel_folder, f"{band}.npy")
if not os.path.isfile(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

data = np.load(file_path)  # Expected shape: (num_segments, segment_length)
segment = data[segment_index]  # This is a 1D array (segment_length,)

# Manually normalize the segment (if it is not already normalized)
mean = np.mean(segment)
std = np.std(segment) + 1e-8
segment_norm = (segment - mean) / std

# === Load the model checkpoint ===
# Adjust the checkpoint path as needed.
latest_ckpt = "./models/theta_z50/ckpt_epoch_5.ckpt"
print("Loading checkpoint:", latest_ckpt)

model = VAEEG(in_channels=1, z_dim=z_dim, negative_slope=0.2, decoder_last_lstm=False)
load_model(model, latest_ckpt)
model.to(device)
model.eval()

# Prepare the input tensor:
# Reshape normalized segment to (batch_size, channels, segment_length)
input_tensor = torch.tensor(segment_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

# === Model Inference ===
with torch.no_grad():
    mu_t, log_var_t, xbar_t = model(input_tensor)
    # Remove batch and channel dimensions to obtain the 1D reconstructed signal.
    reconstructed_norm = xbar_t.squeeze().cpu().numpy()

# === Plotting the Results ===
# Generate a time axis for a 1-second segment.
clip_len = segment_norm.shape[0]  # Should equal 2048 for 1-second at 2048 Hz
t = np.linspace(0, 1, clip_len, endpoint=False)

# Create subplots: original (normalized) on the left, reconstruction on the right.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot normalized original signal.
ax1.plot(t, segment_norm, color=color_orig, label="Normalized Original")
ax1.set_xlabel("Time (s)", fontsize=14)
ax1.set_ylabel("Amplitude (norm.)", fontsize=14)
ax1.set_title("Original (Normalized)", fontsize=16)
ax1.legend(fontsize=12)
ax1.grid(True)

# Plot reconstructed signal.
ax2.plot(t, reconstructed_norm, color=color_recon, label="Reconstruction")
ax2.set_xlabel("Time (s)", fontsize=14)
ax2.set_title("Reconstructed", fontsize=16)
ax2.legend(fontsize=12)
ax2.grid(True)

plt.suptitle(f"Channel 1, {band.capitalize()} Band, Segment {segment_index}", fontsize=18)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
