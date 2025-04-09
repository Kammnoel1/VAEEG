import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from src.model.net.modelA import VAEEG
from src.model.opts.ckpt import load_model

# === Settings ===
channel_folder = "./new_data_downsampled/train/channel_32"  # Folder for channel 32 data
model_base_dir = "./models"  # Base folder for model checkpoints
segment_index = 19    # Starting index for the batch (e.g., 20th segment)
z_dim = 50
sfreq = 256                # Sampling frequency (Hz)
device = "cpu"             # Using CPU; use "cuda" if available

# Use the frequency band you want to infer (e.g., "delta")
band = "delta"
color = "blue"

# === Load the original segments for the chosen frequency band ===
file_path = os.path.join(channel_folder, f"{band}.npy")
if not os.path.isfile(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")
data = np.load(file_path)  # Expected shape: (num_segments, segment_length)
segment = data[segment_index]

# === Load the latest checkpoint for the selected band ===
# For this example, we assume you know the checkpoint path.
latest_ckpt = "./models/delta_z50/ckpt_epoch_5.ckpt"

# === Initialize and load the model ===
model = VAEEG(in_channels=1, z_dim=z_dim, negative_slope=0.2, decoder_last_lstm=False)
load_model(model, latest_ckpt)
model.to(device)
model.eval()

input_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

with torch.no_grad():
    mu_t, log_var_t, xbar_t = model(input_tensor)
    # Move to CPU and convert to NumPy, then squeeze the channel dimension.
    reconstructed = xbar_t.squeeze().cpu().numpy()

# Plot the reconstructed signals from each sample in the batch
t = np.linspace(0, 1, sfreq, endpoint=False)  # 1-second time axis

plt.figure(figsize=(10, 6))
plt.plot(t, reconstructed, label=f"Sample {segment_index}")
plt.xlabel("Time (s)", fontsize=14)
plt.ylabel("Amplitude", fontsize=14)
plt.title(f"Channel 32, Reconstructed '{band}' Band, {segment_index} segment)", fontsize=16)
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()
