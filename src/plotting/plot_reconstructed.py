import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from model.net.modelA import VAEEG
from model.net.modelA import VAEEG
from model.opts.ckpt import load_model
from model.opts.dataset import ClipDataset

# === Settings ===
channel_folder = "./new_data/train/"  
clip_idx = 43          
z_dim = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Select the frequency band to test (e.g., "theta")
band = "alpha"
color_orig = "blue"
color_recon = "red"

# === Load the original segment ===
file_path = os.path.join(channel_folder, f"{band}.npy")
 


# === Load the model checkpoint ===
# Adjust the checkpoint path as needed.
latest_ckpt = "./models/config_z50/ckpt_epoch_5.ckpt"
print("Loading checkpoint:", latest_ckpt)

model = VAEEG(in_channels=1, z_dim=z_dim, negative_slope=0.2, decoder_last_lstm=False, deterministic=False)
load_model(model, latest_ckpt)
model.to(device)
model.eval()

# Import and create dataset

# Create dataset instance for the specific band
dataset = ClipDataset(channel_folder, band)

# Get a single sample from the dataset using clip_idx
clip = dataset[clip_idx][0]
input_tensor = torch.tensor(clip, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


# === Model Inference ===
with torch.no_grad():
    mu_t, log_var_t, xbar_t = model(input_tensor)
    # Remove batch and channel dimensions to obtain the 1D reconstructed signal.
    reconstructed_norm = xbar_t.squeeze().cpu().numpy()

# === Plotting the Results ===
# Generate a time axis for a 1-second segment.
clip_len = clip.shape[0]  # Should equal 2048 for 1-second at 2048 Hz
t = np.linspace(0, 1, clip_len, endpoint=False)

# Create subplots: original (normalized) on the left, reconstruction on the right.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot normalized original signal.
ax1.plot(t, clip, color=color_orig, label="Normalized Original")
ax1.set_xlabel("Time", fontsize=14)
ax1.set_ylabel("Amplitude", fontsize=14)
ax1.set_title("Original", fontsize=16)
ax1.grid(True)

# Plot reconstructed signal.
ax2.plot(t, reconstructed_norm, color=color_recon, label="Reconstruction")
ax2.set_xlabel("Time", fontsize=14)
ax1.set_ylabel("Amplitude", fontsize=14)
ax2.set_title("Reconstructed", fontsize=16)
ax2.grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
