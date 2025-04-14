import os
import glob
import torch
import numpy as np
from src.model.opts.ckpt import load_model
from src.model.net.modelA import VAEEG
from src.model.opts.dataset import ClipDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from scipy.stats import pearsonr

# Set the directory where your checkpoints are saved.
latest_ckpt = "models/theta_z50/ckpt_epoch_5.ckpt"

# Instantiate the model with the same parameters as during training.
model_params = {
    "in_channels": 1,
    "z_dim": 50,  
    "negative_slope": 0.2,
    "decoder_last_lstm": False
}
model = VAEEG(**model_params)

# Load the checkpoint 
aux_info = load_model(model, latest_ckpt)
model.eval()  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set up your test dataset.
test_data_dir = "./new_data_downsampled/test/channel_1"  
band_name = "theta" 
clip_len = 256
batch_size = 8

test_dataset = ClipDataset(data_dir=test_data_dir, band_name=band_name, clip_len=clip_len)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

def compute_pearson(x_np, xhat_np):
    # Compute Pearson correlation for each sample and average.
    correlations = []
    for x, xhat in zip(x_np, xhat_np):
        # Flatten each sample (they are 1D arrays) 
        r, _ = pearsonr(x.flatten(), xhat.flatten())
        correlations.append(r)
    return np.mean(correlations)

# Evaluate the model on test data.
mse_losses, pearson_corrs = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.float()
        batch = batch.to(device)
        mu, log_var, xbar = model(batch)
        mse_loss = F.mse_loss(xbar, batch, reduction="mean")
        # Gather data on the CPU for Pearson computation.
        batch_np = batch.cpu().detach().numpy()
        xbar_np = xbar.cpu().detach().numpy()
        pearson_val = compute_pearson(batch_np, xbar_np)
        mse_losses.append(mse_loss.item())
        pearson_corrs.append(pearson_val)

avg_mse = np.mean(mse_losses)
avg_pearson = np.mean(pearson_corrs)
print(f"Average Reconstruction MSE on Test Data: {avg_mse}")
print(f"Average Pearson Correlation on Test Data: {avg_pearson}")