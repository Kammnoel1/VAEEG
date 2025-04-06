import os
import glob
import torch
import numpy as np
from src.model.opts.ckpt import load_model
from src.model.net.modelA import VAEEG
from src.model.opts.dataset import ClipDataset
from torch.utils.data import DataLoader

# Locate the most recent checkpoint file.
def find_latest_checkpoint(ckpt_dir, pattern="ckpt_epoch_*.ckpt"):
    ckpt_files = glob.glob(os.path.join(ckpt_dir, pattern))
    if not ckpt_files:
        raise FileNotFoundError("No checkpoint files found in " + ckpt_dir)
    # Extract epoch numbers and select the maximum
    latest_ckpt = max(ckpt_files, key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
    return latest_ckpt

# Set the directory where your checkpoints are saved.
checkpoint_dir = "./models/config_z50"

latest_ckpt = find_latest_checkpoint(checkpoint_dir)
print("Loading checkpoint:", latest_ckpt)

# Instantiate the model with the same parameters as during training.
model_params = {
    "in_channels": 1,
    "z_dim": 50,  
    "negative_slope": 0.2,
    "decoder_last_lstm": True
}
model = VAEEG(**model_params)

# Load the checkpoint 
aux_info = load_model(model, latest_ckpt)
model.eval()  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set up your test dataset.
test_data_dir = "./new_data/test"  
band_name = "theta" 
clip_len = 2048  
batch_size = 32

test_dataset = ClipDataset(data_dir=test_data_dir, band_name=band_name, clip_len=clip_len)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

# Evaluate the model on test data.
recon_losses = []
with torch.no_grad():
    for batch in test_loader:
        # Move the input to the proper device
        batch = batch.to(device)
        mu, log_var, xbar = model(batch)
        # Compute reconstruction loss; you can use your recon_loss function.
        loss = torch.nn.functional.mse_loss(xbar, batch, reduction="mean")
        recon_losses.append(loss.item())

avg_loss = np.mean(recon_losses)
print(f"Average Reconstruction Loss on Test Data: {avg_loss}")
