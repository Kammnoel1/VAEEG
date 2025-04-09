import torch
import torch.optim as optim
import torch.utils.data as torch_data
import numpy as np
import matplotlib.pyplot as plt
from src.model.net.modelA import VAEEG
from src.model.net.losses import kl_loss, recon_loss

# === Settings ===
sfreq = 256               # Sampling rate in Hz
num_samples = 100         # Total number of cosine waves to generate
train_ratio = 0.9
n_train = int(num_samples * train_ratio)
n_test = num_samples - n_train
n_epochs = 20
batch_size = 8            # Use mini-batches of 8 samples
z_dim = 50

# Generate time axis for one 1-second segment
t = np.linspace(0, 1, sfreq, endpoint=False)

# === Generate synthetic cosine wave data ===
# Generate 100 cosine waves (each of length sfreq samples) with slight variations.
data_list = []
for i in range(num_samples):
    amplitude = 1.0 + 0.1 * np.random.randn()  # around 1 with small variation
    phase = 0.1 * np.random.randn()             # small phase variation
    frequency = 5.0                             # 5 Hz cosine wave
    cosine_wave = amplitude * np.cos(2 * np.pi * frequency * t + phase)
    data_list.append(cosine_wave)

# Stack data: shape becomes (num_samples, sfreq)
data = np.stack(data_list, axis=0)
# Add channel dimension: now shape becomes (num_samples, 1, sfreq)
data = data[:, None, :]

# Split into training and testing data
train_data = data[:n_train]  # shape: (n_train, 1, sfreq)
test_data  = data[n_train:]  # shape: (n_test, 1, sfreq)

print(f"Train data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

# Convert data to PyTorch tensors
train_tensor = torch.tensor(train_data, dtype=torch.float32)
test_tensor = torch.tensor(test_data, dtype=torch.float32)

# Create TensorDataset and DataLoader for training
train_dataset = torch_data.TensorDataset(train_tensor)  # Each item is a tuple (sample,)
train_loader = torch_data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# === Initialize the Model ===
model = VAEEG(in_channels=1, z_dim=z_dim, negative_slope=0.2, decoder_last_lstm=True)
device = "cpu"  
model.to(device)

optimizer = optim.RMSprop(model.parameters(), lr=0.001)

# === Training Loop ===
for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0.0
    for batch in train_loader:
        # Each batch is a tuple (data_batch,), so extract the tensor.
        input_batch = batch[0].to(device)  # shape: (batch_size, 1, sfreq)
        optimizer.zero_grad()
        
        mu, log_var, xbar = model(input_batch)
        loss_kld = kl_loss(mu, log_var)
        loss_rec = recon_loss(input_batch, xbar)
        loss = loss_rec + loss_kld
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * input_batch.size(0)
    
    avg_loss = epoch_loss / n_train
    print(f"Epoch {epoch+1}/{n_epochs} - Average Loss: {avg_loss:.6f}")

# === Evaluation: Plot one reconstruction from the test set ===
model.eval()
with torch.no_grad():
    # For simplicity, use the entire test tensor as one batch
    mu_t, log_var_t, xbar_t = model(test_tensor.to(device))
    # Get the first test sample (shape becomes (1, sfreq))
    orig = test_tensor[0, 0, :].cpu().numpy()
    recon = xbar_t[0, 0, :].cpu().numpy()

plt.figure(figsize=(10, 4))
plt.plot(t, orig, label="Original")
plt.plot(t, recon, label="Reconstructed", linestyle="--")
plt.xlabel("Time (s)", fontsize=14)
plt.ylabel("Amplitude", fontsize=14)
plt.title("Reconstruction of a 1-Second Cosine Wave", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()
