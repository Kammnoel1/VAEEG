# -*- coding: utf-8 -*-
import argparse
import itertools
import os
import sys
import time
from tqdm import tqdm

import torch
import torch.utils.data
import yaml
from torch import optim
from torch.utils.tensorboard import SummaryWriter

# Ensure the project root is on the path (adjust if necessary)
_CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(_CURRENT_DIR)

from model.opts.dataset import ClipDataset
from model.opts.ckpt import save_model, init_model
from model.opts.viz import batch_imgs
from model.net.modelA import VAEEG
from model.net.losses import recon_loss, kl_loss

torch.autograd.set_detect_anomaly(True)


def save_loss_per_line(target_file, line, header):
    if os.path.isfile(target_file):
        with open(target_file, "r") as fi:
            dat = [l.strip() for l in fi.readlines() if l.strip() != ""]
        if len(dat) == 0 or dat[0] != header:
            with open(target_file, "w") as fo:
                print(header, file=fo)
                print(line, file=fo)
        else:
            with open(target_file, "a") as fo:
                print(line, file=fo)
    else:
        with open(target_file, "w") as fo:
            print(header, file=fo)
            print(line, file=fo)


class Estimator(object):
    def __init__(self, in_model, n_gpus, ckpt_file=None):
        self.model, self.aux, self.device = init_model(in_model, n_gpus=n_gpus, ckpt_file=ckpt_file)

    @staticmethod
    def pearson_index(x, y, dim=-1):
        xy = x * y
        xx = x * x
        yy = y * y
        mx = x.mean(dim)
        my = y.mean(dim)
        mxy = xy.mean(dim)
        mxx = xx.mean(dim)
        myy = yy.mean(dim)
        r = (mxy - mx * my) / torch.sqrt((mxx - mx ** 2) * (myy - my ** 2))
        return r

    def train(self, input_loader, model_dir, n_epoch, lr, beta, n_print):
        summary_dir = os.path.join(model_dir, "save")
        os.makedirs(summary_dir, exist_ok=True)
        loss_file = os.path.join(model_dir, "train_loss.csv")
        writer = SummaryWriter(summary_dir)

        current_epoch = self.aux.get("current_epoch", 0)
        current_step = self.aux.get("current_step", 0)

        optimizer = optim.RMSprop(
            itertools.chain(self.model.encoder.parameters(), self.model.decoder.parameters()),
            lr=lr
        )

        self.model.train()
        start_time = time.time()

        for ie in tqdm(range(n_epoch), desc="Epochs"):
            current_epoch += 1
            for idx, input_x in enumerate(tqdm(input_loader, total=len(input_loader), leave=False, desc="Batches")):
                current_step += 1
                input_x = input_x.float().to(self.device)
                mu, log_var, xbar = self.model(input_x)
                kld = kl_loss(mu, log_var)
                rec = recon_loss(input_x, xbar)
                loss = beta * kld + rec

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if current_step % n_print == 0:
                    writer.add_scalar('loss', loss, current_step)
                    writer.add_scalar('kld_loss', kld, current_step)
                    writer.add_scalar('rec_loss', rec, current_step)

                    error = (input_x - xbar).abs().mean()
                    writer.add_scalar('mae error', error, current_step)

                    pr = self.pearson_index(input_x, xbar)
                    writer.add_scalar('pearsonr', pr.mean(), current_step)

                    cycle_time = (time.time() - start_time) / n_print

                    values = (
                        current_epoch, current_step, cycle_time,
                        loss.cpu().detach().numpy(),
                        kld.cpu().detach().numpy(),
                        rec.cpu().detach().numpy(),
                        error.cpu().detach().numpy(),
                        pr.mean().cpu().detach().numpy()
                    )
                    names = ["current_epoch", "current_step", "cycle_time",
                             "loss", "kld_loss", "rec_loss", "error", "pr"]
                    print("[Epoch %d, Step %d]: (%.3f s / cycle])\n"
                          "  loss: %.3f; kld_loss: %.3f; rec: %.3f;\n"
                          "  mae error: %.3f; pr: %.3f.\n" % values)

                    img = batch_imgs(
                        input_x.cpu().detach().numpy()[:, 0, :],
                        xbar.cpu().detach().numpy()[:, 0, :],
                        2048, 4, 2, fig_size=(8, 5)
                    )
                    writer.add_image("signal", img, current_step, dataformats="HWC")
                    start_time = time.time()

                    n_float = len(values) - 2
                    fmt_str = "%d,%d" + ",%.3f" * n_float
                    save_loss_per_line(loss_file, fmt_str % values, ",".join(names))
            out_ckpt_file = os.path.join(model_dir, f"ckpt_epoch_{current_epoch}.ckpt")
            save_model(self.model, out_file=out_ckpt_file, auxiliary={"current_step": current_step, "current_epoch": current_epoch})
        writer.close()


def train_model_for_band(band, yaml_config, z_dim):
    """
    Train the model for a single frequency band.
    
    Args:
        band: str, frequency band name (e.g., "theta").
        yaml_config: dict, the loaded YAML configuration.
        z_dim: int, dimensionality of the latent space.
    """
    train_params = yaml_config["Train"]
    model_params = yaml_config["Model"]
    dataset_params = yaml_config["DataSet"]
    
    # Overwrite the band for this training run.
    model_params["name"] = band
    # Create a subfolder for saving models for this band.
    model_dir = os.path.join(train_params["model_dir"], f"{band}_z{z_dim}")
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"\nStarting training for the {band} band:")
    print(f"Saving checkpoints and logs to: {model_dir}")
    
    # Instantiate the model.
    model = VAEEG(
        in_channels=model_params["in_channels"],
        z_dim=z_dim,
        negative_slope=model_params["negative_slope"],
        decoder_last_lstm=model_params["decoder_last_lstm"]
    )
    
    # Initialize (or load from checkpoint if provided).
    est = Estimator(model, train_params["n_gpus"], train_params["ckpt_file"])
    
    # The dataset loader expects a file named "<band>.npy" in dataset_params["data_dir"].
    train_ds = ClipDataset(
        data_dir=dataset_params["data_dir"],
        band_name=band,
        clip_len=dataset_params["clip_len"]
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        shuffle=True,
        batch_size=dataset_params["batch_size"],
        drop_last=True,
        num_workers=0
    )
    
    # Run the training loop for this frequency band.
    est.train(
        input_loader=train_loader,
        model_dir=model_dir,
        n_epoch=train_params["n_epoch"],
        lr=train_params["lr"],
        beta=train_params["beta"],
        n_print=train_params["n_print"]
    )

def main():
    parser = argparse.ArgumentParser(description='Training Pipeline for VAEEG across Frequency Bands')
    parser.add_argument('--yaml_file', type=str, required=True,
                        help="Path to the YAML configuration file")
    parser.add_argument('--z_dim', type=int, required=True,
                        help="Latent space dimension")
    args = parser.parse_args()
    
    # Load YAML configuration.
    with open(args.yaml_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update model parameters with provided z_dim.
    config["Model"]["z_dim"] = args.z_dim
    
    # Define frequency bands.
    bands = ["delta", "theta", "alpha", "low_beta", "high_beta"]
    
    # Loop over each frequency band to train a model.
    for band in bands:
        train_model_for_band(band, config, args.z_dim)
    
    print("Training pipeline completed for all bands.")

if __name__ == "__main__":
    main()
