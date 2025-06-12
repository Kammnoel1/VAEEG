# -*- coding: utf-8 -*-
import argparse
import itertools
import os
import sys
import time

import torch
import torch.utils.data
import yaml
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

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
        # read first and check if empty
        with open(target_file, "r") as fi:
            dat = [line.strip() for line in fi.readlines() if line.strip() != ""]

        # new records
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
        self.model, self.aux, self.device = init_model(in_model,
                                                       n_gpus=n_gpus,
                                                       ckpt_file=ckpt_file)

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

        r = (mxy - mx * my) / torch.sqrt((mxx - mx**2) * (myy - my**2))
        return r

    def train(self, input_loader, model_dir, n_epoch, lr, beta, n_print):
        summary_dir = os.path.join(model_dir, "save")
        if not os.path.isdir(summary_dir):
            os.makedirs(summary_dir)

        loss_file = os.path.join(model_dir, "train_loss.csv")
        writer = SummaryWriter(summary_dir)

        current_epoch = self.aux.get("current_epoch", 0)
        current_step = self.aux.get("current_step", 0)
        base_model = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model
        optimizer = optim.RMSprop(
            itertools.chain(
                base_model.encoder.parameters(), base_model.decoder.parameters()
            ),
            lr=lr,
        )

        self.model.train()
        start_time = time.time()
        rank = dist.get_rank() if dist.is_initialized() else 0
        dataset_len = len(input_loader.dataset)
        num_batches = len(input_loader)
        if rank == 0:
            print(f"Dataset has {dataset_len} samples, {num_batches} batches (batch_size={input_loader.batch_size})")
        for ie in range(n_epoch):
            current_epoch = current_epoch + 1
            input_loader.sampler.set_epoch(current_epoch)
            for idx, input_x in enumerate(input_loader, 0):
                current_step = current_step + 1
                input_x = input_x.to(self.device, non_blocking=True)
                mu, log_var, xbar = self.model(input_x)

                kld = kl_loss(mu, log_var)
                rec = recon_loss(input_x, xbar)
                loss = beta * kld + rec

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if current_step % n_print == 0:
                    writer.add_scalar("loss", loss, current_step)
                    writer.add_scalar("kld_loss", kld, current_step)
                    writer.add_scalar("rec_loss", rec, current_step)

                    error = input_x - xbar
                    error = error.abs().mean()
                    writer.add_scalar("mae error", error, current_step)

                    pr = self.pearson_index(input_x, xbar)
                    writer.add_scalar("pearsonr", pr.mean(), current_step)

                    cycle_time = (time.time() - start_time) / n_print

                    values = (
                        current_epoch,
                        current_step,
                        cycle_time,
                        loss.cpu().detach().numpy(),
                        kld.cpu().detach().numpy(),
                        rec.cpu().detach().numpy(),
                        error.cpu().detach().numpy(),
                        pr.mean().cpu().detach().numpy(),
                    )

                    names = [
                        "current_epoch",
                        "current_step",
                        "cycle_time",
                        "loss",
                        "kld_loss",
                        "rec_loss",
                        "error",
                        "pr",
                    ]

                    print(
                        "[Epoch %d, Step %d]: (%.3f s / cycle])\n"
                        "  loss: %.3f; kld_loss: %.3f; rec: %.3f;\n"
                        "  mae error: %.3f; pr: %.3f.\n" % values
                    )

                    img = batch_imgs(
                        input_x.cpu().detach().numpy()[:, 0, :],
                        xbar.cpu().detach().numpy()[:, 0, :],
                        256,
                        4,
                        2,
                        fig_size=(8, 5),
                    )
                    writer.add_image("signal", img, current_step, dataformats="HWC")
                    start_time = time.time()

                    n_float = len(values) - 2
                    fmt_str = "%d,%d" + ",%.3f" * n_float
                    save_loss_per_line(loss_file, fmt_str % values, ",".join(names))

            if rank == 0:
                out_ckpt_file = os.path.join(
                    model_dir, f"ckpt_epoch_{current_epoch}.ckpt"
                )
                save_model(
                    self.model,
                    out_file=out_ckpt_file,
                    auxiliary=dict(
                        current_step=current_step,
                        current_epoch=current_epoch,
                    ),
                )
        if dist.is_initialized():
            dist.barrier()
        writer.close()


def run_training(yaml_file: str, z_dim: int, band_name: str):
    with open(yaml_file, "r") as f:
        cfg = yaml.safe_load(f)
    train_params = cfg["Train"]
    model_params = cfg["Model"]
    dataset_params = cfg["DataSet"]
    model_params["z_dim"] = z_dim
    model_params["name"] = band_name

    m_dir = os.path.join(train_params["model_dir"], model_params["name"], f"z{z_dim}")
    os.makedirs(m_dir, exist_ok=True)

    model = VAEEG(
        in_channels=model_params["in_channels"],
        z_dim=model_params["z_dim"],
        negative_slope=model_params["negative_slope"],
        decoder_last_lstm=model_params["decoder_last_lstm"],
    )
    est = Estimator(
        in_model=model,
        n_gpus=train_params["n_gpus"],
        ckpt_file=train_params.get("ckpt_file"),
    )

    ds = ClipDataset(
        data_dir=dataset_params["data_dir"],
        band_name=model_params["name"],
        clip_len=dataset_params["clip_len"],
    )

    sampler = DistributedSampler(
        ds,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True,
    )
    loader = torch.utils.data.DataLoader(
    ds,
    sampler=sampler,
    batch_size=dataset_params["batch_size"] // dist.get_world_size(),
    num_workers=dataset_params["num_workers"],
    pin_memory=True,
    persistent_workers=True,
    )

    est.train(
        input_loader=loader,
        model_dir=m_dir,
        n_epoch=train_params["n_epoch"],
        lr=train_params["lr"],
        beta=train_params["beta"],
        n_print=train_params["n_print"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Model")
    parser.add_argument(
        "--yaml_file", type=str, required=True, help="configures, path of .yaml file"
    )
    parser.add_argument("--z_dim", type=int, required=True, help="z_dim")
    parser.add_argument("--band_name", type=str, default=None, help="Band name")
    args = parser.parse_args()

    run_training(
        yaml_file=args.yaml_file,
        z_dim=args.z_dim,
        band_name=args.band_name,
    )
