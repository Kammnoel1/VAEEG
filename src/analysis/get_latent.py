import argparse
import os
import sys 

sys.path.insert(0, os.path.dirname(__file__))
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from model.opts.dataset import ClipDataset
from model.opts.ckpt import init_model
from model.net.modelA import VAEEG, re_parameterize


def get_sampled_latent(model, loader, device, z_dim):
    model.eval()
    num_samples = len(loader.dataset)
    z_arr = np.zeros((num_samples, z_dim), dtype=np.float32)  
    idx = 0
    encoder = model.module.encoder if hasattr(model, 'module') else model.encoder
    # Get the deterministic setting from the model
    deterministic = model.module.deterministic if hasattr(model, 'module') else model.deterministic
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            mu, log_var = encoder(x)
            if deterministic:
                z_batch = mu  # In deterministic mode, log_var is None, so just use mu
            else:
                z_batch = re_parameterize(mu, log_var, deterministic=deterministic)
            bs = z_batch.size(0)
            z_arr[idx:idx + bs, :] = z_batch.cpu().numpy()
            idx += bs
    return z_arr


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--yaml_file', required=True)
    p.add_argument('--z_dim', type=int, required=True)
    p.add_argument('--band_name', type=str, required=True)
    args = p.parse_args()

    # load config
    with open(args.yaml_file) as f:
        cfg = yaml.safe_load(f)
    dataset_params = cfg['DataSet']
    train_params = cfg['Train']
    model_params = cfg['Model']
    model_params['z_dim'] = args.z_dim
    model_params['name'] = args.band_name

    # paths
    model_dir = os.path.join(train_params['model_dir'], args.band_name, f"z{args.z_dim}")
    ckpt_file = os.path.join(model_dir, 'ckpt_epoch_{:d}.ckpt'.format(train_params['n_epoch']))

    # init model
    model = VAEEG(
        in_channels=model_params['in_channels'],
        z_dim=model_params['z_dim'],
        negative_slope=model_params['negative_slope'],
        decoder_last_lstm=model_params['decoder_last_lstm'],
        deterministic=model_params.get('deterministic', False)
    )
    model, aux, device = init_model(model, n_gpus=train_params["n_gpus"], ckpt_file=ckpt_file)

    # data loader
    ds = ClipDataset(
        data_dir=dataset_params['data_dir'],
        band_name=model_params['name'],
        clip_len=dataset_params['clip_len'],
    )
    loader = DataLoader(
        ds,
        batch_size=dataset_params['batch_size'],
        num_workers=dataset_params['num_workers'],
        shuffle=False
    )

    # run evaluation and collect all latent vectors
    z_all = get_sampled_latent(
        model=model,
        loader=loader,
        device=device,
        z_dim=model_params['z_dim']
    )

    save_dir = "/ptmp/noka/analysis/tusz/latent/"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir,
        f"{args.band_name}_z{args.z_dim}_test.npy"
    )
    np.save(save_path, z_all)
    print(f"Saved latent vectors to {save_path}")