import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model.opts.dataset import ClipDataset
from model.opts.ckpt import init_model
from model.net.modelA import VAEEG
from model.net.losses import recon_loss, kl_loss
from model.opts.viz import batch_imgs

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

def evaluate(model, loader, writer, device):
    model.eval()
    total_rec = 0.0
    total_r = 0.0
    count = 0
    with torch.no_grad():
        for step, x in enumerate(loader, 1):
            x = x.to(device)
            mu, log_var, xbar = model(x)
            rec = recon_loss(x, xbar)
            pr = pearson_index(x, xbar).mean()
            total_rec += rec.item()
            total_r += pr.item()
            count += 1
            writer.add_scalar('test/rec_loss', rec, step)
            writer.add_scalar('test/pearsonr', pr, step)
            if step == 1:
                img = batch_imgs(
                    x.cpu().detach().numpy()[:, 0, :],
                    xbar.cpu().detach().numpy()[:, 0, :],
                    256,
                    4,
                    2,
                    fig_size=(8, 5)
                )
                writer.add_image('signal', img, step, dataformats='HWC')

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
        decoder_last_lstm=model_params['decoder_last_lstm']
    )
    model, aux, device = init_model(model, n_gpus=train_params["n_gpus"], ckpt_file=ckpt_file)

    # data loader
    ds = ClipDataset(
        data_dir=dataset_params['data_dir'],
        band_name=model_params['name'],
        clip_len=dataset_params['clip_len'],
    )
    loader = DataLoader(ds, batch_size=dataset_params['batch_size'], num_workers=dataset_params['num_workers'], shuffle=False)

    # tensorboard writer
    log_dir = os.path.join(model_dir, 'test_logs')
    writer = SummaryWriter(log_dir)

    # run evaluation
    evaluate(model, loader, writer, device)
    writer.close()
