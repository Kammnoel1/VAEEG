# -*- coding: utf-8 -*-
import os
import warnings

import torch
import torch.nn as nn
import torch.distributed as dist
from utils.decorator import type_assert
from utils.check import *


@type_assert(model=nn.Module, ckpt_file=str)
def load_model(model, ckpt_file):
    """

    Args:
        model: nn.Module
            your net
        ckpt_file: str
            path of checkpoint file

    Returns
        dict for auxiliary information

    """
    state_dict = torch.load(ckpt_file, map_location=torch.device('cpu'))
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state_dict["model"])
    else:
        model.load_state_dict(state_dict["model"])

    return state_dict["auxiliary"]


@type_assert(model=nn.Module)
def save_model(model, out_file, auxiliary=None):
    """

    Args:
        model: nn.Module
            your net
        out_file: str
            path of checkpoint file
        auxiliary: dict or None
            other information to record

    Returns
        None
    """
    check_type("auxiliary", auxiliary, [dict, type(None)])
    data = dict()

    if auxiliary is None:
        data["auxiliary"] = {}
    else:
        data["auxiliary"] = auxiliary

    if isinstance(model, nn.DataParallel):
        data["model"] = model.module.state_dict()
    else:
        data["model"] = model.state_dict()

    torch.save(data, out_file)


@type_assert(m=nn.Module, n_gpus=int)
def init_model(m, n_gpus, ckpt_file=None):
    """
    Initialize model with optional checkpoint, GPU/DP wrapper.
    """
    check_type("ckpt_file", ckpt_file, [str, type(None)])

    # decide device
    n_gpu = min(max(n_gpus, 0), torch.cuda.device_count())
    device = torch.device("cuda" if n_gpu > 0 else "cpu")
    m.to(device)

    # wrap for distributed if needed
    if n_gpu > 1:
        rank = int(os.environ.get("SLURM_PROCID", 0))
        world_size = int(os.environ.get("SLURM_NTASKS", n_gpu))
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
        torch.cuda.set_device(rank % torch.cuda.device_count())
        m.to(device)
        
        # Check if model is in deterministic mode to handle unused parameters
        find_unused_params = getattr(m, 'deterministic', False)
        
        wrapped = nn.parallel.DistributedDataParallel(
            m,
            device_ids=[torch.cuda.current_device()],
            output_device=torch.cuda.current_device(),
            find_unused_parameters=find_unused_params
        )
    else:
        wrapped = m

    aux = {}
    if isinstance(ckpt_file, str) and os.path.isfile(ckpt_file):
        print(f"Initializing model from {ckpt_file}")
        from model.opts.ckpt import load_model
        aux = load_model(wrapped, ckpt_file)
    elif isinstance(ckpt_file, str):
        warnings.warn(f"Checkpoint not found: {ckpt_file}. Initializing randomly.")

    print(f"Model assigned to {device}{' with DDP' if n_gpu>1 else ''}.")
    return wrapped, aux, device