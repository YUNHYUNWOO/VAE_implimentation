from typing import *
import random
import functools
import logging
from io import BytesIO

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset
from torchvision.transforms.functional import to_pil_image
import wandb
import matplotlib.pyplot as plt
from PIL import Image


from FID_calculator import FID_calculator
from model import Vanilla_VAE, VQ_VAE
# from VQ_VAE_auto import VQ_VAE_Auto


def sample_wandb_images_vanilla_vae(model: Vanilla_VAE):
    with torch.no_grad():
        sampled_images_tensor = model.sample(n_sample=6)
    sampled_images = []
    for i in range(sampled_images_tensor.shape[0]):
        image = to_pil_image(sampled_images_tensor[i])
        image = wandb.Image(image)
        sampled_images.append(image)
    return sampled_images

def wandb_image_logger_vq_vae(data: dict):


    def sample_random_test_batch():
        indices = [random.randint(0, len(data['test']['dataset'])) for i in range(6)]
        base_samples = torch.stack([data['test']['dataset'][i][0] for i in indices])
        return base_samples

    def sample_wandb_images_vq_vae(model: VQ_VAE):
        base_samples = sample_random_test_batch().to(model.device)
        with torch.no_grad():
            sampled_images_tensor = model.sample(base_samples)
        sampled_images = []

        for i in range(sampled_images_tensor.shape[0]):
            image = to_pil_image(sampled_images_tensor[i])
            image = wandb.Image(image)
            sampled_images.append(image)

        return sampled_images

    return sample_wandb_images_vq_vae

def cb_plot_logger(model: VQ_VAE):
    cb = model.CodeBook
    
    cluster_size = cb.cluster_size[:cb.K].detach().cpu().numpy()
    d_sq = cb.d_sq[:cb.K].detach().cpu().numpy()

    eps = 1e-5
    n = cluster_size.sum()
    cluster_size_safe = (cluster_size + eps) / (n + cb.K * eps) * n
    var = d_sq / cluster_size_safe

    # ---- matplotlib figure ----
    fig, axs = plt.subplots(1, 3, figsize=(18, 4))

    # ---- cluster_size histogram ----
    axs[0].hist(cluster_size, bins=3000)
    # axs[0].set_xlim(0, 80)
    axs[0].set_title("cluster_size distribution")
    axs[0].set_xlabel("value")
    axs[0].set_ylabel("count")

    # ---- d_sq histogram ----

    axs[1].hist(d_sq, bins=500)
    axs[1].set_title("d_sq distribution")
    axs[1].set_xlabel("value")
    axs[1].set_ylabel("count")

    # ---- var histogram ----
    axs[2].hist(var, bins=500)
    axs[2].set_xlim(0, 6000)
    axs[2].set_title("var distribution")
    axs[2].set_xlabel("value")
    axs[2].set_ylabel("count")
    plt.tight_layout()

    # ---- save figure to buffer ----
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)

    # ---- convert to numpy image ----
    img = Image.open(buf)

    return wandb.Image(img)


def fid_logger_builder(data: dict):

    fid_calcuator = FID_calculator(data['test']['dataset'])

    def get_fid_score(model: Vanilla_VAE, n_samples=1024):
        samples = model.sample(n_sample=n_samples).cpu()
        sample_dataset = TensorDataset(samples)
        
        return fid_calcuator.calc_fid_score(sample_dataset).item()

    return get_fid_score

def log_fn_baseline(default_logging_info: dict, 
                model: nn.Module, 
                prefix: str,
                extra_log: dict[str: Callable[[nn.Module], object]] | None = None
                ):
    #default logging info contains
        # - step
        # - epoch
        # - loss : model_defined loss (return of VAE.loss_fn)
    step = default_logging_info['step']
    epoch = default_logging_info['epoch']
    log_info = default_logging_info['log_info']
    
    # extra logging info
    extra_log_info = {
        key: log_fn(model) for key, log_fn in extra_log.items()
    } if extra_log is not None else {}
    
    # add 'train_' prefix to log_info keys
    log_info = {f'{prefix}_{key}' : log_info[key] for key in log_info}

    final_log = {
        "epoch": epoch,
        **log_info,
        **extra_log_info
    }
    logging.info(final_log)
    wandb.log(final_log, step=step)


builder = {
    'wandb_image_logger_vanilla_vae': lambda data: sample_wandb_images_vanilla_vae,
    'wandb_image_logger_vq_vae': wandb_image_logger_vq_vae,
    'fid_logger_builder': fid_logger_builder,
    'cb_plot_logger': lambda data: cb_plot_logger
}

def get_log_fn(config: dict, data: dict):

    extra_log = {
        log: builder[builder_name](data) for log, builder_name in config['extra_log'].items()
        } if config['extra_log'] is not None else None
    

    return functools.partial(log_fn_baseline, 
                            extra_log=extra_log,
                            prefix=config['prefix'])


