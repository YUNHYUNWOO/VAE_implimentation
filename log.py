from typing import *
import random
import functools
import logging


import torch
from torch import nn
from torch.utils.data import TensorDataset
from torchvision.transforms.functional import to_pil_image
import wandb

from FID_calculator import FID_calculator
from Vanillia_VAE import Vanilla_VAE


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

    def sample_wandb_images_vq_vae(model: Vanilla_VAE):
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
    log_info = {f'{prefix}_{key}' : log_info[key].item() for key in log_info}

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
    'fid_logger_builder': fid_logger_builder
}

def get_log_fn(config: dict, data: dict):

    extra_log = {
        log: builder[builder_name](data) for log, builder_name in config['extra_log'].items()
        } if config['extra_log'] is not None else None
    

    return functools.partial(log_fn_baseline, 
                            extra_log=extra_log,
                            prefix=config['prefix'])


