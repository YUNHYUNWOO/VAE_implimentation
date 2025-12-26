from typing import *
import functools
import logging

import torch
from torch import nn
from torch.utils.data import TensorDataset
from torchvision.transforms.functional import to_pil_image
import wandb

from util.FID_calculator import FID_calculator
from model import Vanilla_VAE


def sample_wandb_images(model: Vanilla_VAE):
    with torch.no_grad():
        sampled_images_tensor = model.sample(n_sample=6)
    sampled_images = []
    for i in range(sampled_images_tensor.shape[0]):
        image = to_pil_image(sampled_images_tensor[i])
        image = wandb.Image(image)
        sampled_images.append(image)

    return sampled_images

def get_fid_score(model: Vanilla_VAE, fid_calcuator: FID_calculator, n_samples=1024):
    samples = model.sample(n_sample=n_samples).cpu()
    sample_dataset = TensorDataset(samples)
    
    return fid_calcuator.calc_fid_score(sample_dataset).item()

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
    }
    
    # add 'train_' prefix to log_info keys
    log_info = {prefix + key : log_info[key].item() for key in log_info}

    final_log = {
        "epoch": epoch,
        **log_info,
        **extra_log_info
    }
    logging.info(final_log)
    wandb.log(final_log, step=step)



def get_train_log_fn():
    return functools.partial(log_fn_baseline, 
                            extra_log={ 'sampled_images' : sample_wandb_images },
                            prefix='train_')

def get_test_log_fn(fid_calcuator: FID_calculator):
    get_fid_score_p = functools.partial(get_fid_score, fid_calcuator=fid_calcuator)
    test_log = functools.partial(log_fn_baseline, 
                                extra_log={
                                    'fid_score' : get_fid_score_p
                                },
                                prefix='test_')
    return test_log