import os
import random
import argparse

import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image
import yaml
import wandb
import functools

from util.FID_calculator import FID_calculator

from model import Vanilla_VAE
from trainer import VAE_Trainer
from dataset import DataPipeline
from log import get_train_log_fn, get_test_log_fn


import os
import random
import numpy as np
import torch

CONFIG_DIR = "./config/"
parser = argparse.ArgumentParser()
parser.add_argument('-n', dest='EXP_NAME', required=True, help="name of experiment (config file name without .yaml), EXP_NAME.yaml file should be in config/ directory")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
   global parser
   args = parser.parse_args()
   # load config file
   
   with open(os.path.join(CONFIG_DIR, f"{args.EXP_NAME}.yaml"), "r") as file:
      config = yaml.safe_load(file)  # Use safe_load to prevent execution of arbitrary Python objects

   def pretty(d, indent=0):
      for key, value in d.items():
         print('\t' * indent + str(key))
         if isinstance(value, dict):
            pretty(value, indent+1)
         else:
            print('\t' * (indent+1) + str(value))
   
   print("your config: ")
   print(pretty(config))  # Output the parsed YAML as a dictionary

   seed_everything(config['seed'])

   data_pipeline = DataPipeline(config['data'])
   # data initialization
   data_pipeline.download_data()
   data = data_pipeline.get_data()

   # model initialization
   if(config['model']['name'] == 'Vanilla_VAE'):
      model = Vanilla_VAE(**config['model']['model_params'])

   # log functions
   train_log = get_train_log_fn()

   fid_calcuator = FID_calculator(data['test']['dataset'])
   test_log = get_test_log_fn(fid_calcuator)
   # test_log = get_test_log_fn(None)


   # trainer initialization
   trainer = VAE_Trainer(model=model,
                         train_dataloader=data['train']['dataloader'],
                         val_dataloader=data['test']['dataloader'],
                         train_log=train_log,
                         val_log=test_log,
                         config=config['trainer']
                         )

   with wandb.init(project="VAEs", entity="hyunwoo629-hanyang-university", config=config, name=args.EXP_NAME):
      trainer.train()

if __name__ == '__main__':
   main()
