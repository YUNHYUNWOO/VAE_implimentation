import os
from typing import *

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchsummary import summary
import wandb


from model import Vanilla_VAE

#def train_log()

class VAE_Trainer():
    def __init__(self, 
                 model: nn.Module, 
                 train_dataloader: DataLoader, 
                 val_dataloader: DataLoader, 
                 train_log: Callable[[dict, nn.Module], None], 
                 val_log: Callable[[dict, nn.Module], None], 
                 config: dict):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.train_log = train_log
        self.val_log = val_log

        self.config = config
        self.opt = self.configure_optimizers()

    def configure_optimizers(self):
        print(self.config['optim'])
        optimizer = optim.Adam(self.model.parameters(), **self.config['optim'])
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x = batch
        output = self.model(x)
        
        loss = self.model.loss_fn(**output)
        log_info = loss
        loss = loss['loss']
        return loss, log_info

    def validate_model(self, batch, batch_idx):
        mean_loss = None
        for i, (x, _) in tqdm(enumerate(self.val_dataloader), desc="Validating Models", total=len(self.val_dataloader)):
            output = self.model(x)

            loss = self.model.loss_fn(**output)
            if mean_loss == None:
                mean_loss = loss
            else:
                mean_loss = {key : mean_loss[key] + loss['key'] for key in loss.keys()}

        mean_loss = {key : mean_loss[key] / float(len(self.val_dataloader)) for key in mean_loss.keys()}
        log_info = mean_loss
        
        return log_info
    def save(self):
        save_name = self.config['checkpoint']['save_name']
        save_dir = self.config['checkpoint']['save_dir']
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, save_name), exist_ok=True)

        save_model_dir = os.path.join(os.path.join(save_dir, save_name), "model.pth")
        torch.save(self.model.cpu(), save_model_dir)
        save_state_dir = os.path.join(os.path.join(save_dir, save_name), "model_state.pth")
        torch.save(self.model.cpu().state_dict(),  save_state_dir)
        
    def train(self):
        self.model.to(self.config['device'])
        total_step = 0

        for epoch in range(self.config['max_epochs']):
            for i, (x, _) in tqdm(enumerate(self.train_dataloader), desc=f"Epoch {epoch}: Training batches", total=len(self.train_dataloader)):
                total_step += 1
                self.model.train()
                x = x.to(self.config['device'])

                loss, log_info = self.training_step(x, i)
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()
                # logging
                if total_step % self.config['logging']['logging_step'] == 0:
                    #default logging info contains
                        # - step
                        # - epoch
                        # - loss
                    default_logging_info = {'step': total_step,
                                            'epoch': epoch,
                                            'log_info': log_info}
                    self.train_log(default_logging_info, self.model)

            with torch.no_grad():
                self.model.eval()
                default_log_info = {'step': total_step,
                                    'epoch': epoch,
                                    'log_info': log_info}
                self.val_log(default_log_info, self.model)

        self.save()

