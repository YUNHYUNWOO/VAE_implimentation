import os

import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import glob
from PIL import Image
import shutil

def is_file_uppacked(path: str):
    def dir_file_count(path: str) -> int:
        total = 0
        for _, _, files in os.walk(path):
            total += len(files)
        return total

    if dir_file_count(path) > 1000:
        return True

def download_celeba() -> str:
    data_dir = os.path.join(os.getcwd(), 'data')
    os.makedirs(data_dir, exist_ok=True)

    os.system(f'kaggle datasets download --path {data_dir} arnrob/celeba-small-images-dataset')
    packed_file_path = f"{data_dir}/celeba-small-images-dataset.zip"
    unpacked_output_dir = f"{data_dir}/celeba-small-images-dataset"
    format = "zip"
    
    print("checking celeba-small unpacked...")
    if is_file_uppacked(unpacked_output_dir):
        print("celeba-small already unpacked", flush=True)
        return unpacked_output_dir
    else:
        print("unpacking celeba-small...", flush=True)
        shutil.unpack_archive(packed_file_path, unpacked_output_dir, format)
        return unpacked_output_dir

def get_transforms_from_config(config: dict) -> transforms.Compose:

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(config['patch_size']),
        transforms.RandomHorizontalFlip()
    ])

    return data_transform

def get_img_paths_from_config(config: dict) -> list[str]:
    data_path = config['data_path']
    paths_train, paths_test = [], []

    if config['name'] == "CelebA":
        paths_train = glob.glob(os.path.join(data_path, 'training', "*.jpg")) + glob.glob(os.path.join(data_path, 'testing', "*.jpg"))
        paths_test = glob.glob(os.path.join(data_path, 'validation', "*.jpg"))

    return paths_train, paths_test


class ImageDataset(Dataset):
    def __init__(self, image_paths: list[str], transform: transforms.Compose):
        self.transform = transform
        self.img_list = image_paths
            
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)

        return img, idx

class DataPipeline():
    def __init__(self, config: dict):
        self.config = config

    def download_data(self):
        assert 'name' in self.config, "dataset name must be specified in config"

        if self.config['name'] == "CelebA":
            download_celeba()
        else :
            raise NotImplementedError(f"{self.config['name']} is not on the supported dataset list.")

    def get_data(self) -> dict:

        cwd = os.getcwd()

        data_transform = get_transforms_from_config(self.config)
        paths_train, paths_test = get_img_paths_from_config(self.config)

        train_dataset = ImageDataset(paths_train, data_transform)
        test_dataset = ImageDataset(paths_test, data_transform)

        train_dataloader = DataLoader(train_dataset,
                                            batch_size=self.config['train_batch_size'], 
                                            pin_memory=self.config['pin_memory'],
                                            num_workers=self.config['num_workers'],
                                            shuffle=True)
        test_dataloader = DataLoader(test_dataset, 
                                        batch_size=self.config['val_batch_size'],
                                        pin_memory=self.config['pin_memory'],
                                        num_workers=self.config['num_workers'],
                                        shuffle=True)
        
        return {
            'train' : {
                'dataset': train_dataset,
                'dataloader': train_dataloader
            },
            'test' : {
                'dataset': test_dataset,
                'dataloader': test_dataloader
            }
        }







