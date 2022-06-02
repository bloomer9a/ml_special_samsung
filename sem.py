import os
import glob
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import numpy as np


class SEMDepthDataset(Dataset):
    def __init__(self, data_path, transforms=False):
        depth_path = os.path.join(data_path, 'Depth')
        self.sem_path = os.path.join(data_path, 'SEM')
        self.sem_list = [f for f in os.listdir(self.sem_path) if f.endswith('.png')]
        self.depth_dict = self.get_depth_dict(depth_path)
        self.transforms = transforms

    def __getitem__(self, idx):
        sem, depth = self.get_sem_and_depth(idx)
        if self.transforms:
            sem = self.random_flip(sem)
        return sem, depth

    def __len__(self):
        return len(self.sem_list)

    def get_depth_dict(self, depth_path):
        depth_file_list = [f for f in os.listdir(depth_path) if f.endswith('.png')]
        depth_file_dict = {}
        for depth_file in depth_file_list:
            file_name = depth_file.split('.png')[0]
            depth_file_dict[file_name] = T.ToTensor()(Image.open(os.path.join(depth_path, depth_file)))

        return depth_file_dict

    def get_sem_and_depth(self, idx):
        sem_file = self.sem_list[idx]
        key = sem_file.split('_itr')[0]
        sem = T.ToTensor()(Image.open(os.path.join(self.sem_path, sem_file)))
        depth = self.depth_dict[key]
        return sem, depth

    def random_flip(self, sem):
        if np.random.rand() > 0.5:
            sem = T.RandomHorizontalFlip(p=1).forward(sem)

        if np.random.rand() > 0.5:
            sem = T.RandomVerticalFlip(p=1).forward(sem)

        return sem






if __name__ == '__main__':
    dataset = SEMDepthDataset(data_path='./data/Train')
    loader = DataLoader(dataset)
    loader_iter = iter(loader)
    sem, depth = next(loader_iter)
    print(f'sem:{sem.shape} depth:{depth.shape}')
