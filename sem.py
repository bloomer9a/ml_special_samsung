import os
import glob
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class SEMDepthDataset(Dataset):
    def __init__(self, data_path, transforms=False, train=True):
        self.sem_path = os.path.join(data_path, 'SEM')
        self.sem_list = [f for f in os.listdir(self.sem_path) if f.endswith('.png')]
        self.train = train
        if train:
            depth_path = os.path.join(data_path, 'Depth')
            self.depth_dict = self.get_depth_dict(depth_path)
            self.transforms = transforms

    def __getitem__(self, idx):
        sem = self.get_sem(idx)
        key = self.get_key(idx)
        if self.train:
            depth = self.get_depth(key)
            if self.transforms:
                sem = self.random_flip(sem)
            return sem, depth

        return sem, key

    def __len__(self):
        return len(self.sem_list)

    def get_depth_dict(self, depth_path):
        depth_file_list = [f for f in os.listdir(depth_path) if f.endswith('.png')]
        depth_file_dict = {}
        for depth_file in depth_file_list:
            file_name = depth_file.split('.png')[0]
            depth_file_dict[file_name] = T.ToTensor()(Image.open(os.path.join(depth_path, depth_file)))

        return depth_file_dict

    def get_depth(self, key):
        key = key.split('_itr')[0]
        depth = self.depth_dict[key]
        return depth

    def get_sem(self, idx):
        sem_file = self.sem_list[idx]
        sem = T.ToTensor()(Image.open(os.path.join(self.sem_path, sem_file)))
        return sem

    def get_key(self, idx):
        sem_file = self.sem_list[idx]
        return sem_file

    def random_flip(self, sem):
        if np.random.rand() > 0.5:
            sem = T.RandomHorizontalFlip(p=1).forward(sem)

        if np.random.rand() > 0.5:
            sem = T.RandomVerticalFlip(p=1).forward(sem)

        return sem
