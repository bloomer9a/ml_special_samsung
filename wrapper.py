import torch
import numpy as np 
from torch.utils.data import Dataset

class ImplicitDataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sem, depth = self.dataset[idx]
        coord = self.to_pixel_samples(sem.contiguous())
        return sem, depth, coord

    def make_coord(self, shape, ranges=None, flatten=True):
        """ Make coordinates at grid centers.
        """
        coord_seqs = []
        for i, n in enumerate(shape):
            if ranges is None:
                v0, v1 = -1, 1
            else:
                v0, v1 = ranges[i]
            r = (v1 - v0) / (2 * n)
            seq = v0 + r + (2 * r) * torch.arange(n).float()
            coord_seqs.append(seq)
        ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
        if flatten:
            ret = ret.view(-1, ret.shape[-1])
        return ret

    def to_pixel_samples(self, img):
        """ Convert the image to coord-RGB pairs.
            img: Tensor, (3, H, W)
        """
        coord = self.make_coord(img.shape[-2:])
        return coord


