import os 
import torch
import cv2
import numpy as np
from omegaconf import OmegaConf
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from PIL import Image

from sem2 import SEMDepthDataset
from wrapper import ImplicitDataset
from models.liif import LIIF


def inference(config):
    ckpt_path = '/workspace/project/ml_special_samsung/output/ckpt/liif_base_unet_encoder_rmse/liif_best.pth.tar'
    output_path = 'output/inference/test_liif_base_unet_encoder_rmse/'
    os.makedirs(output_path, exist_ok=True)
    ckpt = torch.load(ckpt_path)
    model = LIIF().cuda()
    model.load_state_dict(ckpt['state_dict'])
    sem_dataset = SEMDepthDataset(config.data.test_path, train=False)
    dataset = ImplicitDataset(sem_dataset, train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for data in loader:
        sem, key, coord = data
        batch, channel, width, height = sem.shape
        sem = sem.cuda()
        coord = coord.cuda()
        key = key[0]
        output = model(sem, coord)
        output = output.view(batch, width, height, channel)
        output = output.squeeze(0)
        output = output*255
        output.permute(1,2,0)
        output = output.detach().cpu().numpy()
        cv2.imwrite(os.path.join(output_path, key), output)


if __name__ == '__main__':
    config = OmegaConf.load('config.yaml')
    config.merge_with_cli()
    inference(config)