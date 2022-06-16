import os
import torch
import cv2
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from sem import SEMDepthDataset
from wrapper import ImplicitDataset


def get_model_and_dataset(config):
    dataset = SEMDepthDataset(config.data.test_path, train=False)
    if config.model.name == 'unet':
        from models.unet import UNet
        model = UNet().cuda()
        return model, dataset
    elif config.model.name == 'liif':
        from models.liif import LIIF
        model = LIIF().cuda()
        dataset = ImplicitDataset(dataset, train=False)
        return model, dataset
    else:
        raise InvalidModelError('enter a valid model name. model name must be either liif or unet')


def inference(config):
    ckpt_path = config.log.ckpt_path
    output_path = config.log.output_path
    os.makedirs(output_path, exist_ok=True)
    ckpt = torch.load(ckpt_path)
    model, dataset = get_model_and_dataset(config)
    model.load_state_dict(ckpt['state_dict'])
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
        output = output * 255
        output.permute(1, 2, 0)
        output = output.detach().cpu().numpy()
        cv2.imwrite(os.path.join(output_path, key), output)


if __name__ == '__main__':
    config = OmegaConf.load('config.yaml')
    config.merge_with_cli()
    inference(config)
