import os
import torch
import numpy as np
import random
import warnings
import wandb
import torch.optim as optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from omegaconf import OmegaConf
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from sem2 import SEMDepthDataset
from utils2 import prepare_inference, save_output, load_weights


def inference(model, test_loader, config):
    test_loader = tqdm(test_loader, desc=f"Test")
    
    for step, data in enumerate(test_loader):
        model.eval()
        sem, key = data
        sem = sem.cuda()
        output = model(sem)
        if (step + 1) % 100 == 0:
            path = config.log.infer_path
            save_output(output, key, path)
            print(f'{step + 1}th infer saved')



def inference_driver(config):
    
    # Setting up model for inference
    model = prepare_inference(config)
    model.load_state_dict(load_weights(config.test.ckpt_file))
    model.cuda()
    
    test_dataset = SEMDepthDataset(data_path=config.data.test_path, train=False)

    test_sampler = None

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.test.batch_size,
        shuffle=(test_sampler is None),
        num_workers=config.test.workers,
        pin_memory=True,
        sampler=test_sampler,
    )

    inference(model, test_loader, config)
    


if __name__ == '__main__':

    config = OmegaConf.load("config2.yaml")
    config.merge_with_cli()

    inference_driver(config)

