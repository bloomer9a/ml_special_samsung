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

from sem import SEMDepthDataset
from utils import calculate_rmse, prepare_training, save_checkpoint


def main(config):
    if config.train.seed is not None:
        random.seed(config.train.seed)
        torch.manual_seed(config.train.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if config.ddp.dist_url == "env://" and config.ddp.world_size == -1:
        config.ddp.world_size = int(os.environ["WORLD_SIZE"])

    ngpus_per_node = torch.cuda.device_count()
    print(f"Num GPUS: {ngpus_per_node}")

    config.multiprocessing_distributed = True if ngpus_per_node > 1 else False
    config.distributed = config.ddp.world_size > 1 or config.multiprocessing_distributed

    if config.multiprocessing_distributed:
        config.ddp.world_size = ngpus_per_node * config.ddp.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
    else:
        main_worker(config.gpu, ngpus_per_node, config)


def main_worker(gpu, ngpus_per_node, config):
    config.gpu = gpu
    if config.gpu is not None:
        print("Use GPU: {} for training".format(config.gpu))
    if config.distributed:
        if config.ddp.dist_url == "env://" and config.ddp.rank == -1:
            config.ddp.rank = int(os.environ["RANK"])
        if config.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            config.ddp.rank = config.ddp.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=config.ddp.dist_backend,
            init_method=config.ddp.dist_url,
            world_size=config.ddp.world_size,
            rank=config.ddp.rank,
        )
    # Setting up model to train
    model, optimizer, scheduler, criterion, start_epoch, last_epoch, ckpt_path = prepare_training(config)

    if not torch.cuda.is_available():
        print("using CPU will slow down the process for running a program")
    elif config.distributed:
        if config.gpu is not None:
            torch.cuda.set_device(config.gpu)
            model.cuda(config.gpu)
            config.train.batch_size = int(config.train.batch_size / ngpus_per_node)
            config.train.workers = int((config.train.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[config.gpu],
                find_unused_parameters=True,
            )
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif config.gpu is not None:
        torch.cuda.set_device(config.gpu)
        model = model.cuda(config.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    train_dataset = SEMDepthDataset(data_path=config.data.train_path)
    valid_dataset = SEMDepthDataset(data_path=config.data.valid_path)

    if config.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.train.workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.train.workers,
        pin_memory=True,
    )

    if config.log.use_wandb and dist.get_rank() == 0:
        wandb.init(
            project=config.log.project_name,
            name=config.model.arch + config.log.display_name,
            config=OmegaConf.to_container(config),
        )
        os.makedirs(ckpt_path, exist_ok=True)
    best_rmse = 0
    for epoch in range(start_epoch, last_epoch):
        train_rmse, train_loss = train(model, train_loader, optimizer, scheduler, criterion, epoch, dist.get_rank())
        print(f'Epoch: {epoch} Avg RMSE: {train_rmse} Avg Train Loss: {train_loss}')
        valid_rmse, valid_loss = valid(model, valid_loader, criterion, epoch)
        print(f'Epoch: {epoch} Avg RMSE: {valid_rmse} Avg Valid Loss: {valid_loss}')
        if dist.get_rank() == 0:
            if config.log.use_wandb == True:
                log_dict = {"RMSE/train": train_rmse,
                            "Loss/train": train_loss,
                            "RMSE/valid": valid_rmse,
                            "Loss/valid": valid_loss,
                            }
                wandb.log(log_dict, step=epoch)
            is_best = False
            if best_rmse > valid_rmse or epoch == 0:
                best_rmse = valid_rmse
                is_best = True
            if epoch % 10 == 0 or is_best:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "arch": config.model.name,
                        "state_dict": model.module.state_dict(),
                        "optimizer": optimizer,
                    },
                    is_best=is_best,
                    model_name=config.model.name,
                    epoch=epoch,
                    path=ckpt_path
                )
#                print(f'{config.model.name}_e{epoch:03d} has been successfully saved')


def train(model, train_loader, optimizer, scheduler, criterion, epoch, gpu_rank):
    avg_rmse = []
    avg_loss = []
    train_info_dict = dict()
    train_loader = tqdm(train_loader, desc=f"Train)[{epoch:03d}]")
    for step, data in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()
        sem, depth = data
        sem = sem.cuda()
        depth = depth.cuda()
        output = model(sem)
        loss = criterion(output, depth)
        loss.backward()
        optimizer.step()
        avg_loss.append(loss.detach().item())
        rmse = calculate_rmse(depth, output)
        avg_rmse.append(rmse.detach().item())
        train_info_dict['loss'] = np.average(avg_loss)
        train_info_dict['rmse'] = np.average(avg_rmse)
        train_loader.set_postfix(train_info_dict)
    scheduler.step()
    return np.average(avg_rmse), np.average(avg_loss)


def valid(model, valid_loader, criterion, epoch):
    avg_rmse = []
    avg_loss = []
    valid_info_dict = dict()
    valid_loader = tqdm(valid_loader, desc=f"Valid)[{epoch:03d}]")
    for step, data in enumerate(valid_loader):
        model.eval()
        sem, depth = data
        sem = sem.cuda()
        depth = depth.cuda()
        output = model(sem)
        loss = criterion(output, depth)
        avg_loss.append(loss.detach().item())
        rmse = calculate_rmse(output, depth)
        avg_rmse.append(rmse.detach().item())
        valid_info_dict['loss'] = np.average(avg_loss)
        valid_info_dict['rmse'] = np.average(avg_rmse)
        valid_loader.set_postfix(valid_info_dict)
    return np.average(avg_rmse), np.average(avg_loss)


if __name__ == '__main__':
    config = OmegaConf.load("config.yaml")
    config.merge_with_cli()
    os.environ["CUDA_VISIBLE_DEVICES"] = config.ddp.gpu_num
    os.environ["MASTER_ADDR"] = str(config.ddp.master_addr)
    os.environ["MASTER_PORT"] = str(config.ddp.master_port)
    main(config)
