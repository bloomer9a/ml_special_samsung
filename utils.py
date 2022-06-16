import os
import torch
import torch.nn as nn
import torch.optim as optim


def make_coord(shape, ranges=None, flatten=True):
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


class RMSE(nn.Module):
    def __init__(self):
        super(RMSE, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        batch_size = yhat.size(0)
        return torch.sqrt(self.mse(yhat, y) / batch_size)


def calculate_rmse(yhat, y):
    rmse = RMSE()
    return rmse(yhat, y)


def setup_model(config):
    if config.model.name == 'unet':
        from models.unet import UNet
        model = UNet()
        return model
    else:
        Exception("InvalidModelError: you must choose a valid model to train")


def get_criterion(loss):
    if loss == 'l1':
        return nn.L1Loss()
    elif loss == 'mse':
        return nn.MSELoss()
    elif loss == 'rmse':
        return RMSE
    elif loss == 'cross_entropy':
        return nn.CrossEntropyLoss()
    else:
        Exception('InvalidLossError: please choose an appropriate loss')


def resume_training(config, model):
    resume_path = config.model.resume_path
    if os.path.isfile(resume_path):
        print(f"=> loading checkpoint '{resume_path}'")
        if config.gpu is None:
            checkpoint = torch.load(resume_path)
        else:
            # Map model to be loaded to specified single gpu.
            loc = f"cuda:{config.gpu}"
            checkpoint = torch.load(resume_path, map_location=loc)
        config.start_epoch = checkpoint["epoch"]
        if model.module:
            model.module.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint["state_dict"])
        optimizer = checkpoint["optimizer"]
        print(f"=> loaded checkpoint '{resume_path}' (epoch {checkpoint['epoch']})")
        last_epoch = checkpoint["epoch"] - 1
        del checkpoint
        torch.cuda.empty_cache()
    else:
        print(f"=> no checkpoint found at '{config.resume}'")
        last_epoch = -1
        optimizer = None

    return model, optimizer, last_epoch


def prepare_training(config):
    model = setup_model(config)
    if config.model.resume:
        model, optimizer, start_epoch = resume_training(config, model)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999, last_epoch=start_epoch)
    else:
        optimizer = optim.AdamW(params=model.parameters(), lr=config.optimizer.learning_rate)
        start_epoch = 0
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999, last_epoch=-1)
    last_epoch = start_epoch + config.train.epoch
    ckpt_path = os.path.join(config.log.base_path, 'ckpt', config.model.name)
    criterion = get_criterion(config.train.criterion)
    return model, optimizer, scheduler, criterion, start_epoch, last_epoch, ckpt_path


def save_checkpoint(state, is_best, model_name, epoch, path):
    if is_best:
        torch.save(state, os.path.join(path, f"{model_name}_best.pth.tar"))
    else:
        torch.save(state, os.path.join(path, f"{model_name}_e{epoch:03d}.pth.tar"))
