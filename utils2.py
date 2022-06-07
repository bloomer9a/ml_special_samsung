import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image


class RMSE(nn.Module):
    def __init__(self):
        super(RMSE, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')

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


def prepare_inference(config):
    model = setup_model(config)
    
    return model


def save_checkpoint(state, is_best, model_name, epoch, path):
    if is_best:
        torch.save(state, os.path.join(path, f"{model_name}_best.pth.tar"))
    else:
        torch.save(state, os.path.join(path, f"{model_name}_e{epoch:03d}.pth.tar"))


def save_output(output, key, path):
    k = key[0]
    print(k)
    save_image(output, os.path.join(path, f"{k}.png"))


def load_weights(checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location='cpu')['state_dict']
    if from_parallel(state_dict):
        state_dict = unwrap_parallel(state_dict)

    return state_dict


def set_device(x, use_cpu=True):
    multi_gpu = False 

    # When input is tensor 
    if isinstance(x, torch.Tensor): 
        if use_cpu:
            x = x.cpu()

     # When input is model
    elif isinstance(x, nn.Module): 
        if use_cpu:
            x.cpu()
        else:
            torch.cuda.set_device(device[0])
            if multi_gpu:
                x = nn.DataParallel(x, device_ids=device).cuda()
            else: 
                x.cuda(device[0])
    # When input is tuple 
    elif type(x) is tuple or type(x) is list:
        x = list(x)
        for i in range(len(x)):
            x[i] = set_device(x[i], device, use_cpu)
        x = tuple(x) 

    return x 



def from_parallel(state_dict):
    from_parallel = False
    for key, _ in state_dict.items():
        if key.find('module.') != -1:
            from_parallel = True
            break 

    return from_parallel

def unwrap_parallel(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value

    return new_state_dict