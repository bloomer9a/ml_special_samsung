import torch
import torch.nn as nn


def RMSE(yhat, y):
    return torch.sqrt(torch.mean(y - yhat) ** 2)
