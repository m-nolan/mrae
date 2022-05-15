import torch
import numpy as np

def accuracy(est,target):
    return 1 - torch.abs((est - target)/target).mean(dim=1)

def fve(est,target):
    return 1 - (est-target).pow(2).mean(dim=1)/target.var(dim=1)

