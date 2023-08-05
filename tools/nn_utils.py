"""
some tools for neural networks
"""
import copy

import torch
from torch.nn.init import xavier_normal_, kaiming_normal_


def init_weights(model, init_type):
    if init_type not in ['none', 'xavier', 'kaiming']:
        raise ValueError('init must in "none", "xavier" or "kaiming"')

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'xavier':
                xavier_normal_(m.weight.data, gain=1.0)
            elif init_type == 'kaiming':
                kaiming_normal_(m.weight.data, nonlinearity='relu')

    if init_type != 'none':
        model.apply(init_func)


def calculate_param_num(model: torch.nn.Module) -> int:
    cnt = 0
    for name, param in model.named_parameters():
        print(name, param.numel())
        cnt += param.numel()
    return cnt


def copy_model(target_model: torch.nn.Module) -> torch.nn.Module:
    return copy.deepcopy(target_model)
