# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import torch.nn as nn


__all__ = ['activate']


SUPPORTED = {
    'relu': nn.ReLU,
    'relu6': nn.ReLU6,
    'sigmoid': nn.Sigmoid,
    'leakyrelu': nn.LeakyReLU
}


def activate(cfg):
    op_type = cfg.pop('type', None)

    if op_type is None or op_type not in SUPPORTED:
        raise NotImplementedError(
            f'"{op_type}" type convolution is not supported, "{list(SUPPORTED.keys())}" is currently supported'
            )
    
    return SUPPORTED[op_type](**cfg)