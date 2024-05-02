# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import torch.nn as nn


__all__ = ['normalization']


SUPPORTED = {
    'bn': nn.BatchNorm2d,
    'gn': nn.GroupNorm,
}


def normalization(cfg):
    op_type = cfg.pop('type', None)

    if op_type is None or op_type not in SUPPORTED:
        raise NotImplementedError(
            f'"{op_type}" type convolution is not supported, "{list(SUPPORTED.keys())}" is currently supported'
            )
    
    return SUPPORTED[op_type](**cfg)