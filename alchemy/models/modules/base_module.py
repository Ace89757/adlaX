# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import torch.nn as nn

from copy import deepcopy

from alchemy.models.modules.layers import convFunc, normalFucn, actFunc


def baseModule(in_channels=3, 
                out_channels=16, 
                kernel_size=3, 
                stride=1, 
                groups=1,
                dilation=1,
                padding=0,
                bias=False,
                conv_cfg=dict(type='conv'), 
                norm_cfg=dict(type='bn'), 
                act_cfg=dict(type='relu'),
                orders=('conv', 'norm', 'act')):
    # 判断bn是否在conv之前
    if 'conv' in orders and 'norm' in orders:
        if orders.index('norm') < orders.index('conv'):
            norm_before_conv = True
        else:
            norm_before_conv = False
    else:
        norm_before_conv = False

    modules = []

    for layer_type in orders:
        if layer_type == 'conv':
            cfg = deepcopy(conv_cfg)

            # check
            if kernel_size == 1:
                dilation = 1

            cfg.update(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
                groups=groups,
                dilation=dilation
            )

            modules.append(convFunc(cfg))
        
        elif layer_type == 'norm':
            if norm_before_conv:  # norm层在卷积之前
                norm_channels = in_channels
            else:
                norm_channels = out_channels

            cfg = deepcopy(norm_cfg)

            if cfg.get('type') == 'gn':
                cfg.update(num_channels=norm_channels)
            else:
                cfg.update(num_features=norm_channels)

            modules.append(normalFucn(cfg))
        
        elif layer_type == 'act':
            cfg = deepcopy(act_cfg)
            modules.append(actFunc(cfg))
        else:
            raise NotImplementedError(
                f'"{layer_type}" type layer is not supported, "[conv, norm, act, pool]" is currently supported'
                )
    
    if len(modules) > 1:
        return nn.Sequential(*modules)
    else:
        return modules[-1]