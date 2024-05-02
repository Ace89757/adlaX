# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import torch.nn as nn

from alchemy.registry import MODELs
from alchemy.models.modules import base_module


__all__ = ['ResNet']


class BasicBlock(nn.Module):
    expansion = 1
    """
    流程图:
            input ------->----------
           (b, 256)                |
              |                    |
     conv3x3 + norm + act          |
           (b, 256)                |
              |                    |   residual  (conv1x1 + norm)
        conv3x3 + norm             |   (b, 256)
           (b, 256)                |
              |                    |
             add <-------<----------
           (b, 256)
              |
             act
           (b, 256)
              |
            output
           (b, 256)
    """
    def __init__(self,
                 inplanes, 
                 planels, 
                 stride=1,
                 groups=1,
                 dilation=1,
                 base_planes=64,
                 downsample=None,
                 conv_cfg=dict(type='conv'), 
                 norm_cfg=dict(type='bn'), 
                 act_cfg=dict(type='relu', inplace=True)):
        super().__init__()

        if groups != 1 or base_planes != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_planes=64')
        
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')

        # conv3 + norm + act
        self.layer1 = base_module(
            inplanes, 
            planels, 
            kernel_size=3, 
            stride=stride, 
            padding=1,
            groups=1,
            dilation=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            orders=('conv', 'norm', 'act'))
        
        # conv3 + norm
        self.layer2 = base_module(
            planels,
            planels, 
            kernel_size=3, 
            stride=1, 
            padding=1,
            groups=1,
            dilation=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            orders=('conv', 'norm'))
        
        self.act = base_module(act_cfg=act_cfg, orders=('act', ))

        # down sample
        if downsample is None:
            self.downsample = nn.Identity()
        else:
            self.downsample = downsample

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)

        out = out + self.downsample(x)

        out = self.act(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    """
    流程图:
            input ------->----------
           (b, 256)                |
              |                    |
     conv1x1 + norm + act          |
           (b, 64)                 |
              |                    |   residual  (conv1x1 + norm)
     conv3x3 + norm + act          |   (b, 256)
           (b, 64)                 |
              |                    |
        conv1x1 + norm             |
           (b, 256)                |
             add <-------<----------
           (b, 256)
              |
             act
           (b, 256)
              |
            output
           (b, 256)
    """

    def __init__(self,
                 inplanes,
                 planes,
                 stride = 1,
                 groups = 1,
                 dilation = 1,
                 base_planes=64,
                 downsample=None,
                 conv_cfg=dict(type='conv'),
                 norm_cfg=dict(type='bn'),
                 act_cfg=dict(type='relu', inplace=True)):
        super().__init__()

        hidden_planes = int(planes * (base_planes / 64.0)) * groups

        # conv1x1 + norm + act
        self.layer1 = base_module(
            inplanes,
            hidden_planes,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            orders=('conv', 'norm', 'act')
        )

        # conv3x3 + norm + act
        self.layer2 = base_module(
            hidden_planes,
            hidden_planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            groups=groups,
            dilation=dilation,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            orders=('conv', 'norm', 'act')
        )

        # conv1x1 + norm
        self.layer3 = base_module(
            hidden_planes,
            int(planes * self.expansion),
            kernel_size=1, 
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            orders=('conv', 'norm')
        )

        self.act = base_module(act_cfg=act_cfg, orders=('act', ))

        # down sample
        if downsample is None:
            self.downsample = nn.Identity()
        else:
            self.downsample = downsample

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        out = out + self.downsample(x)
        out = self.act(out)

        return out


@MODELs.register_module('resnet')
class ResNet(nn.Module):
    configure = {
        # depth: (block, num_stacks)
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }
    def __init__(self, 
                 depth=18,
                 out_indices=(2, 3, 4, 5),
                 conv_cfg=dict(type='conv'), 
                 norm_cfg=dict(type='bn'), 
                 act_cfg=dict(type='relu')):
        """
        depth (int): 网络深度
        out_indices (Tuple[int]): 网络输出特征层的索引
        conv_cfg (dict): 卷积层相关配置
        norm_cfg (dict): 归一化层相关配置
        act_cfg (dict): 激活层相关配置
        """
        super().__init__()
        # initialize
        self.in_channels = 64
        self.out_channels = []
        self.out_indices = out_indices

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        stage_strides = (1, 2, 2, 2)
        stage_channels = (64, 128, 256, 512)

        # 根据网络深度获取对应的配置
        self.stage_block, self.num_stage_stacks = self.configure[depth]

        # stage0: 2x
        self.c0 = base_module(
            in_channels=3,
            out_channels=self.in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
        
        if 0 in self.out_indices:
            self.out_channels.append(self.in_channels)

        # stage1: 4x
        self.c1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if 1 in self.out_indices:
            self.out_channels.append(self.in_channels)

        # stage2~5: 4x, 8x, 16x, 32x
        for idx in range(4):
            stage_id = idx + 2
            stage = self._make_stages(stage_channels[idx], self.num_stage_stacks[idx], stage_strides[idx])
            self.__setattr__(f'c{stage_id}', stage)

            if stage_id in out_indices:
                self.out_channels.append(self.in_channels)

        # initial weight
        self.init_weight()

    def _make_stages(self, planes, num_stack, stride=1):
        # expansion_planes
        expansion_planes = int(planes * self.stage_block.expansion)

        # downsample
        downsample = None
        if stride != 1 or self.in_channels != expansion_planes:
            downsample = base_module(
                self.in_channels, 
                expansion_planes, 
                kernel_size=1, 
                stride=stride, 
                padding=0,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                orders=('conv', 'norm')
            )

        stage = [
            self.stage_block(
                self.in_channels,
                planes,
                stride=stride,
                downsample=downsample,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            )
        ]

        self.in_channels = expansion_planes

        for _ in range(1, num_stack):
            stage.append(
                self.stage_block(
                    self.in_channels,
                    planes,
                    stride=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
            )

        return nn.Sequential(*stage)
    
    def init_weight(self):
        """
        默认使用Kaiming初始化, 也称为He初始化, 是一种用于初始化深度神经网络权重的方法,特别是针对ReLU激活函数设计的.
        Kaiming初始化的核心思想是让输入和输出的方差保持一致, 从而使得在训练开始时, 各层的激活函数的分布相对稳定.
        Kaiming初始化有两种变体, 分别是fan_in(针对输入)和fan_out(针对输出)，它们基于神经网络中每个神经元的输入和输出的连接数（即“扇出”和“扇入”）来调整权重的初始尺度.
        在实际应用中, 选择fan_in还是fan_out取决于具体的网络结构和激活函数。
        对于大多数卷积神经网络CNN, 通常使用fan_in, 因为卷积核的每个权重是共享的, 所以更关注于输入特征图的尺寸。
        而对于全连接网络, 如果使用了ReLU等激活函数, 通常也会使用fan_in, 因为ReLU在正区间内是线性的, 而fan_in正是为了保持输入的方差.

        """
        # activate type
        act_type = self.act_cfg.get('type', None)
        if act_type not in ['relu', 'leaky_relu']:
            act_type = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if act_type in ['relu', 'leaky_relu']:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity=act_type)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=act_type)

            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        outs = []
        for stage_id in range(6):
            x = self.__getattr__(f'c{stage_id}')(x)

            if stage_id in self.out_indices:
                outs.append(x)
        
        return tuple(outs)