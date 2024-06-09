# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import torch.nn as nn
import torch.nn.functional as F

from alchemy.registry import MODELs
from alchemy.models.modules import baseModule


__all__ = ['FPN']


@MODELs.register_module('fpn')
class FPN(nn.Module):
    def __init__(self, 
                 in_channels,
                 end_level=-1,
                 out_channels=256,
                 out_indices=(0, 1, 2, 3),
                 norm_cfg=dict(type='bn'),
                 act_cfg=dict(type='relu'),
                 conv_cfg=dict(type='conv'),
                 no_norm_on_lateral=False,
                 act_before_extra_conv=False,
                 extra_layers_source='on_input',
                 upsample_cfg=dict(type='nearest')):
        super().__init__()
        self.num_ins = len(in_channels)
        self.num_outs = len(out_indices)

        """
        大致分3中情况:
        1. num_in = num_out:
            正常操作
        2. num_in > num_out:
            不需要额外层, 根据start_level和end_level构建lateral层, 根据out_indices构建fpn层
        3. num_in < num_out
            需要额外层
        """
        self.in_channels = in_channels
        self.out_indices = out_indices
        self.out_channels = out_channels

        self.act_cfg = act_cfg
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        
        self.upsample_cfg = upsample_cfg.copy()
        self.no_norm_on_lateral = no_norm_on_lateral
        self.act_before_extra_conv = act_before_extra_conv

        # 实际使用的backbone的起始层
        self.start_level = out_indices[0]

        # 使用使用backbone的终止层
        if end_level == -1 or end_level == self.num_ins - 1:   # 最后一层
            self.backbone_end_level = self.num_ins
        else:
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins

        # 构建lateral层
        self.lateral_layers = self._lateralLayers()

        # 构建fpn层
        self.fpn_layers = self._fpnLayers()

        # 判断是否需要额外的层
        self.add_extra_layers = False
        self.num_extra_levels = 0      # 需要额外层的数量
        for ind in out_indices:
            if ind + 1 > self.num_ins:
                self.add_extra_layers = True   # 如果输出的层索引大于输入的层数, 就需要额外层
                self.num_extra_levels += 1
        
        # 额外层的输入使用哪个阶段
        assert extra_layers_source in ('on_input', 'on_lateral', 'on_output')
        self.extra_layers_source = extra_layers_source

        # 构建额外层
        self._extraLayers()

        # 构建上采样层
        self.upsample_layers = self._unsampleLayers()

    def _lateralLayers(self):
        lateral_layers = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            lateral_layers.append(
                baseModule(
                    self.in_channels[i],
                    self.out_channels,
                    kernel_size=1,
                    stride=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    orders=('conv', 'norm', 'act') if not self.no_norm_on_lateral else ('conv', 'act')
                    )
            )
        
        return lateral_layers
    
    def _fpnLayers(self):
        fpn_layers = nn.ModuleList()

        for out_ind in self.out_indices:
            if out_ind < self.num_ins:
                fpn_layers.append(
                    baseModule(
                        self.out_channels,
                        self.out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg
                    )
                    )
        
        return fpn_layers
    
    def _extraLayers(self):
        if self.add_extra_layers and self.num_extra_levels > 0:
            for i in range(self.num_extra_levels):
                if i == 0 and self.extra_layers_source == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]   # 使用backbone的end_level作为第一个额外层的输入
                else:
                    in_channels = self.out_channels
                
                extra_fpn_conv = baseModule(
                    in_channels,
                    self.out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)
                
                self.fpn_layers.append(extra_fpn_conv)

    def _unsampleLayers(self):
        upsample_layers = nn.ModuleList()

        upsample_type = self.upsample_cfg.get('type', 'nearest')

        for _ in range(len(self.lateral_layers) - 1):
            if upsample_type == 'deconv':
                upsample = nn.ConvTranspose2d(
                    in_channels=self.out_channels,
                    out_channels=self.out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1
                )
            elif upsample_type == 'nearest':
                upsample = nn.Upsample(scale_factor=2, mode='nearest')
            elif upsample_type == 'bilinear':
                upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            else:
                raise NotImplemented('Currently, upsample supports only deconv, nearest, and bilinear modes')

            upsample_layers.append(upsample)
        
        return upsample_layers

    def forward(self, feats) -> tuple:
        """
        Forward function.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is a 4D-tensor.

        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        assert len(feats) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(feats[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_layers)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for idx, i in enumerate(range(used_backbone_levels - 1, 0, -1)):
            laterals[i - 1] = laterals[i - 1] + self.upsample_layers[idx](laterals[i])
        
        # build outputs
        # part 1: from original levels
        outs = []
        fpn_ind = 0
        for idx, out_ind in enumerate(self.out_indices):
            if out_ind < self.num_ins:
                fpn_ind = idx
                outs.append(self.fpn_layers[fpn_ind](laterals[out_ind - self.start_level]))
                
        # part 2: add extra levels
        if self.add_extra_layers:
            if self.extra_layers_source == 'on_input':
                extra_source = feats[self.backbone_end_level - 1]
            elif self.extra_layers_source == 'on_lateral':
                extra_source = laterals[-1]
            elif self.extra_layers_source == 'on_output':
                extra_source = outs[-1]
            else:
                raise NotImplementedError

            fpn_ind = len(outs)
            outs.append(self.fpn_layers[fpn_ind](extra_source))
            
            for _ in self.out_indices[len(outs):]:
                fpn_ind += 1
                if self.act_before_extra_conv:
                    outs.append(self.fpn_layers[fpn_ind](F.relu(outs[-1])))
                else:
                    outs.append(self.fpn_layers[fpn_ind](outs[-1]))

        return tuple(outs)