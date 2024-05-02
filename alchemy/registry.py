# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

from mmengine.registry import Registry

MODELs = Registry('model', locations=['alchemy.models'])
RUNNERs = Registry('runner', locations=['alchemy.engine'])
DATASETs = Registry('model', locations=['alchemy.datasets'])