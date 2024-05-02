# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

from mmengine.registry import Registry

RUNNERs = Registry('runner', locations=['alchemy.engine'])
MODELs = Registry('model', locations=['alchemy.models'])