# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import os

from loguru import logger
from copy import deepcopy
from torch.utils.data import DataLoader
from mmengine.dataset import pseudo_collate

from alchemy.utils import heading, subheading, keyinfo
from alchemy.registry import RUNNERs, MODELs, DATASETs


__all__ = ['BaseRunner']


@RUNNERs.register_module(name='base')
class BaseRunner(object):
    def __init__(self, cfg, command='train'):
        heading('initialize base runner')

        self.cfg = cfg
        self.command = command

        # initialize
        self._initialize()

        # models
        self._build_model()

        # datasets
        self.train_dataloader, self.test_dataloader = self._build_dataset()

        # optimizer
        self._build_optimizer()

        # scheduler
        self._build_scheduler()
        
    def _initialize(self):
        # directory
        self._init_work_dirs()

        # args
        self.start_epoch = 1
        
    def _init_work_dirs(self):
        subheading('make directory')
        keyinfo('workspace', self.cfg.work_dir)

        self.ckpt_dir = os.path.join(self.cfg.work_dir, 'checkpoints')
        os.makedirs(self.ckpt_dir, exist_ok=True)
        keyinfo('checkpoints', self.ckpt_dir)

        self.tensorboard_dir = os.path.join(self.cfg.work_dir, 'tensorboard')
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        keyinfo('tensorboard', self.tensorboard_dir)

    # ------ models ------
    def _build_model(self):
        subheading('build model')

    # ------ dataset ------
    def _build_dataloader(self, dataloader_cfg):
        dataset = DATASETs.build(dataloader_cfg.pop('dataset'))
        dataloader = DataLoader(
            dataset,
            collate_fn=pseudo_collate,
           **dataloader_cfg
        )

        for k, v in dataloader_cfg.items():
            keyinfo(k, v)

        return dataloader

    def _build_dataset(self):
        subheading(f'build train dataloader')
        train_dataloader = self._build_dataloader(deepcopy(self.cfg.train_dataloader))

        subheading(f'build test dataloader')
        test_dataloader = self._build_dataloader(deepcopy(self.cfg.test_dataloader))

        return train_dataloader, test_dataloader

    # ------ optimizer ------
    def _build_optimizer(self):
        subheading('build optimizer')
    
    # ------ scheduler ------
    def _build_scheduler(self):
        subheading('build scheduler')

    def _train_epoch(self, epoch):
        subheading(f'epoch {epoch}')
        # for data in self.train_dataloader:
        #     print(data['gt_bboxes'])
        #     exit()

    def train(self):
        heading('start training')
        for epoch in range(self.start_epoch, self.start_epoch + self.cfg.epochs):
            self._train_epoch(epoch)


