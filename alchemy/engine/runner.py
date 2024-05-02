# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import os

from loguru import logger

from alchemy.registry import RUNNERs, MODELs
from alchemy.utils import heading, subheading


__all__ = ['BaseRunner']


@RUNNERs.register_module(name='base')
class BaseRunner(object):
    def __init__(self, cfg, command='train'):
        heading('base runner')

        self.cfg = cfg
        self.command = command

        # initialize
        self._initialize()

        # models
        self._build_model()

        # datasets
        self._build_dataset()

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
        subheading('initialize directory')
        logger.info(f'workspace: {self.cfg.work_dir}')

        self.ckpt_dir = os.path.join(self.cfg.work_dir, 'checkpoints')
        os.makedirs(self.ckpt_dir, exist_ok=True)
        logger.info(f'checkpoints: {self.ckpt_dir}')

        self.tensorboard_dir = os.path.join(self.cfg.work_dir, 'tensorboard')
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        logger.info(f'tensorboard: {self.tensorboard_dir}')

    # ------ models ------
    def _build_model(self):
        subheading('build model')

    # ------ dataset ------
    def _build_dataset(self):
        subheading('build dataset')
    
    # ------ optimizer ------
    def _build_optimizer(self):
        subheading('build optimizer')
    
    # ------ scheduler ------
    def _build_scheduler(self):
        subheading('build scheduler')

    def _train_epoch(self, epoch):
        subheading(f'epoch {epoch}')

    def train(self):
        heading('start training')
        for epoch in range(self.start_epoch, self.start_epoch + self.cfg.epochs):
            self._train_epoch(epoch)


