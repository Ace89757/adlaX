# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import os

from loguru import logger
from copy import deepcopy
from torch.utils.data import DataLoader
from mmengine.dataset import pseudo_collate

from alchemy.utils import printHeading, printSubheading, printArgs, printSubheading2
from alchemy.registry import RUNNERs, MODELs, DATASETs


__all__ = ['BaseRunner']


@RUNNERs.register_module(name='base')
class BaseRunner(object):
    printArgs('runner type', 'base')

    def __init__(self, cfg, command='train'):
        self.cfg = cfg
        self.start_epoch = 1
        self.command = command

        # initialize dir
        self.makeDirectory()

        # datasets
        self.train_dataloader, self.test_dataloader = self.buildDataloader()

        # models
        self.buildModel()

        # optimizer
        self.buildOptimizer()

        # scheduler
        self.buildScheduler()
        
    def makeDirectory(self):
        printSubheading('directory')
        
        self.ckpt_dir = os.path.join(self.cfg.work_dir, 'checkpoints')
        os.makedirs(self.ckpt_dir, exist_ok=True)
        
        self.tensorboard_dir = os.path.join(self.cfg.work_dir, 'tensorboard')
        os.makedirs(self.tensorboard_dir, exist_ok=True)

        printArgs('work dir', self.cfg.work_dir)
        printArgs('checkpoints', self.ckpt_dir)
        printArgs('tensorboard', self.tensorboard_dir)

    # ------ models ------
    def buildModel(self):
        printSubheading('model')

    # ------ dataset ------
    def initializeDataloader(self, dataloader_cfg):
        # dataset
        dataset = DATASETs.build(dataloader_cfg.pop('dataset'))

        # dataloader
        printSubheading2('dataloader')
        for k, v in dataloader_cfg.items():
            printArgs(k, v)

        dataloader = DataLoader(
            dataset,
            collate_fn=pseudo_collate,
           **dataloader_cfg
        )

        return dataloader

    def buildDataloader(self):
        printSubheading(f'train dataloader')
        train_dataloader = self.initializeDataloader(deepcopy(self.cfg.train_dataloader))

        printSubheading(f'test dataloader')
        test_dataloader = self.initializeDataloader(deepcopy(self.cfg.test_dataloader))

        return train_dataloader, test_dataloader

    # ------ optimizer ------
    def buildOptimizer(self):
        printSubheading('optimizer')
    
    # ------ scheduler ------
    def buildScheduler(self):
        printSubheading('scheduler')

    def trainStep(self, epoch):
        printSubheading(f'epoch {epoch}')
        # for data in self.train_dataloader:
        #     print(data['gt_bboxes'])
        #     exit()

    def train(self):
        printHeading('start training')
        for epoch in range(self.start_epoch, self.start_epoch + self.cfg.epochs):
            self.trainStep(epoch)


