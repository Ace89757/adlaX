# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import os
import cv2
import numpy as np

from copy import deepcopy
from loguru import logger
from tabulate import tabulate
from torch.utils.data import Dataset

from alchemy.utils import printArgs, printSubheading2
from alchemy.datasets import Compose


class BaseDataset(Dataset):
    dataset_type = 'base'

    def __init__(self, 
                 root_dir, 
                 ann_file,
                 pipeline=None,
                 test_mode=False,
                 class_names=None,
                 data_prefix=dict(img='images'),
                 filter_cfg=dict(min_size=0, keep_empty_gt=False)):
        super().__init__()
        # params
        self.dataset = []
        self.root_dir = root_dir
        self.ann_file = ann_file
        self.filter_cfg = filter_cfg
        self.data_prefix = data_prefix
        self.class_names = class_names
        
        # loading
        self.loadingDataset()

        # filter
        self.filterDataset()

        # pipeline
        self.pipeline = Compose(pipeline)

        # other params
        self.test_mode = test_mode
        self.num_samples = len(self.dataset)
        self.fetch_sample = self.fetchTestSample if self. test_mode else self.fetchTrainSample

        # instances
        self.showDatasetInfo()
    
    def filterDataset(self):
        printSubheading2('filter config')
        valid_samples = []

        min_size = self.filter_cfg.get('min_size', 0)
        keep_empty_gt = self.filter_cfg.get('keep_empty_gt', False)

        printArgs('min size', min_size)
        printArgs('keep empty gt', keep_empty_gt)

        for sample in self.dataset:
            gt_bboxes = sample['gt_bboxes']
            gt_labels = sample['gt_labels']

            new_gt_bboxes = []
            new_gt_labels = []

            for gt_box, gt_label in zip(gt_bboxes, gt_labels):
                w = gt_box[2] - gt_box[0]
                h = gt_box[3] - gt_box[1]

                if w < min_size or h < min_size:
                    continue

                new_gt_bboxes.append(gt_box)
                new_gt_labels.append(gt_label)
            
            if len(new_gt_bboxes) != 0 or keep_empty_gt:
                sample['gt_bboxes'] = np.array(new_gt_bboxes)
                sample['gt_labels'] = np.array(new_gt_labels)

                valid_samples.append(sample)

        self.dataset = valid_samples

        logger.info(f'there are {len(self.dataset)} valid samples left after filtering')
        
    def loadingDataset(self):
        raise NotImplementedError

    def showDatasetInfo(self):
        """
        统计每个类别实例的数量
        """
        printSubheading2('dataset info')
        category_counts = {cls_name: 0 for cls_name in self.class_names}
        for sample in self.dataset:
            for label in sample['gt_labels']:
                cls_name = self.class_names[int(label)]

                category_counts[cls_name] += 1
        
        table = [['category', 'id', 'number']]
        for name, counts in category_counts.items():
            table.append([name, self.class_names.index(name), counts])

        for line in tabulate(table, headers='firstrow', tablefmt='fancy_grid').split('\n'):
            logger.info(line)

    def fetchSample(self, index):
        # get sample
        sample = self.dataset[index]
        
        # load img
        sample['img'] = cv2.imread(sample['img_path'])
    
        return sample

    def fetchTrainSample(self, index):
        sample = None

        while True:
            sample = self.fetchSample(index)

            # preprocess
            sample = self.pipeline(sample)

            if sample is None:
                index = np.random.randint(low=0, high=self.num_samples)
                continue

            break
        
        return sample

    def fetchTestSample(self, index):
        sample = self.fetchSample(index)
        return self.pipeline(sample)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.fetch_sample(index)
