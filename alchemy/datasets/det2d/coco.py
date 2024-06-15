# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import os
import cv2
import numpy as np

from copy import deepcopy
from loguru import logger
from tabulate import tabulate
from pycocotools.coco import COCO

from alchemy.utils import printArgs, printSubheading2
from alchemy.datasets import Compose
from alchemy.registry import DATASETs
from alchemy.datasets.dataset import BaseDataset


__all__ = ['CocoDetectionDataset']


def removeUseless(coco):
    """
    Remove useless info in coco dataset. COCO object is modified inplace.
    This function is mainly used for saving memory (save about 30% mem).
    """
    if isinstance(coco, COCO):
        dataset = coco.dataset
        dataset.pop("info", None)
        dataset.pop("licenses", None)
        for img in dataset["images"]:
            img.pop("license", None)
            img.pop("coco_url", None)
            img.pop("date_captured", None)
            img.pop("flickr_url", None)

        if "annotations" in coco.dataset:
            for anno in coco.dataset["annotations"]:
                anno.pop("segmentation", None)


@DATASETs.register_module('coco')
class CocoDetectionDataset(BaseDataset):
    dataset_type = 'coco'
    
    def loadingDataset(self):
        printSubheading2('dataset')
        printArgs(key_string='dataset type', value_string=self.dataset_type)

        # check
        img_dir = os.path.join(self.root_dir, self.data_prefix['img'])
        # assert os.path.exists(img_dir), FileExistsError(f'{img_dir} is not exists!')
        assert os.path.exists(self.ann_file), FileExistsError(f'{self.ann_file} is not exists!')

        # load coco
        coco = COCO(self.ann_file)
        removeUseless(coco)

        # annotation中的类别映射关系
        ann_cls_mapping = {}
        for cls_info in coco.dataset['categories']:
            ann_cls_mapping[cls_info['id']] = cls_info['name']

        if self.class_names is None:
            # 按cls-id顺序排列
            self.class_names = []
            for cls_id in range(len(ann_cls_mapping)):
                self.class_names.append(ann_cls_mapping[cls_id + 1])

        img_ids = coco.getImgIds()

        for img_id in img_ids:
            img_info = coco.loadImgs(ids=[img_id])[0]

            img_path = os.path.join(img_dir, img_info['file_name'])
            # if not os.path.exists(img_path):
            #     continue

            ann_ids = coco.getAnnIds(imgIds=[int(img_id)], iscrowd=False)
            anns = coco.loadAnns(ann_ids)

            gt_bboxes = []
            gt_labels = []

            for ann in anns:
                # filter category
                category_id = int(ann['category_id'])
                category = ann_cls_mapping[category_id]
                if category not in self.class_names:
                    continue
                
                # bbox
                x1, y1, w, h = [float(x) for x in ann['bbox']]
                gt_bboxes.append([x1, y1, x1 + w, y1 + h])

                # label
                label = self.class_names.index(category)
                gt_labels.append(label)

            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_bboxes[:, [0, 2]] = np.clip(gt_bboxes[:, [0, 2]], a_min=0, a_max=img_info['width'])
            gt_bboxes[:, [1, 3]] = np.clip(gt_bboxes[:, [1, 3]], a_min=0, a_max=img_info['height'])

            sample = deepcopy(img_info)
            sample.update(
                img_path=img_path,
                gt_bboxes=gt_bboxes,
                gt_labels=np.array(gt_labels, dtype=np.int32),
                )

            self.dataset.append(sample)

        logger.info(f'found {len(img_ids)} samplesf from "{self.ann_file}"')
        

if __name__ == '__main__':
    dataset = CocoDetectionDataset(
        root='/Users/ace/workspace/learning/dl/datasets/bdd100k',
        ann_file='/Users/ace/workspace/learning/dl/datasets/bdd100k/annotations/bdd100k_coco_10cls_val_10000_20240502.json'
    )