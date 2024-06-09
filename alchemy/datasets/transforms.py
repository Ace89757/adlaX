# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import cv2
import math
import torch
import numpy as np

from loguru import logger

from alchemy.utils import printArgs, printSubheading2
from alchemy.registry import TRANSFORMs


__all__ = ['Compose', 'LetterBox', 'Normalization']


class Compose:
    """
    Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict, callable], optional): Sequence of transform object or config dict to be composed.
    """

    def __init__(self, transforms):
        printSubheading2('pipeline')
        self.transforms = []

        if transforms is None:
            transforms = []

        for transform in transforms:
            # Compose can be built with config dict with type and corresponding arguments.
            if isinstance(transform, dict):
                transform = TRANSFORMs.build(transform)
                if not callable(transform):
                    raise TypeError(f'transform should be a callable object, but got {type(transform)}')
                self.transforms.append(transform)

            elif callable(transform):
                self.transforms.append(transform)

            else:
                raise TypeError(f'transform must be a callable object or dict, but got {type(transform)}')
            logger.info(transform)

    def __call__(self, sample):
        """
        Call function to apply transforms sequentially.

        Args:
            sample (dict): A result dict contains the sample to transform.

        Returns:
           dict: Transformed sample.
        """
        for trans in self.transforms:
            sample = trans(sample)
            if sample is None:
                return None
        return sample

    def __repr__(self):
        """Print ``self.transforms`` in sequence.

        Returns:
            str: Formatted string.
        """
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string
    

@TRANSFORMs.register_module('letterbox')
class LetterBox:
    def __init__(self, keep_aspect=True, random_interp=False, input_size=(512, 512)):
        self.keep_aspect = keep_aspect
        self.random_interp = random_interp
        
        if isinstance(input_size, int):
            self.input_size = [input_size, input_size]
        else:
            self.input_size = input_size

        self.gt_keys = ('gt_bboxes', )
    
    @staticmethod
    def _randomInterpMethod():
        """
        Randomly select an interp method from given candidates.

        Returns: interp methods
        """
        cv2_interp_codes = {
            'area': cv2.INTER_AREA,
            'bicubic': cv2.INTER_CUBIC,
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'lanczos': cv2.INTER_LANCZOS4
            }
        return cv2_interp_codes[np.random.choice(list(cv2_interp_codes.keys()))]
    
    def __call__(self, sample):
        # resize img
        if sample.get('img', None) is not None:
            raw_img_h, raw_img_w = sample['img'].shape[:2]

            if self.keep_aspect:
                scale_h = scale_w = min(self.input_size[0] / raw_img_h, self.input_size[1] / raw_img_w)
            else:
                scale_h = self.input_size[0] / raw_img_h
                scale_w = self.input_size[1] / raw_img_w

            if scale_h != 1.0 or scale_w != 1.0:
                img_resized = cv2.resize(
                    sample['img'], 
                    (math.ceil(raw_img_w * scale_w), math.ceil(raw_img_h * scale_h)), 
                    interpolation=self._randomInterpMethod() if self.random_interp else cv2.INTER_LINEAR)
            
                sample['img'] = img_resized

            sample['img_shape'] = sample['img'].shape[:2]
            sample['scale_factor'] = (scale_w, scale_h)
        else:
            return None

        # resize instances
        scale_w, scale_h = sample['scale_factor']

        for key in self.gt_keys:
            if sample.get(key, None) is not None:
                sample[key][:, 0::2] *= scale_w
                sample[key][:, 1::2] *= scale_h
        
        return sample
    
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(input_size={self.input_size}, '
        repr_str += f'keep_aspect={self.keep_aspect}, '
        repr_str += f'random_interp={self.random_interp})'

        return repr_str
    
    def _vis(self, sample):
        import os
        crop_img = sample['img'].copy().astype(np.uint8)
        bboxes = sample['gt_bboxes']

        for bbox in bboxes:
            x1, y1, x2, y2 = [int(x) for x in bbox.reshape(-1)]
            cv2.rectangle(crop_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        cv2.imwrite(os.path.join('/raid/xinjin_ai/ace/models/Ace-AlchemyFurnace/alchemy/transforms', f'{img_id}_resize.jpg'), crop_img)
        exit()


@TRANSFORMs.register_module('normalize')
class Normalization:
    def __init__(self, mean=[0, 0, 0], std=[1, 1, 1], normalize=True) -> None:
        self.normalize = normalize
        self.mean = np.array(mean, dtype=np.float32).reshape([1, len(mean), 1, 1])
        self.std = np.array(std, dtype=np.float32).reshape([1, len(std), 1, 1])

    def __call__(self, sample):
        if sample.get('img', None) is not None:
            # img
            img = (sample['img'].transpose((2, 0, 1))[np.newaxis, :, :, :]).astype(np.float32)

            if self.normalize:
                img /= 255.

            img = (img - self.mean) / self.std

            # gt
            for k, v in sample.items():
                if 'gt' in k:
                    sample[k] = torch.from_numpy(v)

            sample['img'] = torch.from_numpy(img)
            return sample

        else:
            return None
    
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean.reshape(-1)}, '
        repr_str += f'std={self.std.reshape(-1)}, '
        repr_str += f'normalize={self.normalize})'

        return repr_str