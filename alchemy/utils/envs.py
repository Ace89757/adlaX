# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import os

from loguru import logger
from mmengine.utils.dl_utils import collect_env

from .logger import heading


__all__ = [
    'get_env_info'
]


def get_env_info(work_dir=None):
    """
    获取运行环境相关信息，并保存相关环境信息
    """
    heading('running environments')

    # 通过mmengine获取环境信息
    env_info = collect_env()

    for env_name in ['OpenCV', 'PyTorch', 'MMEngine', 'TorchVision', 'Python']:
        logger.info(f'{env_name}: {env_info[env_name]}')
    
    # 保存
    if work_dir is not None:
        with open(os.path.join(work_dir, 'environments.txt'), 'w') as env_file:
            for env_name, env_value in env_info.items():
                env_file.write(f'{env_name}: {env_value}\n')