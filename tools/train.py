# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import os
import time
import shutil
import argparse
import datetime

from loguru import logger
from mmengine.config import Config, DictAction

from alchemy.registry import RUNNERs
from alchemy.utils import setup_logger, get_env_info


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path.')
    parser.add_argument('--task',
                        type=str,
                        default='det2d',
                        choices=['det2d'],
                        help='model task, currently supports "det2d".')
    parser.add_argument('--work-dir',
                        type=str,
                        default=None,
                        help='the dir to save logs and models.')
    parser.add_argument('--exp-name',
                        type=str,
                        default='baseline',
                        help='experiment name.')
    parser.add_argument('--resume',
                        action='store_true',
                        help='resume from checkpoint.')
    parser.add_argument('--pretrained',
                        type=str,
                        default=None,
                        help='pretrained model.')
    parser.add_argument('--auto-scale-lr',
                        action='store_true',
                        help='enable automatically scaling LR.')
    parser.add_argument('--cfg-options',
                        nargs='+',
                        action=DictAction,
                        help='override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file. '
                             'If the value to be overwritten is a list, it should be like key="[a,b]" or key=a,b '
                             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
                             'Note that the quotation marks are necessary and that no white space is allowed.'
    )
    args = parser.parse_args()

    return args


@logger.catch()
def main():
    # start time
    st = time.perf_counter()

    # date
    date = datetime.datetime.now().strftime("%Y%m%d%H%M")

    # the args from terminal
    args = parse_args()

    # parse config file
    cfg = Config.fromfile(args.config)

    # merge the terminal args to config
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # resume train
    cfg.resume = args.resume

    # work-dir & checkpoints
    if cfg.resume:
        assert args.work_dir is not None and os.path.exists(args.work_dir)

        cfg.work_dir = os.path.join(args.work_dir, 'resume')
        cfg.load_from = os.path.join(args.work_dir, 'checkpoints', 'last_epoch.pth')
        assert os.path.exists(cfg.load_from)

    else:
        root_dir = './work_dirs'

        if args.work_dir is not None:
            root_dir = args.work_dir

        cfg.work_dir = os.path.join(root_dir, 'train', args.task, args.exp_name, date)
        cfg.load_from = args.pretrained

    cfg.task_prefix = f'{args.task}_{args.exp_name}'

    # logger
    setup_logger(cfg.work_dir, filename=f'log.train.{date[:-4]}{".resume"if cfg.resume else ""}.log')

    # enable scaling lr
    if args.auto_scale_lr:
        if ('auto_scale_lr' in cfg) and ('enable' in cfg.auto_scale_lr) and ('base_batch_size' in cfg.auto_scale_lr):
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError(
                'Can not find "auto_scale_lr" or "auto_scale_lr.enable" or "auto_scale_lr.base_batch_size" in your configuration file.'
            )

    # backup
    if args.resume:
        shutil.copy(cfg.load_from, os.path.join(cfg.work_dir, f'resume.{date}.pth'))

    cfg.dump(
        os.path.join(
            cfg.work_dir,
            f'{os.path.splitext(os.path.basename(args.config))[0]}{".resume"if cfg.resume else ""}.py'
            )
        )

    # environments
    get_env_info(cfg.work_dir)

    # runner
    runner_cfg = cfg.pop('runner_cfg', dict(type='base'))
    runner_cfg.update(cfg=cfg, command='train')
    runner = RUNNERs.build(runner_cfg)

    # train
    runner.train()

    # finish
    timedelta_obj = datetime.timedelta(seconds=time.perf_counter() - st)
    hours, remainder = divmod(timedelta_obj.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.success(
        f'The model training took {int(hours)}h, {int(minutes)}m, {int(seconds)}s.'
    )


if __name__ == '__main__':
    main()
