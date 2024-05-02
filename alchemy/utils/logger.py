# -*- coding: utf-8 -*-
# @Author  : ace
# Copyright by Ace, All Rights Reserved.

import os
import sys
import inspect

from loguru import logger


__all__ = [
    'setup_logger',
    'heading', 'subheading'
    ]


class StreamToLoguru:
    """
    stream object that redirects writes to a logger instance.
    """

    def __init__(self, level='INFO', caller_names=('apex', 'pycocotools')):
        """
        Args:
            level(str): log level string of loguru. Default value: 'INFO'.
            caller_names(tuple): caller names of redirected module. Default value: (apex, pycocotools).
        """
        self.level = level
        self.linebuf = ""
        self.caller_names = caller_names
    
    @staticmethod
    def get_caller_name(depth=0):
        """
        Args:
            depth (int): Depth of caller conext, use 0 for caller depth. Default value: 0.

        Returns:
            str: module name of the caller
        """
        # the following logic is a little bit faster than inspect.stack() logic
        frame = inspect.currentframe().f_back
        for _ in range(depth):
            frame = frame.f_back

        return frame.f_globals['__name__']

    def write(self, buf):
        full_name = self.get_caller_name(depth=1)
        module_name = full_name.rsplit('.', maxsplit=-1)[0]

        if module_name in self.caller_names:
            for line in buf.rstrip().splitlines():
                # use caller level log
                logger.opt(depth=2).log(self.level, line.rstrip())
        else:
            sys.__stdout__.write(buf)

    def flush(self):
        pass


def setup_logger(work_dir, filename='log.txt'):
    """
    配置程序运行中产生的任何信息的输出格式，输出到终端，以及写入日志文件
    Args:
        work_dir (str): 工作路径
        filename (str): 日志文件名称
    """
    # 创建工作目录
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    
    # 删除loguru默认处理程序的配置, 其id=0
    logger.remove(0)

    # 日志文件
    log_file = os.path.join(work_dir, filename)
    if os.path.exists(log_file):
        os.remove(log_file)

    # ---------------- logger.add 参数说明 ----------------
    # sink: 为记录器生成的每条记录指定目的地, 默认情况下，它设置为 sys.stderr输出到终端
    # level: 指定记录器的最低日志级别
    # filter: 用于确定一条记录是否应该被记录
    # ----------------------------------------------------

    # 设置info-level日志处理程序 (INFO (20): 用于记录描述程序正常操作的信息消息)
    info_format = (
        '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | '
        '<white>{level:>10}</white> | '
        '<white>{message}</white>'
        )
    logger.add(
        sink=sys.stderr,
        level="INFO",
        filter=lambda record: record["level"].name == "INFO", format=info_format)
    logger.add(
        sink=log_file,
        level="INFO",
        filter=lambda record: record["level"].name == "INFO", format=info_format)
    
    # 设置warning-level日志处理程序 (WARNING (30): 警告类型，用于指示可能需要进一步调查的不寻常事件)
    warning_format = (
        '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | '
        '<yellow><level>{level:>10}</level></yellow> | '
        '<yellow><level>{module}</level></yellow>:<yellow><level>{line}</level></yellow> - <yellow><level>{message}</level></yellow>'
        )
    logger.add(
        sink=sys.stderr,
        level="WARNING",
        filter=lambda record: record["level"].name == "WARNING", format=warning_format)
    logger.add(
        sink=log_file,
        level="WARNING",
        filter=lambda record: record["level"].name == "WARNING", format=warning_format)
    
    # 设置error-level日志处理程序 (ERROR (40): 错误类型，用于记录影响特定操作的错误条件) (格式里面添加了process和thread记录，方便查看多进程和线程程序)
    error_format = (
        '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | '
        '<red><level>{level:>10}</level></red> | '
        '<magenta>{process}</magenta>:<yellow>{thread}</yellow> | '
        '<red><level>{module}</level></red>:<red><level>{line}</level></red> - <red><level>{message}</level></red>'
        )
    logger.add(
        sink=sys.stderr,
        level="ERROR",
        filter=lambda record: record["level"].name == "ERROR", format=error_format)
    logger.add(
        sink=log_file,
        level="ERROR",
        filter=lambda record: record["level"].name == "ERROR", format=error_format)
    
    # 设置debug-level日志处理程序 (DEBUG (10): 开发人员使用该工具记录调试信息)
    debug_format = (
        '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | '
        '<blue>{level:>10}</blue> | '
        '<blue>{module}</blue>:<blue>{line}</blue> - <blue>{message}</blue>'
        )
    logger.add(
        sink=sys.stderr,
        level="DEBUG",
        filter=lambda record: record["level"].name == "DEBUG", format=debug_format)
    logger.add(
        sink=log_file,
        level="DEBUG",
        filter=lambda record: record["level"].name == "DEBUG", format=debug_format)
    
    # 设置success-level日志处理程序 (SUCCESS (25): 类似于INFO，用于指示操作成功的情况。)
    success_format = (
        '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | '
        '<green>{level:>10}</green> | '
        '<green>{message}</green>'
        )
    logger.add(
        sink=sys.stderr,
        level="SUCCESS",
        filter=lambda record: record["level"].name == "SUCCESS", format=success_format)
    logger.add(
        sink=log_file,
        level="SUCCESS",
        filter=lambda record: record["level"].name == "SUCCESS", format=success_format)

    # 设置trace-level日志处理程序 (TRACE (5): 用于记录程序执行路径的细节信息，以进行诊断) (格式里面添加了process和thread记录，方便查看多进程和线程程序)
    trace_format = (
        '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | '
        '<magenta><level>{level:>10}</level></magenta> | '
        '<magenta>{process}</magenta>:<magenta>{thread}</magenta> | '
        '<magenta><level>{module}</level></magenta>:<magenta><level>{line}</level></magenta> - <magenta><level>{message}</level></magenta>'
        )
    logger.add(
        sink=sys.stderr,
        level="TRACE",
        filter=lambda record: record["level"].name == "TRACE", format=trace_format)
    logger.add(
        sink=log_file,
        level="TRACE",
        filter=lambda record: record["level"].name == "TRACE", format=trace_format)
    
    # 设置critical-level日志处理程序 (CRITICAL (50): 严重类型，用于记录阻止核心功能正常工作的错误条件) (格式里面添加了process和thread记录，方便查看多进程和线程程序)
    critical_format = (
        '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | '
        '<red><level>{level:>10}</level></red> | '
        '<red>{process}</red>:<red>{thread}</red> | '
        '<red><level>{module}</level></red>:<red><level>{line}</level></red> - <red><level>{message}</level></red>'
        )
    logger.add(
        sink=sys.stderr,
        level="CRITICAL",
        filter=lambda record: record["level"].name == "CRITICAL", format=critical_format)
    logger.add(
        sink=log_file,
        level="CRITICAL",
        filter=lambda record: record["level"].name == "CRITICAL", format=critical_format)

    # 设置heading-level日志处理程序 (HEADING (25): 用于记录描述程序正常操作的信息消息)
    logger.level('HEADING', no=25, color="<blue>", icon="@")
    heading_format = (
        '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | '
        '<magenta>{level:>10}</magenta> | '
        '<magenta>{message}</magenta>'
        )
    logger.add(
        sink=sys.stderr,
        level="HEADING",
        filter=lambda record: record["level"].name == "HEADING", format=heading_format)
    logger.add(
        sink=log_file,
        level="HEADING",
        filter=lambda record: record["level"].name == "HEADING", format=heading_format)

    # 设置subheading-level日志处理程序 (SUBHEADING (23): 用于记录描述程序正常操作的信息消息)
    logger.level('SUBHEADING', no=23, color="<blue>", icon='@')
    subheading_format = (
        '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | '
        '<cyan>{level:>10}</cyan> | '
        '<cyan>{message}</cyan>'
        )
    logger.add(
        sink=sys.stderr,
        level="SUBHEADING",
        filter=lambda record: record["level"].name == "SUBHEADING", format=subheading_format)
    logger.add(
        sink=log_file,
        level="SUBHEADING",
        filter=lambda record: record["level"].name == "SUBHEADING", format=subheading_format)

    # 重定向系统输出到loguru
    redirect_logger = StreamToLoguru("INFO")
    sys.stderr = redirect_logger
    sys.stdout = redirect_logger


def format_string(string, mode='heading'):
    if mode == 'heading':
        return f' {string.upper()} '.center(80, '-')

    elif mode == 'subheading':
        return f' {string.title()} '.center(80, '-')
    

def heading(string):
    logger.log('HEADING', format_string(string, mode='heading'))


def subheading(string):
    logger.log('SUBHEADING', format_string(string, mode='subheading'))