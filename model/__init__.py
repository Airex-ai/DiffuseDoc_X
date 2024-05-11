# -*- coding = utf-8 -*-
# @Time : 2023/3/27 14:25
# @Author : 熊文菲
# @File : __init__.py.py
# @Software : PyCharm
import logging
logger = logging.getLogger('base')


def create_model(opt):
    from .model import DDPM as M
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m