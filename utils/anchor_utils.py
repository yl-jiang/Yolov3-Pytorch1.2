#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/16 下午4:54
# @Author  : jyl
# @File    : anchor_utils.py
import numpy as np


def parse_anchors(anchors_dir):
    anchors = np.loadtxt(fname=anchors_dir, delimiter=',', usecols=[0, 1], skiprows=0, dtype=np.float32).reshape(
        [-1, 2])
    anchors_area = np.prod(anchors, axis=-1)
    descend_index = np.argsort(anchors_area)
    anchor_dict = {'small': anchors[descend_index[:3]],
                   'mid': anchors[descend_index[3:6]],
                   'large': anchors[descend_index[6:]]}

    # anchor_dict = {'small': np.asarray([[10, 13], [16, 30], [33, 23]]),
    #                'mid': np.asarray([[30, 61], [62, 45], [59, 119]]),
    #                'large': np.asarray([[116, 90], [156, 198], [373, 326]])}

    return anchor_dict


if __name__ == '__main__':
    print(parse_anchors('./anchors.txt'))
