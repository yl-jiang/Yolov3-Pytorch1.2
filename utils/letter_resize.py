#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/21 上午11:27
# @Author  : jyl
# @File    : letterresize_img.py
import numpy as np
import cv2
import os


def letter_resize(img, target_img_size):
    """
    :param img:
    :param bboxes: format [ymax, xmax, ymin, xmin]
    :param target_img_size: [416, 416]
    :return:
        letterbox_img
        resized_bbox: [ymax, xmax, ymin, xmin]
    """
    if isinstance(img, str) and os.path.exists(img):
        img = cv2.imread(img)
    else:
        assert isinstance(img, np.ndarray)
    letterbox_img = np.full(shape=[target_img_size[0], target_img_size[1], 3], fill_value=128, dtype=np.float32)
    org_img_shape = [img.shape[0], img.shape[1]]  # [height, width]
    ratio = np.min([target_img_size[0] / org_img_shape[0], target_img_size[1] / org_img_shape[1]])
    # resized_shape format : [height, width]
    resized_shape = tuple([int(org_img_shape[0] * ratio), int(org_img_shape[1] * ratio)])
    resized_img = cv2.resize(img, resized_shape[::-1])
    dh = target_img_size[0] - resized_shape[0]
    dw = target_img_size[1] - resized_shape[1]
    letterbox_img[(dh//2):(dh//2+resized_shape[0]), (dw//2):(dw//2+resized_shape[1]), :] = resized_img
    letterbox_img = letterbox_img.astype(np.uint8)
    return letterbox_img, ratio, dh//2, dw//2

