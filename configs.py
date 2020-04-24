#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/15 下午4:08
# @Author  : jyl
# @File    : configs.py
import os
import torch
import logging.config
import random
import numpy as np


class Config:
    # set random seed
    # torch.manual_seed(7)  # cpu
    # torch.cuda.manual_seed(7)  # gpu
    # np.random.seed(7)  # numpy
    # random.seed(7)  # random and transforms
    # torch.backends.cudnn.deterministic = True  # cudnn
    # base_path = os.path.abspath('./')
    # base_path = os.path.split(base_path)[0]
    base_path = '/home/dk/jyl/V3/'

    COCO_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane',
                  'bus', 'train', 'truck', 'boat', 'traffic light',
                  'fire hydrant', 'stop sign', 'parking meter', 'bench',
                  'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
                  'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
                  'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                  'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                  'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife',
                  'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
                  'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                  'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                  'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                  'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                  'teddy bear', 'hair drier', 'toothbrush']

    # training
    batch_size = 8
    epochs = 300
    lr = 1e-7
    pth_lr = lr
    yolo_lr = lr
    optimizer_momentum = 0.9
    optimizer_weight_decay = 0.0005
    match_iou_threshold = 0.5
    use_focal_loss = False
    use_smooth_labels = False
    display_step = 500
    eval_step = 1500
    save_step = 150
    save_every = 5
    # 'SGD | 'Adam
    optimizer = 'SGD'
    # 对那些存在目标的cell下的且与gt匹配的预测框决定是否使用pred_gt_iou作为conf的值
    rescore = True
    model_save_dir = os.path.join(base_path, 'ckpt')

    # path
    COCO2017_dir = '/home/dk/jyl/Data/COCO2017'
    # saved_model_path = '/home/dk/jyl/V3/ckpt/darknet53.pkl'
    saved_model_path = os.path.join(base_path, 'ckpt', f'model_best_{optimizer}.pkl')
    log_config_path = os.path.join(base_path, 'logs', 'log.config')
    log_file_path = os.path.join(base_path, 'logs', f'log_{optimizer}.log')

    # logger
    logging.config.fileConfig('/home/dk/jyl/V3/logs/log.config')
    logger = logging.getLogger('Yolov3Logger')
    summary_writer_path = os.path.join(base_path, 'logs', f'summary_{optimizer}')

    # torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_default_dtype(torch.float32)
    torch.set_default_tensor_type(torch.FloatTensor)
    num_workers = 4

    # anchor
    anchor_num = 3
    kmean_converge = 1e-6
    anchors_path = os.path.join(base_path, 'data', 'anchors.txt')
    # anchors_path = '/home/dk/jyl/V3/data/anchors.txt'

    # grid
    B = 3

    # img
    img_h = 416
    img_w = 416
    img_size = 416
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    aug_threshold = 0.1

    # data
    class_num = 80

    # evaluation
    result_img_dir = os.path.join(base_path, 'data', 'result_imgs')
    coco_train_pkl_path = '/home/dk/jyl/Data/COCO2017/coco2017_train.pkl'
    coco_val_pkl_path = '/home/dk/jyl/Data/COCO2017/coco2017_val.pkl'
    coco_train_img_dir = '/home/dk/jyl/Data/COCO2017/train2017'
    coco_val_img_dir = '/home/dk/jyl/Data/COCO2017/val2017'

    # testing
    score_threshold = 0.3
    iou_threshold = 0.4
    max_boxes_num = 200


opt = Config()


