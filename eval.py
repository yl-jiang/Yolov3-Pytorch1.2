#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/21 上午10:58
# @Author  : jyl
# @File    : eval.py
from train.trainer import Yolov3COCOTrainer
import torch
from configs import opt
from utils import plot_one
import numpy as np
from utils import each_class_nms
from utils import letter_resize
import os
import cv2
from torchvision import transforms


def eval(img_path):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    torch_normailze = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    trainer = Yolov3COCOTrainer()
    trainer.use_pretrain(opt.saved_model_path)

    ori_img = cv2.imread(img_path)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    resized_img, ratio, dh, dw = letter_resize(ori_img, [416, 416])

    with torch.no_grad():
        # numpy
        # img = np.transpose(resized_img, axes=[2, 0, 1])
        # img = img[None, ...] / 255
        # img = torch.from_numpy(img).float().to(opt.device)
        # torch
        img = torch_normailze(resized_img)
        # [3, 416, 416] -> [1, 3, 416, 416]
        img = torch.unsqueeze(img, dim=0).to(opt.device)
        # boxes: [1, (13*13+26*26+52*52)*3, 4] / [xmin, ymin, xmax, ymax]
        # confs: [1, (13*13+26*26+52*52)*3, 1]
        # probs: [1, (13*13+26*26+52*52)*3, 80]
        boxes, confs, probs = trainer.predict(img)

        boxes = boxes.squeeze(0)  # [13*13*5, 4]
        confs = confs.squeeze(0)  # [13*13*5, 1]
        probs = probs.squeeze(0)  # [13*13*5, 80]
        scores = confs * probs  # [13*13*5, 80]

        # box_out: [xmin, ymin, xmax, ymax]
        box_out, score_out, label_out = each_class_nms(boxes, scores, opt.score_threshold, opt.iou_threshold,
                                                       opt.max_boxes_num, 416)
        box_out = box_out.clamp(min=0., max=416.)
        if len(box_out) != 0:
            plot_dict = {'img': ori_img,
                         'ratio': ratio,
                         'dh': dh,
                         'dw': dw,
                         'pred_box': box_out.cpu().numpy(),
                         'pred_score': score_out.cpu().numpy(),
                         'pred_label': label_out.cpu().numpy(),
                         'gt_box': None,
                         'gt_label': None,
                         'img_name': os.path.basename(img_path),
                         'save_path': '/home/dk/Desktop/'}

            plot_one(plot_dict)


if __name__ == '__main__':
    eval('/home/dk/jyl/Data/COCO2017/val2017/000000532481.jpg')
