#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/17 下午2:22
# @Author  : jyl
# @File    : trainer.py
from model import DarkNet53
import torch
import numpy as np
from configs import opt
import os
from utils import parse_anchors
import torch.optim as optim
from .loss_layer import LossLayer
from torchnet.meter import AverageValueMeter
from collections import defaultdict
from torch.optim.lr_scheduler import CosineAnnealingLR


class Yolov3COCOTrainer:
    def __init__(self):
        self.darknet53 = DarkNet53().to(opt.device)
        self.optimizer = self.init_optimizer()
        self.anchors = parse_anchors(opt.anchors_path)
        self.loss_layer = LossLayer(self.anchors)
        self.meter = AverageValueMeter()
        self.loss_dict = defaultdict(dict)
        self.img_size = opt.img_size
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=5, eta_min=0.)

    def use_pretrain(self, model_path, load_optimizer=True):
        if not os.path.exists(model_path):
            raise OSError(2, 'No such file or directory', model_path)
        else:
            print(f'use pretrained model: {model_path}')
            state_dict = torch.load(model_path)
            self.last_loss = state_dict['loss']
            if 'epoch' in state_dict.keys():
                self.epoch_num = state_dict['epoch']
            else:
                self.epoch_num = 0
            if 'total_steps' in state_dict.keys():
                self.total_steps = state_dict['total_steps']
            else:
                self.total_steps = 0
            if 'model' in state_dict.keys():
                print('loading pretrained model ...')
                self.darknet53.load_state_dict(state_dict['model'])
            if load_optimizer and 'optimizer' in state_dict.keys():
                print('loading pretrained optimizer ...')
                self.optimizer.load_state_dict((state_dict['optimizer']))

    def init_optimizer(self):
        if opt.optimizer == 'SGD':
            optimizer = optim.SGD(params=self.darknet53.parameters(),
                                  momentum=opt.optimizer_momentum,
                                  weight_decay=opt.optimizer_weight_decay,
                                  lr=opt.lr,
                                  nesterov=True)
            return optimizer
        elif opt.optimizer == 'Adam':
            optimizer = optim.Adam(params=self.darknet53.parameters(),
                                   weight_decay=opt.optimizer_weight_decay,
                                   lr=opt.lr)
            return optimizer
        else:
            ValueError()

    def save_model(self, epoch, steps, loss, save_path):
        model_state = self.darknet53.state_dict()
        optim_state = self.optimizer.state_dict()
        state_dict = {'model': model_state,
                      'optim': optim_state,
                      'epoch': epoch,
                      'steps': steps,
                      'loss': loss}
        diret = os.path.dirname(save_path)
        if not os.path.exists(diret):
            os.makedirs(diret)
        torch.save(state_dict, save_path)
        print('model saved...')

    def adjust_lr(self, epoch):
        if opt.optimizer == 'SGD':
            if epoch < 5:
                for group in self.optimizer.param_groups:
                    group['lr'] = opt.lr * 0.1
            elif 5 <= epoch < 50:
                for group in self.optimizer.param_gropus:
                    group['lr'] = opt.lr * 0.1
            else:
                for group in self.optimizer.param_groups:
                    group['lr'] = opt.lr * (0.1 ** (2 + epoch // 50))

    def train_step(self, imgs, targets, epoch):
        self.darknet53.train()
        assert len(targets) == 3
        preds = self.darknet53(imgs)
        # compute loss in 3 scale
        # pred: [N, 13, 13, 255]
        # target: [N, 13, 13, 3, 85]
        loss_list_13 = self.loss_layer(preds[0], targets[0])
        # self.print_fun(loss_list_13, 'fm_13')
        loss_list_26 = self.loss_layer(preds[1], targets[1])
        # self.print_fun(loss_list_26, 'fm_26')
        loss_list_52 = self.loss_layer(preds[2], targets[2])
        # self.print_fun(loss_list_52, 'fm_52')

        total_loss = loss_list_13[-1] + loss_list_26[-1] + loss_list_52[-1]
        txty_loss = loss_list_13[0] + loss_list_26[0] + loss_list_52[0]
        twth_loss = loss_list_13[1] + loss_list_26[1] + loss_list_52[1]
        noobj_conf_loss = loss_list_13[2] + loss_list_26[2] + loss_list_52[2]
        obj_conf_loss = loss_list_13[3] + loss_list_26[3] + loss_list_52[3]
        class_loss = loss_list_13[4] + loss_list_26[4] + loss_list_52[4]
        self.loss_dict = {'xy_loss': txty_loss.detach().cpu().item(),
                          'wh_loss': twth_loss.detach().cpu().item(),
                          'obj_conf_loss': obj_conf_loss.detach().cpu().item(),
                          'noobj_conf_loss': noobj_conf_loss.detach().cpu().item(),
                          'class_loss': class_loss.detach().cpu().item(),
                          'total_loss': total_loss.detach().cpu().item()}

        self.meter.add(total_loss.detach().cpu().item())
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # tune learning rate
        if self.scheduler is not None:
            self.scheduler.step(epoch)
        else:
            self.adjust_lr(epoch)

    def reorg_layer(self, preds, anchors):
        grid_size = preds.size(1)
        # ratio format is [h,w]
        ratio = self.img_size / grid_size
        # rescaled_anchors format is [w,h] / make anchors's scale same as predicts
        rescaled_anchors = torch.from_numpy(anchors / ratio).float().to(opt.device)

        # resahpe preds to [N, 13, 13, 3, 85]
        preds = preds.contiguous().view(-1, grid_size, grid_size, 3, 5 + opt.class_num)
        # box_xy: [N, 13, 13, 3, 2] / format [x, y]
        # box_wh: [N, 13, 13, 3, 2] / format [w, h]
        # confs: [N, 13, 13, 3, 1]
        # classes: [N, 13, 13, 3, 80]
        box_xy, box_wh, confs_logit, classes_logit = preds.split([2, 2, 1, opt.class_num], dim=-1)
        box_xy = box_xy.sigmoid()
        grid_x = np.arange(grid_size, dtype=np.float32)
        grid_y = np.arange(grid_size, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)

        xy_offset = np.concatenate([grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)], axis=-1)
        # xy_offset: [13, 13, 1, 2]
        xy_offset = torch.from_numpy(xy_offset).float().to(opt.device)
        xy_offset = xy_offset.contiguous().view(grid_size, grid_size, 1, 2)

        # rescale to input_image scale
        box_xy = (box_xy + xy_offset) * ratio
        # compute in the scale 13
        box_wh = torch.exp(box_wh) * rescaled_anchors
        # rescale to input_image scale
        box_wh = box_wh * ratio

        # reset scaled pred_box to bounding box format [x, y, w, h]
        # bboxes: [N, 13, 13, 3, 4]
        bboxes = torch.cat([box_xy, box_wh], dim=-1)

        return xy_offset, bboxes, confs_logit, classes_logit

    def predict(self, img):
        self.darknet53.eval()
        preds = self.darknet53(img)
        # import pickle
        # pickle.dump(preds, open('/home/dk/Desktop/mykite.pkl', 'wb'))
        result_13 = self.reorg_layer(preds[0], self.anchors['large'])
        result_26 = self.reorg_layer(preds[1], self.anchors['mid'])
        result_52 = self.reorg_layer(preds[2], self.anchors['small'])

        def _reshape(result):
            xy_offset, bbox, conf, prob = result
            grid_size = xy_offset.size(0)
            bbox = bbox.reshape(-1, grid_size * grid_size * opt.anchor_num, 4)
            conf = conf.reshape(-1, grid_size * grid_size * opt.anchor_num, 1)
            prob = prob.reshape(-1, grid_size * grid_size * opt.anchor_num, opt.class_num)
            # bbox: [N, 13*13*3, 4]
            # conf: [N, 13*13*3, 1]
            # prob: [N, 13*13*3, 82]
            return bbox, conf, prob

        bbox_out, conf_out, prob_out = [], [], []
        for result in [result_13, result_26, result_52]:
            bbox, conf, prob = _reshape(result)
            bbox_out.append(bbox)
            conf_out.append(conf.sigmoid())
            prob_out.append(prob.sigmoid())

        # boxes: [N, (13*13+26*26+52*52)*3, 4] / (center_x, center_y, width, height)
        # confs: [N, (13*13+26*26+52*52)*3, 1]
        # probs: [N, (13*13+26*26+52*52)*3, 80]
        boxes = torch.cat(bbox_out, dim=1)
        confs = torch.cat(conf_out, dim=1)
        probs = torch.cat(prob_out, dim=1)

        # [N, (13*13+26*26+52*52)*3, 1]
        xmin = boxes[..., [0]] - boxes[..., [2]] / 2
        ymin = boxes[..., [1]] - boxes[..., [3]] / 2
        xmax = boxes[..., [0]] + boxes[..., [2]] / 2
        ymax = boxes[..., [1]] + boxes[..., [3]] / 2
        # [N, (13*13+26*26+52*52)*3, 4] / [xmin, ymin, xmax, ymax]
        boxes = torch.cat([xmin, ymin, xmax, ymax], dim=-1)
        return boxes, confs, probs

    def print_fun(self, loss_list, name):
        print(name, ':')
        loss_name = ['txty_loss', 'twth_loss', 'noobj_conf_loss', 'obj_conf_loss', 'class_loss', 'total_loss']
        for n, loss in zip(loss_name, loss_list):
            print(n, ':', loss.detach().cpu().item())

