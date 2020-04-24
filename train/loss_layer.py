#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/17 下午2:41
# @Author  : jyl
# @File    : loss_layer.py
import torch.nn as nn
from collections import defaultdict
import torch
from configs import opt
import numpy as np


class LossLayer(nn.Module):
    def __init__(self, anchors, use_focal_scale=False, use_label_smooth=False):
        super(LossLayer, self).__init__()
        self.anchors = anchors
        self.use_focal_scale = use_focal_scale
        self.use_label_smooth = use_label_smooth
        self.img_size = opt.img_size
        self.MSE = nn.MSELoss(reduction='none')
        self.BCE = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, preds, targets):
        # preds: [N, 13, 13, 255]
        # targets: [N, 13, 13, 3, 85]
        assert preds.size(1) == targets.size(1)
        feature_map_size = preds.size(1)
        if feature_map_size == 13:
            loss_list = self.compute_loss(preds, targets, self.anchors['large'])
        elif feature_map_size == 26:
            loss_list = self.compute_loss(preds, targets, self.anchors['mid'])
        else:
            loss_list = self.compute_loss(preds, targets, self.anchors['small'])

        return loss_list

    def reorg_layer(self, preds, anchors):
        grid_size = preds.size(1)
        # ratio format is [h,w]
        ratio = self.img_size / grid_size
        # rescaled_anchors format is [w,h] / make anchors's scale same as predicts
        rescaled_anchors = anchors / ratio

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

    def compute_loss(self, preds, targets, anchors):
        batch_size = preds.size(0)
        grid_size = preds.size(1)
        ratio = self.img_size / grid_size
        anchors = torch.from_numpy(anchors).float().to(opt.device)
        xy_offset, pred_bboxes, confs_logit, classes_logit = self.reorg_layer(preds, anchors)

        # [N, 13, 13, 3, 2]
        preds_xy = pred_bboxes[..., 0:2]
        # [N, 13, 13, 3, 2]
        preds_wh = pred_bboxes[..., 2:4]
        true_xy = targets[..., 0:2] / ratio - xy_offset
        preds_xy = preds_xy / ratio - xy_offset
        true_wh = targets[..., 2:4] / anchors
        preds_wh = preds_wh / anchors
        true_wh[true_wh == 0] = 1.
        preds_wh[preds_wh == 0] = 1.
        true_wh = torch.clamp(true_wh, min=1e-9, max=1e9).log()
        preds_wh = torch.clamp(preds_wh, min=1e-9, max=1e9).log()

        # [N, 13, 13, 3]
        obj_mask = targets[..., 4] > 0.
        # smaller bbox has bigger bbox loss weight/ [batch, 13, 13, 3, 1]
        bbox_loss_scale = 2. - targets[..., 2:3]/self.img_size * targets[..., 3:4] / self.img_size
        txty_loss = self.MSE(input=preds_xy[obj_mask], target=true_xy[obj_mask])
        # [M, 2] & [M, 1] -> [M, 2]
        txty_loss = torch.sum(txty_loss * bbox_loss_scale[obj_mask]) / batch_size
        twth_loss = self.MSE(input=preds_wh[obj_mask], target=true_wh[obj_mask])
        twth_loss = torch.sum(twth_loss * bbox_loss_scale[obj_mask]) / batch_size

        # [N, 13, 13, 3]
        noobj_mask = self.make_noobj_mask(pred_bboxes, targets[..., 0:5])
        noobj_conf_mask = ~obj_mask & noobj_mask
        noobj_conf_loss = self.BCE(input=confs_logit[noobj_conf_mask], target=targets[..., 4:5][noobj_conf_mask])
        obj_conf_loss = self.BCE(input=confs_logit[obj_mask], target=targets[..., 4:5][obj_mask])

        if self.use_focal_scale:
            focal_scale = self.focal_scale(obj_mask.unsqueeze(-1).float(), confs_logit)
            noobj_conf_loss = torch.sum(focal_scale[obj_mask] * noobj_conf_loss) / batch_size
            obj_conf_loss = torch.sum(focal_scale[obj_mask] * obj_conf_loss) / batch_size
        else:
            noobj_conf_loss = noobj_conf_loss.sum() / batch_size
            obj_conf_loss = obj_conf_loss.sum() / batch_size

        if self.use_label_smooth:
            true_class = self.smooth_label(targets[..., 5:], opt.class_num)
        else:
            true_class = targets[..., 5:]
        class_loss = self.BCE(input=classes_logit[obj_mask], target=true_class[obj_mask]) / batch_size

        total_loss = txty_loss + twth_loss + noobj_conf_loss + obj_conf_loss + class_loss

        return txty_loss, twth_loss, noobj_conf_loss, obj_conf_loss, class_loss, total_loss

    def make_noobj_mask(self, preds, targets):
        # targets :[N, 13, 13, 3, 5]
        # preds: [N, 13, 13, 3, 4]
        assert targets.size(-1) == 5 and preds.size(-1) == 4
        grid_size = preds.size(1)
        noobj_mask = torch.ones(opt.batch_size, grid_size, grid_size, opt.B, dtype=torch.bool,
                                requires_grad=False, device=opt.device)

        for b in range(preds.size(0)):
            # [13, 13, 3]
            valid_index = targets[b, ..., 4] > 0.
            # [N, 4]
            valid_bbox = targets[b, ..., 0:4][valid_index]
            # [13, 13, 3, N]
            iou_pred_gt = self.bbox_iou(preds[b], valid_bbox)
            # [13, 13, 3]]
            best_iou = torch.max(iou_pred_gt, dim=-1)[0]
            iou_mask = best_iou >= 0.6
            noobj_mask[b][iou_mask] = False

        return noobj_mask

    @staticmethod
    def bbox_iou(pred_bbox, gt_bbox):
        pred_xy = pred_bbox[..., 0:2]
        pred_wh = pred_bbox[..., 2:4]
        # [13, 13, 3, 2] -> [13, 13, 3, 1, 2]
        pred_xy.unsqueeze_(dim=-2)
        pred_wh.unsqueeze_(dim=-2)

        # [N, 2]
        gt_xy = gt_bbox[:, 0:2]
        gt_wh = gt_bbox[:, 2:4]

        # [13, 13, 3, 1, 2] & [N, 2] -> [13, 13, 3, N, 2]
        intersect_min = torch.max(pred_xy-pred_wh/2, gt_xy-gt_wh/2)
        intersect_max = torch.min(pred_xy+pred_wh/2, gt_xy+gt_wh/2)

        # [13, 13, 3, N, 2] & [13, 13, 3, N, 2] -> [13, 13, 3, N]
        intersect_wh = torch.max(intersect_max - intersect_min, torch.tensor(0).float().to(opt.device))
        intersect_area = torch.prod(intersect_wh, dim=-1)

        gt_area = torch.prod(gt_wh, dim=-1)
        pred_area = torch.prod(pred_wh, dim=-1)
        # [13, 13, 3, 1] & [N] -> [13, 13, 3, N]
        union_area = pred_area + gt_area

        # [13, 13, 3, N]
        iou_pred_gt = intersect_area / (union_area - intersect_area + 1e-9)
        return iou_pred_gt

    @staticmethod
    def smooth_label(labels, class_num):
        delta = 0.01
        smooth = (1 - delta) * labels + delta / class_num
        return smooth

    @staticmethod
    def focal_scale(preds, targets):
        assert preds.size() == targets.size()
        alpha = 1.0
        gamma = 2.0
        focal_scale = alpha * torch.pow(torch.abs(targets - preds.sigmoid()), gamma)
        return focal_scale



