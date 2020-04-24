#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/17 下午8:58
# @Author  : jyl
# @File    : nms.py
import numpy as np
import torch


def cpu_nms(boxes, scores, iou_threshold):
    """
    :param boxes:
        [N, 4] / 'N' means not sure
    :param scores:
        [N, 1]
    :param iou_threshold:
        a scalar
    :return:
        keep_index
    """
    # boxes format : [xmin, ymin, xmax, ymax]
    assert isinstance(boxes, np.ndarray) and isinstance(scores, np.ndarray)
    assert boxes.shape[0] == scores.shape[0]
    box_copy = boxes.copy()
    score_copy = scores.copy()
    keep_index = []
    while np.sum(score_copy) > 0.:
        # mark reserved box
        max_score_index = np.argmax(score_copy)
        box1 = box_copy[[max_score_index]]
        keep_index.append(max_score_index)
        score_copy[max_score_index] = 0.
        ious = cpu_iou(box1, box_copy)
        # mark unuseful box
        # keep_mask shape [N,] / 'N' means uncertain
        del_index = np.greater(ious, iou_threshold)
        score_copy[del_index] = 0.

    return keep_index


def cpu_iou(bbox1, bbox2):
    """
    :param bbox1: [[xmin, ymin, xmax, ymax], ...]
    :param bbox2: [[xmin, ymin, xmax, ymax], ...]
    :return:
    """
    assert bbox1.shape[-1] == bbox2.shape[-1] == 4

    bbox1_area = np.prod(bbox1[:, [2, 3]] - bbox1[:, [0, 1]] + 1, axis=-1)
    bbox2_area = np.prod(bbox2[:, [2, 3]] - bbox2[:, [0, 1]] + 1, axis=-1)

    intersection_ymax = np.minimum(bbox1[:, 3], bbox2[:, 3])
    intersection_xmax = np.minimum(bbox1[:, 2], bbox2[:, 2])
    intersection_ymin = np.maximum(bbox1[:, 1], bbox2[:, 1])
    intersection_xmin = np.maximum(bbox1[:, 0], bbox2[:, 0])

    intersection_w = np.maximum(0., intersection_xmax - intersection_xmin + 1)
    intersection_h = np.maximum(0., intersection_ymax - intersection_ymin + 1)
    intersection_area = intersection_w * intersection_h
    iou_out = intersection_area / (bbox1_area + bbox2_area - intersection_area)

    return iou_out


def gpu_nms(boxes, scores, iou_threshold):
    """
    :param boxes: [M, 4]
    :param scores: [M, 1]
    :param iou_threshold:
    :return:
    """
    assert isinstance(boxes, torch.Tensor) and isinstance(scores, torch.Tensor)
    assert boxes.shape[0] == scores.shape[0]

    box_copy = boxes.detach().clone()
    score_copy = scores.detach().clone()
    keep_index = []
    while torch.sum(score_copy) > 0.:
        # mark reserved box
        max_score_index = torch.argmax(score_copy).item()
        box1 = box_copy[[max_score_index]]
        keep_index.append(max_score_index)
        score_copy[max_score_index] = 0.
        ious = gpu_iou(box1, box_copy)
        ignore_index = ious.gt(iou_threshold)
        score_copy[ignore_index] = 0.

    return keep_index


def gpu_iou(bbox1, bbox2):
    """
    :param bbox1: [[xmin, ymin, xmax, ymax], ...] / type: torch.Tensor
    :param bbox2: [[xmin, ymin, xmax, ymax], ...] / type: torch.Tensor
    :return:
    """
    assert isinstance(bbox1, torch.Tensor)
    assert isinstance(bbox2, torch.Tensor)
    assert (bbox1[:, [2, 3]] >= bbox1[:, [0, 1]]).bool().all()
    assert (bbox2[:, [2, 3]] >= bbox2[:, [0, 1]]).bool().all()
    assert bbox1.shape[-1] == bbox2.shape[-1] == 4
    assert bbox1.device == bbox2.device
    device = bbox1.device

    bbox1_area = torch.prod(bbox1[:, [2, 3]] - bbox1[:, [0, 1]] + 1, dim=-1)
    bbox2_area = torch.prod(bbox2[:, [2, 3]] - bbox2[:, [0, 1]] + 1, dim=-1)

    intersection_ymax = torch.min(bbox1[:, 3], bbox2[:, 3])
    intersection_xmax = torch.min(bbox1[:, 2], bbox2[:, 2])
    intersection_ymin = torch.max(bbox1[:, 1], bbox2[:, 1])
    intersection_xmin = torch.max(bbox1[:, 0], bbox2[:, 0])

    intersection_w = torch.max(torch.tensor(0.).float().to(device), intersection_xmax - intersection_xmin + 1)
    intersection_h = torch.max(torch.tensor(0.).float().to(device), intersection_ymax - intersection_ymin + 1)
    intersection_area = intersection_w * intersection_h
    iou_out = intersection_area / (bbox1_area + bbox2_area - intersection_area)

    return iou_out


def each_class_nms(boxes, scores, score_threshold, iou_threshold, max_box_num, img_size):
    """
    :param boxes: [N, 4]
    :param scores: [N, class_num]
    :param score_threshold:
    :param iou_threshold:
    :param max_box_num:
    :param img_size:
    :return:
     boxes_output shape: [X, 4]
     scores_output shape: [X,]
     labels_output shape: [X,]
    """
    assert isinstance(boxes, torch.Tensor) and isinstance(scores, torch.Tensor)
    assert boxes.dim() == 2 and scores.dim() == 2

    class_num = scores.size(-1)
    device = scores.device

    boxes = boxes.clamp(0., img_size)
    boxes_output = []
    scores_output = []
    labels_output = []
    # [N, class_num]
    score_mask = scores.ge(score_threshold)
    # do nms for each class
    for k in range(class_num):
        valid_mask = score_mask[:, k]  # [M, class_num]
        if valid_mask.sum() == 0:
            continue
        else:
            valid_boxes = boxes[valid_mask]  # [M, 4]
            valid_scores = scores[:, k][valid_mask]  # [M, 1]
            keep_index = gpu_nms(valid_boxes, valid_scores, iou_threshold)
            for keep_box in valid_boxes[keep_index]:
                boxes_output.append(keep_box)
            scores_output.extend(valid_scores[keep_index])
            labels_output.extend([k for _ in range(len(keep_index))])

    assert len(boxes_output) == len(scores_output) == len(labels_output)
    num_out = len(labels_output)
    if num_out == 0:
        return torch.tensor([], device=device), torch.tensor([], device=device), torch.tensor([], device=device)
    else:
        boxes_output = torch.stack(boxes_output, dim=0)
        scores_output = torch.tensor(scores_output, device=device)
        labels_output = torch.tensor(labels_output, device=device)
        assert boxes_output.dim() == 2
        assert labels_output.dim() == scores_output.dim() == 1
        assert boxes_output.size(0) == scores_output.numel() == labels_output.numel()
        if num_out > max_box_num:
            descend_order_index = torch.argsort(scores_output)[::-1]
            output_index = descend_order_index[:max_box_num]
        else:
            output_index = torch.arange(num_out)
        return boxes_output[output_index], scores_output[output_index], labels_output[output_index]


def all_class_nms(boxes, scores, score_threshold, iou_threshold, max_box_num, img_size):
    """
    :param boxes: [N, 4]
    :param scores: [N, class_num]
    :param score_threshold: 0.3
    :param iou_threshold: 0.45
    :param max_box_num:
    :param img_size:
    :return:
     boxes_output shape: [X, 4]
     scores_output shape: [X,]
     labels_output shape: [X,]
    """
    assert isinstance(boxes, torch.Tensor) and isinstance(scores, torch.Tensor)
    assert boxes.dim() == 2 and scores.dim() == 2
    device = scores.device

    boxes = boxes.clamp(0., img_size)
    boxes_output = []
    scores_output = []
    labels_output = []
    # [N, class_num] -> [N, 1]
    scores_mask = scores.max(dim=-1)
    labels = scores_mask[1]
    max_scores = scores_mask[0]
    # [class_num, 1]
    valid_mask = max_scores.ge(score_threshold)

    # do nms for all class
    if valid_mask.sum() != 0:
        valid_boxes = boxes[valid_mask]  # [M, 4]
        valid_scores = max_scores[valid_mask]  # [M, 1]
        valid_labels = labels[valid_mask]
        keep_index = gpu_nms(valid_boxes, valid_scores, iou_threshold)
        for keep_box in valid_boxes[keep_index]:
            boxes_output.append(keep_box)
        scores_output.extend(valid_scores[keep_index])
        labels_output.extend(valid_labels[keep_index])

    assert len(boxes_output) == len(scores_output) == len(labels_output)
    num_out = len(labels_output)
    if num_out == 0:
        return torch.tensor([], device=device), torch.tensor([], device=device), torch.tensor([], device=device)
    else:
        boxes_output = torch.stack(boxes_output, dim=0)
        scores_output = torch.tensor(scores_output)
        labels_output = torch.tensor(labels_output)
        assert boxes_output.dim() == 2
        assert labels_output.dim() == scores_output.dim() == 1
        assert boxes_output.size(0) == scores_output.numel() == labels_output.numel()
        if num_out > max_box_num:
            descend_order_index = torch.argsort(scores_output)[::-1]
            output_index = descend_order_index[:max_box_num]
        else:
            output_index = torch.arange(num_out)
        return boxes_output[output_index], scores_output[output_index], labels_output[output_index]

