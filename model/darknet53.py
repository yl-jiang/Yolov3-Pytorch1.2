#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/17 上午9:26
# @Author  : jyl
# @File    : darknet53.py
from .backbone import Backbone
import torch
import torch.nn as nn


CLASS_NUM = 80


class Concat(nn.Module):
    def __init__(self):
        super(Concat, self).__init__()

    def forward(self, tensor_1, tensor_2):
        return torch.cat([tensor_1, tensor_2], dim=1)


class DarkNet53(nn.Module):
    def __init__(self):
        super(DarkNet53, self).__init__()
        self.backbone = Backbone()
        yolo_1 = [['conv', 1, 1024, 512, 1, 1, 0],
                  ['conv', 1, 512, 1024, 3, 1, 1],
                  ['conv', 1, 1024, 512, 1, 1, 0],
                  ['conv', 1, 512, 1024, 3, 1, 1],
                  ['conv', 1, 1024, 512, 1, 1, 0]]  # to make fm_2 and fm_1
        yolo_2 = [['conv', 1, 512, 1024, 3, 1, 1],
                  ['conv_bias', 1, 1024, 3*(4+1+CLASS_NUM), 1, 1, 0]]  # fm_1
        yolo_3 = [['conv', 1, 512, 256, 1, 1, 0]]  # to make fm_2
        # upsampling
        # concatenate
        yolo_4 = [['conv', 1, 768, 256, 1, 1, 0],
                  ['conv', 1, 256, 512, 3, 1, 1],
                  ['conv', 1, 512, 256, 1, 1, 0],
                  ['conv', 1, 256, 512, 3, 1, 1],
                  ['conv', 1, 512, 256, 1, 1, 0]]  # to make fm_2 and fm_3
        yolo_5 = [['conv', 1, 256, 512, 3, 1, 1],
                  ['conv_bias', 1, 512, 3*(4+1+CLASS_NUM), 1, 1, 0]]  # fm_2
        yolo_6 = [['conv', 1, 256, 128, 1, 1, 0]]  # to make fm_3
        # upsampling
        # concatenate
        yolo_7 = [['conv', 1, 384, 128, 1, 1, 0],
                  ['conv', 1, 128, 256, 3, 1, 1],
                  ['conv', 1, 256, 128, 1, 1, 0],
                  ['conv', 1, 128, 256, 3, 1, 1],
                  ['conv', 1, 256, 128, 1, 1, 0],
                  ['conv', 1, 128, 256, 3, 1, 1],
                  ['conv_bias', 1, 256, 3*(4+1+CLASS_NUM), 1, 1, 0]]  # fm_3

        self.upsample = nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True)
        self.concat = Concat()
        self.yolo_1 = self._make_layers(yolo_1)
        self.yolo_2 = self._make_layers(yolo_2)
        self.yolo_3 = self._make_layers(yolo_3)
        self.yolo_4 = self._make_layers(yolo_4)
        self.yolo_5 = self._make_layers(yolo_5)
        self.yolo_6 = self._make_layers(yolo_6)
        self.yolo_7 = self._make_layers(yolo_7)

    @staticmethod
    def no_bias_conv2d(in_channels, out_channels, kernel_size, stride, padding):
        layer_ = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                            stride=stride, padding=padding, bias=False),
                  nn.BatchNorm2d(num_features=out_channels),
                  nn.LeakyReLU(negative_slope=0.1, inplace=True)]

        return layer_

    @staticmethod
    def bias_conv2d(in_channels, out_channels, kernel_size, stride, padding):
        layer_ = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                            stride=stride, padding=padding, bias=True)]
        return layer_

    def _make_layers(self, layers):
        layer_ = []
        for layer in layers:
            assert isinstance(layer[0], str)
            if layer[0] == 'conv':
                for i in range(layer[1]):
                    layer_ += self.no_bias_conv2d(layer[2], layer[3], layer[4], layer[5], layer[6])
            elif layer[0] == 'residual':
                for i in range(layer[1]):
                    layer_ += self.residual_block(layer[2], layer[3])
            elif layer[0] == 'conv_bias':
                for i in range(layer[1]):
                    layer_ += self.bias_conv2d(layer[2], layer[3], layer[4], layer[5], layer[6])
            else:
                ValueError(f'illegal layer key {layer[0]}')
        return nn.Sequential(*layer_)

    def _init_weights(self):
        for model in self.modules():
            if isinstance(model, nn.Conv2d):
                nn.init.kaiming_normal_(model.weight)
                if model.bias is not None:
                    nn.init.constant_(model.bias, val=0.0)
            if isinstance(model, nn.BatchNorm2d):
                nn.init.constant_(model.weight, val=1.0)
                nn.init.constant_(model.bias, val=0.0)
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight)
                nn.init.constant_(model.bias, val=0.0)

    def forward(self, x):
        route_1, route_2, route_3 = self.backbone(x)
        inter_1 = self.yolo_1(route_3)
        feature_map_13 = self.yolo_2(inter_1)

        inter_1 = self.yolo_3(inter_1)
        inter_1 = self.upsample(inter_1)
        inter_1 = self.concat(inter_1, route_2)
        inter_2 = self.yolo_4(inter_1)
        feature_map_26 = self.yolo_5(inter_2)

        inter_2 = self.yolo_6(inter_2)
        inter_2 = self.upsample(inter_2)
        inter_2 = self.concat(inter_2, route_1)
        feature_map_52 = self.yolo_7(inter_2)

        return feature_map_13.permute(0, 2, 3, 1), feature_map_26.permute(0, 2, 3, 1), feature_map_52.permute(0, 2, 3, 1)


if __name__ == '__main__':
    darknet53 = DarkNet53()
    print(darknet53)
    fm_1, fm_2, fm_3 = darknet53(torch.rand(10, 3, 416, 416))
    print(fm_1.shape, fm_2.shape, fm_3.shape)
