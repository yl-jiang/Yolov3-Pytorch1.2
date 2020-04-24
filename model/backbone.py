#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/15 下午4:08
# @Author  : jyl
# @File    : backbone.py
import torch
import torch.nn as nn


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        # 'conv'(NO BIAS CONV): [layer_type, block_num, in_channels, out_channels, kernel_size, stride, padding]
        # 'residual' : [layer_type, block_num, in_channels, out_channels]
        layer_0 = ['conv', 1, 3, 32, 3, 1, 1]
        layer_1 = ['conv', 1, 32, 64, 3, 2, 1]
        layer_2 = ['residual', 1, 64, 64]
        layer_3 = ['conv', 1, 64, 128, 3, 2, 1]
        layer_4 = ['residual', 2, 128, 128]
        layer_5 = ['conv', 1, 128, 256, 3, 2, 1]
        layer_6 = ['residual', 8, 256, 256]
        layer_7 = ['conv', 1, 256, 512, 3, 2, 1]
        layer_8 = ['residual', 8, 512, 512]
        layer_9 = ['conv', 1, 512, 1024, 3, 2, 1]
        layer_10 = ['residual', 4, 1024, 1024]

        self.conv_0 = self._make_layers(layer_0)
        self.conv_1 = self._make_layers(layer_1)
        self.residual_1 = self._make_layers(layer_2)
        self.conv_2 = self._make_layers(layer_3)
        self.residual_2 = self._make_layers(layer_4)
        self.conv_3 = self._make_layers(layer_5)
        self.residual_3 = self._make_layers(layer_6)
        self.conv_4 = self._make_layers(layer_7)
        self.residual_4 = self._make_layers(layer_8)
        self.conv_5 = self._make_layers(layer_9)
        self.residual_5 = self._make_layers(layer_10)

    @staticmethod
    def no_bias_conv2d(in_channels, out_channels, kernel_size, stride, padding):
        layer_ = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                            stride=stride, padding=padding, bias=False),
                  nn.BatchNorm2d(num_features=out_channels),
                  nn.LeakyReLU(negative_slope=0.1, inplace=True)]

        return nn.Sequential(*layer_)

    def residual_block(self, in_channels, out_channels):
        assert in_channels == out_channels
        layer_ = []
        layer_.extend(self.no_bias_conv2d(in_channels=in_channels, out_channels=out_channels // 2,
                                          stride=1, kernel_size=1, padding=0))
        layer_.extend(self.no_bias_conv2d(in_channels=out_channels // 2, out_channels=out_channels,
                                          stride=1, kernel_size=3, padding=1))

        return nn.Sequential(*layer_)

    def _make_layers(self, layer):
        layer_ = nn.ModuleList([])
        assert isinstance(layer, list) and isinstance(layer[0], str)
        if layer[0] == 'conv':
            for i in range(layer[1]):
                conv = self.no_bias_conv2d(layer[2], layer[3], layer[4], layer[5], layer[6])
                layer_.append(conv)
        elif layer[0] == 'residual':
            for i in range(layer[1]):
                block = self.residual_block(layer[2], layer[3])
                layer_.append(block)
        else:
            ValueError(f'illegal layer key {layer[0]}')
        return layer_

    @staticmethod
    def _residual_forward(layers, x):
        shortcut = x
        for layer in layers:
            feature = layer(shortcut)
            shortcut = feature + shortcut
        return shortcut

    def forward(self, x):
        feature = self.conv_0[0](x)
        feature = self.conv_1[0](feature)
        feature = self._residual_forward(self.residual_1, feature)
        feature = self.conv_2[0](feature)
        feature = self._residual_forward(self.residual_2, feature)
        feature = self.conv_3[0](feature)
        route_1 = self._residual_forward(self.residual_3, feature)
        feature = self.conv_4[0](route_1)
        route_2 = self._residual_forward(self.residual_4, feature)
        feature = self.conv_5[0](route_2)
        route_3 = self._residual_forward(self.residual_5, feature)

        return route_1, route_2, route_3


if __name__ == '__main__':
    backbone = Backbone()
    a, b, c = backbone(torch.rand(10, 3, 416, 416))
    print(a.shape, b.shape, c.shape)

