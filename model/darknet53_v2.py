#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/18 下午2:50
# @Author  : jyl
# @File    : darknet53_v2.py
import torch
import torch.nn as nn


class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()
        self.stage_a_1 = self.conv2d(3, 32, 3, 1)

        self.stage_b_1 = self.conv2d(32, 64, 3, 2)

        self.stage_c_1 = self.residual_block(64, 64)

        self.stage_b_2 = self.conv2d(64, 128, 3, 2)

        self.stage_d_1 = self.residual_block(128, 128)
        self.stage_d_2 = self.residual_block(128, 128)

        self.stage_b_3 = self.conv2d(128, 256, 3, 2)

        self.stage_e_1 = self.residual_block(256, 256)
        self.stage_e_2 = self.residual_block(256, 256)
        self.stage_e_3 = self.residual_block(256, 256)
        self.stage_e_4 = self.residual_block(256, 256)
        self.stage_e_5 = self.residual_block(256, 256)
        self.stage_e_6 = self.residual_block(256, 256)
        self.stage_e_7 = self.residual_block(256, 256)
        self.stage_e_8 = self.residual_block(256, 256)

        self.stage_b_4 = self.conv2d(256, 512, 3, 2)

        self.stage_f_1 = self.residual_block(512, 512)
        self.stage_f_2 = self.residual_block(512, 512)
        self.stage_f_3 = self.residual_block(512, 512)
        self.stage_f_4 = self.residual_block(512, 512)
        self.stage_f_5 = self.residual_block(512, 512)
        self.stage_f_6 = self.residual_block(512, 512)
        self.stage_f_7 = self.residual_block(512, 512)
        self.stage_f_8 = self.residual_block(512, 512)

        self.stage_b_5 = self.conv2d(512, 1024, 3, 2)

        self.stage_g_1 = self.residual_block(1024, 1024)
        self.stage_g_2 = self.residual_block(1024, 1024)
        self.stage_g_3 = self.residual_block(1024, 1024)
        self.stage_g_4 = self.residual_block(1024, 1024)

        self.yolo_fm_a_1 = self.conv2d(1024, 512, 1, 1)
        self.yolo_fm_a_2 = self.conv2d(512, 1024, 3, 1)
        self.yolo_fm_a_3 = self.conv2d(1024, 512, 1, 1)
        self.yolo_fm_a_4 = self.conv2d(512, 1024, 1, 1)
        self.yolo_fm_a_5 = self.conv2d(1024, 512, 3, 1)
        self.yolo_fm_a_6 = self.conv2d(512, 1024, 3, 1)
        self.yolo_fm_a_7 = nn.Conv2d(1024, 255, 1, 1, 0, bias=True)

        self.yolo_inter_1 = self.conv2d(512, 256, 1, 1)

        self.yolo_fm_b_1 = self.conv2d(768, 256, 1, 1)
        self.yolo_fm_b_2 = self.conv2d(256, 512, 3, 1)
        self.yolo_fm_b_3 = self.conv2d(512, 256, 1, 1)
        self.yolo_fm_b_4 = self.conv2d(256, 512, 1, 1)
        self.yolo_fm_b_5 = self.conv2d(512, 256, 3, 1)
        self.yolo_fm_b_6 = self.conv2d(256, 512, 3, 1)
        self.yolo_fm_b_7 = nn.Conv2d(512, 255, 1, 1, 0, bias=True)

        self.yolo_inter_2 = self.conv2d(256, 128, 1, 1)

        self.yolo_fm_c_1 = self.conv2d(384, 128, 1, 1)
        self.yolo_fm_c_2 = self.conv2d(128, 256, 3, 1)
        self.yolo_fm_c_3 = self.conv2d(256, 128, 1, 1)
        self.yolo_fm_c_4 = self.conv2d(128, 256, 1, 1)
        self.yolo_fm_c_5 = self.conv2d(256, 128, 3, 1)
        self.yolo_fm_c_6 = self.conv2d(128, 256, 3, 1)
        self.yolo_fm_c_7 = nn.Conv2d(256, 255, 1, 1, 0, bias=True)

    @staticmethod
    def conv2d(in_channels, out_channels, kernel_size, stride, bias=False):
        layer_ = []
        if kernel_size > 1:
            layer_.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=1, bias=bias))
        else:
            layer_.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=0, bias=bias))

        layer_ += [nn.BatchNorm2d(num_features=out_channels),
                   nn.LeakyReLU(negative_slope=0.1, inplace=True)]
        return nn.Sequential(*layer_)

    def residual_block(self, in_channels, out_channels):
        layer_ = []
        layer_.extend(self.conv2d(in_channels=in_channels, out_channels=out_channels // 2,
                                  kernel_size=1, stride=1))
        layer_.extend(self.conv2d(in_channels=out_channels // 2, out_channels=out_channels,
                                  kernel_size=3, stride=1))

        return nn.Sequential(*layer_)

    def forward(self, x):
        feature = self.stage_a_1(x)
        feature = self.stage_b_1(feature)
        shortcut = feature

        feature = self.stage_c_1(feature)
        feature = feature + shortcut

        feature = self.stage_b_2(feature)
        shortcut = feature

        feature = self.stage_d_1(feature)
        feature = feature + shortcut
        shortcut = feature
        feature = self.stage_d_2(feature)
        feature = feature + shortcut

        feature = self.stage_b_3(feature)
        shortcut = feature

        feature = self.stage_e_1(feature)
        feature = feature + shortcut
        shortcut = feature
        feature = self.stage_e_2(feature)
        feature = feature + shortcut
        shortcut = feature
        feature = self.stage_e_3(feature)
        feature = feature + shortcut
        shortcut = feature
        feature = self.stage_e_4(feature)
        feature = feature + shortcut
        shortcut = feature
        feature = self.stage_e_5(feature)
        feature = feature + shortcut
        shortcut = feature
        feature = self.stage_e_6(feature)
        feature = feature + shortcut
        shortcut = feature
        feature = self.stage_e_7(feature)
        feature = feature + shortcut
        shortcut = feature
        feature = self.stage_e_8(feature)
        feature = feature + shortcut
        route_1 = feature

        feature = self.stage_b_4(feature)
        shortcut = feature

        feature = self.stage_f_1(feature)
        feature = feature + shortcut
        shortcut = feature
        feature = self.stage_f_2(feature)
        feature = feature + shortcut
        shortcut = feature
        feature = self.stage_f_3(feature)
        feature = feature + shortcut
        shortcut = feature
        feature = self.stage_f_4(feature)
        feature = feature + shortcut
        shortcut = feature
        feature = self.stage_f_5(feature)
        feature = feature + shortcut
        shortcut = feature
        feature = self.stage_f_6(feature)
        feature = feature + shortcut
        shortcut = feature
        feature = self.stage_f_7(feature)
        feature = feature + shortcut
        shortcut = feature
        feature = self.stage_f_8(feature)
        feature = feature + shortcut
        route_2 = feature

        feature = self.stage_b_5(feature)
        shortcut = feature

        feature = self.stage_g_1(feature)
        feature = feature + shortcut
        shortcut = feature
        feature = self.stage_g_1(feature)
        feature = feature + shortcut
        shortcut = feature
        feature = self.stage_g_1(feature)
        feature = feature + shortcut
        shortcut = feature
        feature = self.stage_g_1(feature)
        feature = feature + shortcut

        feature_map_1 = self.yolo_fm_a_1(feature)
        feature_map_1 = self.yolo_fm_a_2(feature_map_1)
        feature_map_1 = self.yolo_fm_a_3(feature_map_1)
        feature_map_1 = self.yolo_fm_a_4(feature_map_1)
        feature_map_1 = self.yolo_fm_a_5(feature_map_1)
        inter_1 = feature_map_1
        feature_map_1 = self.yolo_fm_a_6(feature_map_1)
        feature_map_1 = self.yolo_fm_a_7(feature_map_1)

        inter_1 = self.yolo_inter_1(inter_1)
        inter_1 = nn.functional.upsample(inter_1, scale_factor=2.0)
        feature_map_2 = torch.cat([inter_1, route_1], dim=1)
        feature_map_2 = self.yolo_fm_b_1(feature_map_2)
        feature_map_2 = self.yolo_fm_b_2(feature_map_2)
        feature_map_2 = self.yolo_fm_b_3(feature_map_2)
        feature_map_2 = self.yolo_fm_b_4(feature_map_2)
        feature_map_2 = self.yolo_fm_b_5(feature_map_2)
        inter_2 = feature_map_2
        feature_map_2 = self.yolo_fm_b_6(feature_map_2)
        feature_map_2 = self.yolo_fm_b_7(feature_map_2)
       
        inter_2 = self.yolo_inter_2(inter_2)
        inter_2 = nn.functional.upsample(inter_2, scale_factor=2.0)
        feature_map_3 = torch.cat([inter_2, route_2], dim=1)
        feature_map_3 = self.yolo_fm_c_1(feature_map_3)
        feature_map_3 = self.yolo_fm_c_2(feature_map_3)
        feature_map_3 = self.yolo_fm_c_3(feature_map_3)
        feature_map_3 = self.yolo_fm_c_4(feature_map_3)
        feature_map_3 = self.yolo_fm_c_5(feature_map_3)
        feature_map_3 = self.yolo_fm_c_6(feature_map_3)
        feature_map_3 = self.yolo_fm_c_7(feature_map_3)

        return feature_map_1, feature_map_2, feature_map_3


if __name__ == '__main__':
    darknet53 = Darknet53()
    img = torch.rand(1, 3, 416, 416)
    feature_map_1, feature_map_2, feature_map_3 = darknet53(img)
    print('feature_map_1:', feature_map_1.shape)
    print('feature_map_2:', feature_map_2.shape)
    print('feature_map_3:', feature_map_3.shape)


