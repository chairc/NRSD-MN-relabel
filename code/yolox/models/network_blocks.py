#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn


class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


# 激活函数选择
def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


# 基础卷积
class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups,
                              bias=bias, )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(in_channels, in_channels, ksize=ksize, stride=stride, groups=in_channels, act=act, )
        self.pconv = BaseConv(in_channels, out_channels, ksize=1, stride=1, groups=1, act=act)

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, depthwise=False, act="silu", ):
        super().__init__()

        # source bottle function
        # expansion=0.5
        # hidden_channels = int(out_channels * expansion)
        # Conv = DWConv if depthwise else BaseConv
        # self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        # self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)

        # test-1
        # expansion=4
        # hidden_channels = int(in_channels * 4)
        # self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=1,
        #                        bias=False)
        # self.bn = nn.BatchNorm2d(in_channels)
        # self.conv2 = nn.Linear(in_channels, hidden_channels)
        # self.act = get_activation(act, inplace=True)
        # self.conv3 = nn.Linear(hidden_channels, out_channels)

        # test-2
        # expansion=4
        # hidden_channels = int(in_channels * 4)
        # self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=1,
        #                           bias=False)
        # self.bn = nn.BatchNorm2d(in_channels)
        # self.conv2 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0, groups=1,
        #                           bias=False)
        # self.act = get_activation(act, inplace=True)
        # self.conv3 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1,
        #                           bias=False)

        # test-3
        # expansion=0.5
        # hidden_channels = int(out_channels * expansion)
        # self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_channels)
        # self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, groups=1,
        #                        bias=False)
        # self.act = get_activation(act, inplace=True)
        # self.conv3 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=False)

        # our bottleneck function
        # expansion=0.5
        hidden_channels = int(out_channels * expansion)
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.bn = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, groups=1,
                               bias=False)
        self.act = get_activation(act, inplace=True)
        # 使用短连接
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        # source
        # y = self.conv2(self.conv1(x))

        # test-1
        # y = self.conv1(x)
        # y = self.bn(y)
        # y = y.permute(0, 2, 3, 1)
        # y = self.conv2(y)
        # y = self.act(y)
        # y = self.conv3(y)
        # y = y.permute(0, 3, 1, 2)

        # test-2
        # y = self.conv1(x)
        # y = self.bn(y)
        # y = self.conv2(y)
        # y = self.act(y)
        # y = self.conv3(y)

        # test-3
        # y = self.conv1(x)
        # y = self.bn(y)
        # y = self.conv2(y)
        # y = self.act(y)
        # y = self.conv3(y)

        # test-4
        y = self.conv1(x)
        y = self.bn(y)
        y = self.conv2(y)
        y = self.act(y)

        if self.use_add:
            y = y + x
        return y


class ResLayer(nn.Module):
    """
        Residual layer with `in_channels` inputs.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(in_channels, mid_channels, ksize=1, stride=1, act="lrelu")
        self.layer2 = BaseConv(mid_channels, in_channels, ksize=3, stride=1, act="lrelu")

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out


# SPPBottleneck
class SPPBottleneck(nn.Module):
    """
        Spatial pyramid pooling layer used in YOLOv3-SPP
        https://arxiv.org/abs/1406.4729
    """

    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


# SPPFBottleneck
class SPPFBottleneck(nn.Module):
    """
        https://github.com/ultralytics/yolov5/blob/master/models/common.py
        https://github.com/ultralytics/yolov5/blob/3eefab1bb109214a614485b6c5f80f22c122f2b2/models/common.py#L182
    """
    # kernel_sizes=5等价于SPP(kernel_sizes=(5, 9, 13))
    def __init__(self, in_channels, out_channels, kernel_sizes=5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, 1)
        self.conv2 = BaseConv(hidden_channels * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=kernel_sizes, stride=1, padding=kernel_sizes // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.conv2(torch.cat((x, y1, y2, self.m(y2)), dim=1))


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, depthwise=False, act="silu", ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        # Channel becomes 4 times, length and width are halved
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)
