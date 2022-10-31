#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .darknet import CSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv
from .yolo_attention import CBAM, SE, ECA


class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
            self,
            depth=1.0,
            width=1.0,
            in_features=("dark3", "dark4", "dark5"),
            in_channels=[256, 512, 1024],
            depthwise=False,
            act="silu",
    ):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # 注意力通道大小
        self.neck_channels = [512, 256, 512, 1024]
        # CBAM
        # # 对应dark5输出的1024维度通道
        # self.cbam_1 = CBAM(int(in_channels[2] * width))
        # # 对应dark4输出的512维度通道
        # self.cbam_2 = CBAM(int(in_channels[1] * width))
        # # 对应dark3输出的256维度通道
        # self.cbam_3 = CBAM(int(in_channels[0] * width))
        # # 对应FPN第一个CSPLayer维度通道
        # self.cbam_fp1 = ECA(int(self.neck_channels[0] * width))
        # # 对应PAN第一个CSPLayer输出pan_out2和下采样维度通道
        # self.cbam_pa1 = ECA(int(self.neck_channels[1] * width))
        # # 对应PAN第二个CSPLayer输出pan_out1和下采样维度通道
        # self.cbam_pa2 = ECA(int(self.neck_channels[2] * width))
        # # 对应PAN第三个CSPLayer输出pan_out0维度通道
        # self.cbam_pa3 = ECA(int(self.neck_channels[3] * width))

        # ECA
        # 对应dark5输出的维度通道（1024）
        self.eca_1 = ECA(int(in_channels[2] * width))
        # 对应dark4输出的维度通道（512）
        self.eca_2 = ECA(int(in_channels[1] * width))
        # 对应dark3输出的维度通道（256）
        self.eca_3 = ECA(int(in_channels[0] * width))
        # 对应FPN第一个CSPLayer维度通道（512）
        self.eca_fp1 = ECA(int(self.neck_channels[0] * width))
        # 对应PAN第一个CSPLayer输出pan_out2和下采样维度通道（256）
        self.eca_pa1 = ECA(int(self.neck_channels[1] * width))
        # 对应PAN第二个CSPLayer输出pan_out1和下采样维度通道（512）
        self.eca_pa2 = ECA(int(self.neck_channels[2] * width))
        # 对应PAN第三个CSPLayer输出pan_out0维度通道（1024）
        self.eca_pa3 = ECA(int(self.neck_channels[3] * width))

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        # backbone
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features

        # 对输入的特征图使用注意力机制
        # CBAM
        # x0 = self.cbam_1(x0)
        # x1 = self.cbam_2(x1)
        # x2 = self.cbam_3(x2)

        # ECA
        x0 = self.eca_1(x0)
        x1 = self.eca_2(x1)
        x2 = self.eca_3(x2)

        # FPN

        # dark5输出的特征层进行卷积
        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        # 进行上采样
        f_out0 = self.upsample(fpn_out0)  # 512/16
        # 将dark4输出的特征层与上采样结果进行叠加
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        # 将叠加后的的结果进行一个CSPLayer操作
        # 大小不变，通道缩减
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16
        # ECA
        f_out0 = self.eca_fp1(f_out0)

        # 对CSPLayer结果进行基础卷积
        # 大小不变，通道缩减
        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        # 对输出的基础卷积进行上采样
        f_out1 = self.upsample(fpn_out1)  # 256/8
        # 将dark3输出的特征层与上采样结果进行叠加
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        # 输出结果到YOLO HEAD
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8
        # ECA
        pan_out2 = self.eca_pa1(pan_out2)

        # PAN

        # 对pan_out2输出结果进行下采样
        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        # 将pan_out1输出的下采样结果与CSPLayer输出fpn_out1的结果进行叠加
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        # 对CSPLayer结果进行基础卷积
        # 特征提取（普通卷积 + 标准化 + 激活函数）
        # 输出结果到YOLO HEAD
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16
        # ECA
        pan_out1 = self.eca_pa2(pan_out1)

        # 下采样，降低特征图大小
        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        # 将p_out0下采样结果与最下层输出的卷积进行叠加
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        # 对CSPLayer结果进行基础卷积
        # 特征提取（普通卷积 + 标准化 + 激活函数）
        # 输出结果到YOLO HEAD
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32
        # ECA
        pan_out0 = self.eca_pa3(pan_out0)

        outputs = (pan_out2, pan_out1, pan_out0)

        return outputs
