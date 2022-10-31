#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .darknet import CSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv
from .yolo_pafpn_attention import ECA
from .yolo_pafpn_asff import ASFF


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

        # Attention channel size
        self.neck_channels = [512, 256, 512, 1024]

        # ECA
        # dark5 1024
        self.eca_1 = ECA(int(in_channels[2] * width))
        # dark4 512
        self.eca_2 = ECA(int(in_channels[1] * width))
        # dark3 256
        self.eca_3 = ECA(int(in_channels[0] * width))
        # FPN CSPLayer 512
        self.eca_fp1 = ECA(int(self.neck_channels[0] * width))
        # PAN CSPLayer pan_out2 downsample 256
        self.eca_pa1 = ECA(int(self.neck_channels[1] * width))
        # PAN CSPLayer pan_out1 downsample 512
        self.eca_pa2 = ECA(int(self.neck_channels[2] * width))
        # PAN CSPLayer pan_out0 1024
        self.eca_pa3 = ECA(int(self.neck_channels[3] * width))

        # ASFF
        self.asff_1 = ASFF(level=0, multiplier=width)
        self.asff_2 = ASFF(level=1, multiplier=width)
        self.asff_3 = ASFF(level=2, multiplier=width)

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

        # ECA
        x0 = self.eca_1(x0)
        x1 = self.eca_2(x1)
        x2 = self.eca_3(x2)

        # FPN

        # dark5
        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        # upsample
        f_out0 = self.upsample(fpn_out0)  # 512/16
        # dark4 + upsample
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16
        # ECA
        f_out0 = self.eca_fp1(f_out0)

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        # upsample
        f_out1 = self.upsample(fpn_out1)  # 256/8
        # dark3 + upsample
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        #YOLO HEAD
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8
        # ECA
        pan_out2 = self.eca_pa1(pan_out2)

        # PAN

        # pan_out2 downsample
        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        # pan_out1 downsample + CSPLayer fpn_out1
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        # YOLO HEAD
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16
        # ECA
        pan_out1 = self.eca_pa2(pan_out1)

        # downsample
        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        # p_out0 downsample + fpn_out0
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        # YOLO HEAD
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32
        # ECA
        pan_out0 = self.eca_pa3(pan_out0)

        outputs = (pan_out2, pan_out1, pan_out0)

        # ASFF
        pan_out0 = self.asff_1(outputs)
        pan_out1 = self.asff_2(outputs)
        pan_out2 = self.asff_3(outputs)
        outputs = (pan_out2, pan_out1, pan_out0)

        return outputs
