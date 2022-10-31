#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

from .build import *
from .darknet import CSPDarknet
from .losses import IOUloss
from .yolo_head import YOLOXHead
from .yolox import YOLOX

# 原版
# from .yolo_pafpn import YOLOPAFPN
# 原版 + ASFF
# from .yolo_pafpn_asff import YOLOPAFPN
# 原版 + Attention
# from .yolo_pafpn_attention import YOLOPAFPN
# 融合版（ASFF + Attention）
from .yolo_pafpn_best import YOLOPAFPN
