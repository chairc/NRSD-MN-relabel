#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
from torchsummary import summary
from loguru import logger
from yolox.models.yolox import YOLOX
from exps.default.yolox_s import MyExp
from yolox.models.yolo_head import YOLOXHead
from yolox.models.yolo_pafpn import YOLOPAFPN

# 打印网络结构和参数
if __name__ == "__main__":
    # 需要使用device来指定网络在GPU还是CPU运行
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolox = YOLOX().to(device)
    print(yolox)
