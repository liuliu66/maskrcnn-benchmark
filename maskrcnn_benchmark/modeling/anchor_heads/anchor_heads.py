# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling import registry
from .ssd_head import *

def build_anchor_heads(cfg, in_channels):
    func = registry.ANCHOR_HEADS[
        cfg.MODEL.SSD.BOX_HEAD.PREDICTOR
    ]
    return func(cfg, in_channels)