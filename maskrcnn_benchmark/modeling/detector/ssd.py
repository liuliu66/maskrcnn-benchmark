"""
Implements the SSD framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..anchor_heads.anchor_heads import build_anchor_heads

class SSD(nn.Module):
    def __init__(self, cfg):
        super(SSD, self).__init__()
        self.backbone = build_backbone(cfg)
        self.bbox_head = build_anchor_heads(cfg, self.backbone.out_channels)
        if not self.training:
            self.softmax = nn.Softmax(dim=-1)


    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        result, detector_losses = self.bbox_head(features, targets)

        if self.training:
            (cls_scores, bbox_pred) = result
            losses = {}
            losses.update(detector_losses)
            return losses

        return result
