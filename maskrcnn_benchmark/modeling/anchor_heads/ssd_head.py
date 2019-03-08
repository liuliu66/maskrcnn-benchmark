import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling import registry
from .loss import make_multibox_loss_evaluator
from .inference import make_multibox_post_processor


@registry.ANCHOR_HEADS.register("SSDHead")
class SSDHead(nn.Module):
    def __init__(self, cfg, in_channels):
        super(SSDHead, self).__init__()
        self.in_channels_list = cfg.MODEL.SSD.ANCHOR.IN_CHANNELS_LIST
        self.anchor_num_list = [len(x) * 2 + 2 for x in cfg.MODEL.SSD.ANCHOR.ASPECT_RATIOS]
        self.num_classes = cfg.MODEL.SSD.BOX_HEAD.NUM_CLASSES
        cls_layers = []
        reg_layers = []
        for in_channel,anchor_num in zip(self.in_channels_list,
                                         self.anchor_num_list):
            cls_layers += [nn.Conv2d(in_channel,
                                     anchor_num * self.num_classes,
                                     kernel_size=3,
                                     padding=1,
                                     stride=1)]
            reg_layers += [nn.Conv2d(in_channel,
                                     anchor_num * 4,
                                     kernel_size=3,
                                     padding=1,
                                     stride=1)]
        self.cls_conv = nn.ModuleList(cls_layers)
        self.bbox_conv = nn.ModuleList(reg_layers)
        self.loss_evaluator = make_multibox_loss_evaluator(cfg)
        self.post_processor = make_multibox_post_processor(cfg)
        self.softmax = nn.Softmax(dim=-1)
        

    def forward(self, feats, targets):
        cls_scores = []
        bbox_pred = []
        for feat, cls_conv, reg_conv in zip(feats,
                                            self.cls_conv,
                                            self.bbox_conv):
            cls_scores.append(cls_conv(feat))
            bbox_pred.append(reg_conv(feat))

        cls_scores = torch.cat([o.permute(0, 2, 3, 1).contiguous().view(o.size(0), -1) for o in cls_scores], 1)
        bbox_pred = torch.cat([o.permute(0, 2, 3, 1).contiguous().view(o.size(0), -1) for o in bbox_pred], 1)
        cls_scores = cls_scores.view(cls_scores.size(0), -1, self.num_classes)
        bbox_pred = bbox_pred.view(bbox_pred.size(0), -1, 4)

        if not self.training:
            cls_scores = self.softmax(cls_scores)
            result = self.post_processor(cls_scores, bbox_pred)
            return (
                result, {}
            )

        loss_cls, loss_bbox = self.loss_evaluator(cls_scores, bbox_pred, targets)

        return (
            (cls_scores, bbox_pred),
            dict(loss_classifier=loss_cls, loss_box_reg=loss_bbox)
        )




