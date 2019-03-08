'''VGG11/13/16/19 in Pytorch.'''
from collections import namedtuple
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.layers import FrozenBatchNorm2d
from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import BatchNorm2d
from maskrcnn_benchmark.layers import L2Norm
from maskrcnn_benchmark.modeling.make_layers import group_norm
from maskrcnn_benchmark.utils.registry import Registry
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform
from . import fpn as fpn_module

# VGG stage specification
StageSpec = namedtuple(
    "StageSpec",
    [
        "index",  # Index of the stage, eg 1, 2, ..,. 5
        "block_dims",  # Numer of residual blocks in the stage
        "return_features",  # True => return the last feature map from this stage
    ],
)

# -----------------------------------------------------------------------------
# Standard VGG models
# -----------------------------------------------------------------------------
# VGG11
VGG11StagesTo4 = tuple(
    StageSpec(index=i, block_dims=d, return_features=r)
    for (i, d, r) in ((1, (64, 'M'), False), (2, (128, 'M'), False),
    (3, (256, 256, 256, 'M'), False), (4, (512, 512, 'M'), False), (5, (512, 512), True))
)
# VGG11-FPN (including all stages)
VGG11FPNStagesTo5 = tuple(
    StageSpec(index=i, block_dims=d, return_features=r)
    for (i, d, r) in ((1, (64, 'M'), False), (2, (128, 'M'), True),
    (3, (256, 256, 256, 'M'), True), (4, (512, 512, 'M'), True), (5, (512, 512, 'M'), True))
)
# VGG13
VGG13StagesTo4 = tuple(
    StageSpec(index=i, block_dims=d, return_features=r)
    for (i, d, r) in ((1, (64, 64, 'M'), False), (2, (128, 128, 'M'), False),
    (3, (256, 256, 256, 'M'), False), (4, (512, 512, 'M'), False), (5, (512, 512), True))
)
# VGG13-FPN (including all stages)
VGG13FPNStagesTo5 = tuple(
    StageSpec(index=i, block_dims=d, return_features=r)
    for (i, d, r) in ((1, (64, 64, 'M'), False), (2, (128, 128, 'M'), True),
    (3, (256, 256, 256, 'M'), True), (4, (512, 512, 'M'), True), (5, (512, 512, 'M'), True))
)
# VGG16
VGG16StagesTo4 = tuple(
    StageSpec(index=i, block_dims=d, return_features=r)
    for (i, d, r) in ((1, (64, 64, 'M'), False), (2, (128, 128, 'M'), False),
    (3, (256, 256, 256, 'M'), False), (4, (512, 512, 512, 'M'), False), (5, (512, 512, 512), True))
)
# VGG16-SSD
VGG16SSD = tuple(
    StageSpec(index=i, block_dims=d, return_features=r)
    for (i, d, r) in ((1, (64, 64, 'M'), False), (2, (128, 128, 'M'), False),
    (3, (256, 256, 256, 'C'), False), (4, (512, 512, 512), True), (5, ('M', 512, 512, 512), False))
)
# VGG16-FPN (including all stages)
VGG16FPNStagesTo5 = tuple(
    StageSpec(index=i, block_dims=d, return_features=r)
    for (i, d, r) in ((1, (64, 64, 'M'), False), (2, (128, 128, 'M'), True),
    (3, (256, 256, 256, 'M'), True), (4, (512, 512, 512, 'M'), True), (5, (512, 512, 512, 'M'), True))
)
# VGG19
VGG19StagesTo4 = tuple(
    StageSpec(index=i, block_dims=d, return_features=r)
    for (i, d, r) in ((1, (64, 64, 'M'), False), (2, (128, 128, 'M'), False),
    (3, (256, 256, 256, 256, 'M'), False), (4, (512, 512, 512, 512, 'M'), False), (5, (512, 512, 512, 512), True))
)
# VGG19-FPN (including all stages)
VGG19FPNStagesTo5 = tuple(
    StageSpec(index=i, block_dims=d, return_features=r)
    for (i, d, r) in ((1, (64, 64, 'M'), False), (2, (128, 128, 'M'), True),
    (3, (256, 256, 256, 256, 'M'), True), (4, (512, 512, 512, 512, 'M'), True), (5, (512, 512, 512, 512, 'M'), True))
)

class VGG(nn.Module):
    extra_setting = {
        300: ((256, 'S', 512), (128, 'S', 256), (128, 256), (128, 256)),
        512: ((256, 'S', 512), (128, 'S', 256), (128, 'S', 256), (128, 'S', 256), (128, 'S', 256)),
    }

    def __init__(self, cfg):
        super(VGG, self).__init__()
        stage_specs = _STAGE_SPECS[cfg.MODEL.BACKBONE.CONV_BODY]
        self.stages = []
        self.return_features = {}
        self.out_channels = []
        in_channels = 3
        self.L2Norm = L2Norm(512, 20)
        for stage_spec in stage_specs:
            name = "layer" + str(stage_spec.index)
            module, in_channels = _make_stage(stage_spec.block_dims, in_channels)
            self.add_module(name, module)
            self.stages.append(name)
            self.return_features[name] = stage_spec.return_features
            self.out_channels = in_channels
        # for SSD, add extra layers
        if cfg.MODEL.META_ARCHITECTURE == 'SSD':
            layers = [nn.MaxPool2d(kernel_size=3, stride=1, padding=1), 
                      Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
                      nn.ReLU(inplace=True),
                      Conv2d(1024, 1024, kernel_size=1),
                      nn.ReLU(inplace=True)]
            name = "extra_block1"
            self.add_module(name, nn.Sequential(*layers))
            self.stages.append(name)
            self.return_features[name] = True
            in_channels = 1024
            for i, setting in enumerate(self.extra_setting[cfg.MODEL.SSD.INPUT_SIZE]):
                module, in_channels = _make_extra_stage(setting, in_channels)
                name = "extra_block" + str(i + 2)
                self.add_module(name, module)
                self.stages.append(name)
                self.return_features[name] = True
            self.out_channels = in_channels

  
    def forward(self, x):
        outputs = []
        for stage_name in self.stages:
            x = getattr(self, stage_name)(x)
            if self.return_features[stage_name] and len(outputs) == 0:
                x = self.L2Norm(x)
            if self.return_features[stage_name]:
                outputs.append(x)
        return outputs


def _make_extra_stage(setting, in_channels):
    layers = []
    kernel = 1
    stride = 1
    padding = 0
    for dim in setting:
        if dim == 'S':
            stride = 2
            padding = 1
            continue
        else:
            layers += [Conv2d(in_channels, dim, kernel_size=kernel, stride=stride, padding=padding)]
            in_channels = dim
            stride = 1
            padding = 0
        kernel = 3

    return nn.Sequential(*layers), in_channels


def _make_stage(block_dims, in_channels):
    layers = []
    for dim in block_dims:
        if dim == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif dim == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            layers += [Conv2d(in_channels, dim, kernel_size=3, padding=1),
                       BatchNorm2d(dim),
                       nn.ReLU(inplace=True)]
            in_channels = dim

    return nn.Sequential(*layers), in_channels


@registry.BACKBONES.register("VGG11")
@registry.BACKBONES.register("VGG13")
@registry.BACKBONES.register("VGG16")
@registry.BACKBONES.register("VGG16-SSD")
@registry.BACKBONES.register("VGG19")
def build_vgg_backbone(cfg):
    body = VGG(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = body.out_channels
    return model


@registry.BACKBONES.register("VGG11-FPN")
@registry.BACKBONES.register("VGG13-FPN")
@registry.BACKBONES.register("VGG16-FPN")
@registry.BACKBONES.register("VGG19-FPN")
def build_vgg_fpn_backbone(cfg):
    body = VGG(cfg)
    in_channels_list = [128, 256, 512, 512]
    out_channels = body.out_channels
    fpn = fpn_module.FPN(
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


_STAGE_SPECS = Registry({
    "VGG11": VGG11StagesTo4,
    "VGG11-FPN": VGG11FPNStagesTo5,
    "VGG13": VGG13StagesTo4,
    "VGG13-FPN": VGG13FPNStagesTo5,
    "VGG16": VGG16StagesTo4,
    "VGG16-SSD": VGG16SSD,
    "VGG16-FPN": VGG16FPNStagesTo5,
    "VGG19": VGG19StagesTo4,
    "VGG19-FPN": VGG19FPNStagesTo5,
})
