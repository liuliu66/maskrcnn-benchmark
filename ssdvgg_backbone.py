import cv2
import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model

cfg.merge_from_file('experiments/cfgs/e2e_ssd300_vgg16.yaml')
model = build_detection_model(cfg)

print(model)
#print(model.backbone.out_channels)
