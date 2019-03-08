# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.anchor_heads.box_utils import decode, generate_anchors



class PostProcessor(nn.Module):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, cfg):
        super(PostProcessor, self).__init__()
        self.device = cfg.MODEL.DEVICE
        self.num_classes = cfg.MODEL.SSD.BOX_HEAD.NUM_CLASSES
        self.background_label = 0
        self.top_k = cfg.MODEL.SSD.BOX_HEAD.TOP_K
        # Parameters used in nms.
        self.nms_thresh = cfg.MODEL.SSD.BOX_HEAD.NMS
        if self.nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = cfg.MODEL.SSD.BOX_HEAD.THRESH_TEST
        self.variance = cfg.MODEL.SSD.ANCHOR.VARIANCE
        self.anchors = generate_anchors(cfg).to(self.device)
        self.image_shape = (cfg.MODEL.SSD.INPUT_SIZE, ) * 2

    def forward(self, cls_scores, bbox_pred):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        num_batch = bbox_pred.size(0)  # batch size
        anchors = self.anchors[:bbox_pred.size(1), :]
        num_anchors = anchors.size(0)
        #cls_scores = cls_scores.view(num_batch, num_anchors,
                                    #self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        scale = torch.Tensor([self.image_shape[1], self.image_shape[0],
                             self.image_shape[1], self.image_shape[0]]).to(self.device)
        results = []
        for i in range(num_batch):
            decoded_boxes = decode(bbox_pred[i], anchors, self.variance) * scale
            prob = cls_scores[i]
            boxlist = self.prepare_boxlist(decoded_boxes, prob, self.image_shape)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = self.filter_results(boxlist)
            results.append(boxlist)
        return results

    def prepare_boxlist(self, boxes, scores, image_shape):
        """
        :param boxes: predicted boxes for current batch shape: [8732, 4]
        :param scores: predicted boxes for current batch shape: [8732, 2]
        :param image_shape: image_size : (300, 300)
        :return: boxlist for current batch
        """
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1, self.num_classes)
        boxlist = BoxList(boxes, image_shape, mode="xyxy")
        boxlist.add_field("scores", scores)
        return boxlist

    def filter_results(self, boxlist):
        """
        :param boxlist: boxlist for current batch
        :param num_classes: num classes
        :return: detection result after nms and top_k filter
        """
        boxes = boxlist.bbox.reshape(-1, 4)
        scores = boxlist.get_field("scores")
        result = []
        # For each class, perform nms
        conf_scores = scores.clone()

        for cl in range(1, self.num_classes):
            c_mask = conf_scores[:,cl].gt(self.conf_thresh)
            scores = conf_scores[:,cl][c_mask]
            if scores.dim() == 0:
                continue
            l_mask = c_mask.unsqueeze(1).expand_as(boxes)
            boxes = boxes[l_mask].view(-1, 4)
            boxlist_for_class = BoxList(boxes, boxlist.size, mode="xyxy")
            boxlist_for_class.add_field("scores", scores)
            # idx of highest scoring and non-overlapping boxes per class
            boxlist_for_class = boxlist_nms(
                boxlist_for_class, self.nms_thresh
            )
            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field(
                "labels", torch.full((num_labels,), cl, dtype=torch.int64, device=self.device)
            )
            result.append(boxlist_for_class)

        result = cat_boxlist(result)
        number_of_detections = len(result)

        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.top_k > 0:
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.top_k + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]

        return result


def make_multibox_post_processor(cfg):

    postprocessor = PostProcessor(cfg)
    return postprocessor
