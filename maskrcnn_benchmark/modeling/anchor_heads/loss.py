import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.anchor_heads.box_utils import match, log_sum_exp, generate_anchors


class MultiBoxLossComputation(object):
    """
    This class computes the SSD Multibox loss.
    """

    def __init__(self, cfg):
        self.device = cfg.MODEL.DEVICE
        self.anchors = generate_anchors(cfg).to(self.device)
        self.num_classes = cfg.MODEL.SSD.BOX_HEAD.NUM_CLASSES
        self.negpos_ratio = cfg.MODEL.SSD.BOX_HEAD.NEGPOS_RATIO
        self.threshold = cfg.MODEL.SSD.BOX_HEAD.THRESH_TRAIN
        self.variance = cfg.MODEL.SSD.ANCHOR.VARIANCE


    def __call__(self, cls_scores, bbox_pred, targets):
        """
        Arguments:
            anchors (list[BoxList])
            objectness (list[Tensor])
            box_regression (list[Tensor])
            targets (list[BoxList])

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor
        """
        num_batch = cls_scores.size(0)
        anchors = self.anchors[:bbox_pred.size(1), :]
        num_anchors = (anchors.size(0))

        cls_labels = torch.LongTensor(num_batch, num_anchors).to(self.device)
        bbox_gt = torch.Tensor(num_batch, num_anchors, 4).to(self.device)
        for i in range(num_batch):
            truths = targets[i].bbox
            labels = targets[i].get_field('labels') - 1
            match(self.threshold, truths, anchors.data, self.variance, labels,
                  bbox_gt, cls_labels, i)

        pos = cls_labels > 0
        num_pos = pos.sum(dim=1, keepdim=True)

        # bbox Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(bbox_pred)
        loss_bbox = smooth_l1_loss(bbox_pred[pos_idx].view(-1, 4),
                                   bbox_gt[pos_idx].view(-1, 4),
                                   beta=1. / 9,
                                   size_average=False
                                   )

        # Compute max conf across batch for hard negative mining
        batch_conf = cls_scores.view(-1, self.num_classes)
        loss_cls = F.cross_entropy(batch_conf, cls_labels.view(-1), ignore_index=-1, reduction='none')
        loss_cls = loss_cls.view(num_batch, -1)

        # Hard Negative Mining
        loss_cls = loss_cls.view(pos.size()[0], pos.size()[1])  # add line
        pos_loss_cls = loss_cls[pos]
        loss_cls[pos] = 0  # filter out pos boxes for now
        _, loss_idx = loss_cls.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        neg_loss_cls = loss_cls[neg]

        loss_cls = pos_loss_cls.sum() + neg_loss_cls.sum()

        N = num_pos.data.sum().float()
        loss_bbox = loss_bbox.double()
        loss_cls = loss_cls.double()
        loss_bbox /= N
        loss_cls /= N

        return loss_cls, loss_bbox



def make_multibox_loss_evaluator(cfg):
    loss_evaluator = MultiBoxLossComputation(cfg)
    return loss_evaluator
