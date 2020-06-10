import mmcv
import numpy as np
import pycocotools.mask as mask_util
import torch

import wwtool
from mmdet.core import mask_target

from ..registry import HEADS
from mmdet.models.mask_heads.fcn_mask_head import FCNMaskHead
from ..builder import build_loss


@HEADS.register_module
class MaskFootprintHead(FCNMaskHead):
    def __init__(self,
                loss_footprint=dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
                *args, 
                **kwargs):
        super(MaskFootprintHead, self).__init__(*args, **kwargs)
        self.loss_footprint = build_loss(loss_footprint)

    def loss(self, footprint_pred, footprint_targets, labels):
            loss = dict()
            if self.class_agnostic:
                loss_footprint = self.loss_mask(footprint_pred, footprint_targets,
                                        torch.zeros_like(labels))
            else:
                loss_footprint = self.loss_mask(footprint_pred, footprint_targets, labels)
            loss['loss_footprint'] = loss_footprint
            return loss


