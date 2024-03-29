import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES
from .utils import weight_reduce_loss
from .utils import weighted_loss

def mask_centermap(pred, target, label, reduction='mean', avg_factor=None, mask_weights=None):
    # TODO: handle these two reserved arguments
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.mse_loss(
        pred_slice, target, reduction='mean')[None]

@weighted_loss
def weighted_mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')

def mask_weight_centermap(pred, target, label, reduction='mean', avg_factor=None, mask_weights=None):
    # TODO: handle these two reserved arguments
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return weighted_mse_loss(
        pred_slice, target, weight=mask_weights, reduction='mean')[None]


@LOSSES.register_module
class CenterMapLoss(nn.Module):

    def __init__(self,
                 use_mask_weight=False,
                 use_mask=False,
                 reduction='mean',
                 loss_weight=1.0):
        super(CenterMapLoss, self).__init__()
        assert (use_mask_weight is False) or (use_mask is False)
        self.use_mask_weight = use_mask_weight
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight

        if self.use_mask_weight:
            self.cls_criterion = mask_weight_centermap
        elif self.use_mask:
            self.cls_criterion = mask_centermap
        else:
            self.cls_criterion = None

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls
