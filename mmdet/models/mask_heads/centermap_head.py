from mmdet.ops import ConvModule
from ..registry import HEADS
from .fcn_mask_head import FCNMaskHead


@HEADS.register_module
class CenterMapHead(FCNMaskHead):

    def __init__(self, with_conv_res=True, *args, **kwargs):
        super(CenterMapHead, self).__init__(*args, **kwargs)

    def loss(self, mask_pred, mask_targets, labels):
        mask_targets = mask_targets / 255.0
        loss = dict()
        if self.class_agnostic:
            loss_mask = self.loss_mask(mask_pred, mask_targets,
                                       torch.zeros_like(labels))
        else:
            loss_mask = self.loss_mask(mask_pred, mask_targets, labels)
        loss['loss_mask'] = loss_mask
        return loss
