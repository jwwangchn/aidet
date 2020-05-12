import mmcv
import numpy as np
import pycocotools.mask as mask_util
import torch

import wwtool
from mmdet.core import mask_target

from ..registry import HEADS
from .fcn_mask_head import FCNMaskHead


@HEADS.register_module
class CenterMapHead(FCNMaskHead):

    def __init__(self, with_conv_res=True, *args, **kwargs):
        super(CenterMapHead, self).__init__(*args, **kwargs)

    def loss(self, mask_pred, mask_targets, labels, mask_weights):
        mask_targets = mask_targets / 255.0
        if mask_weights is not None:
            mask_weights = mask_weights / 255.0 + 1.0
        loss = dict()
        if self.class_agnostic:
            loss_mask = self.loss_mask(mask_pred, 
                                       mask_targets,
                                       torch.zeros_like(labels),
                                       mask_weights=mask_weights)
        else:
            loss_mask = self.loss_mask(mask_pred, mask_targets, labels, mask_weights=mask_weights)
        loss['loss_mask'] = loss_mask
        return loss

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg,
                      ori_shape, scale_factor, rescale):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor or ndarray): shape (n, #class+1, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            img_shape (Tensor): shape (3, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape: original image size

        Returns:
            list[list]: encoded masks
        """
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.cpu().numpy()
        assert isinstance(mask_pred, np.ndarray)
        # when enabling mixed precision training, mask_pred may be float16
        # numpy array
        mask_pred = mask_pred.astype(np.float32)

        cls_segms = [[] for _ in range(self.num_classes - 1)]
        bboxes = det_bboxes.cpu().numpy()[:, :4]
        labels = det_labels.cpu().numpy() + 1

        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0

        for i in range(bboxes.shape[0]):
            if not isinstance(scale_factor, (float, np.ndarray)):
                scale_factor = scale_factor.cpu().numpy()
            bbox = (bboxes[i, :] / scale_factor).astype(np.int32)
            label = labels[i]
            w = max(bbox[2] - bbox[0] + 1, 1)
            h = max(bbox[3] - bbox[1] + 1, 1)

            if not self.class_agnostic:
                mask_pred_ = mask_pred[i, label, :, :]
            else:
                mask_pred_ = mask_pred[i, 0, :, :]

            bbox_mask = mmcv.imresize(mask_pred_, (w, h))
            
            # visualization
            # bbox_mask_ = np.pad(bbox_mask, ((25, 25), (25, 25)), 'constant', constant_values = (0, 0))
            # wwtool.show_grayscale_as_heatmap(bbox_mask_)

            bbox_mask = (bbox_mask > rcnn_test_cfg.mask_thr_binary).astype(
                np.uint8)
                
            if rcnn_test_cfg.get('crop_mask', False):
                im_mask = bbox_mask
            else:
                im_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                im_mask[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = bbox_mask

            if rcnn_test_cfg.get('rle_mask_encode', True):
                rle = mask_util.encode(
                    np.array(im_mask[:, :, np.newaxis], order='F'))[0]
                cls_segms[label - 1].append(rle)
            else:
                cls_segms[label - 1].append(im_mask)

        return cls_segms
