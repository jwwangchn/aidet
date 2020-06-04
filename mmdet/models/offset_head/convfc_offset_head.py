import mmcv
import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

from mmcv.cnn import kaiming_init, normal_init
from mmdet.core import auto_fp16, force_fp32, mask_target
from mmdet.ops import ConvModule, build_upsample_layer
from mmdet.ops.carafe import CARAFEPack
from ..builder import build_loss
from ..registry import HEADS


@HEADS.register_module
class ConvFCOffsetHead(nn.Module):

    def __init__(self,
                 num_convs=4,
                 num_fcs=2,
                 roi_feat_size=7,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_offset=dict(
                     type='MSELoss', loss_weight=1.0)):
        super(ConvFCOffsetHead, self).__init__()
        
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.loss_mask = build_loss(loss_offset)

        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))
    
        roi_feat_size = _pair(roi_feat_size)
        roi_feat_area = roi_feat_size[0] * roi_feat_size[1]
        self.fcs = nn.ModuleList()
        for i in range(num_fcs):
            in_channels = (
                self.conv_out_channels *
                roi_feat_area if i == 0 else self.fc_out_channels)
            self.fcs.append(nn.Linear(in_channels, self.fc_out_channels))

        self.fc_offset = nn.Linear(self.fc_out_channels, 2)
        self.relu = nn.ReLU()
        self.loss_offset = build_loss(loss_offset)

    def init_weights(self):
        for conv in self.convs:
            kaiming_init(conv)
        for fc in self.fcs:
            kaiming_init(
                fc,
                a=1,
                mode='fan_in',
                nonlinearity='leaky_relu',
                distribution='uniform')
        normal_init(self.fc_offset, std=0.01)

    @auto_fp16()
    def forward(self, x):
        for conv in self.convs:
            x = self.relu(conv(x))
        x = x.view(x.size(0), -1)
        for fc in self.fcs:
            x = self.relu(fc(x))
        offset = self.fc_offset(x)
        return offset

    def get_target(self, sampling_results, gt_offsets, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
                                   gt_masks, rcnn_train_cfg)
        return mask_targets

    def loss(self, offset_pred, offset_targets, labels):
        loss = dict()
        loss_offset = self.loss_iou(offset_pred,
                                      offset_targets)
        loss['loss_offset'] = loss_offset
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
            mask_pred = mask_pred.sigmoid().cpu().numpy()
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
