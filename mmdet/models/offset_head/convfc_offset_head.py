import mmcv
import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

from mmcv.cnn import kaiming_init, normal_init
from mmdet.core import auto_fp16, force_fp32, offset_target, delta2offset, offset2delta
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
                 target_means=[0., 0.],
                 target_stds=[1.0, 1.0],
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_offset=dict(
                     type='MSELoss', loss_weight=1.0)):
        super(ConvFCOffsetHead, self).__init__()
        
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.target_means = target_means
        self.target_stds = target_stds
        self.loss_offset = build_loss(loss_offset)

        self.convs = nn.ModuleList()
        for i in range(num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    conv_kernel_size,
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
        for module_list in [self.fcs, self.fc_offset]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

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
        offset_targets = offset_target(pos_proposals, 
                                       pos_assigned_gt_inds,
                                       gt_offsets,
                                       rcnn_train_cfg,
                                       target_means=self.target_means,
                                       target_stds=self.target_stds)
        return offset_targets

    def loss(self, offset_pred, offset_targets, labels):
        loss = dict()
        loss_offset = self.loss_offset(offset_pred,
                                      offset_targets)
        loss['loss_offset'] = loss_offset
        return loss

    def get_offsets(self, 
                    offset_pred, 
                    det_bboxes, 
                    rcnn_test_cfg, 
                    scale_factor, 
                    rescale,
                    img_shape=[1024, 1024]):
        if offset_pred is not None:
            offsets = delta2offset(det_bboxes, 
                                   offset_pred, 
                                   self.target_means, 
                                   self.target_stds, 
                                   img_shape)
        else:
            offsets = torch.zeros((det_bboxes.size()[0], 2))

        if isinstance(offsets, torch.Tensor):
            offsets = offsets.cpu().numpy()
        assert isinstance(offsets, np.ndarray)

        offsets = offsets.astype(np.float32)

        return offsets
