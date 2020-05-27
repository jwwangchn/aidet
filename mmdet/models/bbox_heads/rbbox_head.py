import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import HEADS
from .bbox_head import BBoxHead
from ..builder import build_loss
from ..losses import accuracy

from mmdet.ops import ConvModule
from mmdet.core import delta2thetaobb, delta2pointobb, delta2hobb
from mmdet.core import thetaobb_rescale, pointobb_rescale, hobb_rescale
from mmdet.core import thetaobb_nms_by_bbox_nms
from mmdet.core import rbbox_target


@HEADS.register_module
class RBBoxHead(BBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

                                /-> cls convs -> cls fcs -> cls
    shared convs -> shared fcs
                                \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 out_dim_reg=5,
                 encode=None,
                 loss_rbbox_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_rbbox=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 conv_cfg=None,
                 norm_cfg=None,
                 *args,
                 **kwargs):
        super(RBBoxHead, self).__init__(*args, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.out_dim_reg = out_dim_reg
        self.encode = encode
        self.loss_rbbox_cls = loss_rbbox_cls
        self.loss_rbbox = loss_rbbox
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.loss_rbbox_cls = build_loss(loss_rbbox_cls)
        self.loss_rbbox = build_loss(loss_rbbox)

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes)
        if self.with_reg:
            out_dim_reg = (self.out_dim_reg if self.reg_class_agnostic else self.out_dim_reg *
                           self.num_classes)
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        super(RBBoxHead, self).init_weights()
        # conv layers are already initialized by ConvModule
        for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred

    def get_target(self, sampling_results, gt_rbboxes, gt_labels, rbbox_test_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        rbbox_targets = rbbox_target(pos_proposals,
                                    neg_proposals,
                                    pos_assigned_gt_inds,
                                    gt_rbboxes,
                                    gt_labels,
                                    rbbox_test_cfg,
                                    self.target_means,
                                    self.target_stds,
                                    self.out_dim_reg)
        return rbbox_targets

    def loss(self,
             cls_score,
             rbbox_pred,
             labels,
             label_weights,
             rbbox_targets,
             rbbox_weights,
             reduction_override=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                losses['loss_rbbox_cls'] = self.loss_rbbox_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['acc'] = accuracy(cls_score, labels)
        if rbbox_pred is not None:
            pos_inds = labels > 0
            if pos_inds.any():
                if self.reg_class_agnostic:
                    pos_rbbox_pred = rbbox_pred.view(
                        rbbox_pred.size(0), self.out_dim_reg)[pos_inds.type(torch.bool)]
                else:
                    pos_rbbox_pred = rbbox_pred.view(
                        rbbox_pred.size(0), -1,
                        self.out_dim_reg)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]

                losses['loss_rbbox'] = self.loss_rbbox(
                    pos_rbbox_pred,
                    rbbox_targets[pos_inds.type(torch.bool)],
                    rbbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=rbbox_targets.size(0),
                    reduction_override=reduction_override)
        return losses

    def get_det_rbboxes_parallel(self,
                                rois,
                                cls_score,
                                rbbox_pred, 
                                img_shape,
                                scale_factor,
                                rescale=False,
                                cfg=None,
                                bbox_cls_inds=None, 
                                bbox_keep_inds=None):
        delta2rbbox = {"thetaobb": delta2thetaobb,
                       "pointobb": delta2pointobb,
                       "hobb": delta2hobb}
        rbbox_rescale = {"thetaobb": thetaobb_rescale,
                       "pointobb": pointobb_rescale,
                       "hobb": hobb_rescale}
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        if rbbox_pred is not None:
            rbboxes = delta2rbbox[cfg.encode](rois[:, 1:], 
                                    rbbox_pred, 
                                    self.target_means,
                                    self.target_stds, 
                                    img_shape)
        else:
            rbboxes = rois[:, 1:]

        if rescale:
            rbboxes = rbbox_rescale[cfg.encode](rbboxes, scale_factor, reverse_flag=True)

        if cfg is None:
            return rbboxes, scores
        else:
            det_rbboxes, det_labels = thetaobb_nms_by_bbox_nms(rbboxes, 
                scores,
                bbox_cls_inds,
                bbox_keep_inds,
                cfg.max_per_img,
                out_dim_reg=self.out_dim_reg)
            # det_thetaobbs, det_labels = multiclass_thetaobb_nms(
            #     thetaobbs, scores, cfg.score_thr, cfg.polygon_nms_iou_thr, cfg.max_per_img)

            return det_rbboxes, det_labels