import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from .. import builder
from ..registry import DETECTORS
from .two_stage import TwoStageDetector


@DETECTORS.register_module
class CenterMapOBB(TwoStageDetector):

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 mask_roi_extractor,
                 mask_head,
                 semantic_roi_extractor=None,
                 semantic_head=None,
                 semantic_fusion=('bbox', 'mask'),
                 fusion_operation='add',
                 train_cfg=None,
                 test_cfg=None,
                 neck=None,
                 shared_head=None,
                 pretrained=None):
        super(CenterMapOBB, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

        if semantic_head is not None:
            self.semantic_roi_extractor = builder.build_roi_extractor(
                semantic_roi_extractor)
            self.semantic_head = builder.build_head(semantic_head)

        self.semantic_fusion = semantic_fusion
        self.fusion_operation = fusion_operation

    @property
    def with_semantic(self):
        if hasattr(self, 'semantic_head') and self.semantic_head is not None:
            return True
        else:
            return False

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_semantic_seg=None,
                      gt_heatmap_weight=None,
                      proposals=None):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_metas,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_metas, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        if self.with_semantic:
            semantic_pred, semantic_feat = self.semantic_head(x)
            loss_seg = self.semantic_head.loss(semantic_pred, gt_semantic_seg, gt_heatmap_weight)
            losses['loss_semantic_seg'] = loss_seg
        else:
            semantic_feat = None       

        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[i],
                                                     gt_bboxes[i],
                                                     gt_bboxes_ignore[i],
                                                     gt_labels[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        # bbox head forward and loss
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)

            # fusion the feature of semantic branch and bbox branch
            if self.with_semantic and 'bbox' in self.semantic_fusion:
                bbox_semantic_feat = self.semantic_roi_extractor([semantic_feat],
                                                                rois)
                if bbox_semantic_feat.shape[-2:] != bbox_feats.shape[-2:]:
                    bbox_semantic_feat = F.adaptive_avg_pool2d(
                        bbox_semantic_feat, bbox_feats.shape[-2:])
                if self.fusion_operation == 'attention':
                    bbox_semantic_feat = self.conv_attention1(bbox_semantic_feat)
                    bbox_semantic_feat = torch.relu(bbox_semantic_feat)
                    bbox_semantic_feat = self.conv_attention2(bbox_semantic_feat)
                    bbox_semantic_feat = torch.sigmoid(bbox_semantic_feat)
                    bbox_feats = bbox_feats * bbox_semantic_feat + bbox_feats
                elif self.fusion_operation == 'add':
                    bbox_feats += bbox_semantic_feat
                elif self.fusion_operation == 'mul':
                    bbox_feats *= bbox_semantic_feat

            cls_score, bbox_pred = self.bbox_head(bbox_feats)

            bbox_targets = self.bbox_head.get_target(
                sampling_results, gt_bboxes, gt_labels, self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *bbox_targets)
            losses.update(loss_bbox)

        # mask head forward and loss
        if self.with_mask:
            if not self.share_roi_extractor:
                pos_rois = bbox2roi(
                    [res.pos_bboxes for res in sampling_results])
                mask_feats = self.mask_roi_extractor(
                    x[:self.mask_roi_extractor.num_inputs], pos_rois)
                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
            else:
                pos_inds = []
                device = bbox_feats.device
                for res in sampling_results:
                    pos_inds.append(
                        torch.ones(
                            res.pos_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                    pos_inds.append(
                        torch.zeros(
                            res.neg_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                pos_inds = torch.cat(pos_inds)
                mask_feats = bbox_feats[pos_inds]

            if mask_feats.shape[0] > 0:
                # fusion the feature of semantic branch and mask branch
                if self.with_semantic and 'mask' in self.semantic_fusion:
                    mask_semantic_feat = self.semantic_roi_extractor([semantic_feat],
                                                                    pos_rois)
                    if mask_semantic_feat.shape[-2:] != mask_feats.shape[-2:]:
                        mask_semantic_feat = F.adaptive_avg_pool2d(
                            mask_semantic_feat, mask_feats.shape[-2:])
                    if self.fusion_operation == 'attention':
                        mask_semantic_feat = self.conv_attention1(mask_semantic_feat)
                        mask_semantic_feat = torch.relu(mask_semantic_feat)
                        mask_semantic_feat = self.conv_attention2(mask_semantic_feat)
                        mask_semantic_feat = torch.sigmoid(mask_semantic_feat)
                        mask_feats = mask_feats * mask_semantic_feat + mask_feats
                    elif self.fusion_operation == 'add':
                        mask_feats += mask_semantic_feat
                    elif self.fusion_operation == 'mul':
                        mask_feats *= mask_semantic_feat

                mask_pred = self.mask_head(mask_feats)
                mask_targets = self.mask_head.get_target(
                    sampling_results, gt_masks, self.train_cfg.rcnn)
                pos_labels = torch.cat(
                    [res.pos_gt_labels for res in sampling_results])
                loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                                pos_labels)
                losses.update(loss_mask)

        return losses

    def _bbox_forward_test(self, x, rois, semantic_feat=None):
        bbox_feats = self.bbox_roi_extractor(
            x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        if self.with_semantic and 'bbox' in self.semantic_fusion:
            bbox_semantic_feat = self.semantic_roi_extractor([semantic_feat],
                                                             rois)
            if bbox_semantic_feat.shape[-2:] != bbox_feats.shape[-2:]:
                bbox_semantic_feat = F.adaptive_avg_pool2d(
                    bbox_semantic_feat, bbox_feats.shape[-2:])
            if self.fusion_operation == 'attention':
                bbox_semantic_feat = self.conv_attention1(bbox_semantic_feat)
                bbox_semantic_feat = torch.relu(bbox_semantic_feat)
                bbox_semantic_feat = self.conv_attention2(bbox_semantic_feat)
                bbox_semantic_feat = torch.sigmoid(bbox_semantic_feat)
                bbox_feats = bbox_feats * bbox_semantic_feat + bbox_feats
            elif self.fusion_operation == 'add':
                bbox_feats += bbox_semantic_feat
            elif self.fusion_operation == 'mul':
                bbox_feats *= bbox_semantic_feat
        cls_score, bbox_pred = self.bbox_head(bbox_feats)
        return cls_score, bbox_pred

    def _mask_forward_test(self, x, bboxes, semantic_feat=None):
        mask_rois = bbox2roi([bboxes])
        mask_feats = self.mask_roi_extractor(
            x[:len(self.mask_roi_extractor.featmap_strides)], mask_rois)
        if self.with_semantic and 'mask' in self.semantic_fusion:
            mask_semantic_feat = self.semantic_roi_extractor([semantic_feat],
                                                             mask_rois)
            if mask_semantic_feat.shape[-2:] != mask_feats.shape[-2:]:
                mask_semantic_feat = F.adaptive_avg_pool2d(
                    mask_semantic_feat, mask_feats.shape[-2:])
            if self.fusion_operation == 'attention':
                mask_semantic_feat = self.conv_attention1(mask_semantic_feat)
                mask_semantic_feat = torch.relu(mask_semantic_feat)
                mask_semantic_feat = self.conv_attention2(mask_semantic_feat)
                mask_semantic_feat = torch.sigmoid(mask_semantic_feat)
                mask_feats = mask_feats * mask_semantic_feat + mask_feats
            elif self.fusion_operation == 'add':
                mask_feats += mask_semantic_feat
            elif self.fusion_operation == 'mul':
                mask_feats *= mask_semantic_feat
            mask_pred = self.mask_head(mask_feats)
        return mask_pred

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        x = self.extract_feat(img)
        proposal_list = self.simple_test_rpn(
            x, img_metas,
            self.test_cfg.rpn) if proposals is None else proposals

        if self.with_semantic:
            _, semantic_feat = self.semantic_head(x)
        else:
            semantic_feat = None

        img_shape = img_metas[0]['img_shape']
        ori_shape = img_metas[0]['ori_shape']
        scale_factor = img_metas[0]['scale_factor']

        rcnn_test_cfg = self.test_cfg.rcnn

        rois = bbox2roi(proposal_list)

        cls_score, bbox_pred = self._bbox_forward_test(
            x, rois, semantic_feat=semantic_feat)
            
        bbox_label = cls_score.argmax(dim=1)
        rois = self.bbox_head.regress_by_class(rois, bbox_label, bbox_pred,
                                            img_metas[0])

        det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head.num_classes)

        if self.with_mask:
            if det_bboxes.shape[0] == 0:
                mask_classes = self.mask_head.num_classes - 1
                segm_result = [[] for _ in range(mask_classes)]
            else:
                _bboxes = (
                    det_bboxes[:, :4] *
                    scale_factor if rescale else det_bboxes)

                mask_rois = bbox2roi([_bboxes])
                mask_feats = self.mask_roi_extractor(
                    x[:len(self.mask_roi_extractor.featmap_strides)], mask_rois)
                if self.with_semantic and 'mask' in self.semantic_fusion:
                    mask_semantic_feat = self.semantic_roi_extractor(
                        [semantic_feat], mask_rois)
                    if self.fusion_operation == 'attention':
                        mask_semantic_feat = self.conv_attention1(mask_semantic_feat)
                        mask_semantic_feat = torch.relu(mask_semantic_feat)
                        mask_semantic_feat = self.conv_attention2(mask_semantic_feat)
                        mask_semantic_feat = torch.sigmoid(mask_semantic_feat)
                        mask_feats = mask_feats * mask_semantic_feat + mask_feats
                    elif self.fusion_operation == 'add':
                        mask_feats += mask_semantic_feat
                    elif self.fusion_operation == 'mul':
                        mask_feats *= mask_semantic_feat

                mask_pred = self.mask_head(mask_feats)
                segm_result = self.mask_head.get_seg_masks(
                    mask_pred, _bboxes, det_labels, rcnn_test_cfg,
                    ori_shape, scale_factor, rescale)
        else:
            return bbox_result

        return bbox_result, segm_result

    def aug_test(self, imgs, img_metas, proposals=None, rescale=False):
        pass