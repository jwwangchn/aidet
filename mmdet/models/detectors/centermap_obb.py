import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

import mmcv
import wwtool
import pycocotools.mask as maskUtils

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler, bbox_mapping, merge_aug_bboxes, multiclass_nms, merge_aug_masks, get_classes, tensor2imgs
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
                      gt_mask_weights=None,
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
                if gt_mask_weights is not None:
                    mask_weights = self.mask_head.get_target(
                        sampling_results, gt_mask_weights, self.train_cfg.rcnn)
                else:
                    mask_weights = None
                pos_labels = torch.cat(
                    [res.pos_gt_labels for res in sampling_results])
                loss_mask = self.mask_head.loss(mask_pred, 
                                                mask_targets,
                                                pos_labels,
                                                mask_weights)
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

        if not self.with_mask:
            return bbox_result
        else:
            if det_bboxes.shape[0] == 0:
                mask_classes = self.mask_head.num_classes - 1
                segm_result = [[] for _ in range(mask_classes)]
            else:
                # if det_bboxes is rescaled to the original image size, we need to
                # rescale it back to the testing scale to obtain RoIs.
                if rescale and not isinstance(scale_factor, float):
                    scale_factor = torch.from_numpy(scale_factor).to(
                        det_bboxes.device)
                _bboxes = (
                    det_bboxes[:, :4] *
                    scale_factor if rescale else det_bboxes)

                mask_pred = self._mask_forward_test(x, _bboxes, semantic_feat=semantic_feat)

                segm_result = self.mask_head.get_seg_masks(
                    mask_pred, 
                    _bboxes, 
                    det_labels, 
                    rcnn_test_cfg,
                    ori_shape, 
                    scale_factor, 
                    rescale)
            
            return bbox_result, segm_result

    def aug_test(self, imgs, img_metas, proposals=None, rescale=False):
        if self.with_semantic:
            semantic_feats = [
                self.semantic_head(feat)[1]
                for feat in self.extract_feats(imgs)
            ]
        else:
            semantic_feats = [None] * len(img_metas)

        # recompute feats to save memory
        proposal_list = self.aug_test_rpn(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn)

        rcnn_test_cfg = self.test_cfg.rcnn
        aug_bboxes = []
        aug_scores = []
        for x, img_meta, semantic in zip(
                self.extract_feats(imgs), img_metas, semantic_feats):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']

            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip)

            rois = bbox2roi([proposals])
            cls_score, bbox_pred = self._bbox_forward_test(
                x, rois, semantic_feat=semantic)

            bboxes, scores = self.bbox_head.get_det_bboxes(
                rois,
                cls_score,
                bbox_pred,
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
                                                rcnn_test_cfg.score_thr,
                                                rcnn_test_cfg.nms,
                                                rcnn_test_cfg.max_per_img)
        # if rescale:
        #     _det_bboxes = det_bboxes
        # else:
        #     _det_bboxes = det_bboxes.clone()
        #     _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head.num_classes)

        if self.with_mask:
            if det_bboxes.shape[0] == 0:
                segm_result = [[]
                               for _ in range(self.mask_head.num_classes -
                                              1)]
            else:
                aug_masks = []
                aug_img_metas = []
                for x, img_meta, semantic in zip(
                        self.extract_feats(imgs), img_metas, semantic_feats):
                    img_shape = img_meta[0]['img_shape']
                    scale_factor = img_meta[0]['scale_factor']
                    flip = img_meta[0]['flip']
                    _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
                                           scale_factor, flip)
                    mask_pred = self._mask_forward_test(x, _bboxes, semantic_feat=semantic)
                    aug_masks.append(mask_pred.cpu().numpy())
                    aug_img_metas.append(img_meta)

                merged_masks = merge_aug_masks(aug_masks, aug_img_metas,
                                               self.test_cfg.rcnn)

                ori_shape = img_metas[0][0]['ori_shape']
                segm_result = self.mask_head.get_seg_masks(
                    merged_masks,
                    det_bboxes,
                    det_labels,
                    rcnn_test_cfg,
                    ori_shape,
                    scale_factor=1.0,
                    rescale=False)
        else:
            return bbox_result
        
        return bbox_result, segm_result
    
    def show_result(self, 
                    data, 
                    result, 
                    dataset=None, 
                    score_thr=0.3, 
                    out_file=None,
                    show_flag=None,
                    thickness=2):
        # RGB
        DOTA_COLORS = {'harbor': (60, 180, 75), 'ship': (230, 25, 75), 'small-vehicle': (255, 225, 25), 'large-vehicle': (245, 130, 200), 
        'storage-tank': (230, 190, 255), 'plane': (245, 130, 48), 'soccer-ball-field': (0, 0, 128), 'bridge': (255, 250, 200), 
        'baseball-diamond': (240, 50, 230), 'tennis-court': (70, 240, 240), 'helicopter': (0, 130, 200), 'roundabout': (170, 255, 195), 
        'swimming-pool': (250, 190, 190), 'ground-track-field': (170, 110, 40), 'basketball-court': (0, 128, 128)}
        
        if isinstance(result, tuple):
            bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None

        if isinstance(data, dict):
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)
        else:
            imgs = [mmcv.imread(data)]
            img_metas = [{'img_shape': imgs[0].shape}]

        if dataset is None:
            class_names = self.CLASSES
        elif isinstance(dataset, str):
            class_names = get_classes(dataset)
        elif isinstance(dataset, (list, tuple)):
            class_names = dataset
        else:
            raise TypeError(
                'dataset must be a valid dataset name or a sequence'
                ' of class names, not {}'.format(type(dataset)))
        
        for img, img_meta in zip(imgs, img_metas):
            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :]

            bboxes = np.vstack(bbox_result)
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ]
            labels = np.concatenate(labels)
            colors = [DOTA_COLORS[class_names[label]][::-1] for label in labels]

            # draw segmentation masks
            if segm_result is not None and (show_flag == 2 or show_flag == 0):
                segms = mmcv.concat_list(segm_result)
                inds = np.where(bboxes[:, -1] > score_thr)[0]
                for ind in inds:
                    color_mask = np.random.randint(
                        0, 256, (1, 3), dtype=np.uint8)
                    mask = maskUtils.decode(segms[ind]).astype(np.bool)
                    
                    segms[ind]['counts'] = segms[ind]['counts'].decode()
                    thetaobb, pointobb = wwtool.segm2rbbox(segms[ind])

                    for idx in range(-1, 3, 1):
                        cv2.line(img_show, (int(pointobb[idx * 2]), int(pointobb[idx * 2 + 1])), (int(pointobb[(idx + 1) * 2]), int(pointobb[(idx + 1) * 2 + 1])), colors[ind], thickness=thickness)

                    # img_show[mask] = img_show[mask] * 0.5 + color_mask * 0.5
            
            scores = bboxes[:, -1]
            inds = scores > score_thr
            labels = labels[inds]
            bboxes = bboxes[inds, :]
            # draw bounding boxes
            if (show_flag == 2 or show_flag == 1):
                for idx, (bbox, label) in enumerate(zip(bboxes, labels)):
                    bbox_int = bbox.astype(np.int32)
                    left_top = (bbox_int[0], bbox_int[1])
                    right_bottom = (bbox_int[2], bbox_int[3])
                    cv2.rectangle(
                        img_show, left_top, right_bottom, colors[idx], thickness=thickness)

        wwtool.show_image(img_show, win_size=800, save_name=out_file)
