import torch
import torch.nn as nn
import numpy as np
import cv2
import wwtool

from mmdet.core import bbox2result, rbbox2result, bbox2roi, build_assigner, build_sampler, tensor2imgs, get_classes
from .. import builder

from ..registry import DETECTORS
from .two_stage import TwoStageDetector
from .test_mixins import RBBoxTestMixin

@DETECTORS.register_module
class RBBoxRCNN(TwoStageDetector, RBBoxTestMixin):

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 rbbox_roi_extractor,
                 rbbox_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 pretrained=None):
        super(RBBoxRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

        if rbbox_head is not None:
            if rbbox_roi_extractor is not None:
                self.rbbox_roi_extractor = builder.build_roi_extractor(
                    rbbox_roi_extractor)
                self.share_roi_extractor = False
            else:
                self.share_roi_extractor = True
                self.rbbox_roi_extractor = self.bbox_roi_extractor
            self.rbbox_head = builder.build_head(rbbox_head)

        self.init_extra_weights()

    @property
    def with_rbbox(self):
        return hasattr(self, 'rbbox_head') and self.rbbox_head is not None

    def init_extra_weights(self):
        self.rbbox_head.init_weights()
        if not self.share_roi_extractor:
            self.rbbox_roi_extractor.init_weights()

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(device=img.device)
        # bbox head
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)
            outs = outs + (cls_score, bbox_pred)
        # rbbox head
        if self.with_rbbox:
            rbbox_feats = self.rbbox_roi_extractor(
                x[:self.rbbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                rbbox_feats = self.shared_head(rbbox_feats)
            cls_score, rbbox_pred = self.rbbox_head(rbbox_feats)
            outs = outs + (cls_score, rbbox_pred)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_rbboxes=None,
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

            gt_rbbox (None | Tensor) : true oriented bounding boxes for each box
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

        # assign gts and sample proposals
        if self.with_bbox or self.with_rbbox:
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
            cls_score, bbox_pred = self.bbox_head(bbox_feats)

            bbox_targets = self.bbox_head.get_target(sampling_results,
                                                     gt_bboxes, gt_labels,
                                                     self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *bbox_targets)
            losses.update(loss_bbox)

        # rbbox head forward and loss
        if self.with_rbbox:
            if not self.share_roi_extractor:
                rois = bbox2roi([res.bboxes for res in sampling_results])
                rbbox_feats = self.rbbox_roi_extractor(
                    x[:self.rbbox_roi_extractor.num_inputs], rois)
            else:
                rbbox_feats = bbox_feats

            cls_score, rbbox_pred = self.rbbox_head(rbbox_feats)

            rbbox_targets = self.rbbox_head.get_target(sampling_results,
                                                     gt_rbboxes, 
                                                     gt_labels,
                                                     self.test_cfg.rbbox)
            loss_rbbox = self.rbbox_head.loss(cls_score, rbbox_pred,
                                            *rbbox_targets)
            losses.update(loss_rbbox)

        return losses


    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        x = self.extract_feat(img)
        
        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        det_bboxes, det_labels, bbox_cls_inds, bbox_keep_inds = self.simple_test_bboxes_with_rbbox(
            x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale, rbbox=True)
        bbox_results = bbox2result(det_bboxes, 
                                   det_labels,
                                   self.bbox_head.num_classes)

        if not self.with_rbbox:
            return bbox_results
        else:
            if self.test_cfg.rbbox.parallel:
                det_rbboxes, det_labels = self.simple_test_rbbox_parallel(
                    x, 
                    img_meta, 
                    proposal_list, 
                    self.test_cfg.rbbox, 
                    rescale=rescale,
                    bbox_cls_inds=bbox_cls_inds, 
                    bbox_keep_inds=bbox_keep_inds)
                rbbox_results = rbbox2result(det_rbboxes, 
                                             det_labels,
                                             self.bbox_head.num_classes)
            else:
                rbbox_results = self.simple_test_rbbox_serial(
                    x, 
                    img_meta, 
                    det_bboxes, 
                    det_labels, 
                    rescale=rescale, 
                    num_classes=self.rbbox_head.num_classes)
            
            return bbox_results, rbbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        proposal_list = self.aug_test_rpn(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn)
        det_bboxes, det_labels = self.aug_test_bboxes(
            self.extract_feats(imgs), img_metas, proposal_list,
            self.test_cfg.rcnn)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_thetaobb:
            segm_results = self.aug_test_thetaobb(
                self.extract_feats(imgs), img_metas, det_bboxes, det_labels)
            return bbox_results, segm_results
        else:
            return bbox_results

    def show_result(self, 
                    data, 
                    result, 
                    dataset=None, 
                    score_thr=0.5, 
                    out_file=None,
                    show_flag=0,
                    thickness=2):
        # RGB
        DOTA_COLORS = {'harbor': (60, 180, 75), 'ship': (230, 25, 75), 'small-vehicle': (255, 225, 25), 'large-vehicle': (245, 130, 200), 
        'storage-tank': (230, 190, 255), 'plane': (245, 130, 48), 'soccer-ball-field': (0, 0, 128), 'bridge': (255, 250, 200), 
        'baseball-diamond': (240, 50, 230), 'tennis-court': (70, 240, 240), 'helicopter': (0, 130, 200), 'roundabout': (170, 255, 195), 
        'swimming-pool': (250, 190, 190), 'ground-track-field': (170, 110, 40), 'basketball-court': (0, 128, 128)}
        
        if isinstance(result, tuple):
            bbox_result, rbbox_result = result
        else:
            bbox_result, rbbox_result = result, None

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
            if rbbox_result is not None and (show_flag == 2 or show_flag == 0):
                rbboxes = np.vstack(rbbox_result)
                
                inds = np.where(bboxes[:, -1] > score_thr)[0]
                for ind in inds:
                    rbbox = rbboxes[ind]
                    if self.test_cfg.rbbox.encode == 'thetaobb':
                        img_show = wwtool.show_thetaobb(img_show, rbbox[:-1], color=colors[ind])
                    elif self.test_cfg.rbbox.encode == 'pointobb':
                        img_show = wwtool.show_pointobb(img_show, rbbox[:-1], color=colors[ind])
                    elif self.test_cfg.rbbox.encode == 'hobb':
                        img_show = wwtool.show_hobb(img_show, rbbox[:-1], color=colors[ind])
            
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