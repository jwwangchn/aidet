import torch
import numpy as np

from mmdet.ops.nms import nms_wrapper

def multiclass_nms_with_index(multi_bboxes, multi_scores, score_thr, nms_cfg, max_num=-1):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class)
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.shape[1]
    # print("bbox num_classes: ", num_classes)
    bboxes, labels = [], []
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = getattr(nms_wrapper, nms_type)
    bbox_cls_inds, bbox_keep_inds = [], []
    for i in range(1, num_classes):
        cls_inds = multi_scores[:, i] > score_thr
        bbox_cls_inds.append(cls_inds)
        if not cls_inds.any():
            continue
        # get bboxes and scores of this class
        # print("bbox multi_bboxes number: ", multi_bboxes.shape[0])
        if multi_bboxes.shape[1] == 4:
            _bboxes = multi_bboxes[cls_inds, :]
        else:
            _bboxes = multi_bboxes[cls_inds, i * 4:(i + 1) * 4]
        _scores = multi_scores[cls_inds, i]
        cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1)
        # print("bbox cls_dets number: ", cls_dets.shape[0])
        cls_dets, _ = nms_op(cls_dets, **nms_cfg_)
        bbox_keep_inds.append(_)
        # print("bbox keep number: ", _.shape)
        cls_labels = multi_bboxes.new_full(
            (cls_dets.shape[0], ), i - 1, dtype=torch.long)
        bboxes.append(cls_dets)
        labels.append(cls_labels)
    if bboxes:
        bboxes = torch.cat(bboxes)
        labels = torch.cat(labels)
        if bboxes.shape[0] > max_num:
            _, inds = bboxes[:, -1].sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds]
            labels = labels[inds]
    else:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)

    return bboxes, labels, bbox_cls_inds, bbox_keep_inds

def thetaobb_nms_by_bbox_nms(multi_bboxes, multi_scores, bbox_cls_inds, bbox_keep_inds, max_num=-1, out_dim_reg=5):
    """Polygon NMS for multi-class polygons.

    Args:
        multi_bboxes (Tensor): shape (n, #class * 5) or (n, 5)
        multi_scores (Tensor): shape (n, #class)
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.shape[1]
    bboxes, labels = [], []
    # print("thetaobb num_classes: ", num_classes)
    for i in range(1, num_classes):
        cls_inds = bbox_cls_inds[i - 1]
        if not cls_inds.any():
            continue
        # get bboxes and scores of this class
        # print("thetaobb multi_bboxes number: ", multi_bboxes.shape[0])
        if multi_bboxes.shape[1] == out_dim_reg:
            _bboxes = multi_bboxes[cls_inds, :]
        else:
            _bboxes = multi_bboxes[cls_inds, i * out_dim_reg:(i + 1) * out_dim_reg]
        _scores = multi_scores[cls_inds, i]
        cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1)
        # start = time.time()
        # print("thetaobb cls_dets number: ", cls_dets.shape[0])
        # cls_dets, _ = nms_wrapper.thetaobb_nms(cls_dets, polygon_nms_iou_thr)
        # cls_dets, _ = nms_wrapper.thetaobb_nms(cls_dets, polygon_nms_iou_thr)
        _ = bbox_keep_inds.pop(0)
        cls_dets = cls_dets[_, :]
        # end = time.time()
        # print("thetaobb keep number: ", _.shape[0])
        cls_labels = multi_bboxes.new_full(
            (cls_dets.shape[0], ), i - 1, dtype=torch.long)
        bboxes.append(cls_dets)
        labels.append(cls_labels)
    if bboxes:
        bboxes = torch.cat(bboxes)
        labels = torch.cat(labels)
        if bboxes.shape[0] > max_num:
            _, inds = bboxes[:, -1].sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds]
            labels = labels[inds]
    else:
        bboxes = multi_bboxes.new_zeros((0, out_dim_reg + 1))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)

    return bboxes, labels