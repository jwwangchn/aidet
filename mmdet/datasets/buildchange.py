# Modified from https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/datasets/cityscapes.py # noqa
# and https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalInstanceLevelSemanticLabeling.py # noqa

import glob
import os
import os.path as osp
import tempfile

import mmcv
import numpy as np
import pycocotools.mask as maskUtils

from mmdet.utils import print_log
from .coco import CocoDataset
from .registry import DATASETS


@DATASETS.register_module
class BuildChangeDataset(CocoDataset):

    CLASSES = ('building')
    def __init__(self,
                ann_file,
                pipeline,
                data_root=None,
                img_prefix='',
                seg_prefix=None,
                proposal_file=None,
                test_mode=False,
                filter_empty_gt=True,
                bbox_type='roof'):
        super(BuildChangeDataset, self).__init__(ann_file,
                                                pipeline,
                                                data_root,
                                                img_prefix,
                                                seg_prefix,
                                                proposal_file,
                                                test_mode,
                                                filter_empty_gt)
        self.bbox_type = bbox_type

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            img_id = img_info['id']
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            ann_info = self.coco.loadAnns(ann_ids)
            all_iscrowd = all([_['iscrowd'] for _ in ann_info])
            if self.filter_empty_gt and (self.img_ids[i] not in ids_with_ann
                                         or all_iscrowd):
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_footprint_masks = []
        gt_offsets = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            
            if self.bbox_type == 'roof':
                x1, y1, w, h = ann['bbox']
            elif self.bbox_type == 'building':
                x1, y1, w, h = ann['building_bbox']

            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])
                if 'offset' in ann:
                    gt_offsets.append(ann['offset'])
                else:
                    gt_offsets.append([0, 0])
                
                if 'footprint_mask' in ann:
                    gt_footprint_masks.append(ann['footprint_mask'])
                else:
                    gt_footprint_masks.append([0, 0])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            gt_offsets = np.array(gt_offsets, dtype=np.float32)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            gt_offsets = np.array((0, 2), dtype=np.float32)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            footprint_masks=gt_footprint_masks,
            seg_map=seg_map,
            offsets=gt_offsets)

        return ann

    def _segm2json(self, results):
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            det, seg, offset = results[idx]
            for label in range(len(det)):
                # bbox results
                bboxes = det[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    bbox_json_results.append(data)

                # segm results
                # some detectors use different scores for bbox and mask
                if isinstance(seg, tuple):
                    segms = seg[0][label]
                    mask_score = seg[1][label]
                else:
                    segms = seg[label]
                    mask_score = [bbox[4] for bbox in bboxes]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(mask_score[i])
                    data['category_id'] = self.cat_ids[label]
                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    segm_json_results.append(data)
        return bbox_json_results, segm_json_results