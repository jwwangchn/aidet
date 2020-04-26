import glob
import os
import os.path as osp
import tempfile
import shutil

import mmcv
import numpy as np
import pycocotools.mask as maskUtils

from mmdet.utils import print_log
from .coco import CocoDataset
from .registry import DATASETS

from wwtool import segm2rbbox, Email
from wwtool.datasets.dota import mergebypoly, mergebyrec, dota_eval_task1, dota_eval_task2


@DATASETS.register_module
class DOTADataset(CocoDataset):

    CLASSES = ('harbor', 'ship', 'small-vehicle', 'large-vehicle', 'storage-tank', 'plane', 'soccer-ball-field', 'bridge', 'baseball-diamond', 'tennis-court', 'helicopter', 'roundabout', 'swimming-pool', 'ground-track-field', 'basketball-court')

    CLASSES_OFFICIAL = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court', 'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter')

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True):
        super(DOTADataset, self).__init__(ann_file,
                                          pipeline,
                                          data_root,
                                          img_prefix,
                                          seg_prefix,
                                          proposal_file,
                                          test_mode,
                                          filter_empty_gt)
        self.dota_eval_functions = {"hbb": dota_eval_task1,
                                    "obb": dota_eval_task2}
        self.txt_save_dir = {"hbb": 'dota_hbb',
                             "obb": 'dota_obb'}
        self.mergetxt_save_dir = {"hbb": 'merge_dota_hbb',
                                  "obb": 'merge_dota_obb'}
        self.txt_file_prefix = {"hbb": 'Task1',
                                "obb": 'Task2'}

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.filter_empty_gt and self.img_ids[i] not in ids_with_ann:
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

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

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
            seg_map=seg_map)

        return ann

    def results2txt(self, results, submit_path, logger):
        """Write results .pkl file to dota txt format files

        This function provides an all-in-one method to convert .pkl to .txt

        Args:
            dataset (obj): DotaDataset class, contains all annotations information
            results (tuple): load from .pkl file, contain 'bbox' and 'rbbox' information
            submit_path (str): path to save txt result files
        
        Returns:
            No
        """
        # 1. convert pkl to dict
        hbb_obb_results = []
        prog_bar = mmcv.ProgressBar(len(self))
        for idx in range(len(self)):
            det, seg = results[idx]
            img_id = self.img_ids[idx]
            img_info = self.img_infos[idx]

            # bboxes shape = [15, N], per-class
            for label in range(len(det)):
                bboxes, segms = det[label], seg[label]

                instance_num = bboxes.shape[0]
                for idx in range(instance_num):
                    score = float(bboxes[idx][4])
                    if score < 0.01:
                        continue
                    data = dict()
                    data['file_name'] = img_info['filename']
                    data['image_id'] = img_id
                    data['score'] = score
                    data['category_id'] = self.cat_ids[label]
                    data['bbox'] = bboxes[idx][:-1].tolist()
                    segms[idx]['counts'] = segms[idx]['counts'].decode()
                    data['segmentation'] = segms[idx]
                    data['thetaobb'], data['rbbox'] = segm2rbbox(segms[idx])
                    hbb_obb_results.append(data)
            prog_bar.update()

        # 2. convert dict to list
        bboxes, labels, scores, filenames = {'hbb': [], 'obb': []}, [], [], []

        for hbb_obb_result in hbb_obb_results:
            bboxes['hbb'].append(hbb_obb_result['bbox'])
            bboxes['obb'].append(hbb_obb_result['rbbox'])

            labels.append(hbb_obb_result['category_id'])
            scores.append(hbb_obb_result['score'])
            filenames.append(hbb_obb_result['file_name'])

        # 3. generate subimage results
        print_log("\n------------------------------start write results to txt-----------------------------", logger=logger)
        for task in ['hbb', 'obb']:
            self.format_results(submit_path, filenames, bboxes[task], scores, labels, task)

        # 4. generate original image results
        print_log("\n------------------------------start merge txt file-----------------------------", logger=logger)
        for task in ['hbb', 'obb']:
            self.merge_txt(submit_path, task)

    def format_results(self, 
                       submit_path, 
                       filenames, 
                       bboxes, 
                       scores, 
                       labels, 
                       task='hbb'):
        txt_path = os.path.join(submit_path, self.txt_save_dir[task])
        if not os.path.exists(txt_path):
            os.makedirs(txt_path)
        else:
            shutil.rmtree(txt_path)
            os.makedirs(txt_path)

        write_handle = {}
        
        for classname in DOTADataset.CLASSES:
            txt_file_name = "{}_{}.txt".format(self.txt_file_prefix[task], classname)
            write_handle[classname] = open(os.path.join(txt_path, txt_file_name), 'a+')

        for i, bbox in enumerate(bboxes):
            if task == 'hbb':
                command_bbox = '%s %.3f %.1f %.1f %.1f %.1f\n' % (filenames[i], scores[i], bbox[0], bbox[1], bbox[2], bbox[3])
            else:
                command_bbox = '%s %.3f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f\n' % (filenames[i], scores[i], bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5], bbox[6], bbox[7])
            write_handle[DOTADataset.CLASSES[labels[i] - 1]].write(command_bbox)

        for classname in DOTADataset.CLASSES:
            write_handle[classname].close()

    def merge_txt(self, submit_path, task='hbb'):
        txt_path = os.path.join(submit_path, self.txt_save_dir[task])
        mergetxt_path = os.path.join(submit_path, self.mergetxt_save_dir[task])

        if not os.path.exists(mergetxt_path):
            os.makedirs(mergetxt_path)
        else:
            shutil.rmtree(mergetxt_path)
            os.makedirs(mergetxt_path)

        if task == 'hbb':
            mergebyrec(txt_path, mergetxt_path)
        else:
            mergebypoly(txt_path, mergetxt_path)
        
    def evaluate(self,
                 results,
                 metric=['hbb', 'obb'],
                 submit_path='./results/dota/common_submit',
                 annopath='./data/dota/v0/evaluation_sample/labelTxt-v1.0/{:s}.txt',
                 imageset_file='./data/dota/v0/evaluation_sample/testset.txt',
                 logger=None):
        tasks = metric
        mmcv.mkdir_or_exist(submit_path)
        filename_prefix = {'hbb': "/Task1_{:s}.txt",
                           'obb': "/Task2_{:s}.txt"}

        # convert results to txt file and save file (DOTA format)
        self.results2txt(results, submit_path, logger)

        # evaluating tasks of DOTA
        for task in tasks:
            msg = "Evaluating in DOTA {} Task".format(task)
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)
            result_path = os.path.join(submit_path, self.mergetxt_save_dir[task] + filename_prefix[task])

            self._evaluation_dota(result_path, annopath, imageset_file, task, logger)

    def _evaluation_dota(self, detpath, annopath, imagesetfile, task, logger):
        # mean_metrics = [mean ap, mean precision, mean reccall]
        mean_metrics = 0
        # metrics = {"class name": [ap, precision, reccall]}
        class_metrics = dict()

        for idx, classname in enumerate(DOTADataset.CLASSES_OFFICIAL):
            reccall, precision, ap = self.dota_eval_functions[task](detpath,
                annopath,
                imagesetfile,
                classname,
                ovthresh=0.5,
                use_07_metric=True)

            class_metrics[DOTADataset.CLASSES_OFFICIAL[idx]] = ap
            mean_metrics = mean_metrics + float(ap)

        mean_metrics = mean_metrics/(len(DOTADataset.CLASSES_OFFICIAL))
        
        print_log('mean metrics: {}'.format(mean_metrics), logger=logger)
        print_log('class metrics: {}'.format(class_metrics), logger=logger)
        
        return mean_metrics, class_metrics
