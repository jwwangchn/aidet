import glob
import os
import os.path as osp
import tempfile
import shutil
import pandas as pd
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import mmcv
import numpy as np
import pycocotools.mask as maskUtils

from mmdet.utils import print_log
from .coco import CocoDataset
from .registry import DATASETS

import wwtool
from wwtool import segm2rbbox
from wwtool.datasets.dota import mergebypoly, mergebypoly_mp, mergebyrec, mergebyrec_mp, dota_eval_task1, dota_eval_task2


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
                 heatmap_weight_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 min_area=0,
                 max_small_length=0,
                 evaluation_iou_threshold=0.5,
                 classwise_nms_threshold=True):
        super(DOTADataset, self).__init__(ann_file,
                                          pipeline,
                                          data_root,
                                          img_prefix,
                                          seg_prefix,
                                          proposal_file,
                                          test_mode,
                                          filter_empty_gt)
        self.dota_eval_functions = {"hbb": dota_eval_task2,
                                    "obb": dota_eval_task1}
        self.txt_save_dir = {"hbb": 'dota_hbb',
                             "obb": 'dota_obb'}
        self.mergetxt_save_dir = {"hbb": 'merge_dota_hbb',
                                  "obb": 'merge_dota_obb'}
        self.txt_file_prefix = {"hbb": 'Task2',
                                "obb": 'Task1'}
        self.heatmap_weight_prefix = heatmap_weight_prefix
        self.min_area = min_area
        self.max_small_length = max_small_length
        self.evaluation_iou_threshold = evaluation_iou_threshold
        self.classwise_nms_threshold = classwise_nms_threshold

        # join paths if data_root is specified
        if self.data_root is not None:
            if not (self.heatmap_weight_prefix is None or osp.isabs(self.heatmap_weight_prefix)):
                self.heatmap_weight_prefix = osp.join(self.data_root, self.heatmap_weight_prefix)

    def pre_pipeline(self, results):
        super(DOTADataset, self).pre_pipeline(results)
        if self.heatmap_weight_prefix:
            results['heatmap_weight_prefix'] = self.heatmap_weight_prefix

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
        gt_rbboxes = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= self.min_area or max(w, h) < self.max_small_length:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['pointobb'])
                gt_rbboxes.append(ann['pointobb'])
                gt_masks_ann.append([ann['pointobb']])

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

        if gt_rbboxes:
            gt_rbboxes = np.array(gt_rbboxes, dtype=np.float32)
        else:
            gt_rbboxes = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')
        heatmap_weight = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map,
            rbboxes=gt_rbboxes)
            heatmap_weight=heatmap_weight)

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
        if type(results) == str:
            # loading results from pkl file
            results = mmcv.load(results)

        hbb_obb_results = []
        prog_bar = mmcv.ProgressBar(len(self))

        for idx in range(len(self)):
            det, seg = results[idx]
            img_id = self.img_ids[idx]
            img_info = self.img_infos[idx]

            # bboxes shape = [15, N], per-class
            for label in range(len(det)):
                bboxes, segms = det[label], seg[label]
                for idx, (bbox, segm) in enumerate(zip(bboxes, segms)):
                    
                    data = dict()
                    score = float(bbox[4])
                    data['file_name'] = img_info['filename']
                    data['image_id'] = img_id
                    data['score'] = score
                    data['category_id'] = self.cat_ids[label]
                    data['bbox'] = bbox[:-1].tolist()
                    segm['counts'] = segm['counts'].decode()
                    data['segmentation'] = segm
                    data['thetaobb'], data['rbbox'] = segm2rbbox(segm)
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
        print_log("\nStart write results to txt", logger=logger)
        for task in ['hbb', 'obb']:
            self.format_dota_results(submit_path, filenames, bboxes, scores, labels, task)

        # 4. generate original image results
        print_log("\nStart merge txt file", logger=logger)
        for task in ['hbb', 'obb']:
            self.merge_txt(submit_path, task)

    def format_dota_results(self, 
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

        for i, bbox in enumerate(bboxes[task]):
            if task == 'hbb':
                command_bbox = '%s %.3f %.1f %.1f %.1f %.1f\n' % (filenames[i], scores[i], bbox[0], bbox[1], bbox[2], bbox[3])
            else:
                if DOTADataset.CLASSES[labels[i] - 1] == 'storage-tank':
                    bbox = wwtool.bbox2pointobb(bboxes['hbb'][i])
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
        
        # average = 0.15
        hbb_nms_thr = {'harbor': 0.4, 'ship': 0.4, 'small-vehicle': 0.4, 'large-vehicle': 0.5, 'storage-tank': 0.1, 'plane': 0.25, 'soccer-ball-field': 0.2, 'bridge': 0.5, 'baseball-diamond': 0.15, 'tennis-court': 0.2, 'helicopter': 0.2, 'roundabout': 0.15, 'swimming-pool': 0.2, 'ground-track-field': 0.15, 'basketball-court': 0.2}
            
        # average = 0.4
        obb_nms_thr = {'harbor': 0.1, 'ship': 0.05, 'small-vehicle': 0.15, 'large-vehicle': 0.5, 'storage-tank': 0.35, 'plane': 0.2, 'soccer-ball-field': 0.2, 'bridge': 0.45, 'baseball-diamond': 0.2, 'tennis-court': 0.1, 'helicopter': 0.1, 'roundabout': 0.15, 'swimming-pool': 0.05, 'ground-track-field': 0.4, 'basketball-court': 0.2}

        if self.classwise_nms_threshold:
            pass
        else:
            for class_name in DOTADataset.CLASSES:
                hbb_nms_thr[class_name] = 0.3
                obb_nms_thr[class_name] = 0.3
            
        if task == 'hbb':
            mergebyrec_mp(txt_path, mergetxt_path, nms_thresh=hbb_nms_thr)
        else:
            mergebypoly_mp(txt_path, mergetxt_path, o_thresh=obb_nms_thr)
        
    def evaluate(self,
                 results,
                 metric=['hbb', 'obb'],
                 submit_path='./results/dota/common_submit',
                 annopath='./data/dota/v0/evaluation_sample/labelTxt-v1.0/{:s}.txt',
                 imageset_file='./data/dota/v0/evaluation_sample/testset.txt',
                 PR_path=None,
                 logger=None,
                 excel=None,
                 jsonfile_prefix=None):
        tasks = metric
        mmcv.mkdir_or_exist(submit_path)
        filename_prefix = {'hbb': "/Task2_{:s}.txt",
                           'obb': "/Task1_{:s}.txt"}

        # convert results to txt file and save file (DOTA format)
        self.results2txt(results, submit_path, logger)

        # evaluating tasks of DOTA
        two_task_aps = []
        two_task_prs = []
        for task in tasks:
            msg = "Evaluating in DOTA {} Task".format(task)
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)
            result_path = os.path.join(submit_path, self.mergetxt_save_dir[task] + filename_prefix[task])

            mean_AP, class_AP, class_PR = self._evaluation_dota(result_path, annopath, imageset_file, task, logger)
            class_AP['mAP'] = mean_AP
            two_task_aps.append(class_AP)
            two_task_prs.append(class_PR)

        eval_results = {**two_task_aps[0], **two_task_aps[1]}
        for key, value in eval_results.items():
            if key in two_task_aps[0] and key in two_task_aps[1]:
                eval_results[key] = [value , two_task_aps[0][key]]

        df = pd.DataFrame(data=two_task_aps)
        writer = pd.ExcelWriter(excel, engine='xlsxwriter')
        df=df.style.set_properties(**{'text-align': 'center'})
        df.to_excel(writer, sheet_name='Sheet1')
        worksheet = writer.sheets['Sheet1']
        worksheet.set_column("B:R", 12)
        writer.save()

        if type(results) == str:
            # loading results from pkl file
            results = mmcv.load(results)
        self.format_results(results, jsonfile_prefix)

        if PR_path:
            mmcv.mkdir_or_exist(PR_path)
            for task_idx, task in enumerate(tasks):
                pr_fn = "{}.pdf".format(task)
                pr_file = os.path.join(PR_path, pr_fn)
                fig, ax = plt.subplots(figsize=(12, 10))
                for classname in DOTADataset.CLASSES_OFFICIAL:
                    ap = two_task_aps[task_idx][classname]
                    recall, precision = two_task_prs[task_idx][classname]
                    
                    ax.plot(recall, precision, label='{} ({})'.format(classname, ap))
                
                ax.set_title('PR curve of {} task, mAP = {}'.format(task, two_task_aps[task_idx]['mAP']))
                ax.set_xlabel('recall')
                ax.set_ylabel('precision')
                ax.set_xlim(0, 1)
                ax.set_ylim(0.2, 1)
                ax.legend()
                plt.savefig(pr_file, bbox_inches='tight', dpi=600, pad_inches=0.1)
                plt.clf()

        return eval_results

    def _evaluation_dota(self, detpath, annopath, imagesetfile, task, logger):
        # mean_metrics = [mean ap, mean precision, mean recall]
        mean_AP = 0
        class_PR = dict()
        class_AP = dict()
        class_AP['Task'] = task
        for idx, classname in enumerate(DOTADataset.CLASSES_OFFICIAL):
            recall, precision, ap = self.dota_eval_functions[task](detpath,
                annopath,
                imagesetfile,
                classname,
                ovthresh=self.evaluation_iou_threshold,
                use_07_metric=True)
            ap_format = round(ap * 100.0, 2)
            class_AP[DOTADataset.CLASSES_OFFICIAL[idx]] = ap_format
            class_PR[DOTADataset.CLASSES_OFFICIAL[idx]] = [recall, precision]
            mean_AP = mean_AP + ap_format

        mean_AP = mean_AP/(len(DOTADataset.CLASSES_OFFICIAL))
        mean_AP = round(mean_AP, 2)
        
        print_log('mAP: {}'.format(mean_AP), logger=logger)
        print_log('class metrics: {}'.format(class_AP), logger=logger)
        
        return mean_AP, class_AP, class_PR