import os
import cv2
import numpy as np
import mmcv
import wwtool
import pandas as pd
import argparse
import pycocotools.mask as maskUtils
from pycocotools.coco import COCO

from mmdet.apis import init_detector, inference_detector, show_result

def parse_args():
    parser = parser = argparse.ArgumentParser(description='DOTA Testing')
    parser.add_argument('--dataset', default='dota', help='dataset name')
    parser.add_argument('--dataset_version', default='v1', help='dataset name')
    parser.add_argument('--config_version', default='dota_v015_centermap_net_r101_v4_trainval', help='version of experiments (DATASET_V#NUM)')
    parser.add_argument('--imageset', default='test', help='imageset of evaluation')
    parser.add_argument('--epoch', default=12, help='epoch')
    parser.add_argument('--show', action='store_true', help='show flag')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    # config and result files
    config_file = './configs/{}/{}.py'.format(args.dataset, args.config_version)
    checkpoint_file = './work_dirs/{}/epoch_{}.pth'.format(args.config_version, args.epoch)
    img_dir = './data/{}/v1/{}/images'.format(args.dataset, args.imageset)
    
    single_classes = ['swimming-pool', 'helicopter']
    # single_class = 'roundabout'
    for single_class in single_classes:
        output_dir = f'./results/{args.dataset}/{args.config_version}/vis/{single_class}'
        wwtool.mkdir_or_exist(output_dir)
        anno_file = './data/{}/v1/coco/annotations/dota_test_v1_1.0_best_keypoint.json'.format(args.dataset, args.imageset)
        coco = COCO(anno_file)

        catIds = coco.getCatIds(catNms=[single_class])
        imgIds = coco.getImgIds(catIds=catIds)

        print(config_file)
        model = init_detector(config_file, checkpoint_file, device='cuda:0')

        for idx, imgId in enumerate(imgIds):
            img_info = coco.loadImgs(imgIds[idx])[0]
            print(idx, img_info['file_name'])
            img = cv2.imread(os.path.join(img_dir, img_info['file_name']))
            img_origin = img.copy()

            output_file = os.path.join(output_dir, img_info['file_name'])
            annIds = coco.getAnnIds(imgIds=img_info['id'], catIds=catIds, iscrowd=None)
            anns = coco.loadAnns(annIds)
            for ann in anns:
                pointobb = ann['pointobb']
                img_origin = wwtool.show_pointobb(img_origin, pointobb, color=(200, 130, 0))

            bbox_result, segm_result = inference_detector(model, img)
            bboxes = np.vstack(bbox_result)
            if len(bboxes) == 0:
                continue
            detected_image = model.show_result(img, (bbox_result, segm_result), 'dota', score_thr=0.3, show_flag=0, thickness=3, out_file=output_file, wait_time=50, single_color=(75, 25, 230), single_class=single_class)
                
            white_background = 255 * np.ones((1024, 20, 3), dtype=np.uint8)
            merged_img = np.hstack((img_origin, white_background))
            merged_img = np.hstack((merged_img, detected_image))

            wwtool.show_image(merged_img, win_size=800, save_name=output_file, wait_time=10, no_resize=True)
