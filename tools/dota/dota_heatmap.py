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
    parser.add_argument('--config_version', default='centermap_net_tgrs_mask_weight_V4', help='version of experiments (DATASET_V#NUM)')
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
    img_dir = '/home/jwwangchn/Documents/Nutstore/100-Work/110-Projects/2019-DOTA/05-CVPR/supplementary/dota_selective_2'
    
    vis_image_list = os.listdir(img_dir)

    anno_file = './data/{}/v1/coco/annotations/dota_test_v1_1.0_best_keypoint.json'.format(args.dataset)
    coco = COCO(anno_file)
    output_dir = f'./results/{args.dataset}/{args.config_version}/vis/'
    wwtool.mkdir_or_exist(output_dir)

    print(config_file)
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    for idx, image_fn in enumerate(vis_image_list):
        print(idx, image_fn)

        img = cv2.imread(os.path.join(img_dir, image_fn))
        img_origin = img.copy()

        bbox_result, segm_result = inference_detector(model, img)
        bboxes = np.vstack(bbox_result)
        if len(bboxes) == 0:
            continue

        detected_image = model.show_result(img, 
                                          (bbox_result, segm_result), 
                                          'dota', 
                                          score_thr=0.5, 
                                          show_flag=0, 
                                          thickness=3, 
                                          out_file=None, 
                                          wait_time=0, 
                                          single_color=None, 
                                          single_class=None,
                                          show=True)
