import os
import cv2
import numpy as np
import mmcv
import wwtool
import pandas as pd
import argparse
import pycocotools.mask as maskUtils
import wwtool

from mmdet.apis import init_detector, inference_detector, show_result

def parse_args():
    parser = parser = argparse.ArgumentParser(description='DOTA Testing')
    parser.add_argument('--dataset', default='buildchange', help='dataset name')
    parser.add_argument('--dataset_version', default='v2', help='dataset version')
    parser.add_argument('--city', default='xian_fine', help='city name')
    parser.add_argument('--config_version', default='bc_v001_mask_rcnn_r50_v1_roof', help='version of experiments (DATASET_V#NUM)')
    parser.add_argument('--imageset', default='val', help='imageset of evaluation')
    parser.add_argument('--epoch', default=12, help='epoch')
    parser.add_argument('--show', action='store_true', help='show flag')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    # config and result files
    config_file = './configs/{}/{}.py'.format(args.dataset, args.config_version)
    checkpoint_file = './work_dirs/{}/epoch_{}.pth'.format(args.config_version, args.epoch)
    img_dir = './data/{}/{}/{}/images'.format(args.dataset, args.dataset_version, args.city)
    save_dir = f'results/buildchange/{args.config_version}/{args.city}/vis/'
    wwtool.mkdir_or_exist(save_dir)

    print(config_file)
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    img_list = os.listdir(img_dir)
    prog_bar = mmcv.ProgressBar(len(img_list))

    firstfile = True
    for img_name in img_list:
        image_basename = wwtool.get_basename(img_name)
        if image_basename not in ['arg_L18_104408_210384__512_1024', 'arg_L18_104408_210384__1024_512', 'google_L18_104528_210368__512_1024']:
            continue
        
        out_file = os.path.join(save_dir, img_name)
        img_file = os.path.join(img_dir, img_name)
        img = cv2.imread(img_file)
        if args.show:
            wwtool.show_image(img, win_name='original')
        bbox_result, segm_result = inference_detector(model, img)
        bboxes = np.vstack(bbox_result)
        if len(bboxes) == 0:
            continue
        show_result(img, (bbox_result, segm_result), model.CLASSES, score_thr=0.5, out_file=out_file)
        # wwtool.imshow_bboxes(img, bbox_result[0][:, 0:-1], show=True)
