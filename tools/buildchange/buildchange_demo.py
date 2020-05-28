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
    parser.add_argument('--city', default='shanghai', help='city name')
    parser.add_argument('--config_version', default='bc_v002_mask_rcnn_r50_v2_jinan_roof', help='version of experiments (DATASET_V#NUM)')
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

    imageset_file = f'./data/{args.dataset}/{args.dataset_version}/{args.city}/valset.txt'
    image_name_list = []
    imageset_handle = open(imageset_file, 'r')
    image_name_lines = imageset_handle.readlines()
    for image_name_line in image_name_lines:
        image_name_line = image_name_line.strip('\n')
        image_name_list.append(image_name_line)

    firstfile = True
    for img_name in img_list:
        image_basename = wwtool.get_basename(img_name)
        origin_image_name = "_".join(image_basename.split('__')[0].split('_')[1:])
        if origin_image_name not in image_name_list:
            continue
        print(origin_image_name, img_name)
        
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
