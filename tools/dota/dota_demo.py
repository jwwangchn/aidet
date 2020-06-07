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
    
    vis_image_list = ['P0088__1.0__0___684', 'P1442__1.0__0___0', 'P2195__1.0__440___0', 'P0155__1.0__0___0', 'P0628__1.0__0___88']
    # single_classes = ['harbor', 'ship', 'small-vehicle', 'large-vehicle', 'storage-tank', 'plane', 'soccer-ball-field', 'bridge', 'baseball-diamond', 'tennis-court', 'helicopter', 'roundabout', 'swimming-pool', 'ground-track-field', 'basketball-court']
    single_classes = ['tennis-court', 'roundabout', 'swimming-pool']

    anno_file = './data/{}/v1/coco/annotations/dota_test_v1_1.0_best_keypoint.json'.format(args.dataset)
    coco = COCO(anno_file)

    output_dir = f'./results/{args.dataset}/{args.config_version}/vis/'
    wwtool.mkdir_or_exist(output_dir)
    print(output_dir)
    
    catIds = coco.getCatIds(catNms=' ')
    imgIds = coco.getImgIds(catIds=catIds)

    catIds = coco.getCatIds(catNms=' ')

    print(config_file)
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    for idx, imgId in enumerate(imgIds):
        img_info = coco.loadImgs(imgIds[idx])[0]
        print(idx, img_info['file_name'])
        if wwtool.get_basename(img_info['file_name']) not in vis_image_list:
            continue
        img = cv2.imread(os.path.join(img_dir, img_info['file_name']))
        img_origin = img.copy()

        annIds = coco.getAnnIds(imgIds=img_info['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        for ann in anns:
            pointobb = ann['pointobb']
            img_origin = wwtool.show_pointobb(img_origin, pointobb, color=(200, 130, 0))

        bbox_result, segm_result = inference_detector(model, img)
        bboxes = np.vstack(bbox_result)
        if len(bboxes) == 0:
            continue
        detected_image = model.show_result(img, (bbox_result, segm_result), 'dota', score_thr=0.5, show_flag=0, thickness=3, out_file=None, wait_time=50)
            
        white_background = 255 * np.ones((1024, 20, 3), dtype=np.uint8)
        # merged_img = np.hstack((img_origin, white_background))
        # merged_img = np.hstack((merged_img, detected_image))
        output_file_gt = os.path.join(output_dir, wwtool.get_basename(img_info['file_name']) + '_gt.png')
        output_file_pred = os.path.join(output_dir, wwtool.get_basename(img_info['file_name']) + '_pred.png')
        print(output_file_gt)
        cv2.imwrite(output_file_gt, img_origin)
        cv2.imwrite(output_file_pred, detected_image)

        wwtool.show_image(detected_image, win_size=800, wait_time=10, no_resize=True)
