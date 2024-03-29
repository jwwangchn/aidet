"""
Evaluating in DOTA hbb Task
mAP: 72.66
class metrics: {'Task': 'hbb', 'plane': 88.56, 'baseball-diamond': 80.5, 'bridge': 54.28, 'ground-track-field': 62.25, 'small-vehicle': 78.36, 'large-vehicle': 77.71, 'ship': 85.62, 'tennis-court': 89.58, 'basketball-court': 77.49, 'storage-tank': 81.27, 'soccer-ball-field': 47.45, 'roundabout': 64.05, 'harbor': 72.28, 'swimming-pool': 70.65, 'helicopter': 59.81}

Evaluating in DOTA obb Task
mAP: 71.47
class metrics: {'Task': 'obb', 'plane': 88.67, 'baseball-diamond': 80.07, 'bridge': 50.05, 'ground-track-field': 62.46, 'small-vehicle': 78.14, 'large-vehicle': 74.07, 'ship': 86.11, 'tennis-court': 90.11, 'basketball-court': 77.56, 'storage-tank': 80.85, 'soccer-ball-field': 47.51, 'roundabout': 61.49, 'harbor': 65.08, 'swimming-pool': 66.61, 'helicopter': 63.28}


with gt:

Evaluating in DOTA hbb Task
mAP: 73.85
class metrics: {'Task': 'hbb', 'plane': 88.44, 'baseball-diamond': 83.2, 'bridge': 58.46, 'ground-track-field': 64.04, 'small-vehicle': 78.57, 'large-vehicle': 78.15, 'ship': 86.08, 'tennis-court': 89.8, 'basketball-court': 77.77, 'storage-tank': 82.95, 'soccer-ball-field': 49.53, 'roundabout': 68.76, 'harbor': 72.37, 'swimming-pool': 71.03, 'helicopter': 58.53}

Evaluating in DOTA obb Task
mAP: 72.59
class metrics: {'Task': 'obb', 'plane': 88.58, 'baseball-diamond': 82.9, 'bridge': 53.55, 'ground-track-field': 64.39, 'small-vehicle': 78.58, 'large-vehicle': 74.3, 'ship': 86.58, 'tennis-court': 90.06, 'basketball-court': 77.95, 'storage-tank': 82.32, 'soccer-ball-field': 49.49, 'roundabout': 64.36, 'harbor': 65.16, 'swimming-pool': 66.99, 'helicopter': 63.61}

"""
# model settings
model = dict(
    type='CenterMapOBB',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[8],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=dict(
        type='SharedFCBBoxHead',
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=16,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
    mask_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=14, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    mask_head=dict(
        type='CenterMapHead',
        num_convs=10,
        in_channels=256,
        conv_out_channels=256,
        num_classes=16,
        loss_mask=dict(
            type='CenterMapLoss', use_mask_weight=True, use_mask=False, loss_weight=3.0)))
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        mask_size=28,
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05,
        nms=dict(type='nms', iou_thr=0.5),
        max_per_img=1000,
        mask_thr_binary=0.5))
# dataset settings
dataset_type = 'DOTADataset'
dataset_version = 'v1'
train_rate = '1.0'                  # 1.0_0.5 or 1.0
val_rate = '1.0'                    # 1.0_0.5 or 1.0
test_rate = '1.0'
data_root = './data/dota/{}/coco/'.format(dataset_version)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', 
        with_bbox=True, 
        with_mask=True, 
        with_reverse_mask_weight=True,
        poly2mask=False, 
        poly2centermap=True, 
        centermap_encode='centerness', 
        centermap_rate=0.5, 
        centermap_factor=4),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_mask_weights']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/dota_trainval_{}_{}_best_keypoint.json'.format(dataset_version, train_rate),
        img_prefix=data_root + 'trainval/',
        pipeline=train_pipeline,
        min_area=36,
        max_small_length=8),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/dota_test_{}_{}_best_keypoint_no_ground_truth.json'.format(dataset_version, val_rate),
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/dota_test_{}_{}_best_keypoint.json'.format(dataset_version, val_rate),
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric=['bbox', 'segm'])
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/dota_v013_centermap_obb_r50_10conv_v1_trainval'
load_from = None
resume_from = None
workflow = [('train', 1)]
