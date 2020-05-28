"""
Evaluating in DOTA hbb Task
mAP: 53.57
class metrics: {'Task': 'hbb', 'plane': 76.92, 'baseball-diamond': 47.53, 'bridge': 26.45, 'ground-track-field': 51.92, 'small-vehicle': 48.17, 'large-vehicle': 72.24, 'ship': 78.27, 'tennis-court': 88.33, 'basketball-court': 45.62, 'storage-tank': 68.7, 'soccer-ball-field': 33.24, 'roundabout': 46.19, 'harbor': 51.06, 'swimming-pool': 27.43, 'helicopter': 41.54}

Evaluating in DOTA obb Task
mAP: 46.32
class metrics: {'Task': 'obb', 'plane': 77.31, 'baseball-diamond': 46.05, 'bridge': 18.47, 'ground-track-field': 49.17, 'small-vehicle': 27.57, 'large-vehicle': 59.7, 'ship': 62.43, 'tennis-court': 89.8, 'basketball-court': 49.74, 'storage-tank': 67.5, 'soccer-ball-field': 36.74, 'roundabout': 46.07, 'harbor': 27.49, 'swimming-pool': 19.83, 'helicopter': 16.96}
"""
# model settings
model = dict(
    type='RBBoxRCNN',
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
    rbbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    rbbox_head=dict(
        type='RBBoxHead',
        num_shared_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=16,
        out_dim_reg=5,
        target_means=[0., 0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2, 0.1],
        reg_class_agnostic=False,
        encode='thetaobb',
        loss_rbbox_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_rbbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)))
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
        score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=1000),
    rbbox=dict(
        encode='thetaobb',
        score_thr=0.05,
        polygon_nms_iou_thr=0.5,
        max_per_img=1000,
        parallel=True)
    # soft-nms is also supported for rcnn testing
    # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)
)
# dataset settings
dataset_type = 'DOTADataset'
dota_version = 'v1.0'
dataset_version = 'v1'
train_rate = '1.0'                  # 1.0_0.5 or 1.0
val_rate = '1.0'                    # 1.0_0.5 or 1.0
data_root = './data/dota/{}/coco/'.format(dataset_version)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_rbbox=True),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Pointobb2RBBox', encoding_method='thetaobb'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_rbboxes']),
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
        encode='thetaobb'),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/dota_test_{}_{}_best_keypoint_no_ground_truth.json'.format(dataset_version, val_rate),
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline,
        encode='thetaobb'),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/dota_test_{}_{}_best_keypoint_no_ground_truth.json'.format(dataset_version, val_rate),
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline,
        evaluation_iou_threshold=0.5,
        encode='thetaobb'))
evaluation = dict(interval=1, metric='bbox')
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
work_dir = './work_dirs/dota_v002_theta_obb_r50_v1_train'
load_from = None
resume_from = None
workflow = [('train', 1)]
