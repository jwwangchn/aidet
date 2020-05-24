norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
# model settings
model = dict(
    type='CenterMapOBB',
    pretrained='torchvision://resnet101',
    backbone=dict(
        type='ResNet',
        depth=101,
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
            type='CenterMapLoss', use_mask_weight=True, use_mask=False, loss_weight=3.0)),
    semantic_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=14, sample_num=2),
        out_channels=256,
        featmap_strides=[4]),
    semantic_head=dict(
        type='WeightedPseudoSegmentationHead',
        num_convs=1,
        in_channels=256,
        inside_channels=128,
        conv_out_channels=256,
        num_classes=16,
        ignore_label=255,
        loss_weight=1.0,
        use_focal_loss=True,
        with_background_reweight=True,
        reweight_version='v1',
        norm_cfg=norm_cfg))
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1,
            gpu_assign_thr=512),
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
            ignore_iof_thr=-1,
            gpu_assign_thr=512),
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
        nms=dict(type='soft_nms', iou_thr=0.5),
        max_per_img=1000,
        mask_thr_binary=0.5))
# dataset settings
dataset_type = 'DOTADataset'
dataset_version = 'v4'
train_rate = '1.0_0.5'                  # 1.0_0.5 or 1.0
val_rate = '1.0_0.5'                    # 1.0_0.5 or 1.0
test_rate = '1.0_0.5'
data_root = './data/dota/{}/coco/'.format(dataset_version)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', 
        with_bbox=True, 
        with_mask=True,
        with_mask_weight=True,
        with_seg=True,
        with_heatmap_weight=True, 
        poly2mask=False, 
        poly2centermap=True, 
        centermap_encode='centerness', 
        centermap_rate=0.5, 
        centermap_factor=4),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomRotate', rotate_ratio=1.0, choice=(0, 90, 180, 270)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='SegRescale', scale_factor=1 / 4),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg', 'gt_heatmap_weight', 'gt_mask_weights']),
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
        seg_prefix=data_root + 'pseudo_segmentation/',
        heatmap_weight_prefix=data_root + 'heatmap_weight/',
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
        ann_file=data_root + 'annotations/dota_test_{}_{}_best_keypoint_no_ground_truth.json'.format(dataset_version, val_rate),
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline))
evaluation = dict(interval=2, 
                  metric=['hbb', 'obb'], 
                  submit_path='./results/dota/centermap_net_tgrs_mask_weight_V4', 
                  annopath='./data/dota/v0/test/labelTxt-v1.0/{:s}.txt', 
                  imageset_file='./data/dota/v0/test/testset.txt', 
                  excel='./results/dota/centermap_net_tgrs_mask_weight_V4/centermap_net_tgrs_mask_weight_V4.xlsx', 
                  jsonfile_prefix='./results/dota/centermap_net_tgrs_mask_weight_V4')
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
work_dir = './work_dirs/centermap_net_tgrs_mask_weight_V4'
load_from = None
resume_from = None
workflow = [('train', 1)]