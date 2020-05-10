"""
description: test mask loss weight parameters in centerness, keep rate = 0.5 
    + baseline                              (v200)
    + heatmap_encode = centerness

parameter:
    centerness:
        factor = 4
        rate = 0.5
        num_conv = 4

results:
-----------------------------------rbbox---------------------------------
mAP: 0.750171906754792
classaps:  {'plane': 0.8994695029326246, 'baseball-diamond': 0.8634491024879912, 'bridge': 0.5360252570634132, 'ground-track-field': 0.7377472668286548, 'small-vehicle': 0.7432763709555644, 'large-vehicle': 0.7136906067961293, 'ship': 0.8590867052521359, 'tennis-court': 0.9066761192635302, 'basketball-court': 0.8380549625227149, 'storage-tank': 0.8683529511778888, 'soccer-ball-field': 0.5711044215134543, 'roundabout': 0.6942587460409484, 'harbor': 0.7256642112262935, 'swimming-pool': 0.5998309248235874, 'helicopter': 0.6958914524369475}
-----------------------------------bbox---------------------------------
mAP: 0.7747130782916638
classaps:  {'plane': 0.900712064782724, 'baseball-diamond': 0.8645322769129618, 'bridge': 0.6122123531670073, 'ground-track-field': 0.7426270596096667, 'small-vehicle': 0.786316355506039, 'large-vehicle': 0.7230137080059935, 'ship': 0.8575815883027148, 'tennis-court': 0.904278457649805, 'basketball-court': 0.8374066637689531, 'storage-tank': 0.8691790558684818, 'soccer-ball-field': 0.5696684666701123, 'roundabout': 0.7093049336858089, 'harbor': 0.7813398719364196, 'swimming-pool': 0.7646924074711645, 'helicopter': 0.6978309110371044}


"""
# model settings
model = dict(
    type='MaskOBBRCNN',
    pretrained='open-mmlab://resnext101_32x4d',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
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
        use_sigmoid_cls=True),
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
        reg_class_agnostic=False),
    mask_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=14, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    mask_head=dict(
        type='FCNMaskOBBHead',
        num_convs=4,
        in_channels=256,
        conv_out_channels=256,
        num_classes=16,
        loss_function='mask_mse',
        loss_weight=3.0),
    semantic_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=14, sample_num=2),
        out_channels=256,
        featmap_strides=[2]),
    semantic_head=dict(
        type='PanopticPseudomaskSemanticHead',
        num_ins=5,
        fusion_level=1,
        num_convs=1,
        in_channels=256,
        inside_channels=128,
        conv_out_channels=256,
        num_classes=16,
        ignore_label=255,
        loss_weight=1.0,
        use_focal_loss=True,
        with_background_reweight=True,
        reweight_version='v1'))   # mask_cross_entropy or mask_lovase_softmax
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
        smoothl1_beta=1 / 9.0,
        debug=False),
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
        nms=dict(type='soft_nms', iou_thr=0.5),
        max_per_img=1000,
        mask_thr_binary=0.5))
# dataset settings
dataset_type = 'DotaDataset'
dota_version = 'v1.0'
dataset_version = 'v4'
train_rate = '1.0_0.5'                  # 1.0_0.5 or 1.0
val_rate = '1.0_0.5'                    # 1.0_0.5 or 1.0
test_rate = '1.0_0.5'
mask2rbbox_function = 'minAreaRect'         # minAreaRect or quadrilateral or ellipse
data_root = './data/dota/{}/coco/'.format(dataset_version)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/dota_trainval_{}_{}_best_keypoint.json'.format(dataset_version, train_rate),
        img_prefix=data_root + 'trainval/',
        img_scale=[(1024, 1024), (896, 896), (768, 768)],
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=True,
        with_crowd=True,
        with_label=True,
        random_rotation=True,
        with_heatmapmask=True,
        heatmap_encode='centerness',
        heatmap_rate=0.5,
        heatmap_factor=4,
        with_semantic_seg=True,
        seg_prefix=data_root + 'obb_seg/',
        seg_scale_factor=1 / 2,
        with_pseudomask=True,
        pseudomask_prefix=data_root + 'centerness_seg'),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/dota_test_{}_{}_best_keypoint.json'.format(dataset_version, val_rate),
        img_prefix=data_root + 'test/',
        img_scale=[(1024, 1024), (896, 896), (768, 768)],
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=True,
        with_crowd=True,
        with_label=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/dota_test_{}_{}_best_keypoint.json'.format(dataset_version, val_rate),
        img_prefix=data_root + 'test/',
        img_scale=[(1024, 1024), (896, 896), (768, 768)],
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True))
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
work_dir = './work_dirs/dota_v9225'
load_from = None
resume_from = None
workflow = [('train', 1)]
