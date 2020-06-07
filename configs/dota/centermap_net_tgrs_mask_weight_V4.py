"""
IOU = 0.5
Task	baseball-diamond	basketball-court	bridge	ground-track-field	harbor	helicopter	large-vehicle	mAP	plane	roundabout	ship	small-vehicle	soccer-ball-field	storage-tank	swimming-pool	tennis-court
hbb	85.09	84.43	59.92	69.43	79.38	64.8	80.69	77.44	89.79	69.1	86.6	79.14	56.44	86.12	80.55	90.16
obb	84.74	84.96	54.47	70.38	73.76	66.06	79.54	76.23	89.89	69.16	87.18	78.39	57.58	85.34	71.53	90.45

Evaluating in DOTA hbb Task
mAP: 77.44
class metrics: {'Task': 'hbb', 'plane': 89.79, 'baseball-diamond': 85.09, 'bridge': 59.92, 'ground-track-field': 69.43, 'small-vehicle': 79.14, 'large-vehicle': 80.69, 'ship': 86.6, 'tennis-court': 90.16,  'basketball-court': 84.43, 'storage-tank': 86.12, 'soccer-ball-field': 56.44, 'roundabout': 69.1, 'harbor': 79.38, 'swimming-pool': 80.55, 'helicopter': 64.8}

Evaluating in DOTA obb Task
mAP: 76.23
class metrics: {'Task': 'obb', 'plane': 89.89, 'baseball-diamond': 84.74, 'bridge': 54.47, 'ground-track-field': 70.38, 'small-vehicle': 78.39, 'large-vehicle': 79.54, 'ship': 87.18, 'tennis-court': 90.45, 'basketball-court': 84.96, 'storage-tank': 85.34, 'soccer-ball-field': 57.58, 'roundabout': 69.16, 'harbor': 73.76, 'swimming-pool': 71.53, 'helicopter': 66.06}


IOU=0.7
Evaluating in DOTA hbb Task
mAP: 62.26
class metrics: {'Task': 'hbb', 'plane': 84.86, 'baseball-diamond': 70.54, 'bridge': 36.18, 'ground-track-field': 64.27, 'small-vehicle': 67.77, 'large-vehicle': 71.17, 'ship': 76.64, 'tennis-court': 88.95, 'basketball-court': 77.5, 'storage-tank': 75.29, 'soccer-ball-field': 47.81, 'roundabout': 45.32, 'harbor': 60.02, 'swimming-pool': 33.75, 'helicopter': 33.83}

Evaluating in DOTA obb Task
mAP: 54.13
class metrics: {'Task': 'obb', 'plane': 79.46, 'baseball-diamond': 61.22, 'bridge': 20.28, 'ground-track-field': 62.23, 'small-vehicle': 47.59, 'large-vehicle': 54.89, 'ship': 63.03, 'tennis-court': 89.56, 'basketball-court': 78.38, 'storage-tank': 74.09, 'soccer-ball-field': 50.44, 'roundabout': 43.73, 'harbor': 41.36, 'swimming-pool': 18.36, 'helicopter': 27.39}

official:
mAP: 0.7612941118523574
ap of each class: plane:0.8983448785172954, baseball-diamond:0.8435986272925508, bridge:0.5473957651887944, ground-track-field:0.7030901542828081, small-vehicle:0.7771216296268206, large-vehicle:0.7851383004608076, ship:0.8722357225953453, tennis-court:0.9064080739220957, basketball-court:0.8502347537198977, storage-tank:0.8540361396813172, soccer-ball-field:0.5732232539009823, roundabout:0.6920088837539076, harbor:0.7407532505639888, swimming-pool:0.715299174741412, helicopter:0.6605230695373374


no soft-nms

mAP: 0.7603393385541147
ap of each class: plane:0.8983235169332907, baseball-diamond:0.8441050257762618, bridge:0.545956295058921, ground-track-field:0.7024872434867073, small-vehicle:0.7766137074062566, large-vehicle:0.783176339035878, ship:0.8718588452659245, tennis-court:0.9065517022043408, basketball-court:0.8488915831920468, storage-tank:0.8526865520313235, soccer-ball-field:0.564610058143582, roundabout:0.6922926707328085, harbor:0.7412754925764213, swimming-pool:0.7156452946474171, helicopter:0.6606157518205397

mAP: 77.33
class metrics: {'Task': 'hbb', 'plane': 89.7, 'baseball-diamond': 84.92, 'bridge': 59.72, 'ground-track-field': 67.96, 'small-vehicle': 79.16, 'large-vehicle': 80.66, 'ship': 86.61, 'tennis-court': 90.47, 'basketball-court': 84.47, 'storage-tank': 86.19, 'soccer-ball-field': 56.42, 'roundabout': 69.0, 'harbor': 79.33, 'swimming-pool': 80.53, 'helicopter': 64.81}


"""
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
        nms=dict(type='nms', iou_thr=0.5),
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
        pipeline=test_pipeline,
        evaluation_iou_threshold=0.5))
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
