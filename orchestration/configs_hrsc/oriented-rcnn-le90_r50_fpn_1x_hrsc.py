# Vendored rotcert HRSC2016-MS training config (3rd oriented-detection dataset arm,
# 2026-07-13). Self-contained sibling of the DIOR-R config
# (oriented-rcnn-le90_r50_fpn_1x_dior.py): the model is the mmrotate Oriented R-CNN
# R-50 FPN 1x recipe at commit 3ff004eb21ea040455b5585db229edba4037f1bf, restated
# inline (no dataset _base_ dependency), with:
#   * num_classes 20 -> 1  (HRSC canonical single-class 'ship' benchmark; the
#     fine-grained HRSC ship-type ids are collapsed, mirroring mmrotate's default
#     HRSCDataset(classwise=False) -- HRSC contributes ONE per-class Mondrian cell).
#   * dataset: DOTADataset over the HRSC2016-MS->DOTA-format conversion
#     (prepare_hrsc.py: robndbox5 <robndbox> XML -> 8-point-poly annfiles), source
#     (canonical) train split; images resized to 800x800 keep_ratio (HRSC images are
#     ~1166x753, not uniform 1024 crops).
#   * SAME disclosed stability fixes as DIOR-R: mmdet FilterAnnotations
#     (min_gt_bbox_wh=(1e-2,1e-2)) after Resize (zero-area-GT NaN guard) + AdamW.
# Deploy into /root/mmrotate/configs/rotcert_hrsc/.
_base_ = ['../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py']

# --- HRSC2016-MS (DOTA-format conversion) dataset roots (env-independent absolutes) ---
dataset_type = 'DOTADataset'
data_root = '/root/autodl-tmp/hrsc_dota/'
hrsc_metainfo = dict(classes=('ship',))

angle_version = 'le90'
model = dict(
    type='mmdet.FasterRCNN',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        boxtype2tensor=False),
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='OrientedRPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='mmdet.AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
            use_box_type=True),
        bbox_coder=dict(
            type='MidpointOffsetCoder',
            angle_version=angle_version,
            target_means=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5]),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss',
            beta=0.1111111111111111,
            loss_weight=1.0)),
    roi_head=dict(
        type='mmdet.StandardRoIHead',
        bbox_roi_extractor=dict(
            type='RotatedSingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlignRotated',
                out_size=7,
                sample_num=2,
                clockwise=True),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='mmdet.Shared2FCBBoxHead',
            predict_box_type='rbox',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            reg_predictor_cfg=dict(type='mmdet.Linear'),
            cls_predictor_cfg=dict(type='mmdet.Linear'),
            bbox_coder=dict(
                type='DeltaXYWHTRBBoxCoder',
                angle_version=angle_version,
                norm_factor=None,
                edge_swap=True,
                proj_xy=True,
                target_means=(.0, .0, .0, .0, .0),
                target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
            loss_bbox=dict(
                type='mmdet.SmoothL1Loss', beta=1.0, loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='mmdet.MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='RBbox2HBboxOverlaps2D')),
            sampler=dict(
                type='mmdet.RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='mmdet.MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                iou_calculator=dict(type='RBboxOverlaps2D'),
                ignore_iof_thr=-1),
            sampler=dict(
                type='mmdet.RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms_rotated', iou_threshold=0.1),
            max_per_img=2000)))

# AdamW (DIOR-R disclosed recipe delta, carried over: SGD NaNs on this small-object
# oriented data; AdamW lr 1e-4 + grad clip is the mmrotate-sanctioned stable recipe).
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.05),
    clip_grad=dict(max_norm=35, norm_type=2))

# Degenerate-annotation guard (mmdet FilterAnnotations after Resize) -- same fix the
# DIOR-R/DOTA arms disclose; qbox (8-point) load matches the DOTA-format annfiles.
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=None),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(800, 800), keep_ratio=True),
    dict(type='mmdet.FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.PackDetInputs')
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        metainfo=hrsc_metainfo,
        data_root=data_root,
        ann_file='train/annfiles/',
        data_prefix=dict(img_path='train/images/'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline))

# Train-only: validation/test disabled (inference is done by score_rtmdet.py, not
# tools/test.py; keeps the config free of HRSC eval scaffolding).
val_cfg = None
val_dataloader = None
val_evaluator = None
test_cfg = None
test_dataloader = None
test_evaluator = None
