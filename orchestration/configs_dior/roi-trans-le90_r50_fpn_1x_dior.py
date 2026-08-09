# Vendored rotcert DIOR-R RoI Transformer config (Config B cross-family breadth,
# 2026-07-13). Merge of two vendored parents:
#   MODEL block  <- configs_dota_zoo/roi-trans-le90_r50_fpn_1x_dota.py (the mmrotate
#     dev-1.x RoI Transformer R-50 FPN 1x recipe at commit
#     3ff004eb21ea040455b5585db229edba4037f1bf: mmdet.CascadeRCNN, 2 stages, rotated
#     RoI refinement). This is a dev-1.x config -- so unlike the DOTA-zoo *checkpoint*
#     (which the DOTA arm can consume inference-only), in-house DIOR-R training with a
#     dev-1.x config has NO version-incompatibility issue; nothing here is v0.1.0-era.
#   DATASET / SCHEDULE / OPTIMIZER conventions <- configs_dior/
#     oriented-rcnn-le90_r50_fpn_1x_dior.py (DIOR-R base dataset, 1x schedule, AdamW +
#     FilterAnnotations stability fixes -- restated below verbatim).
#
# INTENTIONAL differences vs the two parents (everything else is byte-identical):
#   vs the DOTA RoI-Trans parent:
#     * _base_ dataset  dota.py -> dior.py  (DIORDataset oriented annotations).
#     * bbox_head[0].num_classes 15 -> 20  and  bbox_head[1].num_classes 15 -> 20
#       (DIOR-R has 20 classes; verified against oriented-rcnn-le90_r50_fpn_1x_dior.py
#       roi_head.bbox_head.num_classes=20 and rotated_rtmdet_l-3x-dior.py num_classes=20).
#     * optimizer  SGD lr=0.005 (DOTA parent's trailing optim_wrapper)  ->  AdamW lr=1e-4
#       + grad clip  (the DIOR-R disclosed stability recipe: SGD diverges to NaN on
#       DIOR-R; carried over from the DIOR ORCNN/RTMDet arms verbatim).
#     * train_pipeline + train_dataloader restated with mmdet.FilterAnnotations
#       (min_gt_bbox_wh=(1e-2,1e-2)) after Resize -- the disclosed zero-area-GT NaN guard
#       (DIOR-R trainval carries 2 zero-area boxes; the base dior.py pipeline lacks it).
#   vs the DIOR ORCNN parent:
#     * the whole model block is RoI Transformer (CascadeRCNN 2-stage), not Oriented R-CNN.
# angle_version stays le90 (RoI-Trans DOTA parent + both DIOR parents all le90).
# Deploy into mmrotate/configs/roi_trans/ (training runs with cwd=/root/mmrotate so the
# base dior.py's relative data_root 'data/DIOR/' resolves against the box data symlink).
_base_ = [
    '../_base_/datasets/dior.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

angle_version = 'le90'
model = dict(
    type='mmdet.CascadeRCNN',
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
        type='mmdet.RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='mmdet.AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
            use_box_type=True),
        bbox_coder=dict(
            type='DeltaXYWHHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
            use_box_type=True),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        type='mmdet.CascadeRoIHead',
        num_stages=2,
        stage_loss_weights=[1, 1],
        bbox_roi_extractor=[
            dict(
                type='mmdet.SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            dict(
                type='RotatedSingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlignRotated',
                    out_size=7,
                    sample_num=2,
                    clockwise=True),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
        ],
        bbox_head=[
            dict(
                type='mmdet.Shared2FCBBoxHead',
                predict_box_type='rbox',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=20,
                reg_predictor_cfg=dict(type='mmdet.Linear'),
                cls_predictor_cfg=dict(type='mmdet.Linear'),
                bbox_coder=dict(
                    type='DeltaXYWHTHBBoxCoder',
                    angle_version=angle_version,
                    norm_factor=2,
                    edge_swap=True,
                    target_means=(.0, .0, .0, .0, .0),
                    target_stds=(0.1, 0.1, 0.2, 0.2, 0.1),
                    use_box_type=True),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='mmdet.CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(
                    type='mmdet.SmoothL1Loss', beta=1.0, loss_weight=1.0)),
            dict(
                type='mmdet.Shared2FCBBoxHead',
                predict_box_type='rbox',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=20,
                reg_predictor_cfg=dict(type='mmdet.Linear'),
                cls_predictor_cfg=dict(type='mmdet.Linear'),
                bbox_coder=dict(
                    type='DeltaXYWHTRBBoxCoder',
                    angle_version=angle_version,
                    norm_factor=None,
                    edge_swap=True,
                    proj_xy=True,
                    target_means=[0., 0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1, 0.05]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='mmdet.CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(
                    type='mmdet.SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]),
    # model training and testing settings
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
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='mmdet.MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type='RBbox2HBboxOverlaps2D')),
                sampler=dict(
                    type='mmdet.RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='mmdet.MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type='RBboxOverlaps2D')),
                sampler=dict(
                    type='mmdet.RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms_rotated', iou_threshold=0.1),
            max_per_img=2000)))

# DIOR-R disclosed stability recipe (carried over verbatim from
# oriented-rcnn-le90_r50_fpn_1x_dior.py): SGD diverges to NaN on DIOR-R; AdamW
# lr 1e-4 + grad clip is the mmrotate-sanctioned stable recipe. This _delete_=True
# block replaces the DOTA RoI-Trans parent's trailing `optim_wrapper=dict(
# optimizer=dict(lr=0.005))` (SGD).
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.05),
    clip_grad=dict(max_norm=35, norm_type=2))


# Degenerate-annotation guard (2026-07-10, probed live): DIOR-R trainval
# carries exactly 2 zero-area oriented boxes (images 04137, 07007); with a
# fixed shuffle seed they deterministically NaN the loss at ~iter 200. Fix =
# mmrotate's own sanctioned in-pipeline filter (cf. _base_/datasets/hrsid.py):
# FilterAnnotations(min_gt_bbox_wh=(1e-2, 1e-2)) after Resize. The base dior.py
# pipeline lacks it, so train_pipeline + train_dataloader are restated here with
# the single filter insertion (identical to the DIOR ORCNN/RTMDet arms).
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
    dataset=dict(
        type='ConcatDataset',
        ignore_keys=['DATASET_TYPE'],
        datasets=[
            dict(
                type='DIORDataset',
                data_root='data/DIOR/',
                ann_file='ImageSets/Main/train.txt',
                data_prefix=dict(img_path='JPEGImages-trainval'),
                filter_cfg=dict(filter_empty_gt=True),
                pipeline=train_pipeline),
            dict(
                type='DIORDataset',
                data_root='data/DIOR/',
                ann_file='ImageSets/Main/val.txt',
                data_prefix=dict(img_path='JPEGImages-trainval'),
                filter_cfg=dict(filter_empty_gt=True),
                pipeline=train_pipeline,
                backend_args=None)
        ]))
