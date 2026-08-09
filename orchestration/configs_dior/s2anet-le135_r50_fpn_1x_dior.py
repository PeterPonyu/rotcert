# Vendored rotcert DIOR-R S2A-Net config (Config B cross-family breadth, 2026-07-13).
# Merge of two vendored parents:
#   MODEL block  <- configs_dota_zoo/s2anet-le135_r50_fpn_1x_dota.py (the mmrotate
#     dev-1.x S2A-Net R-50 FPN 1x recipe at commit
#     3ff004eb21ea040455b5585db229edba4037f1bf: RefineSingleStageDetector, one-stage
#     feature-alignment refine). This is a dev-1.x config -- the DOTA zoo EXCLUDED
#     S2A-Net from its inference arm because the released S2A-Net *checkpoint* is
#     v0.1.0-era and incompatible with mmrotate dev-1.x; in-house DIOR-R TRAINING with
#     this dev-1.x config has no such incompatibility (we build + train the model from
#     the dev-1.x config, never load a v0.1.0 checkpoint), so S2A-Net is admissible here.
#   DATASET / SCHEDULE / OPTIMIZER conventions  <- configs_dior/
#     oriented-rcnn-le90_r50_fpn_1x_dior.py (DIOR-R base dataset, 1x schedule, AdamW +
#     FilterAnnotations stability fixes -- restated below verbatim).
#
# INTENTIONAL differences vs the two parents (everything else is byte-identical):
#   vs the DOTA S2A-Net parent:
#     * _base_ dataset  dota.py -> dior.py  (DIORDataset oriented annotations).
#     * bbox_head_init.num_classes 15 -> 20  and  bbox_head_refine[0].num_classes 15 -> 20
#       (DIOR-R has 20 classes; verified against oriented-rcnn-le90_r50_fpn_1x_dior.py
#       num_classes=20 and rotated_rtmdet_l-3x-dior.py num_classes=20).
#     * ADD an AdamW optim_wrapper (the DOTA S2A-Net parent has NO optim override -> it
#       would inherit schedule_1x's SGD, which NaNs on DIOR-R). Same disclosed AdamW
#       lr=1e-4 + grad-clip recipe the DIOR ORCNN/RTMDet arms use.
#     * train_pipeline + train_dataloader restated with mmdet.FilterAnnotations
#       (min_gt_bbox_wh=(1e-2,1e-2)) after Resize -- the disclosed zero-area-GT NaN guard.
#   vs the DIOR ORCNN parent:
#     * the whole model block is S2A-Net (RefineSingleStageDetector), not Oriented R-CNN.
#     * angle_version stays le135 -- this is S2A-Net's NATIVE convention (its
#       DeltaXYWHTRBBoxCoder / FakeRotatedAnchorGenerator are tuned for le135), a
#       PER-DETECTOR field, not a dataset field; the DIOR ORCNN arm is le90. This is
#       inert downstream: score_rtmdet.py canonicalizes every prediction to le90 before
#       writing the detections JSONL, so certification is convention-uniform regardless.
# Deploy into mmrotate/configs/s2anet/ (training runs with cwd=/root/mmrotate so the base
# dior.py's relative data_root 'data/DIOR/' resolves against the box data symlink).
_base_ = [
    '../_base_/datasets/dior.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

angle_version = 'le135'
model = dict(
    type='RefineSingleStageDetector',
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
        zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head_init=dict(
        type='S2AHead',
        num_classes=20,
        in_channels=256,
        stacked_convs=2,
        feat_channels=256,
        anchor_generator=dict(
            type='FakeRotatedAnchorGenerator',
            angle_version=angle_version,
            scales=[4],
            ratios=[1.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHTRBBoxCoder',
            angle_version=angle_version,
            norm_factor=1,
            edge_swap=False,
            proj_xy=True,
            target_means=(.0, .0, .0, .0, .0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0),
            use_box_type=False),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='mmdet.SmoothL1Loss', beta=0.11, loss_weight=1.0)),
    bbox_head_refine=[
        dict(
            type='S2ARefineHead',
            num_classes=20,
            in_channels=256,
            stacked_convs=2,
            feat_channels=256,
            frm_cfg=dict(
                type='AlignConv',
                feat_channels=256,
                kernel_size=3,
                strides=[8, 16, 32, 64, 128]),
            anchor_generator=dict(
                type='PseudoRotatedAnchorGenerator',
                strides=[8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHTRBBoxCoder',
                angle_version=angle_version,
                norm_factor=1,
                edge_swap=False,
                proj_xy=True,
                target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
                target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
            loss_cls=dict(
                type='mmdet.FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(
                type='mmdet.SmoothL1Loss', beta=0.11, loss_weight=1.0))
    ],
    train_cfg=dict(
        init=dict(
            assigner=dict(
                type='mmdet.MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='RBboxOverlaps2D')),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        refine=[
            dict(
                assigner=dict(
                    type='mmdet.MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.4,
                    min_pos_iou=0,
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type='RBboxOverlaps2D')),
                allowed_border=-1,
                pos_weight=-1,
                debug=False)
        ],
        stage_loss_weights=[1.0]),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms_rotated', iou_threshold=0.1),
        max_per_img=2000))

# DIOR-R disclosed stability recipe (carried over verbatim from
# oriented-rcnn-le90_r50_fpn_1x_dior.py): SGD diverges to NaN on DIOR-R; AdamW lr 1e-4
# + grad clip is the mmrotate-sanctioned stable recipe. ADDED here (the DOTA S2A-Net
# parent carries no optim override and would inherit schedule_1x's SGD).
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
