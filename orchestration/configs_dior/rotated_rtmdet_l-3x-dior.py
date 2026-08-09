# Vendored rotcert DIOR-R training config (design A2 addendum 2026-07-10).
# Derived from rotated_rtmdet_l-3x-dota.py at mmrotate commit
# 3ff004eb21ea040455b5585db229edba4037f1bf with three changes:
#   dataset base dota_rr.py -> ../_base_/datasets/dior.py (800x800 flip-aug
#   recipe, consistent with the ORCNN arm + AOPG conventions; drops DOTA's
#   RandomRotate aug -- recorded as a recipe delta in the prereg);
#   num_classes 15 -> 20; SyncBN -> BN (single-GPU non-distributed train --
#   SyncBN requires an initialized process group; identical semantics on 1 GPU).
# Deploy into mmrotate/configs/rotated_rtmdet/.
_base_ = [
    './_base_/default_runtime.py', './_base_/schedule_3x.py',
    '../_base_/datasets/dior.py'
]
checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-l_8xb256-rsb-a1-600e_in1k-6a760974.pth'  # noqa

angle_version = 'le90'
model = dict(
    type='mmdet.RTMDet',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        boxtype2tensor=False,
        # DIOR test images are not uniformly 800x800 (keep_ratio resize leaves
        # e.g. ~792-px sides); CSPNeXt-PAFPN's upsample-concat needs stride-32
        # divisible inputs or it crashes ("Expected size 50 but got size 49",
        # epoch-12 val, 2026-07-11). The DOTA parents never hit this because
        # DOTA crops are uniformly 1024x1024, so they omit the divisor pad.
        pad_size_divisor=32,
        batch_augments=None),
    backbone=dict(
        type='mmdet.CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=1,
        widen_factor=1,
        channel_attention=True,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint=checkpoint)),
    neck=dict(
        type='mmdet.CSPNeXtPAFPN',
        in_channels=[256, 512, 1024],
        out_channels=256,
        num_csp_blocks=3,
        expand_ratio=0.5,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU')),
    bbox_head=dict(
        type='RotatedRTMDetSepBNHead',
        num_classes=20,
        in_channels=256,
        stacked_convs=2,
        feat_channels=256,
        angle_version=angle_version,
        anchor_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(
            type='DistanceAnglePointCoder', angle_version=angle_version),
        loss_cls=dict(
            type='mmdet.QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='RotatedIoULoss', mode='linear', loss_weight=2.0),
        with_objectness=False,
        exp_on_reg=True,
        share_conv=True,
        pred_kernel_size=1,
        use_hbbox_loss=False,
        scale_angle=False,
        loss_angle=None,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU')),
    train_cfg=dict(
        assigner=dict(
            type='mmdet.DynamicSoftLabelAssigner',
            iou_calculator=dict(type='RBboxOverlaps2D'),
            topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms_rotated', iou_threshold=0.1),
        max_per_img=2000),
)

# batch_size set by the training chain via --cfg-options
# (RTMDET_R_BATCH, design A2: ~8 on 24GB at DIOR's 800x800)
train_dataloader = dict(num_workers=4)


# Degenerate-annotation guard (2026-07-10, probed live): DIOR-R trainval
# carries exactly 2 zero-area oriented boxes (images 04137, 07007); with a
# fixed shuffle seed they deterministically NaN the loss at ~iter 200 under
# SGD 0.005 / SGD 0.0025+clip / AdamW 1e-4 alike. Fix = mmrotate's own
# sanctioned in-pipeline filter (cf. _base_/datasets/hrsid.py):
# FilterAnnotations(min_gt_bbox_wh=(1e-2, 1e-2)) after Resize. The base
# dior.py pipeline lacks it, so train_pipeline + train_dataloader are
# restated here with the single filter insertion.
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
