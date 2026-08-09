# Vendored rotcert HRSC2016-MS RTMDet-R-l 3x config (3rd dataset arm, 2026-07-13).
# Self-contained sibling of the DIOR-R config (rotated_rtmdet_l-3x-dior.py): the model
# is the mmrotate RTMDet-R-l recipe at commit 3ff004eb..., restated inline, with:
#   * num_classes 20 -> 1 (HRSC canonical single-class 'ship'; classwise=False).
#   * dataset: DOTADataset over the HRSC2016-MS->DOTA-format conversion (prepare_hrsc.py),
#     source (canonical) train split, 800x800 keep_ratio flip-aug recipe (consistent
#     with the ORCNN arm; drops DOTA's RandomRotate -- recorded recipe delta).
#   * SyncBN -> BN (single-GPU non-distributed) and pad_size_divisor=32 (HRSC images
#     are not uniform 800x800 after keep_ratio resize; CSPNeXt-PAFPN needs stride-32
#     divisible inputs -- same fix disclosed for DIOR-R).
#   * FilterAnnotations(min_gt_bbox_wh=(1e-2,1e-2)) zero-area-GT NaN guard.
# Inherits the rotated_rtmdet base runtime + 3x schedule (AdamW; the DIOR-R arm used
# these exact bases successfully). Deploy into /root/mmrotate/configs/rotated_rtmdet/.
_base_ = ['./_base_/default_runtime.py', './_base_/schedule_3x.py']
checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-l_8xb256-rsb-a1-600e_in1k-6a760974.pth'  # noqa

dataset_type = 'DOTADataset'
data_root = '/root/autodl-tmp/hrsc_dota/'
hrsc_metainfo = dict(classes=('ship',))

angle_version = 'le90'
model = dict(
    type='mmdet.RTMDet',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        boxtype2tensor=False,
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
        num_classes=1,
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

# Degenerate-annotation guard + qbox load (matches DOTA-format annfiles).
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
    batch_size=8,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        metainfo=hrsc_metainfo,
        data_root=data_root,
        ann_file='train/annfiles/',
        data_prefix=dict(img_path='train/images/'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline))

# Train-only: disable validation/test (inference via score_rtmdet.py). Also drop any
# base PipelineSwitchHook (stage-2 pipeline switch) so the single train_pipeline above
# is authoritative for all epochs.
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='mmdet.ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
]
val_cfg = None
val_dataloader = None
val_evaluator = None
test_cfg = None
test_dataloader = None
test_evaluator = None
