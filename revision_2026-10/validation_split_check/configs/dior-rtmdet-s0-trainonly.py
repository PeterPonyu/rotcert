# dior-rtmdet-s0-trainonly: frozen DIOR-R validation-split check (DESIGN-FROZEN.md sha256 a3cfcfe14effa987295b7b1b0d3b6d9662a59db92e58aeb6258ef74cf8bd6aad).
# Formal dior-rtmdet-s0 input config (sha256 8070fbe1a36ed659e788b2354899cc803756c1a6846650a8c20b458a5cfd3819) with only these changes: test_dataloader.dataset.ann_file, test_dataloader.dataset.data_prefix.img_path, test_evaluator.out_file_path, train_dataloader.dataset.datasets, work_dir.
angle_version = 'le90'
backend_args = None
base_lr = 0.00025
checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-l_8xb256-rsb-a1-600e_in1k-6a760974.pth'
custom_hooks = [{'type': 'mmdet.NumClassCheckHook'},
 {'ema_type': 'mmdet.ExpMomentumEMA',
  'momentum': 0.0002,
  'priority': 49,
  'type': 'EMAHook',
  'update_buffers': True}]
data_root = 'data/DIOR/'
dataset_type = 'DIORDataset'
default_hooks = {'checkpoint': {'by_epoch': True,
                'interval': 1,
                'max_keep_ckpts': 1,
                'save_best': None,
                'save_last': True,
                'type': 'CheckpointHook'},
 'logger': {'interval': 50, 'type': 'LoggerHook'},
 'param_scheduler': {'type': 'ParamSchedulerHook'},
 'sampler_seed': {'type': 'DistSamplerSeedHook'},
 'timer': {'type': 'IterTimerHook'},
 'visualization': {'type': 'mmdet.DetVisualizationHook'}}
default_scope = 'mmrotate'
env_cfg = {'cudnn_benchmark': False,
 'dist_cfg': {'backend': 'nccl'},
 'mp_cfg': {'mp_start_method': 'fork', 'opencv_num_threads': 0}}
interval = 12
load_from = None
log_level = 'INFO'
log_processor = {'by_epoch': True, 'type': 'LogProcessor', 'window_size': 50}
max_epochs = 36
model = {'backbone': {'act_cfg': {'type': 'SiLU'},
              'arch': 'P5',
              'channel_attention': True,
              'deepen_factor': 1,
              'expand_ratio': 0.5,
              'init_cfg': {'checkpoint': 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-l_8xb256-rsb-a1-600e_in1k-6a760974.pth',
                           'prefix': 'backbone.',
                           'type': 'Pretrained'},
              'norm_cfg': {'type': 'BN'},
              'type': 'mmdet.CSPNeXt',
              'widen_factor': 1},
 'bbox_head': {'act_cfg': {'type': 'SiLU'},
               'anchor_generator': {'offset': 0, 'strides': [8, 16, 32], 'type': 'mmdet.MlvlPointGenerator'},
               'angle_version': 'le90',
               'bbox_coder': {'angle_version': 'le90', 'type': 'DistanceAnglePointCoder'},
               'exp_on_reg': True,
               'feat_channels': 256,
               'in_channels': 256,
               'loss_angle': None,
               'loss_bbox': {'loss_weight': 2.0, 'mode': 'linear', 'type': 'RotatedIoULoss'},
               'loss_cls': {'beta': 2.0,
                            'loss_weight': 1.0,
                            'type': 'mmdet.QualityFocalLoss',
                            'use_sigmoid': True},
               'norm_cfg': {'type': 'BN'},
               'num_classes': 20,
               'pred_kernel_size': 1,
               'scale_angle': False,
               'share_conv': True,
               'stacked_convs': 2,
               'type': 'RotatedRTMDetSepBNHead',
               'use_hbbox_loss': False,
               'with_objectness': False},
 'data_preprocessor': {'batch_augments': None,
                       'bgr_to_rgb': False,
                       'boxtype2tensor': False,
                       'mean': [103.53, 116.28, 123.675],
                       'pad_size_divisor': 32,
                       'std': [57.375, 57.12, 58.395],
                       'type': 'mmdet.DetDataPreprocessor'},
 'neck': {'act_cfg': {'type': 'SiLU'},
          'expand_ratio': 0.5,
          'in_channels': [256, 512, 1024],
          'norm_cfg': {'type': 'BN'},
          'num_csp_blocks': 3,
          'out_channels': 256,
          'type': 'mmdet.CSPNeXtPAFPN'},
 'test_cfg': {'max_per_img': 2000,
              'min_bbox_size': 0,
              'nms': {'iou_threshold': 0.1, 'type': 'nms_rotated'},
              'nms_pre': 2000,
              'score_thr': 0.05},
 'train_cfg': {'allowed_border': -1,
               'assigner': {'iou_calculator': {'type': 'RBboxOverlaps2D'},
                            'topk': 13,
                            'type': 'mmdet.DynamicSoftLabelAssigner'},
               'debug': False,
               'pos_weight': -1},
 'type': 'mmdet.RTMDet'}
optim_wrapper = {'optimizer': {'lr': 0.00025, 'type': 'AdamW', 'weight_decay': 0.05},
 'paramwise_cfg': {'bias_decay_mult': 0, 'bypass_duplicate': True, 'norm_decay_mult': 0},
 'type': 'OptimWrapper'}
param_scheduler = [{'begin': 0, 'by_epoch': False, 'end': 1000, 'start_factor': 1e-05, 'type': 'LinearLR'},
 {'T_max': 18,
  'begin': 18,
  'by_epoch': True,
  'convert_to_iter_based': True,
  'end': 36,
  'eta_min': 1.25e-05,
  'type': 'CosineAnnealingLR'}]
randomness = {'deterministic': False, 'seed': 0}
resume = False
test_cfg = {'type': 'TestLoop'}
test_dataloader = {'batch_size': 1,
 'dataset': {'ann_file': 'ImageSets/Main/trainval.txt',
             'backend_args': None,
             'data_prefix': {'img_path': 'JPEGImages-trainval'},
             'data_root': '/root/autodl-tmp/rotcert-planB/data/dior/',
             'pipeline': [{'backend_args': None, 'type': 'mmdet.LoadImageFromFile'},
                          {'keep_ratio': True, 'scale': (800, 800), 'type': 'mmdet.Resize'},
                          {'box_type': 'qbox', 'type': 'mmdet.LoadAnnotations', 'with_bbox': True},
                          {'box_type_mapping': {'gt_bboxes': 'rbox'}, 'type': 'ConvertBoxType'},
                          {'meta_keys': ('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'),
                           'type': 'mmdet.PackDetInputs'}],
             'test_mode': True,
             'type': 'DIORDataset'},
 'drop_last': False,
 'num_workers': 2,
 'persistent_workers': True,
 'sampler': {'shuffle': False, 'type': 'DefaultSampler'}}
test_evaluator = {'out_file_path': '/root/autodl-tmp/rotcert-planB/run/jobs/dior-rtmdet-s0-trainonly/predictions.pkl',
 'type': 'DumpResults'}
test_pipeline = [{'backend_args': None, 'type': 'mmdet.LoadImageFromFile'},
 {'keep_ratio': True, 'scale': (800, 800), 'type': 'mmdet.Resize'},
 {'meta_keys': ('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'),
  'type': 'mmdet.PackDetInputs'}]
train_cfg = {'max_epochs': 36, 'type': 'EpochBasedTrainLoop', 'val_interval': 37}
train_dataloader = {'batch_sampler': None,
 'batch_size': 8,
 'dataset': {'datasets': [{'ann_file': 'ImageSets/Main/train.txt',
                           'data_prefix': {'img_path': 'JPEGImages-trainval'},
                           'data_root': '/root/autodl-tmp/rotcert-planB/data/dior/',
                           'filter_cfg': {'filter_empty_gt': True},
                           'pipeline': [{'backend_args': None, 'type': 'mmdet.LoadImageFromFile'},
                                        {'box_type': 'qbox',
                                         'type': 'mmdet.LoadAnnotations',
                                         'with_bbox': True},
                                        {'box_type_mapping': {'gt_bboxes': 'rbox'}, 'type': 'ConvertBoxType'},
                                        {'keep_ratio': True, 'scale': (800, 800), 'type': 'mmdet.Resize'},
                                        {'min_gt_bbox_wh': (0.01, 0.01), 'type': 'mmdet.FilterAnnotations'},
                                        {'direction': ['horizontal', 'vertical', 'diagonal'],
                                         'prob': 0.75,
                                         'type': 'mmdet.RandomFlip'},
                                        {'type': 'mmdet.PackDetInputs'}],
                           'type': 'DIORDataset'}],
             'ignore_keys': ['DATASET_TYPE'],
             'type': 'ConcatDataset'},
 'num_workers': 8,
 'persistent_workers': True,
 'sampler': {'shuffle': True, 'type': 'DefaultSampler'}}
train_pipeline = [{'backend_args': None, 'type': 'mmdet.LoadImageFromFile'},
 {'box_type': 'qbox', 'type': 'mmdet.LoadAnnotations', 'with_bbox': True},
 {'box_type_mapping': {'gt_bboxes': 'rbox'}, 'type': 'ConvertBoxType'},
 {'keep_ratio': True, 'scale': (800, 800), 'type': 'mmdet.Resize'},
 {'min_gt_bbox_wh': (0.01, 0.01), 'type': 'mmdet.FilterAnnotations'},
 {'direction': ['horizontal', 'vertical', 'diagonal'], 'prob': 0.75, 'type': 'mmdet.RandomFlip'},
 {'type': 'mmdet.PackDetInputs'}]
val_cfg = None
val_dataloader = None
val_evaluator = None
val_pipeline = [{'backend_args': None, 'type': 'mmdet.LoadImageFromFile'},
 {'keep_ratio': True, 'scale': (800, 800), 'type': 'mmdet.Resize'},
 {'box_type': 'qbox', 'type': 'mmdet.LoadAnnotations', 'with_bbox': True},
 {'box_type_mapping': {'gt_bboxes': 'rbox'}, 'type': 'ConvertBoxType'},
 {'meta_keys': ('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'),
  'type': 'mmdet.PackDetInputs'}]
vis_backends = [{'type': 'LocalVisBackend'}]
visualizer = {'name': 'visualizer', 'type': 'RotLocalVisualizer', 'vis_backends': [{'type': 'LocalVisBackend'}]}
work_dir = '/root/autodl-tmp/rotcert-planB/run/jobs/dior-rtmdet-s0-trainonly'
