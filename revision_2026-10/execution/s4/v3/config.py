"""S4 fixed design; primary family is HCP, not every diagnostic method."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'))
CLASSES=['airplane','airport','baseballfield','basketballcourt','bridge','chimney','expressway-service-area','expressway-toll-station','dam','golffield','groundtrackfield','harbor','overpass','ship','stadium','storagetank','tenniscourt','trainstation','vehicle','windmill']
MODELS=['dior-orcnn-s0','dior-roit-s0','dior-rtmdet-s0','dior-s2anet-s0']
CONDITIONS=['identity','brightness_070','contrast_070','gamma_150','saturation_035','blue_cast_115']
METHODS=['class_mondrian_pooled','class_mondrian_hcp']
PROTOCOL_ID='rotcert-dior-photometric-v1'
B=5000
SEED=20260928
TAUS=[.5,.6,.7]
CORE_SHIFTS=['brightness_070','contrast_070']
# Absolute arms: clean anchor then clean-cal/shift, shifted-cal/shift for each shift.
ARMS=[dict(name='clean_anchor',cal=0,eval=0)]+[dict(name=c+'__'+tag,cal=cal,eval=i) for i,c in enumerate(CONDITIONS) if i for tag,cal in [('clean_cal',0),('recalibrated',i)]]
CONTRASTS=[dict(name=c+'__'+kind,shift=c,left=1+2*(i-1)+offset,right=(0 if offset==0 else 1+2*(i-1)),kind=kind) for i,c in enumerate(CONDITIONS) if i for kind,offset in [('shift_drop',0),('recalibration_recovery',1)]]
METRICS=['scene_coverage','object_coverage','scene_all_tp_coverage','q_px','finite_q_rate',
 'normalized_center_radius','angle_informative_rate','angle_full_rate','angle_halfwidth_lower_rad','angle_halfwidth_upper_rad',
 'joint_center_angle_coverage_lower','joint_center_angle_coverage_upper',
 'all_gt_scene_recall','all_gt_object_recall','all_gt_scene_e2e','all_gt_object_e2e',
 'tp_source_rate','zero_tp_source_rate','gt_positive_zero_tp_source_rate',
 'tp_count','fp_count','fn_count','gt_count','detection_count',
 'tp_source_count','gt_positive_source_count','zero_tp_source_count',
 'cal_tp_source_count','cal_tp_object_count','cal_class_absent',
 'all_detection_finite_q_rate','all_detection_normalized_center_radius','all_detection_informative_angle_rate']
SUPPORT_NAMES=['all_source_copies','distinct_drawn_sources','gt_positive_source_copies','tp_positive_source_copies','gt_object_copies','tp_object_copies','det_object_copies']

ALL_SOURCE_METRICS=['all_source_copies','gt_positive_source_copies','tp_positive_source_copies','zero_tp_source_copies','gt_positive_zero_tp_source_copies','gt_objects','tp_objects','fp_objects','fn_objects','detections','all_gt_scene_recall','all_gt_object_recall','all_gt_scene_e2e','all_gt_object_e2e','mixed_tp_scene_coverage','mixed_tp_object_coverage']
