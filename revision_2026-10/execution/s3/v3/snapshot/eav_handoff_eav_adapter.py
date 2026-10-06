"""EAV method adapters, source pin fb99da7b0f49edffcd1fa9e45a4424e9ae86e1eb.

M-native: author score, strict confidence/IoU, class-agnostic Hungarian, pose
pooled linear quantile. CPU IoU is explicitly a reference backend, not MMCV parity.
M-corrected: changes ONLY the quantile rank on the same native scores/groups.
M-common: frozen class-exact greedy matches shared by both geometries; no native
pose/quantile selection. Calibrator is the caller's frozen common contract.

No detector, network, training, filesystem writes, GPU or remote actions here.
"""
from __future__ import annotations
import math
import re
from decimal import Decimal, ROUND_CEILING
from typing import Callable
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from shapely.geometry import Polygon

PIN = 'fb99da7b0f49edffcd1fa9e45a4424e9ae86e1eb'
CONDITIONS = ('30m_30deg', '30m_90deg', '60m_30deg', '60m_90deg', '100m_30deg', '100m_90deg', 'unknown')
TARGETS = {
    'M-native': 'Object-pooled containment among class-agnostic Hungarian matches, within filename flight group. No scene/PAC or recall guarantee.',
    'M-corrected': 'Same population/event as M-native; order-statistic correction only. Object dependence remains.',
    'M-common': 'New class-k TP-positive source scene, then uniform class-k TP, when caller uses HCP and its assumptions. Class-exact common matches. The EAV event is containment; the GWD event is OBB membership in a GWD ball; no identical label-set assertion.',
}


def obbs(value):
    x = np.asarray(value)
    if x.size == 0: return np.empty((0, 5), dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 5 or not np.all(np.isfinite(x)) or np.any(x[:,2:4] <= 0):
        raise ValueError('Expected finite, positive-side (N,5) OBBs')
    if x.dtype.kind != 'f': x=x.astype(np.float64)
    return x


def corners(obb):
    """Author-compatible float32 corner construction, radians, image XY."""
    b = np.asarray(obb)
    obbs(b[None, :])
    cx, cy, w, h, a = b
    local = np.array([[-w/2,-h/2],[w/2,-h/2],[w/2,h/2],[-w/2,h/2]],dtype=np.float32)
    out = np.zeros_like(local)
    c,s=np.cos(a),np.sin(a)
    for i,(x,y) in enumerate(local):
        rx=c*x-s*y;ry=s*x+c*y
        out[i]=[cx+rx,cy+ry]
    return out


def lines(poly):
    """Same outward-normal selection and numeric thresholds as pinned author code."""
    p=np.asarray(poly)
    if p.shape!=(4,2) or not np.isfinite(p).all(): raise ValueError('Expected 4 finite corners')
    center=p.mean(axis=0); result=[]
    for i in range(4):
        u,v=p[i],p[(i+1)%4]
        a,b=u[1]-v[1],v[0]-u[0]; c=-a*u[0]-b*u[1]
        norm=np.sqrt(a*a+b*b); mid=(u+v)/2
        if norm>1e-10 and np.linalg.norm(mid+0.1*np.array([a,b])/norm-center)<np.linalg.norm(mid-center):
            a,b,c=-a,-b,-c
        result.append((float(a),float(b),float(c)))
    return result


def distance(point,line):
    a,b,c=line; n=np.sqrt(a*a+b*b)
    return 0.0 if n<1e-10 else float((a*point[0]+b*point[1]+c)/n)


def contains(inner,outer):
    edges=lines(outer)
    return all(distance(point,edge)<=1e-3 for point in inner for edge in edges)


def margin(predicted,truth):
    if contains(truth,predicted): return 0.0
    return max(0.0,max(distance(p,e) for p in truth for e in lines(predicted)))


def expand(poly,q):
    q=float(q)
    if not math.isfinite(q):
        raise ValueError('Infinite q is an abstract full set; do not manufacture finite corners')
    if q<0: raise ValueError('Negative margin')
    if q==0:return np.asarray(poly).copy()
    edges=[]
    for a,b,c in lines(poly):
        norm=np.sqrt(a*a+b*b)
        if norm<1e-10:raise ValueError('Degenerate edge')
        edges.append((a,b,c-q*norm))
    out=np.zeros_like(poly)
    for i,(a2,b2,c2) in enumerate(edges):
        a1,b1,c1=edges[(i-1)%4];det=a1*b2-a2*b1
        if abs(det)<1e-10:raise ValueError('Parallel adjacent lines')
        out[i]=[(b1*c2-b2*c1)/det,(a2*c1-a1*c2)/det]
    return out.astype(np.float32)


def set_readout(prediction,q):
    if q is None:return {'status':'missing_group','corners':None,'footprint_area':None,'orientation_projection':None}
    if q==math.inf:return {'status':'no_finite_certificate','corners':None,'footprint_area':math.inf,'orientation_projection':'all_RP1'}
    p=expand(corners(prediction),q)
    return {'status':'finite','corners':p,'footprint_area':float(Polygon(p).area),
            'orientation_projection':'all_RP1_without_minimum_size_or_IoU_restriction'}


def parse_pose(filename):
    # Preserve author regex, including inability to parse whitespace between m and _.
    try:
        m=re.search(r'(\d+)m_(\d+)c',filename)
        return f'{m[1]}m_{m[2]}deg' if m else 'unknown'
    except (TypeError,AttributeError):return 'unknown'


def target_pose(target):
    """Do not silently inject metadata missing from the official loader."""
    return parse_pose(target.get('filename','unknown_file'))


def _margins(values):
    a=np.asarray(values,dtype=float)
    if a.ndim!=1 or not np.isfinite(a).all() or np.any(a<0):raise ValueError('Invalid margins')
    return a


def quantile(values,alpha=0.1,variant='M-native'):
    a=_margins(values)
    if not 0<alpha<1:raise ValueError('alpha must be in (0,1)')
    if variant not in ('M-native','M-corrected'):raise ValueError('Common protocol must name its own calibrator')
    if variant=='M-native':return None if len(a)==0 else float(np.quantile(a,1-alpha,method='linear'))
    # Interpret declared decimal alpha exactly; ceil must not grow due to binary multiplication noise.
    rank=int((Decimal(len(a)+1)*(Decimal(1)-Decimal(str(alpha)))).to_integral_value(rounding=ROUND_CEILING))
    return math.inf if rank>len(a) else float(np.partition(a,rank-1)[rank-1])


def fit_pose_groups(records,alpha=0.1,variant='M-native'):
    values={g:[] for g in CONDITIONS}; counts={g:0 for g in CONDITIONS}
    for record in records:
        parsed=target_pose(record);group=parsed if parsed in values else 'unknown'
        counts[group]+=1;values[group].extend(_margins(record['margins']).tolist())
    if not any(values.values()):raise ValueError('No non-conformity scores collected')
    groups={}
    for group,scores in values.items():
        q=quantile(scores,alpha,variant)
        groups[group]={'q':q,'n':len(scores),'images':counts[group],
                       'status':'missing_group' if q is None else ('no_finite_certificate' if math.isinf(q) else 'finite')}
    pooled=[v for group_scores in values.values() for v in group_scores]
    return {'variant':variant,'alpha':alpha,'target':TARGETS[variant],
            'groups':groups,'global':{'q':quantile(pooled,alpha,variant),'n':len(pooled)}}


def group_threshold(fit,target):
    # Author calibration redirects nonstandard parsed groups to unknown; inference does NOT.
    group=target_pose(target)
    result=fit['groups'].get(group)
    if result is None or result['status']=='missing_group':raise ValueError(f'No q for flight condition {group}; no fallback')
    return result['q']


def scores_and_classes(logits):
    x=np.asarray(logits)
    if x.ndim!=2 or x.shape[1]==0 or not np.isfinite(x).all():raise ValueError('Finite query-by-class logits required')
    if x.dtype.kind!='f':x=x.astype(np.float64)
    # Use the author's actual sigmoid/max operators: NumPy/SciPy float32 rounding
    # can move an edge case across the strict confidence threshold.
    with torch.no_grad():
        p=torch.nn.functional.sigmoid(torch.from_numpy(np.ascontiguousarray(x)))
        values,indices=p.max(dim=-1)
    return values.numpy(),indices.numpy()


def polygon_iou_matrix(pred,gt):
    """CPU double-precision geometric reference, explicitly not author MMCV kernel."""
    pred,gt=obbs(pred),obbs(gt)
    def polygon(b):
        x,y,w,h,t=map(float,b);c,s=math.cos(t),math.sin(t)
        p=np.array([[-w/2,-h/2],[w/2,-h/2],[w/2,h/2],[-w/2,h/2]])
        return Polygon(p @ np.array([[c,s],[-s,c]]) + [x,y])
    a,b=list(map(polygon,pred)),list(map(polygon,gt))
    out=np.empty((len(a),len(b)))
    for i,p in enumerate(a):
        for j,q in enumerate(b):
            inter=p.intersection(q).area;out[i,j]=inter/(p.area+q.area-inter)
    return out


def _match_inputs(pred,gt,scores,pc,gc,ious):
    pred,gt=obbs(pred),obbs(gt);scores=np.asarray(scores)
    pc,gc=np.asarray(pc),np.asarray(gc)
    if scores.shape!=(len(pred),) or pc.shape!=(len(pred),) or gc.shape!=(len(gt),):raise ValueError('Length mismatch')
    if not np.isfinite(scores).all() or np.any((scores<0)|(scores>1)):raise ValueError('Invalid confidence')
    ious=np.asarray(ious)
    if ious.shape!=(len(pred),len(gt)) or not np.isfinite(ious).all() or np.any((ious<0)|(ious>1)):raise ValueError('Invalid IoU matrix')
    return pred,gt,scores,pc,gc,ious


def _match_result(pairs,kept,n_gt,pc,gc,ious,backend,mode):
    pi={i for i,j in pairs};gi={j for i,j in pairs}
    return {'mode':mode,'iou_backend':backend,'pairs':[list(p) for p in pairs],
            'ious':[float(ious[i,j]) for i,j in pairs],
            'selected_pred_indices':list(map(int,kept)),
            'fp_indices':[int(i) for i in kept if i not in pi],
            'fn_indices':[j for j in range(n_gt) if j not in gi],
            'class_mismatch_pairs':[list(p) for p in pairs if pc[p[0]]!=gc[p[1]]],
            'tp_count':len(pairs),'gt_count':n_gt,'selected_detection_count':len(kept)}


def match_native(pred,gt,scores,pred_classes,gt_classes,*,ious=None,iou_backend='cpu_polygon_reference',score_threshold=0.5,iou_threshold=0.5):
    if not 0<=score_threshold<=1 or not 0<=iou_threshold<=1:raise ValueError('Invalid threshold')
    if ious is None:
        ious=polygon_iou_matrix(pred,gt);iou_backend='cpu_polygon_reference'
    elif iou_backend=='cpu_polygon_reference':iou_backend='supplied_matrix_unverified_backend'
    pred,gt,scores,pc,gc,ious=_match_inputs(pred,gt,scores,pred_classes,gt_classes,ious)
    kept=np.flatnonzero(scores>score_threshold)
    pairs=[]
    # Preserve 1 - IoU then 1 - cost, including original dtype rounding.
    cost=1.0-ious[kept]
    if cost.shape[0] and cost.shape[1]:
        rows,cols=linear_sum_assignment(cost)
        pairs=[(int(kept[i]),int(j)) for i,j in zip(rows,cols) if 1.0-cost[i,j]>iou_threshold]
    out=_match_result(pairs,kept,len(gt),pc,gc,ious,iou_backend,'native_hungarian_class_agnostic')
    out['below_threshold_indices']=np.flatnonzero(scores<=score_threshold).tolist()
    return out


def match_common(pred,gt,scores,pred_classes,gt_classes,*,score_threshold,iou_threshold,ious=None):
    if not 0<=score_threshold<=1 or not 0<=iou_threshold<=1:raise ValueError('Invalid threshold')
    backend='cpu_polygon_reference' if ious is None else 'supplied_matrix_unverified_backend'
    if ious is None:ious=polygon_iou_matrix(pred,gt)
    pred,gt,scores,pc,gc,ious=_match_inputs(pred,gt,scores,pred_classes,gt_classes,ious)
    kept=np.flatnonzero(scores>=score_threshold);used=set();pairs=[]
    for i in sorted(kept,key=lambda i:(-float(scores[i]),int(i))):
        candidates=[j for j in range(len(gt)) if j not in used and pc[i]==gc[j] and ious[i,j]>=iou_threshold]
        if candidates:
            j=min(candidates,key=lambda j:(-float(ious[i,j]),j));used.add(j);pairs.append((int(i),int(j)))
    pairs.sort()
    out=_match_result(pairs,kept,len(gt),pc,gc,ious,backend,'common_class_exact_greedy')
    out['below_threshold_indices']=np.flatnonzero(scores<score_threshold).tolist()
    return out


def pixel_boxes(normalized,image_hw):
    b=obbs(normalized).copy();h,w=image_hw
    if h<=0 or w<=0:raise ValueError('Positive image size required')
    b[:,:4]*=np.asarray([w,h,w,h],dtype=b.dtype)
    return b


def process_native(outputs,target,image_hw,*,ious=None,iou_backend='cpu_polygon_reference'):
    """Input is one image's author-shaped normalized outputs/target. All GT survive accounting."""
    boxes=np.asarray(outputs['pred_boxes']);logits=np.asarray(outputs['pred_logits'])
    if boxes.ndim!=3 or boxes.shape[0]!=1 or boxes.shape[2]!=5 or logits.ndim!=3 or logits.shape[:2]!=boxes.shape[:2]:
        raise ValueError('Exactly one image with aligned query boxes/logits required')
    pred=obbs(boxes[0]);gt=obbs(target['boxes'])
    scores,classes=scores_and_classes(logits[0])
    matched=match_native(pred,gt,scores,classes,target['labels'],ious=ious,iou_backend=iou_backend)
    pp,gg=pixel_boxes(pred,image_hw),pixel_boxes(gt,image_hw)
    matched['margins']=[margin(corners(pp[i]),corners(gg[j])) for i,j in matched['pairs']]
    matched['pose']=target_pose(target)
    matched['filename_missing']='filename' not in target
    matched['target']=TARGETS['M-native'];matched['pin']=PIN
    return matched


def evaluate_native(outputs,target,image_hw,q,**kwargs):
    result=process_native(outputs,target,image_hw,**kwargs)
    if q is None:raise ValueError('Missing group cannot be evaluated')
    pp=pixel_boxes(np.asarray(outputs['pred_boxes'])[0],image_hw);gg=pixel_boxes(target['boxes'],image_hw)
    result['covered']=[True if q==math.inf else contains(corners(gg[j]),expand(corners(pp[i]),q)) for i,j in result['pairs']]
    result['certificate_status']='no_finite_certificate' if q==math.inf else 'finite'
    return result


def common_geometry_rows(pred,gt,scores,pred_classes,gt_classes,*,source_id,score_threshold,iou_threshold,gwd_score:Callable):
    """Only geometry changes: both scores use exactly the returned shared pairs.

    The caller supplies its frozen GWD function and applies its same calibrator to
    either score. This function never fits pose quantiles or selects on outcomes.
    """
    if not source_id:raise ValueError('Source scene ID is mandatory')
    pred,gt=obbs(pred),obbs(gt)
    matching=match_common(pred,gt,scores,pred_classes,gt_classes,score_threshold=score_threshold,iou_threshold=iou_threshold)
    rows=[]
    for i,j in matching['pairs']:
        g=float(gwd_score(pred[i],gt[j]))
        if not math.isfinite(g) or g<0:raise ValueError('Invalid GWD score')
        rows.append({'source_id':source_id,'pred_index':i,'gt_index':j,'class':int(gt_classes[j]),
                     'eav_margin':margin(corners(pred[i]),corners(gt[j])),'gwd_score':g})
    return {'variant':'M-common','target':TARGETS['M-common'],'matching':matching,'rows':rows,
            'calibrator':'external frozen identical per geometry','pose_grouping':'not silently substituted',
            'events':{'eav':'GT polygon contained in expanded prediction (1e-3 author numeric tolerance)',
                      'gwd':'GT OBB in calibrated Gaussian Wasserstein ball'}}


def population_difference(native,common):
    a=set(map(tuple,native['pairs']));b=set(map(tuple,common['pairs']))
    return {'native_tp':len(a),'common_tp':len(b),'intersection':len(a&b),
            'native_only':sorted(a-b),'common_only':sorted(b-a),
            'native_class_mismatch_count':len(native['class_mismatch_pairs'])}


def evaluate_common_geometry(bundle,pred,gt,*,eav_thresholds,gwd_thresholds):
    """Measure paired common events; never re-match or refit a calibrator.

    q dictionaries are supplied by the same frozen external calibrator, by class.
    Raw GT/FN/FP counts are retained. Infinite q remains in unconditional accounting;
    no useful finite certificate is claimed. Exact score and author's tolerance
    containment are separate observables.
    """
    if bundle.get('variant')!='M-common':raise ValueError('Need a common matched bundle')
    pred,gt=obbs(pred),obbs(gt)
    expected=[list(p) for p in bundle['matching']['pairs']]
    if expected!=[[r['pred_index'],r['gt_index']] for r in bundle['rows']]:raise ValueError('Rows/pairs differ')
    result=[]
    for row in bundle['rows']:
        cls=row['class'];q_e=eav_thresholds.get(cls);q_g=gwd_thresholds.get(cls)
        for q in [q_e,q_g]:
            if q is not None and (math.isnan(float(q)) or q<0 or q==-math.inf):raise ValueError('Invalid threshold')
        i,j=row['pred_index'],row['gt_index']
        item=dict(row)
        item.update({'eav_q':q_e,'gwd_q':q_g,
                     'eav_score_le_q':None if q_e is None else row['eav_margin']<=q_e,
                     'eav_containment_author_tolerance':None if q_e is None else (True if q_e==math.inf else contains(corners(gt[j]),expand(corners(pred[i]),q_e))),
                     'gwd_ball_membership':None if q_g is None else row['gwd_score']<=q_g,
                     'eav_certificate_status':'missing_class' if q_e is None else ('no_finite_certificate' if q_e==math.inf else 'finite'),
                     'gwd_certificate_status':'missing_class' if q_g is None else ('no_finite_certificate' if q_g==math.inf else 'finite')})
        result.append(item)
    return {'variant':'M-common','rows':result,'matching':bundle['matching'],'target':bundle['target'],
            'not_a_recall_certificate':'FN remain in matching; TP event comparison alone does not certify recall.',
            'not_an_area_ratio':'EAV footprint and GWD center slice must not be divided as comparable sizes.'}
