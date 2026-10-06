"""Frozen score definitions and pinned-author S14 numerics, local snapshots only."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[k]='1'
import importlib.util
import math
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'snapshot'))
from rotcert.gwd import canonicalize_le90, obb_gwd
from rotcert import scores_ext as X

NAMES=['gwd','gwd_normalized','naive-coord','wrapped-coord','doubled','coord_max_additive',
       'max_rank','coord_max_multiplicative','hull','iou','kld','hellinger','bhattacharyya',
       'corner_max','expansion_margin','expansion_analytic']
IDS=['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S12','S12','S13','S14','S14']
VECTOR={'naive-coord':5,'wrapped-coord':5,'doubled':6,'hull':4}
ARMS=[dict(name=n,score=n,score_id=s,calibrator='common_class_scene_HCP_bonferroni' if n in VECTOR else 'common_class_scene_HCP') for n,s in zip(NAMES,IDS)]
ARMS += [dict(name=n+'__legacy_object',score=n,score_id=IDS[NAMES.index(n)],calibrator='legacy_class_object_SCP_bonferroni') for n in VECTOR]
ARMS += [dict(name='expansion_margin__legacy_'+tag,score='expansion_margin',score_id='S14',calibrator=mode) for tag,mode in [('uncorrected','legacy_class_object_linear_quantile'),('scp','legacy_class_object_SCP')]]
ALPHA_NUM=np.arange(80,1201,10,dtype=np.int64);ALPHA_DEN=4000


def official_module():
    path=Path(os.environ.get('ROTCERT_EAV_GEOMETRY',ROOT/'snapshot/eav_geometry_official.py'))
    if not path.is_file():
        raise FileNotFoundError('Pinned upstream EAV geometry is not redistributed. Set ROTCERT_EAV_GEOMETRY to tools/geometry_utils.py from the documented EAV-DETR commit.')
    spec=importlib.util.spec_from_file_location('s3_pinned_eav',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def author_corners(obbs):
    b=np.asarray(obbs,dtype=float)
    local=np.stack([np.stack([-b[:,2]/2,-b[:,3]/2],axis=1),
                    np.stack([b[:,2]/2,-b[:,3]/2],axis=1),
                    np.stack([b[:,2]/2,b[:,3]/2],axis=1),
                    np.stack([-b[:,2]/2,b[:,3]/2],axis=1)],axis=1).astype(np.float32)
    c=np.cos(b[:,4])[:,None];s=np.sin(b[:,4])[:,None]
    x=(b[:,0,None]+(c*local[:,:,0].astype(float)-s*local[:,:,1].astype(float))).astype(np.float32)
    y=(b[:,1,None]+(s*local[:,:,0].astype(float)+c*local[:,:,1].astype(float))).astype(np.float32)
    return np.stack([x,y],axis=2)


def author_lines(corners):
    p=corners;v=np.roll(p,-1,axis=1)
    a=p[:,:,1]-v[:,:,1];b=v[:,:,0]-p[:,:,0]
    c=-a*p[:,:,0]-b*p[:,:,1]
    norm=np.sqrt(a*a+b*b)
    mid=(p+v)/np.float32(2.)
    with np.errstate(divide='ignore',invalid='ignore'):
        test=mid+np.float32(.1)*np.stack([a,b],axis=2)/norm[:,:,None]
    center=p.mean(axis=1)
    flip=(norm>1e-10)&(np.linalg.norm(test-center[:,None,:],axis=2)<np.linalg.norm(mid-center[:,None,:],axis=2))
    return np.stack([np.where(flip,-a,a),np.where(flip,-b,b),np.where(flip,-c,c)],axis=2).astype(float)


def author_margin(pred,gt):
    """Vectorized pinned score: double containment precheck, float32 point arithmetic."""
    cp,cg=author_corners(pred),author_corners(gt);ll=author_lines(cp)
    a,b,c=(ll[:,:,i] for i in range(3));den=np.sqrt(a*a+b*b)
    # is_point_inside explicitly converts coordinates to Python float.
    dist64=(a[:,None,:]*cg[:,:,0,None].astype(float)+b[:,None,:]*cg[:,:,1,None].astype(float)+c[:,None,:])/den[:,None,:]
    inside=np.max(dist64,axis=(1,2))<=.001
    # calculate_expansion_margin passes np.float32 coordinates. NumPy2 weak
    # Python scalar arithmetic matches float32 A*px+B*py+C numerator.
    a32,b32,c32=a.astype(np.float32),b.astype(np.float32),c.astype(np.float32)
    numerator=a32[:,None,:]*cg[:,:,0,None]+b32[:,None,:]*cg[:,:,1,None]+c32[:,None,:]
    distance=numerator.astype(float)/den[:,None,:]
    distance=np.where(den[:,None,:]<1e-10,0,distance)
    q=np.maximum(0,np.max(distance,axis=(1,2)))
    return np.where(inside,0,q)


def coordinate_arrays(pred,gt):
    pw,ph,pt=canonicalize_le90(pred[:,2],pred[:,3],pred[:,4]);gw,gh,tt=canonicalize_le90(gt[:,2],gt[:,3],gt[:,4])
    xy=np.abs(pred[:,:2]-gt[:,:2]);wh=np.column_stack([np.abs(pw-gw),np.abs(ph-gh)]);dt=np.abs(pt-tt)
    wrapped=np.minimum(dt,np.pi-dt)
    hp=np.column_stack([abs(np.cos(pt))*pw+abs(np.sin(pt))*ph,abs(np.sin(pt))*pw+abs(np.cos(pt))*ph])/2
    hg=np.column_stack([abs(np.cos(tt))*gw+abs(np.sin(tt))*gh,abs(np.sin(tt))*gw+abs(np.cos(tt))*gh])/2
    return {'naive-coord':np.column_stack([xy,wh,dt]),'wrapped-coord':np.column_stack([xy,wh,wrapped]),
            'doubled':np.column_stack([xy,wh,abs(np.cos(2*pt)-np.cos(2*tt)),abs(np.sin(2*pt)-np.sin(2*tt))]),
            'hull':np.column_stack([xy,abs(hp-hg)])}


def all_static_scores(cache):
    p,g=cache['tp_pred'],cache['tp_gt'];out=coordinate_arrays(p,g)
    scalar={name:fn(p,g) for name,fn in X.EXT_SCORES.items()}
    scalar['expansion_analytic']=scalar.pop('expansion_margin')
    scalar['expansion_margin']=author_margin(p,g)
    scalar['gwd']=obb_gwd(p,g);scalar['gwd_normalized']=scalar['gwd']/np.sqrt(p[:,2]*p[:,3])
    scalar['iou']=1-cache['tp_iou']
    for name,v in scalar.items():out[name]=np.asarray(v)[:,None]
    return out


class WeightedMaxRank:
    """Weighted replicate-exact original design object ECDF; strict-left ties."""
    def __init__(self,residuals,weights):
        take=np.asarray(weights)>0;residuals=residuals[take];weights=np.asarray(weights,dtype=np.int64)[take]
        if not len(residuals):raise ValueError('empty max-rank design support')
        self.total=int(weights.sum());self.values=[];self.prefix=[]
        for j in range(residuals.shape[1]):
            order=np.argsort(residuals[:,j],kind='stable')
            self.values.append(residuals[order,j]);self.prefix.append(np.r_[0,np.cumsum(weights[order])])
    def score(self,residuals):
        return np.maximum.reduce([self.prefix[j][np.searchsorted(self.values[j],residuals[:,j],side='left')] for j in range(residuals.shape[1])])[:,None]
    def invert(self,q):
        q=np.asarray(q);out=np.full((len(q),5),np.inf)
        finite=np.isfinite(q)&(q<self.total)
        for j in range(5):
            ix=np.searchsorted(self.prefix[j][1:],q[finite],side='right')
            out[finite,j]=self.values[j][ix]
        return out
