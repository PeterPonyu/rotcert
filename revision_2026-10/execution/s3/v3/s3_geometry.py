"""Explicit geometry semantics; undefined projections never get fake useful radii."""
from fractions import Fraction
import math
import numpy as np
from s3_scores import canonicalize_le90
from s3_native import native_readout

GEOMETRY_REASON={
    'gwd':'exact free-shape GWD center projection and set-valued RP1 angle projection',
    'gwd_normalized':'prediction-scale converted exact GWD projection',
    'naive-coord':'exact canonical-coordinate angle arc; center rectangle circumradius',
    'wrapped-coord':'exact RP1 geodesic arc; center rectangle circumradius',
    'doubled':'exact doubled-angle feasible arc union total length; no single symmetric halfwidth',
    'coord_max_additive':'exact separable local-coordinate center and angle projection',
    'coord_max_multiplicative':'exact separable normalized local-coordinate center and angle projection',
    'max_rank':'exact strict-left weighted design-ECDF inverse coordinate bounds',
    'hull':'center rectangle defined; free-shape orientation projection not derived (undefined)',
    'corner_max':'center enclosing radius<=q by averaging matched corner displacement; orientation undefined',
    'expansion_margin':'strict faithful score<=q; finite strict-score geometry not established and NA; native expand(float32)+contains is a separate event with emitted polygon descriptors only',
    'expansion_analytic':'exact expanded analytic rectangle containment footprint; full RP1',
    'iou':'free-shape center/orientation projection not derived; undefined rather than fixed-shape slice',
    'kld':'free-shape center/orientation projection not derived; undefined rather than fixed-shape slice',
    'hellinger':'free-shape center/orientation projection not derived; undefined rather than fixed-shape slice',
    'bhattacharyya':'free-shape center/orientation projection not derived; undefined rather than fixed-shape slice',
}
METRICS=['score_coverage','finite_threshold_rate','bounded_region_rate','center_defined_rate',
         'normalized_center_radius','angle_defined_rate','informative_angle_rate','full_angle_rate',
         'orientation_total_length_rad','symmetric_half_extent_rad','joint_center_angle_coverage',
         'containment_footprint_area_px2',
         'eav_native_containment_coverage','eav_native_only_rate','eav_strict_only_rate',
         'eav_native_emitted_polygon_area_px2','eav_native_emitted_polygon_normalized_circumradius']
STRATA=['AR_lt1p2','AR_1p2_to3','AR_ge3','seam_le5deg','interior_gt5deg']


def gwd_halfwidth(w,h,q):
    """Free-shape set projection; stable side gap and exact boundary repair.

    Never form 1-h/w to locate qstar: cancellation misses near-square full sets.
    Compare q/(w-h) instead; exact Fraction comparison and scaled residual repair
    are used near 1/sqrt(8). Positive finite sides are part of the input contract.
    """
    w,h,q=np.broadcast_arrays(np.asarray(w,float),np.asarray(h,float),np.asarray(q,float))
    if np.any(~np.isfinite(w)) or np.any(~np.isfinite(h)) or np.any(w<=0) or np.any(h<=0) or np.any(q<0) or np.any(np.isnan(q)):
        raise ValueError('invalid GWD projection inputs')
    lw=np.maximum(w,h);sh=np.minimum(w,h);delta=lw-sh
    with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
        ratio=np.divide(q,delta,out=np.full(w.shape,np.inf),where=delta>0)
    boundary=1/np.sqrt(8.)
    full=(ratio>=boundary)|(delta==0)|np.isposinf(q)
    # Subtraction is exact for near-square inputs (Sterbenz); for separated sides
    # its relative error is bounded. A broad 64eps ratio band covers both cases.
    near=(delta>0)&np.isfinite(q)&(abs(ratio-boundary)<=64*np.finfo(float).eps*boundary)
    residual=np.full(w.shape,np.nan)
    for i in np.flatnonzero(near):
        a,b,r=map(Fraction.from_float,(float(lw.flat[i]),float(sh.flat[i]),float(q.flat[i])))
        gap=(a-b)*(a-b);eight=8*r*r
        full.flat[i]=eight>=gap
        if eight<gap:residual.flat[i]=float((gap-eight)/(8*gap))
    half=np.full(w.shape,np.pi/2);use=~full
    t=sh[use]/lw[use];r=ratio[use];x=q[use]/lw[use]
    d1=np.where(np.isnan(residual[use]),.125-r*r,residual[use])
    # Cancel (w-h)^2 before sqrt, avoiding underflow for nearly square boxes.
    top=r*np.sqrt(np.maximum((1+t*t)/4-x*x,0))
    bottom=np.sqrt(np.maximum(d1,0))*np.sqrt(np.maximum((1+t)**2/8-x*x,0))
    angle=.5*np.arctan2(top,bottom)
    half[use]=np.where(q[use]==0,0,np.nextafter(angle,np.inf))
    return half,full


def doubled_length(theta,qc,qs):
    """Exact finite-arc partition on phi=2theta ∈ [0,2pi)."""
    theta,qc,qs=np.broadcast_arrays(theta,qc,qs);result=np.empty(theta.shape)
    for idx in range(theta.size):
        t,cq,sq=theta.flat[idx],qc.flat[idx],qs.flat[idx]
        if cq>=2 and sq>=2:result.flat[idx]=np.pi;continue
        c0,s0=np.cos(2*t),np.sin(2*t)
        bounds=[0.,2*np.pi]
        for val in [c0-cq,c0+cq]:
            if -1<val<1:
                a=math.acos(val);bounds.extend([a,2*np.pi-a])
        for val in [s0-sq,s0+sq]:
            if -1<val<1:
                a=math.asin(val)%(2*np.pi);bounds.extend([a,(np.pi-a)%(2*np.pi)])
        b=np.unique(bounds);mid=(b[1:]+b[:-1])/2
        keep=(abs(np.cos(mid)-c0)<=cq)&(abs(np.sin(mid)-s0)<=sq)
        result.flat[idx]=float(np.sum(np.diff(b)[keep])/2)
    return result


def endpoints(name,pred,gt,q,covered,rank_model=None):
    n=len(pred);q=np.asarray(q,float)
    if q.ndim==1:q=q[:,None]
    pw,ph,pt=canonicalize_le90(pred[:,2],pred[:,3],pred[:,4]);gw,gh,tt=canonicalize_le90(gt[:,2],gt[:,3],gt[:,4])
    dt=abs(tt-pt);wrap=np.minimum(dt,np.pi-dt)
    dx=gt[:,0]-pred[:,0];dy=gt[:,1]-pred[:,1]
    u=np.cos(pt)*dx+np.sin(pt)*dy;v=-np.sin(pt)*dx+np.cos(pt)*dy
    rad=np.full(n,np.nan);length=np.full(n,np.nan);half=np.full(n,np.nan);area=np.full(n,np.nan)
    center_inside=np.full(n,np.nan);angle_inside=np.full(n,np.nan)
    finite=np.isfinite(q).all(axis=1);bounded=finite.astype(float);s=q[:,0]
    native_metrics=np.full((n,5),np.nan)
    if name in ('gwd','gwd_normalized'):
        rad=s*np.sqrt(pw*ph) if name=='gwd_normalized' else s
        half,full=gwd_halfwidth(pw,ph,rad);length=2*half
        center_inside=np.hypot(dx,dy)<=rad
        angle_inside=np.where(gw==gh,full,wrap<=half)
    elif name in ('naive-coord','wrapped-coord','doubled','hull'):
        rad=np.hypot(q[:,0],q[:,1]);center_inside=(abs(dx)<=q[:,0])&(abs(dy)<=q[:,1])
        bounded=np.isfinite(q[:,:4]).all(axis=1)
        if name=='naive-coord':
            lo=np.maximum(-np.pi/2,pt-q[:,4]);hi=np.minimum(np.pi/2,pt+q[:,4]);length=hi-lo
            angle_inside=(tt>=lo)&(tt<=hi)
        elif name=='wrapped-coord':
            half=np.minimum(q[:,4],np.pi/2);length=2*half;angle_inside=wrap<=half
        elif name=='doubled':
            length=doubled_length(pt,q[:,4],q[:,5])
            angle_inside=(abs(np.cos(2*tt)-np.cos(2*pt))<=q[:,4])&(abs(np.sin(2*tt)-np.sin(2*pt))<=q[:,5])
    elif name in ('coord_max_additive','coord_max_multiplicative','max_rank'):
        if name=='coord_max_additive':
            bx=by=s;bt=2*s/pw
        elif name=='coord_max_multiplicative':
            bx=s*np.maximum(pw,1);by=s*np.maximum(ph,1);bt=2*s
        else:
            inverse=rank_model.invert(s);bx,by,bt=inverse[:,0],inverse[:,1],inverse[:,4]
            bounded=finite&(s<rank_model.total)
        rad=np.hypot(bx,by);half=np.minimum(bt,np.pi/2);length=2*half
        center_inside=(abs(u)<=bx)&(abs(v)<=by);angle_inside=wrap<=half
    elif name=='corner_max':
        rad=s;center_inside=np.hypot(dx,dy)<=rad
    elif name=='expansion_margin':
        native,native_area,native_radius=native_readout(pred,gt,s)
        native_metrics=np.column_stack([native,native&~covered,covered&~native,
                                        native_area,native_radius/np.sqrt(pw*ph)])
        # No proof equates the float32 strict-score level set with a q+eps
        # rectangle. Finite score-set projections remain undefined in v2.
        bounded=np.where(finite,np.nan,0.)
        infinite=np.isposinf(s)
        rad[infinite]=np.inf;length[infinite]=np.pi;half[infinite]=np.pi/2
        center_inside[infinite]=1.;angle_inside[infinite]=1.;area[infinite]=np.inf
    elif name=='expansion_analytic':
        rad=np.hypot(pw/2+s,ph/2+s);area=(pw+2*s)*(ph+2*s)
        length=np.full(n,np.pi);half=np.full(n,np.pi/2);angle_inside=np.ones(n,dtype=bool)
        center_inside=np.hypot(dx,dy)<=rad  # enclosing-radius diagnostic event
    elif name=='iou':bounded=finite&(s<1)
    elif name=='hellinger':bounded=finite&(s<1)
    angle_defined=~np.isnan(length);center_defined=~np.isnan(rad)
    full=np.where(angle_defined,length>=np.pi,np.nan)
    informative=np.where(angle_defined,length<np.pi,np.nan)
    joint=np.where(angle_defined&center_defined,np.asarray(angle_inside,float)*np.asarray(center_inside,float),np.nan)
    result=np.column_stack([covered,finite,bounded,center_defined,rad/np.sqrt(pw*ph),
                            angle_defined,informative,full,length,half,joint,area,native_metrics])
    seam=np.pi/2-abs(pt)<=np.deg2rad(5)
    strata=np.column_stack([pw/ph<1.2,(pw/ph>=1.2)&(pw/ph<3),pw/ph>=3,seam,~seam])
    return result,strata
