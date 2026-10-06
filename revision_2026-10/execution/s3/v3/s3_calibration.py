"""Conservative weighted source-HCP and legacy comparators; no permissive epsilon."""
from fractions import Fraction
import math
import numpy as np


def _exact_prefix(weights, counts, end):
    # Group by per-scene TP count, not one Fraction per object.
    totals=np.bincount(counts[:end],weights=weights[:end]).astype(np.int64)
    return sum((Fraction(int(v),i) for i,v in enumerate(totals) if i and v),Fraction())


def quantile_grid(values,scene,multiplicity,alpha_num,alpha_den,mode='hcp'):
    values=np.asarray(values,float)
    if values.ndim==1:values=values[:,None]
    if not np.isfinite(values).all():raise ValueError('nonfinite score')
    scene=np.asarray(scene,dtype=np.int64);multiplicity=np.asarray(multiplicity,dtype=np.int64)
    if np.any(multiplicity<0):raise ValueError('negative source count')
    n,d=values.shape;alpha_num=np.asarray(alpha_num,dtype=np.int64)
    if np.any(alpha_num<=0) or np.any(alpha_num>=alpha_den):raise ValueError('invalid alpha')
    if n==0:return np.full((len(alpha_num),d),np.inf)
    object_counts=np.bincount(scene,minlength=len(multiplicity))
    scene_K=int(multiplicity[object_counts>0].sum())
    copies=multiplicity[scene]
    total_objects=int(copies.sum())
    q=np.full((len(alpha_num),d),np.inf)
    if not scene_K:return q
    for j in range(d):
        order=np.argsort(values[:,j],kind='stable')
        take=copies[order]>0;order=order[take]
        sorted_values=values[order,j];w=copies[order]
        denominator=alpha_den*d if mode in ('hcp','object_bonferroni') else alpha_den
        if mode in ('object_bonferroni','object_scp'):
            ranks=((denominator-alpha_num)*(total_objects+1)+denominator-1)//denominator
            prefix=np.cumsum(w,dtype=np.int64)
            idx=np.searchsorted(prefix,ranks,side='left')
        elif mode=='linear':
            # Literal linear quantile of the replicated object sample; no repeat array.
            rank_num=(denominator-alpha_num)*(total_objects-1)
            low=rank_num//denominator;rem=rank_num%denominator
            prefix=np.cumsum(w,dtype=np.int64)
            lo=sorted_values[np.searchsorted(prefix,low+1,side='left')]
            hi=sorted_values[np.searchsorted(prefix,np.minimum(low+2,total_objects),side='left')]
            q[:,j]=lo+(hi-lo)*(rem/denominator)
            continue
        elif mode=='hcp':
            m=object_counts[scene[order]]
            cum=np.cumsum(w.astype(np.longdouble)/m.astype(np.longdouble),dtype=np.longdouble)
            target=(np.longdouble(scene_K+1)*(denominator-alpha_num))/denominator
            eps=np.finfo(np.longdouble).eps
            # Conservative guard for positive division, accumulation, target arithmetic.
            err=np.longdouble(8)*(len(order)+4)*eps*(scene_K+1)
            idx=np.searchsorted(cum,target,side='left')
            for z,k in enumerate(idx):
                lower=0 if k==0 else cum[k-1]
                upper=cum[k] if k<len(cum) else np.longdouble(scene_K)
                if abs(lower-target[z])<=err or abs(upper-target[z])<=err:
                    t=Fraction(int((denominator-alpha_num[z])*(scene_K+1)),int(denominator))
                    a=max(0,int(np.searchsorted(cum,target[z]-err,side='left')))
                    b=min(len(cum),int(np.searchsorted(cum,target[z]+err,side='left'))+1)
                    # Find first cumulative mass >= exact rational target.
                    while a<b:
                        mid=(a+b)//2
                        mass=_exact_prefix(w,m,min(mid+1,len(w)))
                        if mass>=t:b=mid
                        else:a=mid+1
                    idx[z]=a
        else:raise ValueError('unknown calibrator')
        valid=idx<len(sorted_values)
        q[valid,j]=sorted_values[idx[valid]]
    return q


def coverage_curve(values,q,scene,multiplicity):
    """Monotone q(alpha): histogram of last covered grid point, not NxAxD."""
    values=np.asarray(values,float);q=np.asarray(q,float)
    if values.ndim==1:values=values[:,None]
    if not len(values):return np.full(len(q),np.nan)
    cnt=np.bincount(scene,minlength=len(multiplicity));den=int(multiplicity[cnt>0].sum())
    if not den:return np.full(len(q),np.nan)
    weights=multiplicity[scene]/cnt[scene]/den
    length=np.full(len(values),len(q),dtype=np.int64)
    for j in range(values.shape[1]):
        if np.any(q[1:,j]>q[:-1,j]):raise ValueError('nonmonotone threshold grid')
        length=np.minimum(length,np.searchsorted(-q[:,j],-values[:,j],side='right'))
    h=np.bincount(length,weights=weights,minlength=len(q)+1)
    return np.cumsum(h[::-1])[::-1][1:]


def source_weights(scene,multiplicity,mask=None):
    if mask is None:mask=np.ones(len(scene),dtype=bool)
    counts=np.bincount(scene[mask],minlength=len(multiplicity))
    den=int(multiplicity[counts>0].sum())
    weights=np.zeros(len(scene))
    if den:
        sel=mask&(multiplicity[scene]>0)
        weights[sel]=multiplicity[scene[sel]]/counts[scene[sel]]/den
    return weights,den,int(np.sum((counts>0)&(multiplicity>0)))


def weighted_metric(values,weights):
    select=weights>0
    if not np.any(select):return np.nan
    v=np.asarray(values)[select]
    if np.any(np.isnan(v)):return np.nan
    if np.any(np.isposinf(v)):return np.inf
    return float(np.dot(v,weights[select])/weights[select].sum())


def choose_alpha(curves):
    if np.any(~np.isfinite(curves)):return None
    macro=curves.mean(axis=0)
    # alpha ascending, argmin stable first minimizes exact stored abs difference.
    return int(np.argmin(np.abs(macro-.9)))


def strict_macro(values,axis=0):
    # np.mean intentionally propagates missing classes and infinity.
    return np.mean(values,axis=axis)
