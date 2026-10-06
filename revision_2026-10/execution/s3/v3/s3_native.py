"""Pinned official finite expand(float32), then recomputed-edge containment.

The score and this event are distinct. No q+epsilon surrogate is used. The
canonical inputs for this S3 cache are float64 pixel OBBs. Infinite q denotes an
abstract full set and is never passed to expand. Source-only, no external writes.
"""
import numpy as np
from s3_scores import author_corners, author_lines


def expand_batch(poly, q):
    poly=np.asarray(poly)
    q=np.broadcast_to(np.asarray(q,dtype=float),(len(poly),))
    if poly.shape!=(len(q),4,2) or poly.dtype!=np.float32:
        raise ValueError('Expected float32 (N,4,2) polygons')
    if np.any(~np.isfinite(q)) or np.any(q<0):
        raise ValueError('Only finite nonnegative q may enter native expansion')
    out=poly.copy();take=q>0
    if not take.any():return out
    ll=author_lines(poly[take]);a,b,c=(ll[:,:,i] for i in range(3))
    norm=np.sqrt(a*a+b*b)
    if np.any(norm<1e-10):raise ValueError('Degenerate official edge')
    c=c-q[take,None]*norm
    a1,b1,c1=(np.roll(v,1,axis=1) for v in (a,b,c))
    det=a1*b-a*b1
    if np.any(abs(det)<1e-10):raise ValueError('Parallel official adjacent lines')
    x=(b1*c-b*c1)/det;y=(a*c1-a1*c)/det
    out[take]=np.stack((x,y),axis=2).astype(np.float32)
    if not np.isfinite(out).all():raise ValueError('Native expansion float32 overflow')
    return out


def contains_batch(inner,outer):
    if inner.shape!=outer.shape or inner.shape[1:]!=(4,2):raise ValueError('polygon shape')
    ll=author_lines(outer);a,b,c=(ll[:,:,i] for i in range(3));den=np.sqrt(a*a+b*b)
    # is_point_inside_obb casts px/py to Python float before signed distance.
    xx=inner[:,:,0,None].astype(float);yy=inner[:,:,1,None].astype(float)
    with np.errstate(divide='ignore',invalid='ignore'):
        distance=(a[:,None,:]*xx+b[:,None,:]*yy+c[:,None,:])/den[:,None,:]
    distance=np.where(den[:,None,:]<1e-10,0.,distance)
    if not np.isfinite(distance).all():raise ValueError('Nonfinite containment arithmetic')
    return np.all(distance<=1e-3,axis=(1,2))


def native_readout(pred,gt,q):
    """Returns actual native event + expanded polygon area/circumradius.

    Polygon area/radius describe the emitted polygon only. They are not the
    footprint/center projection of either the strict-score set or tolerant event.
    """
    q=np.asarray(q,float)
    if q.shape!=(len(pred),) or np.any(np.isnan(q)) or np.any(q<0):raise ValueError('invalid q')
    event=np.ones(len(pred),dtype=bool);area=np.full(len(pred),np.inf);radius=area.copy()
    finite=np.isfinite(q)
    if finite.any():
        pp=author_corners(pred[finite]);gg=author_corners(gt[finite])
        expanded=expand_batch(pp,q[finite]);event[finite]=contains_batch(gg,expanded)
        xy=expanded.astype(float)
        # Translation before shoelace reduces cancellation, no change of polygon.
        zz=xy-xy[:,0:1,:]
        area[finite]=abs(np.sum(zz[:,:,0]*np.roll(zz[:,:,1],-1,axis=1)-zz[:,:,1]*np.roll(zz[:,:,0],-1,axis=1),axis=1))/2
        radius[finite]=np.max(np.linalg.norm(xy-pred[finite,None,:2],axis=2),axis=1)
    return event,area,radius
