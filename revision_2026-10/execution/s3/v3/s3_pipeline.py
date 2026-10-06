"""Full calibration/design/evaluation functional with shared source multiplicities."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[k]='1'
import time
from pathlib import Path
import numpy as np
from s3_scores import ARMS,ALPHA_NUM,ALPHA_DEN,all_static_scores,WeightedMaxRank,X
from s3_calibration import quantile_grid,coverage_curve,source_weights,weighted_metric,choose_alpha
from s3_geometry import endpoints,METRICS,STRATA
from rotcert.splits import three_way_scene_split


def roles_from_cache(c):
    universe=np.unique(c['tp_scene']);names=[str(c['scene_names'][i]) for i in universe]
    lookup={str(c['scene_names'][i]):int(i) for i in universe}
    split=three_way_scene_split(names,seed=0,cal_frac=.4,match_frac=.2)
    return [np.array([lookup[x] for x in split[k]],dtype=np.int64) for k in ('calibration','matching','eval')]


def multiplicities(c,roles,cell_index,draw):
    n=len(c['scene_names']);out=[]
    for role_id,ids in enumerate(roles):
        v=np.zeros(n,dtype=np.int64)
        if draw<0:v[ids]=1
        else:
            seed=np.random.SeedSequence([20260928,3,cell_index,draw,role_id])
            gen=np.random.Generator(np.random.PCG64(seed))
            v+=np.bincount(gen.choice(ids,size=len(ids),replace=True),minlength=n)
        out.append(v)
    return out


class Cell:
    def __init__(self,cache,static=None):
        self.cache=cache;self.sc=cache['tp_scene'];self.cl=cache['tp_cls']
        self.nclass=len(cache['class_names']);self.roles=roles_from_cache(cache)
        self.static=all_static_scores(cache) if static is None else static
        self.residuals=X.wrapped_coord_residuals(cache['tp_pred'],cache['tp_gt'])
        self.role_mask=[np.isin(self.sc,ids) for ids in self.roles]
        self.class_role=[[np.flatnonzero(mask&(self.cl==k)) for k in range(self.nclass)] for mask in self.role_mask]

    def run(self,cell_index,draw):
        started=time.process_time();mult=multiplicities(self.cache,self.roles,cell_index,draw)
        cm,dm,em=mult;sc=self.sc;c=self.cache
        design=self.role_mask[1]
        rank=WeightedMaxRank(self.residuals[design],dm[sc[design]])
        scores=dict(self.static);scores['max_rank']=rank.score(self.residuals).astype(float)
        # arm x policy (nominal/tuned) x class x metric
        shape=(len(ARMS),2,self.nclass,len(METRICS))
        values=np.full(shape,np.nan);strat=np.full((len(ARMS),2,self.nclass,len(STRATA),3),np.nan)
        # stratum endpoints: score coverage, informative-angle, normalized radius.
        alpha=np.full((len(ARMS),2),np.nan);design_cover=np.full((len(ARMS),2,self.nclass),np.nan)
        curves=np.full((len(ARMS),self.nclass,len(ALPHA_NUM)),np.nan)
        qvalues=np.full((len(ARMS),2,self.nclass,6),np.nan)
        support=np.zeros((3,self.nclass,3),dtype=np.int64) # duplicated sources,distinct sources,objects with multiplicity
        for ri,mm in enumerate(mult):
            for k,rows in enumerate(self.class_role[ri]):
                _,duplicated,distinct=source_weights(sc[rows],mm)
                support[ri,k]=duplicated,distinct,int(mm[sc[rows]].sum())
        time_arms=[]
        for ai,arm in enumerate(ARMS):
            mark=time.process_time();name=arm['score'];v=scores[name]
            mode=('object_bonferroni' if 'bonferroni' in arm['calibrator'] and 'legacy' in arm['calibrator']
                  else 'linear' if 'linear' in arm['calibrator'] else 'object_scp' if 'legacy' in arm['calibrator'] else 'hcp')
            grids=[]
            for k in range(self.nclass):
                cal=self.class_role[0][k];des=self.class_role[1][k]
                q=quantile_grid(v[cal],sc[cal],cm,ALPHA_NUM,ALPHA_DEN,mode)
                grids.append(q)
                curves[ai,k]=coverage_curve(v[des],q,sc[des],dm)
            tuned=choose_alpha(curves[ai]);nominal=int(np.flatnonzero(ALPHA_NUM==400)[0])
            for policy,idx in enumerate((nominal,tuned)):
                if idx is None:continue
                alpha[ai,policy]=ALPHA_NUM[idx]/ALPHA_DEN
                design_cover[ai,policy]=curves[ai,:,idx]
                for k in range(self.nclass):
                    ev=self.class_role[2][k];q=grids[k][idx];dim=len(q)
                    # Raw integer rank threshold is normalized only for reported q.
                    qvalues[ai,policy,k,:dim]=q/rank.total if name=='max_rank' else q
                    if not len(ev) or not support[2,k,0]:continue
                    counts=em[sc[ev]];active=counts>0;ev=ev[active]
                    covered=np.all(v[ev]<=q[None,:],axis=1)
                    data,bins=endpoints(name,c['tp_pred'][ev],c['tp_gt'][ev],np.broadcast_to(q,(len(ev),dim)),covered,rank)
                    weights,_,_=source_weights(sc[ev],em)
                    values[ai,policy,k]=[weighted_metric(data[:,j],weights) for j in range(len(METRICS))]
                    for st in range(len(STRATA)):
                        ww,_,_=source_weights(sc[ev],em,bins[:,st])
                        strat[ai,policy,k,st]=[weighted_metric(data[:,j],ww) for j in [0,6,4]]
            time_arms.append(time.process_time()-mark)
        macro=np.mean(values,axis=2) # intentionally strict missing classes
        return dict(values=values,macro=macro,strata=strat,selected_alpha=alpha,
                    design_coverage=design_cover,design_curves=curves,q=qvalues,
                    support=support,multiplicity=np.stack(mult),rank_design_object_count=np.array(rank.total),
                    arm_cpu_seconds=np.array(time_arms),total_cpu_seconds=np.array(time.process_time()-started))
