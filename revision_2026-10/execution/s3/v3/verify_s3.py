#!/usr/bin/env python
"""Check every canonical bootstrap draw; pilot has no inferential intervals."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import argparse
import json
import math
import numpy as np
from scipy.stats import binom
from s3_io import ROOT,lock,verify,cap,load_result,write_json,now,sha
from s3_integrity import InputGuard,verify_roles,validate_science,checkpoint_prefix
from s3_geometry import METRICS

def interval(values,B):
    a=np.asarray(values)
    if len(a)!=B:raise ValueError('incomplete B')
    if np.any(np.isnan(a)):return dict(status='unavailable_required_draw_missing',missing=int(np.isnan(a).sum()),B=B)
    ordered=np.sort(a)
    def quant(p):
        x=(B-1)*p;i=math.floor(x);j=math.ceil(x)
        if i==j or ordered[i]==ordered[j]:return float(ordered[i])
        if np.isinf(ordered[j]):return float(ordered[j])
        return float(ordered[i]+(x-i)*(ordered[j]-ordered[i]))
    tails=[]
    for p in [.025,.975]:
        lo=int(binom.ppf(.025,B,p));hi=int(binom.ppf(.975,B,p))+1
        tails.append(dict(p=p,point=quant(p),order_statistic_95_MC_bracket=[float(ordered[max(0,lo-1)]),float(ordered[min(B-1,hi-1)])],order_indices=[max(1,lo),min(B,hi)]))
    blocks=[]
    if B==5000:
        for b in range(5):blocks.append(interval(a[b*1000:(b+1)*1000],1000)['percentile95'])
    return dict(status='available_marginal_exploratory',B=B,percentile95=[quant(.025),quant(.975)],tail_MC=tails,five_blocks1000=blocks)


def main(mode):
    frozen=verify();cap();cells=json.loads((ROOT/'cells.json').read_text());B=2 if mode=='pilot' else 5000
    completed=json.loads((ROOT/mode/'COMPLETE.json').read_text())
    if completed['frozen_sha256']!=frozen or completed['draws_per_cell']!=B:raise RuntimeError('mode completion mismatch')
    receipts=[];v1_changes=[]
    for c in cells:
        cid=c['cell_id'];directory=ROOT/mode/cid
        with lock(ROOT/'locks'/('cell-'+cid+'.lock')):
            guard=InputGuard(c['cache_path'],c['cache_sha256'])
            with np.load(c['cache_path'],allow_pickle=False) as z:cache={k:z[k] for k in z.files}
            roles=verify_roles(ROOT/'prepared'/cid,cache,c,frozen)
            if checkpoint_prefix(directory,B)!=[-1]+list(range(B)):raise RuntimeError('not all predetermined draws present')
            matrices=[];per_class=[]
            for draw in [-1]+list(range(B)):
                p=directory/('base.npz' if draw<0 else f'draw-{draw:04d}.npz')
                r=load_result(p,frozen,c['cell_index'],draw)
                validate_science(r,cache,roles,c['cell_index'],draw);guard.check()
                if draw>=0 and draw%100==0:guard.check(force_hash=True);verify(check_inputs=False)
                if mode=='pilot':
                    with np.load(ROOT.parent/'pilot'/cid/p.name,allow_pickle=False) as z:old={k:z[k] for k in z.files}
                    delta={}
                    for k in ['selected_alpha','design_coverage','design_curves','q','support','multiplicity','rank_design_object_count']:
                        if not np.array_equal(old[k],r[k],equal_nan=True):
                            mask=np.isfinite(old[k])&np.isfinite(r[k])
                            same=(old[k]==r[k])|(np.isnan(old[k])&np.isnan(r[k]))
                            delta[k]=dict(changed=int(np.sum(~same)),max_abs=float(np.max(abs(old[k][mask]-r[k][mask]))) if mask.any() else None)
                    # Numeric score roundoff repair may change an exact last bit;
                    # roles, support and RNG must always be identical.
                    for k in ['support','multiplicity','rank_design_object_count']:
                        np.testing.assert_array_equal(old[k],r[k])
                    v1_changes.append(dict(cell=cid,draw=draw,nongeometry_differences=delta))
                if draw>=0:matrices.append(r['macro']);per_class.append(r['values'])
            if mode=='full':
                mm=np.stack(matrices);cc=np.stack(per_class);out={}
                for a in range(mm.shape[1]):
                    for policy in range(2):
                        out[f'arm{a}/policy{policy}/macro']=[interval(mm[:,a,policy,j],B) for j in range(mm.shape[-1])]
                        for k in range(cc.shape[-2]):out[f'arm{a}/policy{policy}/class{k}']=[interval(cc[:,a,policy,k,j],B) for j in range(cc.shape[-1])]
                p=directory/'MARGINAL-INTERVALS.json'
                if not p.exists():write_json(p,dict(frozen_sha256=frozen,metrics=METRICS,intervals=out),immutable=True)
            guard.check(force_hash=True)
            receipts.append(dict(cell=cid,checkpoints=B+1,exact_RNG_and_canonical_roles='PASS',metadata_and_support='PASS',strict_macro='PASS'))
    verify();p=ROOT/mode/'VERIFIED.json'
    receipt=dict(status='PASS',utc=now(),mode=mode,frozen_sha256=frozen,cells=len(cells),draws_per_cell=B,receipts=receipts,
                 v1_pilot_comparison=v1_changes,pilot_has_no_CI=(mode=='pilot'),interpretation='95% marginal exploratory only; undefined or missing required draws make endpoint CI unavailable')
    if not p.exists():write_json(p,receipt,immutable=True)
    print(json.dumps({k:receipt[k] for k in ['status','mode','cells','draws_per_cell','frozen_sha256']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['pilot','full']);a=p.parse_args()
    with lock(ROOT/'locks/scheduler.lock'):main(a.mode)
