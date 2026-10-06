#!/usr/bin/env python3
"""No-resampling replay against parent-owned S5 point snapshot (read-only)."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import json,math,time
import numpy as np
from bootstrap_core import CellData,ROLES,ROUTES,METRICS

HERE=Path(__file__).resolve().parent


def run():
    protocol=json.loads((HERE/'source_snapshot/s5/PROTOCOL.json').read_text())
    rows=[];errors=[]
    for cell in protocol['cells']:
        start=time.monotonic();data=CellData(cell)
        weights={r:np.ones(len(data.role_names[r]),dtype=np.int64) for r in ROLES}
        result,receipt=data.compute(weights,0,point_seed=True)
        point=json.loads((HERE/'source_snapshot/s5/results'/f'{cell["name"]}.json').read_text())
        max_error=0.;known_na=0
        for k,row in enumerate(point['classes']):
            expected_q=[row['marginal_HCP']['q'],row['G1']['representative']['q'],row['G1']['EB']['q'],
                        row['direct_E2E']['0.2']['certificate']['q'],row['direct_E2E']['0.1']['certificate']['q']]
            evaluations=[row['marginal_HCP']['evaluation'],row['evaluation']['representative'],row['evaluation']['EB'],
                         row['direct_E2E']['0.2']['evaluation'],row['direct_E2E']['0.1']['evaluation']]
            for r,q in enumerate(expected_q):
                expected_status=0 if q is None else 2 if q=='infinity' else 1
                if result['status'][k,r]!=expected_status:errors.append([cell['name'],k,r,'status'])
                if q is None:
                    assert np.all(np.isnan(result['metrics'][k,r]))
                    known_na+=int(evaluations[r] is not None)
                    continue
                q=math.inf if q=='infinity' else q
                if result['metrics'][k,r,0]!=q:errors.append([cell['name'],k,r,'q'])
                for j,name in enumerate(METRICS):
                    if name=='TP_scene_coverage':expected=1-evaluations[r]['TP_scene_risk'] if evaluations[r]['TP_scene_risk'] is not None else math.nan
                    else:
                        value=evaluations[r][name];expected=math.nan if value is None else math.inf if value=='infinity' else value
                    actual=result['metrics'][k,r,j]
                    if math.isnan(expected):
                        if not math.isnan(actual):errors.append([cell['name'],k,r,name,'NA'])
                    elif math.isinf(expected):
                        if actual!=expected:errors.append([cell['name'],k,r,name,'inf'])
                    else:
                        error=abs(actual-expected);max_error=max(max_error,error)
                        if error>1e-10:errors.append([cell['name'],k,r,name,error])
            for b,beta in enumerate(['0.2','0.1']):
                if result['g2_pass'][k,b]!=row['G2'][beta]['reported']:errors.append([cell['name'],k,b,'G2_pass'])
                if result['g2_upper'][k]!=row['G2'][beta]['bound']['upper']:errors.append([cell['name'],k,b,'G2_upper'])
                for route,r in [('representative',5+b),('EB',7+b)]:
                    expected=row['composition'][beta][route]['reported']
                    if bool(result['status'][k,r])!=expected:errors.append([cell['name'],k,r,'joint_status'])
        rows.append({'cell':cell['name'],'seconds':time.monotonic()-start,'max_numeric_error':max_error,
                     'expected_no_certificate_deployment_NA_instead_of_point_fallback':known_na})
    out={'status':'PASS' if not errors else 'FAIL','scope':'identity multiplicities and point representative seed; NOT bootstrap draws or confidence intervals',
         'cells':rows,'errors':errors}
    (HERE/'evidence/identity-parity.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2));return not errors

if __name__=='__main__':raise SystemExit(0 if run() else 1)
