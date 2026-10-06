"""S5 initial real-cache certificate audit. Fixed roles/operating point, CPU only.

Uses a finalized owned copy of the independently tested exact contract library.
Point observations are not bootstrap CIs; external assumptions remain explicit.
"""
from pathlib import Path
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from datetime import datetime,timezone
from fractions import Fraction
import fcntl,hashlib,json,math,sys,time
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'vendor'))
from rotcert_contracts.pac import (Provenance,representative_pac_g1,fixed_sequence_g1,
                                  fixed_lambda_g2,compose_pac,fixed_sequence_e2e,FIXED_Q_GRID)
from rotcert_contracts.bounds import empirical_bernstein
from rotcert_contracts.exact import hcp_threshold
from gwd import obb_gwd
from rotcert_contracts.geometry import angle_projection


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def now():return datetime.now(timezone.utc).isoformat()
def mean(v):return float(np.mean(v)) if len(v) else None

def support(cache,idx):
 ns=len(cache['scene_names']);nk=len(cache['class_names']);out=[]
 for k in range(nk):
  tp=np.bincount(cache['tp_scene'][cache['tp_cls']==k],minlength=ns)
  fn=np.bincount(cache['fn_scene'][cache['fn_cls']==k],minlength=ns)
  fp=np.bincount(cache['fp_scene'][cache['fp_cls']==k],minlength=ns)
  out.append({'class':str(cache['class_names'][k]),'sources':len(idx),'TP_positive_sources':int(np.count_nonzero(tp[idx])),
              'GT_positive_sources':int(np.count_nonzero((tp+fn)[idx])),'TPs':int(tp[idx].sum()),'GTs':int((tp+fn)[idx].sum()),'FPs':int(fp[idx].sum())})
 return out


def fast_halfwidth(w,h,q):
    # Numerical utility diagnostic only; exact interval arithmetic resolves the
    # critical branch and audits fixed first/last informative examples below.
    out=np.full(len(w),math.pi/2)
    if math.isinf(q):return out
    gap=(w-h)/(2*np.sqrt(2));inform=(w>h)&(q<gap)
    if q==0:out[inform]=0.;return out
    wi,hi=w[inform],h[inform];scale=wi;ww=wi/scale;hh=hi/scale;qq=q/scale
    t=(ww*ww+hh*hh)/4;d=(ww-hh)*(ww+hh)/8
    out[inform]=.5*np.arcsin(np.clip((qq/d)*np.sqrt(np.maximum(t-qq*qq,0)),0,1))
    near=np.flatnonzero(np.abs(q-gap)<=64*np.finfo(float).eps*np.maximum.reduce([np.abs(gap),np.full(len(w),q),np.full(len(w),1e-300)]))
    for i in near:out[i]=angle_projection(float(w[i]),float(h[i]),float(q)).half_width_upper
    candidates=np.flatnonzero(inform)
    for i in candidates[[0,-1]] if len(candidates) else []:
        exact=angle_projection(float(w[i]),float(h[i]),float(q)).half_width_upper
        if abs(out[i]-exact)>1e-10:raise RuntimeError('Geometry diagnostic disagrees with exact reference')
    return out


def evaluated(cache,scores,idx,k,q):
 ns=len(cache['scene_names']);tpm=cache['tp_cls']==k;tp_sc=cache['tp_scene'][tpm];s=scores[tpm]
 tp=np.bincount(tp_sc,minlength=ns);fn=np.bincount(cache['fn_scene'][cache['fn_cls']==k],minlength=ns);gt=tp+fn
 bad=np.bincount(tp_sc,weights=s>q,minlength=ns)
 good_tp=idx[tp[idx]>0];good_gt=idx[gt[idx]>0]
 rows=np.flatnonzero(tpm & np.isin(cache['tp_scene'],idx))
 p=cache['tp_pred'][rows];g=cache['tp_gt'][rows];sc=cache['tp_scene'][rows]
 w=np.maximum(p[:,2],p[:,3]);h=np.minimum(p[:,2],p[:,3]);half=fast_halfwidth(w,h,q)
 # per-source then uniform class TP: same weighting as the G1 target.
 objw=(1/tp[sc])/len(good_tp) if len(good_tp) else np.empty(0)
 infinite=math.isinf(q)
 return {'q':'infinity' if infinite else q,'n_TP_positive_eval':len(good_tp),'n_GT_positive_eval':len(good_gt),
         'TP_scene_risk':mean((bad[good_tp]/tp[good_tp]).tolist()),'GT_scene_FNR':mean((fn[good_gt]/gt[good_gt]).tolist()),
         'GT_scene_E2E_risk':mean(((fn[good_gt]+bad[good_gt])/gt[good_gt]).tolist()),
         'object_TP_risk':float(bad[idx].sum()/tp[idx].sum()) if tp[idx].sum() else None,
         'infinite_fraction':float(infinite) if len(rows) else None,
         'normalized_radius_scene_mean':'infinity' if infinite and len(rows) else float(objw@(q/np.sqrt(w*h))) if len(rows) else None,
         'angle_halfwidth_scene_mean_radians':float(objw@half) if len(rows) else None,
         'informative_angle_scene_fraction':float(objw@(half<math.pi/2)) if len(rows) else None,
         'risk_is_empirical_heldout':'sampling noise; not a direct observation of population certificate failure'}


def run_cell(cell,protocol,signature):
 with np.load(cell['cache'],allow_pickle=False) as z:c={k:z[k] for k in z.files}
 assert sha(cell['cache'])==cell['cache_sha256'];assert sha(cell['roles'])==cell['roles_sha256']
 roles=json.loads(Path(cell['roles']).read_text());names=c['scene_names'].astype(str);lookup={v:i for i,v in enumerate(names)}
 ids={role:np.asarray([lookup[x] for x in roles[role]],dtype=int) for role in ['G1_cal','G2_cal','eval']}
 for a,b in [('G1_cal','G2_cal'),('G1_cal','eval'),('G2_cal','eval')]:assert not set(ids[a])&set(ids[b])
 assert sum(len(v) for v in ids.values())==len(names)
 assert np.all(c['tp_conf']>=.05) and np.all(c['tp_iou']>=.5)
 scores=obb_gwd(c['tp_pred'],c['tp_gt']);nk=len(c['class_names']);ns=len(names)
 shared={'protocol_sha256':signature['protocol_sha256'],'source_population':cell['dataset']+' frozen source scenes; class-conditional populations differ by TP/GT support',
         'sampling_unit':'image/source image; tile instances retained','detector_sha256':cell['detector_checkpoint_sha256'],
         'matching_sha256':signature['matching_source_sha256'],'preprocessing_sha256':cell['export_provenance_sha256'],
         'source_roles_sha256':cell['roles_sha256'],'class_dictionary':tuple(c['class_names'].astype(str)),
         'training_ids':tuple(roles['train']),'design_ids':tuple(roles['design']),'operating_lambda':Fraction(1,20),
         'representative_seed':20260928}
 p1=Provenance(calibration_ids=tuple(roles['G1_cal']),**shared);p2=Provenance(calibration_ids=tuple(roles['G2_cal']),**shared)
 report={'cell':cell['name'],'utc':now(),'classes':[],'support':{role:support(c,ix) for role,ix in ids.items()},
         'signature':signature,'input':cell,'iid_and_training_independence':'assumed; no train source-ID overlap; geographic independence not proved',
         'certificate_family':'per route class family delta=.05, G1=.025 G2=.025; separate beta targets/routes not selected by eval',
         'status':'POINT_CERTIFICATE_AUDIT_ONLY_BOOTSTRAP_PENDING'}
 for k,classname in enumerate(c['class_names'].astype(str)):
  use=c['tp_cls']==k;sc=c['tp_scene'][use];vals=scores[use];tp=np.bincount(sc,minlength=ns)
  fn=np.bincount(c['fn_scene'][c['fn_cls']==k],minlength=ns);gt=tp+fn
  records={str(names[i]):vals[sc==i] for i in ids['G1_cal']}
  recallrecords={str(names[i]):(int(gt[i]),int(tp[i])) for i in ids['G2_cal']}
  rep=representative_pac_g1(records,classname,p1);eb=fixed_sequence_g1(records,classname,p1)
  marginal=hcp_threshold(list(records.values()),'0.1')
  routes={'representative':rep,'EB':eb};row={'class':classname,'marginal_HCP':{'q':'infinity' if math.isinf(marginal) else marginal,
         'interpretation':'marginal G1, not eligible for PAC composition','evaluation':evaluated(c,scores,ids['eval'],k,marginal)},'G1':routes,'G2':{},'composition':{},'direct_E2E':{}}
  for key,item in routes.items():
   q=math.inf if item['q'] in [None,'infinity'] else item['q']
   # Don't mutate a signed component certificate to append evaluation.
   row.setdefault('evaluation',{})[key]=evaluated(c,scores,ids['eval'],k,q)
  for beta in ['0.2','0.1']:
   g2=fixed_lambda_g2(recallrecords,classname,p2,beta);row['G2'][beta]=g2
   row['composition'][beta]={key:compose_pac(item,g2) for key,item in routes.items()}
   target=Fraction('0.1')+Fraction(beta)
   e2records={str(names[i]):(int(gt[i]),vals[sc==i]) for i in ids['G2_cal']}
   direct=fixed_sequence_e2e(e2records,classname,p2,eps=target)
   chosen=direct['q'];qq=math.inf if chosen=='infinity' else chosen
   row['direct_E2E'][beta]={'certificate':direct,'reported':direct['reported'],'finite':direct['status']=='finite_certificate',
     'evaluation':None if chosen is None else evaluated(c,scores,ids['eval'],k,qq)}
  report['classes'].append(row)
 summary={}
 for beta in ['0.2','0.1']:
  summary[beta]={'class_count':nk,'G2_pass':sum(x['G2'][beta]['reported'] for x in report['classes']),
      'representative_finite_G1':sum(x['G1']['representative']['status']=='finite_certificate' for x in report['classes']),
      'EB_finite_G1':sum(x['G1']['EB']['status']=='finite_certificate' for x in report['classes']),
      'representative_finite_joint':sum(x['composition'][beta]['representative']['reported'] and x['composition'][beta]['representative']['G1_useful_finite'] for x in report['classes']),
      'EB_finite_joint':sum(x['composition'][beta]['EB']['reported'] and x['composition'][beta]['EB']['G1_useful_finite'] for x in report['classes']),
      'direct_finite':sum(x['direct_E2E'][beta]['finite'] for x in report['classes'])}
 report['summary']=summary
 dump(HERE/'results'/f'{cell["name"]}.json',report);return {'cell':cell['name'],'summary':summary}


def main():
 with (HERE/'point-writer.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  (HERE/'results').mkdir(exist_ok=True)
  protocol=json.loads((HERE/'PROTOCOL.json').read_text());frozen=json.loads((HERE/'SOURCE-FREEZE.json').read_text())
  for path,h in frozen['hashes'].items():assert sha(path)==h,path
  signature={'protocol_sha256':sha(HERE/'PROTOCOL.json'),'source_freeze_sha256':sha(HERE/'SOURCE-FREEZE.json'),
             'matching_source_sha256':frozen['matching_source_sha256'],'phase':'point support and certification only, pipeline bootstrap pending'}
  started=time.monotonic();rows=[]
  for cell in protocol['cells']:
   rows.append(run_cell(cell,protocol,signature));print(json.dumps(rows[-1]),flush=True)
  for path,h in frozen['hashes'].items():assert sha(path)==h,path
  dump(HERE/'POINT-COMPLETE.json',{'utc':now(),'status':'POINT_ANALYSIS_COMPLETE_AWAITING_REVIEW','rows':rows,'signature':signature,'seconds':time.monotonic()-started})
if __name__=='__main__':main()
