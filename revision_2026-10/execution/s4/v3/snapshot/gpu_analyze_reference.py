#!/usr/bin/env python3
"""CPU-only paired calibration analysis. Uses owned snapshots, preserving FP/FN and all sources."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from supplement import checked, sha, load_bundle, new_output, write_json, write_rows, require, utc, json_lines


def verify_run(work, bundle, model, condition):
    work=Path(work).resolve()
    r=json.loads((work/'run.json').read_text())
    require(r['status']=='PASS' and r['scope']=='INFERENCE_ONLY_SCIENCE','Only complete science runs can enter analysis')
    require(r['bundle_sha256']==sha(Path(bundle)/'bundle.json') and r['model']==model and r['condition']==condition, 'Wrong run identity')
    for path,h in r['output_hashes'].items():checked(work/path,h)
    return r

def matched_sources(work, expected, score_floor=.05, tau=.5):
    """Match every fixed source, retaining FP, FN, and no-TP scenes. No survivor intersection."""
    import numpy as np
    from analysis_vendor.matching import greedy_match
    from analysis_vendor.gwd import obb_gwd
    detections=defaultdict(list); gt=defaultdict(list)
    inventory=list(json_lines(Path(work)/'image_inventory.jsonl'))
    require({r['id'] for r in inventory}==set(expected) and len(inventory)==len(expected),'Missing all-image inventory')
    for r in inventory:
        require(r['source_id']==expected[r['id']]['source_id'] and r['role']==expected[r['id']]['role'],'Role or source changed')
    for d in json_lines(Path(work)/'detections.jsonl'):
        require(d['image_id'] in expected,'Extra prediction source')
        if d['score']>=score_floor:detections[d['image_id']].append(d)
    for g in json_lines(Path(work)/'ground_truth.jsonl'):
        require(g['image_id'] in expected,'Extra GT source');gt[g['image_id']].append(g)
    result={}
    for source,e in expected.items():
        ds=detections[source];gs=gt[source];byclass={};tp=[];fp=[];fn=[]
        for cls in sorted(set(g['class'] for g in gs)|set(d['class'] for d in ds)):
            di=[i for i,d in enumerate(ds) if d['class']==cls];gi=[i for i,g in enumerate(gs) if g['class']==cls]
            match=greedy_match([ds[i] for i in di],[gs[i] for i in gi],iou_thr=tau)
            tpc=[]
            for pair in match['matches']:
                di0=di[pair['det_index']];gi0=gi[pair['gt_index']];d=ds[di0];g=gs[gi0]
                score=float(obb_gwd(np.asarray(d['obb']),np.asarray(g['obb'])))
                tpc.append({'gt_id':g['gt_id'],'det_index':di0,'gt_index':gi0,'class':cls,
                            'confidence':d['score'],'iou':pair['iou'],'gwd':score,
                            'prediction_area':d['obb'][2]*d['obb'][3],'pred_obb':d['obb'],'gt_obb':g['obb']})
            fpc=[di[i] for i in match['unmatched_det_indices']];fnc=[gs[gi[i]]['gt_id'] for i in match['unmatched_gt_indices']]
            byclass[cls]={'tp':len(tpc),'fp':len(fpc),'fn':len(fnc),'gt':len(gi),'detections':len(di)}
            tp.extend(tpc);fp.extend(fpc);fn.extend(fnc)
        require(len(tp)+len(fn)==len(gs) and len(tp)+len(fp)==len(ds),'TP/FP/FN conservation failed')
        result[source]={'image_id':source,'source_id':e['source_id'],'role':e['role'],
                        'n_gt':len(gs),'n_det':len(ds),'tp':tp,'fp_det_indices':sorted(fp),
                        'fn_gt_ids':fn,'class_counts':byclass,
                        'detections':[{'class':d['class'],'score':d['score'],'area':d['obb'][2]*d['obb'][3],'obb':d['obb']} for d in ds],
                        'gt_classes':[g['class'] for g in gs]}
    return result

def finite_stats(values):
    import numpy as np
    values=list(values); finite=[x for x in values if math.isfinite(x)]
    return {'n':len(values),'n_finite':len(finite),'n_infinite':sum(math.isinf(x) for x in values),
            'infinite_fraction':sum(math.isinf(x) for x in values)/len(values) if values else None,
            'finite_mean':float(np.mean(finite)) if finite else None,
            'finite_median':float(np.median(finite)) if finite else None,
            'finite_p90':float(np.quantile(finite,.9)) if finite else None}

def coverage_and_region(cal,ev,method,alpha=.10):
    import numpy as np
    from analysis_vendor import scene
    groups=[np.array([t['gwd'] for t in r['tp']],dtype=float) for r in cal.values()]
    q_functions={'pooled':scene.pooled_threshold,'hcp':scene.hcp_threshold,'scene_max':scene.scene_max_threshold}
    fixed_q=q_functions[method](groups,alpha) if method!='crc_d' else None
    per=[];all_areas=[];tp_areas=[];ratios=[];q_values=[];byclass=defaultdict(lambda:{'covered':0,'tp':0,'source_ids':set()})
    for sid,r in ev.items():
        q=scene.crc_object_adaptive_threshold(groups,alpha,r['n_det']) if method=='crc_d' else fixed_q
        require(q>=0,'Negative radius is not a useful localization region')
        q_values.append(q);n_tp=len(r['tp']); covered=sum(t['gwd']<=q for t in r['tp'])
        area=math.pi*q*q
        all_areas.extend([area]*r['n_det']);tp_areas.extend([area]*n_tp)
        ratios.extend(area/d['area'] for d in r['detections'])
        for t in r['tp']:
            c=byclass[t['class']];c['covered']+=t['gwd']<=q;c['tp']+=1;c['source_ids'].add(sid)
        per.append({'source_id':sid,'n_gt':r['n_gt'],'n_det':r['n_det'],'tp':n_tp,'fp':len(r['fp_det_indices']),
                    'fn':len(r['fn_gt_ids']),'covered_tp':covered,'q':float(q) if math.isfinite(q) else 'inf',
                    'center_slice_area':float(area) if math.isfinite(area) else 'inf',
                    'tp_object_fraction':covered/n_tp if n_tp else None,
                    'all_tp_covered':bool(covered==n_tp) if n_tp else None,
                    'all_gt_localized_fraction':covered/r['n_gt'] if r['n_gt'] else None,
                    'all_gt_recall':n_tp/r['n_gt'] if r['n_gt'] else None})
    n_tp=sum(x['tp'] for x in per);n_gt=sum(x['n_gt'] for x in per);covered=sum(x['covered_tp'] for x in per)
    def avg(field):
        v=[x[field] for x in per if x[field] is not None]
        return float(np.mean(v)) if v else None
    summary={'method':method,'alpha':alpha,'calibration_all_sources':len(cal),
      'calibration_tp_sources':sum(bool(r['tp']) for r in cal.values()),'calibration_tp_count':sum(len(r['tp']) for r in cal.values()),
      'eval_all_sources':len(ev),'eval_tp_sources':sum(bool(r['tp']) for r in ev.values()),
      'eval_zero_tp_sources':sum(not r['tp'] for r in ev.values()),'eval_gt_positive_zero_tp_sources':sum(r['n_gt']>0 and not r['tp'] for r in ev.values()),
      'tp':n_tp,'fp':sum(x['fp'] for x in per),'fn':sum(x['fn'] for x in per),'gt':n_gt,
      'tp_object_coverage':covered/n_tp if n_tp else None,'tp_scene_weighted_coverage':avg('tp_object_fraction'),
      'tp_scene_simultaneous_coverage':avg('all_tp_covered'),
      'all_gt_object_localization_success':covered/n_gt if n_gt else None,
      'all_gt_scene_localization_success':avg('all_gt_localized_fraction'),
      'all_gt_object_recall':n_tp/n_gt if n_gt else None,'all_gt_scene_recall':avg('all_gt_recall'),
      'q':finite_stats(q_values),'all_detection_center_slice_area':finite_stats(all_areas),
      'tp_center_slice_area':finite_stats(tp_areas),'all_detection_center_slice_area_over_predicted_area':finite_stats(ratios),
      'useful_region_boundary':'Fixed-shape/fixed-angle GWD CENTER SLICE only; not full box union/footprint. No efficiency ranking across unequal coverage.',
      'global_q_per_class_object_coverage_diagnostic':{k:{'covered':v['covered'],'tp':v['tp'],'tp_sources':len(v['source_ids']),'coverage':v['covered']/v['tp'] if v['tp'] else None} for k,v in byclass.items()}}
    return summary,per

def mondrian_endpoints(cal,ev,method,alpha=.10):
    """Primary original-method classwise target; absent classes remain explicitly undefined."""
    from supplement import CLASSES
    def only_class(data,cls):
        result={}
        for sid,r in data.items():
            tp=[x for x in r['tp'] if x['class']==cls]
            ds=[x for x in r['detections'] if x['class']==cls]
            n_gt=r['gt_classes'].count(cls)
            result[sid]=dict(r,tp=tp,detections=ds,n_det=len(ds),n_gt=n_gt,
                fp_det_indices=list(range(len(ds)-len(tp))),
                fn_gt_ids=['count-only']*(n_gt-len(tp)),gt_classes=[cls]*n_gt)
        return result
    classes={};per_source=[]
    for cls in CLASSES:
        summary,rows=coverage_and_region(only_class(cal,cls),only_class(ev,cls),method,alpha)
        summary['support_status']=('NO_CALIBRATION_SUPPORT' if summary['calibration_tp_sources']==0 else 'NO_EVALUATION_SUPPORT' if summary['eval_tp_sources']==0 else 'AVAILABLE')
        summary['eligible_primary_scene_coverage']=summary['tp_scene_weighted_coverage'] if summary['support_status']=='AVAILABLE' else None
        classes[cls]=summary
        per_source.extend(dict(x,**{'class':cls}) for x in rows if x['n_gt'] or x['n_det'])
    valid=[v['eligible_primary_scene_coverage'] for v in classes.values()]
    macro=sum(valid)/len(CLASSES) if all(x is not None for x in valid) else None
    return {'target':'Class-Mondrian conditional-on-class-TP source then uniform class TP; primary original target',
            'method':method,'class_macro_scene_weighted_coverage':macro,
            'undefined_class_count':sum(x is None for x in valid),'fixed_class_count':len(CLASSES),
            'classes':classes,'no_simultaneous_or_power_claim':True},per_source


def recall_and_e2e_diagnostics(cal,ev):
    import numpy as np
    from analysis_vendor import g2,e2e
    out={'claims':'Numerical rejection diagnostics only; assumptions and multiplicity across conditions require root statistics review.', 'g1_score_floor':.05,'g2_operating_threshold':'Classwise selected lambda_max on fixed grid; distinct from G1 floor','e2e_operating_threshold':.30,'composition_computed':False,'composition_reason':'G1/G2 operating thresholds and targets differ; no composition bound is emitted.', 'g2':{},'e2e':{}}
    from supplement import CLASSES
    grid=np.array(g2.FIXED_LAMBDA_GRID)
    for cls in CLASSES:
        def rows(data):
            rr=[]
            for r in data.values():
                n=r['gt_classes'].count(cls)
                if n:
                    conf=np.array([t['confidence'] for t in r['tp'] if t['class']==cls])
                    rr.append(1-(conf[:,None]>=grid[None,:]).sum(axis=0)/n)
            return np.asarray(rr).reshape(-1,len(grid))
        cm,em=rows(cal),rows(ev)
        if not len(cm) or not len(em):
            out['g2'][cls]={'status':'NO_SUPPORT','cal_sources':len(cm),'eval_sources':len(em)};continue
        d=g2.certify_risk(cm,grid,.20,.05,family_size=len(CLASSES));rejected=d.pop('certified');d['numerical_rejection']=rejected
        d['guarantee_claimed']=False;d['eval_sources']=len(em)
        d['eval_risk']=float(em[:,np.flatnonzero(grid==d['lambda_max'])[0]].mean()) if rejected else None
        out['g2'][cls]=d
    def e2e_rows(data):
        records=[]
        for r in data.values():
            conf=[t['confidence'] for t in r['tp']]+[float('nan')]*len(r['fn_gt_ids'])
            scores=[t['gwd'] for t in r['tp']]+[float('nan')]*len(r['fn_gt_ids'])
            records.append((np.asarray(conf),np.asarray(scores)))
        return e2e.e2e_loss_matrix(records,e2e.FIXED_Q_GRID_PX,.30)
    cm,em=e2e_rows(cal),e2e_rows(ev)
    for eps in [.10,.20,.30]:
        if not len(cm) or not len(em):out['e2e'][str(eps)]={'status':'NO_SUPPORT'};continue
        d=e2e.certify_e2e(cm,e2e.FIXED_Q_GRID_PX,eps,.05);rejected=d.pop('certified');d['numerical_rejection']=rejected;d['guarantee_claimed']=False
        d['eval_loss']=float(em[:,np.flatnonzero(e2e.FIXED_Q_GRID_PX==d['q_star'])[0]].mean()) if rejected else None
        out['e2e'][str(eps)]=d
    return out

def analyze(bundle_path,runs_root,output,model,score_floor=.05,tau=.5):
    bp=Path(bundle_path).resolve();b=load_bundle(bp)
    require(model in b['models'],'Model outside bundle')
    require(score_floor in b['protocol']['fixed_analysis']['sensitivity_score_thresholds'] and tau in b['protocol']['fixed_analysis']['sensitivity_iou_thresholds'],'Analysis point not frozen')
    expected={r['id']:r for r in json_lines(bp/'sources.jsonl')}
    run_root=Path(runs_root).resolve();paths={c:run_root/(model+'__'+c+'__science') for c in b['conditions']}
    for c,w in paths.items():verify_run(w,bp,model,c)
    out=new_output(output,[bp,run_root]); matched={}
    for c,w in paths.items():
        matched[c]=matched_sources(w,expected,score_floor,tau)
        write_rows(out/(c+'.matched.jsonl'),matched[c].values())
    role=lambda data,r:{k:v for k,v in data.items() if v['role']==r}
    clean_cal=role(matched['identity'],'calibration');summaries=[];paired=[];diag={}
    for c,data in matched.items():
        ev=role(data,'evaluation');shift_cal=role(data,'calibration');arms=[('clean_cal',clean_cal)]
        if c!='identity': arms.append(('in_condition_cal',shift_cal))
        per_method={}
        for arm,cal in arms:
            for method in b['protocol']['fixed_analysis']['analyzer_method_families']:
                s,per=coverage_and_region(cal,ev,method);s.update(condition=c,arm=arm,model=model)
                summaries.append(dict(s,target='global mixture diagnostic'));per_method[(arm,method)]=per
                ms,mper=mondrian_endpoints(cal,ev,method);ms.update(condition=c,arm=arm,model=model)
                summaries.append(ms)
                write_rows(out/f'{c}.{arm}.{method}.classwise.per_source.jsonl',mper)
                write_rows(out/f'{c}.{arm}.{method}.per_source.jsonl',per)
            diag[c+'.'+arm]=recall_and_e2e_diagnostics(cal,ev) if score_floor==.05 and tau==.5 else {'status':'PRIMARY_MATCH_ONLY','composition_computed':False,'reason':'G2/E2E diagnostic grid is bound to primary score .05 / IoU .5; sensitivity outputs are descriptive localization/recall only.'}
        if c!='identity':
            for method in b['protocol']['fixed_analysis']['analyzer_method_families']:
                for x,y in zip(per_method[('clean_cal',method)],per_method[('in_condition_cal',method)]):
                    require(x['source_id']==y['source_id'],'Pairing drift')
                    paired.append({'source_id':x['source_id'],'condition':c,'method':method,
                        'delta_covered_tp':y['covered_tp']-x['covered_tp'],
                        'delta_all_gt_success':None if x['all_gt_localized_fraction'] is None else y['all_gt_localized_fraction']-x['all_gt_localized_fraction'],
                        'clean_cal_area':x['center_slice_area'],'in_condition_cal_area':y['center_slice_area'],
                        'delta_area_finite_pairs_only':y['center_slice_area']-x['center_slice_area'] if isinstance(x['center_slice_area'],float) and isinstance(y['center_slice_area'],float) else None})
    write_rows(out/'paired_sources.jsonl',paired)
    write_json(out/'summary.json',{'model':model,'score_threshold':score_floor,'iou_threshold':tau,'summaries':summaries,
         'interpretation':'Single fixed split; no empirical-power, cross-condition guarantee, or population CI claim. See root statistics protocol.'})
    write_json(out/'g2-e2e-diagnostics.json',diag)
    write_json(out/'analysis-receipt.json',{'status':'PASS','bundle_sha256':sha(bp/'bundle.json'),
         'inputs':{c:sha(w/'run.json') for c,w in paths.items()},'outputs':{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()},'utc':utc()})
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['bundle','runs-root','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--model',required=True);p.add_argument('--score-floor',type=float,default=.05);p.add_argument('--iou',type=float,default=.5)
    a=p.parse_args();print(analyze(a.bundle,a.runs_root,a.out,a.model,a.score_floor,a.iou))
if __name__=='__main__':main()
