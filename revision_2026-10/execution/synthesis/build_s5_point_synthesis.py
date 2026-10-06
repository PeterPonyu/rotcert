"""Publish only reviewed point-stage S5 claims, with status and fallback explicit."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,math
E=Path(__file__).resolve().parents[1]
O=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
review=E/'review/s5/VERDICT.json';v=json.loads(review.read_text())
assert v['verdict']=='PASS_POINT_STAGE_WITH_REPORTING_LIMITS' and not v['point_rerun_required']
for p,h in v['source_hashes'].items():assert sha(Path(p))==h
protocol=json.loads((E/'s5/PROTOCOL.json').read_text());rows=[];sources={str(review):sha(review)}
for cell in protocol['cells']:
 p=E/'s5/results'/f"{cell['name']}.json";data=json.loads(p.read_text());sources[str(p)]=sha(p)
 classes=[]
 for k in data['classes']:
  rs={}
  for route in ['representative','EB']:
   cert=k['G1'][route];evaluation=k['evaluation'][route]
   rs[route]={'reported':cert['reported'],'status':cert['status'],'q':cert['q'],'evaluation_q':evaluation['q'],
    'evaluation_semantics':'reported region diagnostic' if cert['reported'] else 'infinite mathematical fallback only; no certified deployment',
    'TP_scene_coverage_percent':100*(1-evaluation['TP_scene_risk']) if evaluation['TP_scene_risk'] is not None else None,
    'FNR_percent':100*evaluation['GT_scene_FNR'] if evaluation['GT_scene_FNR'] is not None else None,
    'E2E_risk_percent':100*evaluation['GT_scene_E2E_risk'] if evaluation['GT_scene_E2E_risk'] is not None else None,
    'angle_informative_percent':100*evaluation['informative_angle_scene_fraction'] if evaluation['informative_angle_scene_fraction'] is not None else None,
    'angle_halfwidth_degrees':evaluation['angle_halfwidth_scene_mean_radians']*180/math.pi if evaluation['angle_halfwidth_scene_mean_radians'] is not None else None,
    'normalized_radius':evaluation['normalized_radius_scene_mean'],'infinite_fraction':evaluation['infinite_fraction'],
    'support_TP_G1cal':cert['n_TP_positive'],'empirical_heldout_not_population_risk':True}
  classes.append({'class':k['class'],'marginal_HCP_q':k['marginal_HCP']['q'],'G1_routes':rs,
    'G2':{beta:{key:x for key,x in c.items() if key in ['reported','status','n_GT_positive','risk_limit_exact','bound']} for beta,c in k['G2'].items()},
    'modular':k['composition'],
    'direct':{beta:{key:x for key,x in c.items() if key in ['reported','finite','evaluation']} for beta,c in k['direct_E2E'].items()}})
 rows.append({'cell':cell['name'],'role_counts':cell['role_counts'],'summary':data['summary'],'classes':classes})
out={'utc':datetime.now(timezone.utc).isoformat(),'status':'REVIEWED_POINT_STAGE_ONLY_BOOTSTRAP_PENDING',
 'reporting_corrections_applied':['S5-R1: direct E2E has separate confidence family, shares G2 calibration data; not independent estimates',
 'S5-R2: reported/finite/no_certificate and infinite mathematical fallback retained; no failed route becomes successful by fallback'],
 'confidence':'Per fixed route/target/cell with declared class budget; no combined .95 across routes, beta targets, cells or detectors',
 'modular_risk_targets':{'alpha.1_beta.2':'E2E risk<=.30; success>=70%','alpha.1_beta.1':'E2E risk<=.20; success>=80%'},
 'bootstrap_interval_claims':False,'iid_geographic_independence':'assumed, not established by source IDs',
 'sources':sources,'rows':rows}
p=O/'S5-REVIEWED-POINT-SYNTHESIS.json'
if p.exists():raise SystemExit('Refusing overwrite; explicit version required')
p.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
lines=['# S5 已复核点估计与修稿边界\n',
'本阶段六条件、96类行已独立审查通过，尚未完成B5000全流程不确定性，因此不是完整S5验收。15,564项核验断言是工程/算术验证，不是新增统计样本。\n',
'**原始G1/G2目标已保留并加强量词。** HCP保留TP条件定位的边际保证；新增PAC代表分数和EB两条合法G1，分别与同lambda的G2组合。每条模块化路线采用定位.025/召回.025的置信预算，再分配到固定类别字典。直接E2E是另一个置信族并共享G2校准数据，估计不独立，也无跨路线/模型/风险目标统一95%声明。\n',
'在alpha=.10,beta=.20时模块化目标是端到端风险≤.30，不能叫90%端到端成功；beta=.10时风险≤.20。下面是单个冻结划分的实际输出数量，不能当总体发证概率。\n']
for r in rows:
 a=r['summary']['0.2'];b=r['summary']['0.1'];K=a['class_count']
 lines.append(f"- **{r['cell']}**：有限代表G1 {a['representative_finite_G1']}/{K}，有限EB G1 {a['EB_finite_G1']}/{K}；beta=.20有限模块化组合{a['representative_finite_joint']}/{K}，beta=.10为{b['representative_finite_joint']}/{K}。直接E2E风险.30/.20的有限输出分别{a['direct_finite']}/{K}与{b['direct_finite']}/{K}。\n")
lines += ['\n四个DIOR检测器共同通过beta=.20的类别是groundtrackfield。定位半径有限并不说明召回也能发证；DIOR G2未通过既涉及观测风险偏高，也涉及置信上界宽。DOTA/HRSID不通过不能被说成理论错误，亦不能被说成总体风险一定超标。\n',
'**具体效用不能略掉。** O-RCNN groundtrackfield代表G1的q=20.6663，留出定位风险5.3045%、召回漏检风险4.6233%、E2E风险9.7661%；它仍只获得预定.30目标的证书，不能用9.7661%点估计升级为.10证书。该类代表G1约52.6523%的场景加权TP方向输出有信息，其余仍是全角度。对airplane，代表G1方向有信息比例约9.3071%，说明有限半径和实用方向约束应分别报告。\n',
'**有限性与统计功效分开。** 本预算下代表G1的最低TP-positive校准源支持：20类为每类64、15类为61、单类为36。DOTA最大类仅50、HRSID仅19，故当前设计无法产出有限代表证书。更多bootstrap或GPU迭代不增加源支持；必要时要新的独立场景或另作预先设计的更有效验证路线，不能暗改本轮目标。\n',
'**拒绝、全集与证书状态分开。** 原始point实现为缺失q保留∞数学回退并计算零定位误覆盖，但reported仍为false。汇总输出中保留status、reported、q与evaluation_q；此回退不计成功部署。B5000适配器应使未报告部署风险为NA，只有明确报告的全空间证书才保留其无限代价。两种schema差异须公开。\n',
'**还不能写的结论。** 这里未估计已实现power；没有对固定calibration的总体PAC风险作观测验证；不同路线的发证数量还受校准数据分配与置信预算影响，不是纯粹的模块化代价消融。全流程B5000将描述约定重抽模型下的变化，不能证明iid或增加独立样本。\n',
'来源：`review/s5/VERDICT.json`、`review/s5/REVIEW.md`、`s5/POINT-COMPLETE.json`及六个逐类结果。审查提出的S5-R1、S5-R2已在此汇总处理，冻结源码和旧结果未修改。\n']
(O/'S5-POINT-RESULTS.md').write_text('\n'.join(lines))
print('S5 reviewed point synthesis and two reporting corrections applied')
