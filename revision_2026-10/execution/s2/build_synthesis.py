"""Post-run presentation only. Does not alter frozen algorithms or results."""
import json
import math
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parent
source=json.loads((ROOT/'full/summary.json').read_text())
verified=json.loads((ROOT/'full/VERIFIED.json').read_text())
assert verified['status']=='PASS'
rows=source['results']

def stat(cid,method,metric):
    return rows[cid]['methods'][method][metric]

def value(cid,method,metric):
    s=stat(cid,method,metric)
    return f"{100*s['mean']:.3f}%（MCSE {100*s['mcse']:.3f}pp）"

def effect(cid):
    s=rows[cid]['paired_contrasts']['pooled_minus_hcp_population_scene_risk']
    return f"{100*s['mean']:+.3f}pp（配对MCSE {100*s['mcse']:.3f}pp）"

def risk_range(method,metric):
    vals=[r['methods'][method][metric]['mean'] for r in rows]
    return [min(vals),max(vals)]

contrasts=[]
for a,b in [(6,8),(18,20),(30,32)]:
    x=stat(a,'pooled','population_scene_risk_lower');y=stat(b,'pooled','population_scene_risk_lower')
    contrasts.append(dict(contrast='rho.9_minus_rho0_at_gamma0_heterogeneous',
        ncal=rows[a]['condition']['ncal'],condition_ids=[a,b],
        pooled_population_scene_risk_difference=y['mean']-x['mean'],
        mcse_independent_conditions=math.hypot(x['mcse'],y['mcse']),
        interpretation='controlled synthetic effect; not empirical spatial causation'))

cells=[]
for r in rows:
    c=r['condition']
    cells.append(dict(condition=c,replications=r['replications'],
        paired_hcp_minus_pooled_scene_coverage_effect=r['paired_contrasts']['pooled_minus_hcp_population_scene_risk'],
        all_methods=r['methods'],additional_paired_contrasts=r['paired_contrasts'],
        population_E_m=r['population_E_m']))

synthesis=dict(schema='s2-complete-scientific-synthesis-v1',utc=datetime.now(timezone.utc).isoformat(),
    source='full/summary.json',verification='full/VERIFIED.json',status='COMPLETE_VERIFIED',
    original_design_changed=False,
    total_calibration_replications=72000,total_fresh_evaluation_scenes=144000000,
    nominal_risk=.1,coverage_effect_sign='positive = HCP higher scene coverage than pooled',
    main_reported_risk='population conditional risk, averaged across2000 calibration draws; fresh2000-scene unbiased expected-object estimator remains in all-method results',
    error_bars='MCSE of Monte Carlo mean, not confidence bounds for observed real datasets or simultaneous inference',
    poisson_FP_tail_bracket_bound=source['poisson65plus_tail_bound'],
    rho_contrasts=contrasts,cells=cells,
    limitations=[
        'The latent Gaussian intervention identifies mechanisms only inside the frozen synthetic data-generating model.',
        'The mean over calibration randomness is not a PAC bound for a realized calibration set.',
        'Scene-uniform and object-count-weighted risks differ; neither guarantee transfers to the other.',
        'Finite-output subset risks are selected diagnostics; infinity rates must accompany coverage.',
        'No theorem is proven by these36 simulation cells; mathematical proof is separate.',
        'S2 does not resolve geometry superiority, EAV novelty, actual data iid, or shift robustness.',
    ],requested_model='gpt-6-astra',actual_model_verification='parent responsibility; not inferred')
synthesis['ranges']={
    'hcp_population_scene_risk':risk_range('hcp','population_scene_risk_lower'),
    'scene_max_fresh_eval_any_failure':risk_range('scene_max','scene_any_failure'),
    'crc_m_population_object_risk':risk_range('crc_m','population_object_risk_lower'),
    'crc_d_m_population_object_risk':risk_range('crc_d_m','population_object_risk_lower'),
    'crc_d_poisson_population_object_risk':risk_range('crc_d_m_plus_poisson5','population_object_risk_lower'),
}
(ROOT/'SCIENTIFIC-SYNTHESIS.json').write_text(json.dumps(synthesis,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

lines=[
'# S2 已完成：权重错配、依赖和计数修正的可分辨机制',
'',
'结论：36个冻结条件均完成，2000次独立校准重复/条件，每次2000个新评估场景；共72,000次校准重复、1.44亿个fresh evaluation scenes。六个方法使用配对数据。完整结果经过独立验收，未删除零效应或反向条件。',
'',
'实验与实现仅写本S2目录。请求模型gpt-6-astra；实际runtime模型核实归父任务。本报告不是新颖性审稿判决，也不改变原稿的angle-aware G1/G2目标。',
'',
'## 主要科学发现',
'',
'**1. 计数—难度关联足以造成持续的场景/对象目标错配，即使rho=0。**',
'',
'异质计数、gamma=−.75、rho=0时，稠密场景更容易，池化对象分位数过度代表它们。ncal=1000：pooled场景风险 '+value(27,'pooled','population_scene_risk_lower')+'，HCP '+value(27,'hcp','population_scene_risk_lower')+'；HCP场景覆盖增加 '+effect(27)+'。pooled对象风险仅 '+value(27,'pooled','population_object_risk_lower')+'。所以约29pp的场景差可以主要来自估计目标的加权错配，而不是必须归因场景内相关；本条件不同场景的m仍随机，rho0仅排除给定m后的U相关。',
'',
'**2. 方向能够反转，不能宣称HCP在所有指标上支配pooled。**',
'',
'异质计数、gamma=+.75、rho=0、ncal=1000，稠密场景更难，pooled场景风险降至 '+value(33,'pooled','population_scene_risk_lower')+'，HCP为 '+value(33,'hcp','population_scene_risk_lower')+'；覆盖差 '+effect(33)+'。HCP覆盖更接近其场景目标而不是更高；对应HCP对象风险 '+value(33,'hcp','population_object_risk_lower')+'，并不违背场景定理。这个不利目标结果必须保留，禁止把HCP对象风险解释为也获90%保证。',
'',
'**3. 没有size-score关联时，潜在相关仍能造成有限校准池化失真，且随ncal增大衰减。**',
'',
'固定gamma=0、异质m，单对象边际分布与rho无关。rho从0增加到.9时：',
]
for c in contrasts:
    a,b=c['condition_ids']
    lines.append(f"- ncal={c['ncal']}：pooled场景风险从 {value(a,'pooled','population_scene_risk_lower')} 变为 {value(b,'pooled','population_scene_risk_lower')}；跨独立条件差 {100*c['pooled_population_scene_risk_difference']:+.3f}pp，MCSE {100*c['mcse_independent_conditions']:.3f}pp。")
lines += [
'',
'这与rho保持单对象边际、但改变校准分位数抽样分布的机制相符。是本合成生成机制内的受控证据，不是对真实遥感空间相关的因果识别。',
'',
'**4. CRC-D 能显著减少固定M造成的无穷输出，但必须守住对象风险口径。**',
'',
'异质计数、ncal=50的9条件里，CRC-M无穷输出占场景99.15%–99.75%；CRC-D(D=m)为4.945%–4.999%，加入Poisson(5)误检后为5.295%–5.515%。这说明小样本下固定M=64的保守代价极大；单报接近零风险会把几乎全是无穷集合的结果误读为实用优势。',
'',
'场景比例不能代替对象比例：同一组ncal50异质条件里，CRC-D(m)仍有53.620%–53.997%的评估对象得到无穷集合，CRC-D(m+Poisson5)为54.310%–54.730%。约5%的无穷场景正是高计数场景。对象加权实用覆盖仍有明显代价，不能把约95%有限场景写成约95%有限对象。',
'',
'ncal=1000、gamma=−.75、rho=0时，CRC-D对象风险 '+value(27,'crc_d_m','population_object_risk_lower')+'，但其场景风险 '+value(27,'crc_d_m','population_scene_risk_lower')+'。它准确地说明两个目标不同；不能把CRC-D称为自动修复HCP场景风险的替代。与CRC-M的同目标配对差、输出率和q均在完整结果保存。',
'',
'**5. 误检计数带来保守成本，而不是额外定位信息。**',
'',
'D=m+Poisson5与D=m使用同一匹配分数分布。增加D逐点提高或保持阈值，可能产生无穷输出；36条件的总体对象风险均降低。例ncal50/gamma−.75/rho0：'+value(3,'crc_d_m','population_object_risk_lower')+' → '+value(3,'crc_d_m_plus_poisson5','population_object_risk_lower')+'。这不代表后者定位能力更好。',
'',
'**6. 平均边际风险控制与单次校准可靠性明显不同。**',
'',
'例ncal1000/gamma−.75/rho0，HCP的平均场景风险9.936%，但46.4%的校准实现其条件总体场景风险超过10%；CRC-D(D=m)平均对象风险9.855%，但42.0%的校准实现其条件对象风险超过10%。这些跨校准频率仅是已知模拟模型下的描述性量，直接支持修复量词的必要性，不能把边际模拟达标包装成PAC保证。',
'',
'## 全部方法的风险与零结果',
'',
'- HCP平均总体场景风险范围8.234%–9.989%；常数m下ncal50保守，ncal1000接近10%。这与边际定理相符，但36个模拟条件不证明普适性。',
'- scene-max的fresh-eval场景任意失败率均值范围9.724%–10.048%。有9个条件的均值略高于10%，最大名义越界仅1.64个MCSE；这些结果完整保留，不能把每一个Monte Carlo均值≤10%设成定理通过门槛。连续场景最大值分布下，单次split conformal的ensemble风险理论值为(n+1−ceil(.9(n+1)))/(n+1)，分别约9.804%、9.950%、9.990%，与模拟尺度一致。',
'- CRC-M、CRC-D(m)、CRC-D(m+Poisson5)36条件平均总体对象风险范围分别0.002%–9.942%、0.484%–9.942%、0.274%–9.892%。低风险要结合无穷率与尺寸解释。',
'- 常数m时，同策略下对象/场景权重差逐次精确为0；rho仍会改变有限校准行为。pooled与HCP的阈值并不完全相同：例如ncal50时覆盖差约1.72–1.73pp来自不同校正强度，ncal1000缩至约0.087–0.088pp。不能用常数m强行要求校准器输出完全一致。',
'- gamma=0、rho0、ncal1000，pooled场景风险9.998%（MCSE0.009pp），HCP9.936%（MCSE0.018pp）；配对差仅+0.062pp（MCSE0.016pp）。这里没有一个与负gamma条件相似的巨大“空间校正优势”。',
'',
'## 效应量与精度口径',
'',
'以下附录报告所有36条件、六方法。风险值和效应都用百分点表达；“±”后一项是一个MCSE，不是95%CI或多重校正后显著性声明。主要风险摘要使用给定校准集下可解析的population marginal，再跨2000校准取均值，避免fresh eval噪声掩盖机制。严格按原冻结要求生成的2000 fresh eval场景的scene risk、E[N]/已知E[m]估计、有限测试比值、run SD和2.5/50/97.5分位数都保留在full/summary.json、SCIENTIFIC-SYNTHESIS.json与原始NPZ。CRC-D-FP的解析Poisson求和尾余量≤2.396e−48；不把这个截断界称为浮点严格外包或置信区间。',
'',
'配对HCP覆盖差 = pooled场景风险−HCP场景风险；正值表示HCP场景覆盖较高，负值表示pooled更保守。scene-max报告2000 fresh eval下任意对象失败的scene率；其余主目标为population风险。各方法的目标不同，不对六列直接排“优劣”。',
'',
'## 36条件完整摘要',
]
for r in rows:
    c=r['condition'];cid=c['condition_id'];ms=r['methods'];p=r['paired_contrasts']['pooled_minus_hcp_population_scene_risk']
    lines += ['',f"### C{cid:02d}: ncal={c['ncal']}, {c['regime']}, gamma={c['gamma']:+.2f}, rho={c['rho']:.1f}",'']
    for method,metric,label in [('pooled','population_scene_risk_lower','pooled场景'),('hcp','population_scene_risk_lower','HCP场景'),('scene_max','scene_any_failure','scene-max场景任意失败'),('crc_m','population_object_risk_lower','CRC-M对象'),('crc_d_m','population_object_risk_lower','CRC-D(m)对象'),('crc_d_m_plus_poisson5','population_object_risk_lower','CRC-D(m+Pois5)对象')]:
        s=ms[method][metric];inf=ms[method]['infinite_scene_rate']['mean'];infobj=ms[method]['infinite_object_rate']['mean']
        lines.append(f"- {label}风险：{100*s['mean']:.3f}% ± {100*s['mcse']:.3f}pp MCSE；run SD {100*s['sd']:.3f}pp；无穷输出场景 {100*inf:.3f}%、对象 {100*infobj:.3f}%。")
    lines.append(f"- 配对HCP−pooled场景覆盖：{100*p['mean']:+.3f}pp ± {100*p['mcse']:.3f}pp MCSE；run SD {100*p['sd']:.3f}pp。")
lines += [
'',
'## 对论文四个高风险点的作用与未解决项',
'',
'- R1-2/R3-2：S2支持明确区分scene-uniform、object-weighted与scene-simultaneous estimand，解释为什么边际保证不能升级为一次校准后的PAC保证；数值实现的整数边界与独立测试补上实现一致性证据。',
'- R1-1/R3-1：S2提供“加权目标错配与依赖分开”的正向/反向/零控制机制图景，不再用20pp观察差泛称空间因果。它不单独构成角度几何新颖性，也不能代替EAV方法及系统对照。',
'- 原始angle-aware G1/G2目标保留。S2是机制与量词证据；S3仍需在相同目标、相同校准/设计/评估流程下检验角度信息量、有限区域率、中心半径与强基线代价。真实数据的地理独立性和受控shift结果另有验收门槛。',
'',
'## 执行、验收与交付',
'',
'实际试算360rep：0.620 CPU秒、0.738墙钟秒；按×200×1.5安全系数预计186.085 CPU秒、221.403墙钟秒，预测内存449.97MiB，符合父任务<25分钟/≤2GiB门槛，启动前已通知。完整运行实际83.404 CPU秒、90.734墙钟秒，峰值100.735MiB；没有GPU/SSH或依赖安装。',
'',
'14项独立科学/完整性测试通过；完整结果43200项摘要标量独立复算通过，最大舍入差2.14e−14；72次首尾rep重放通过；36条件前10次与pilot完全相等；所有源/输入/检查点hash一致。full/VERIFIED.json记录验收，不需要为展示再重复运行。',
'',
'本S2职责完成。可以承接父任务另行限定范围和文件所有权的S3 equal-target geometry pipeline；本次没有启动任何S3工作。',
]
(ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print('Wrote RESULTS.md and SCIENTIFIC-SYNTHESIS.json for all36 conditions; no simulation rerun.')
