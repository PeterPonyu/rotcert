"""Reduce independently verified S1 receipts without reading partial draws."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, math
E = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def pp(values):
    return None if values is None else [100 * float(x) for x in values]

def interval_record(x):
    return {
        'status': x['status'], 'level': x['level'],
        'interval_pp': pp(x['interval']), 'valid_draws': x['valid_draws'],
        'missing_draws': x['missing_draws'],
        'bootstrap_mean_pp': 100*x['mean'] if 'mean' in x else None,
        'quantile_mc_brackets_pp': [pp(z['bounds']) for z in x.get('quantile_mc', [])],
        'five_block_intervals_pp': [pp(z) for z in x.get('five_block_intervals', [])]
    }

def main():
    receipt_path = E/'s1/LOCAL-S1-COMPLETE.json'
    receipt = json.loads(receipt_path.read_text())
    assert receipt['status'] == 'FULL_PRIMARY_AND_SENSITIVITY_VERIFIED'
    source_hashes = {'s1/LOCAL-S1-COMPLETE.json': sha(receipt_path)}
    by_kind = {}
    for kind in ('primary', 'fixed-role'):
        name = f'{kind}-VERIFIED.json'
        path = E/'s1/results'/name
        assert sha(path) == receipt['verification_receipts'][name]
        d = json.loads(path.read_text())
        assert d['status'] == 'S1_COMPLETE_INDEPENDENTLY_VERIFIED'
        source_hashes[f's1/results/{name}'] = sha(path)
        rows = []
        for c in d['cells']:
            assert c['status'] == 'VERIFIED' and c['B'] == 5000
            point = c['point']
            classes = []
            for i, k in enumerate(c['per_class']):
                classes.append({
                    'class': k['class'],
                    'point_delta_pp': 100 * point['per_class_delta'][i],
                    'point_coverages_percent': [100*v[i] for v in point['per_class_coverage']],
                    'delta_ci': interval_record(k['delta_interval95_marginal']),
                    'coverage_ci': [interval_record(z) for z in k['coverage_intervals95_marginal']],
                    'cal_unique_min': k['cal_unique_min'],
                    'eval_unique_min': k['eval_unique_min'],
                    'missing_inner_evaluations': k['missing_inner_evaluations'],
                    'infinite_threshold_fraction_pooled_hcp': k['infinite_threshold_fraction'],
                })
            row = {
                'cell': c['cell'], 'B': c['B'], 'R_inner': c['R_inner'],
                'point_delta_pp': 100 * point['class_macro_delta'],
                'point_macro_coverages_percent': [100*sum(v)/len(v) for v in point['per_class_coverage']],
                'macro_delta_ci': interval_record(c['macro_delta_interval']),
                'unique_support': c['unique_support'], 'classes': classes,
                'verification_scope': c['independent_recalibration'],
            }
            row['class_marginal_positive_count'] = sum(
                x['delta_ci']['interval_pp'] is not None and x['delta_ci']['interval_pp'][0] > 0 for x in classes)
            row['class_marginal_negative_count'] = sum(
                x['delta_ci']['interval_pp'] is not None and x['delta_ci']['interval_pp'][1] < 0 for x in classes)
            row['class_intervals_unavailable'] = [x['class'] for x in classes if x['delta_ci']['interval_pp'] is None]
            rows.append(row)
        by_kind[kind] = rows
    synthesis = {
        'utc': datetime.now(timezone.utc).isoformat(),
        'status': 'VERIFIED_RECEIPTS_SYNTHESIZED', 'source_hashes': source_hashes,
        'contrast': 'class-Mondrian scene-HCP minus class-Mondrian object-pooled; same frozen detector, TP matches and GWD',
        'estimand': 'fixed-dictionary macro of per-class source-uniform-TP localization coverage',
        'primary_point': 'original data mean over 20 frozen source partitions',
        'primary_bootstrap': '5000 outer source resamples; 20 duplicate-safe inner source partitions each',
        'sensitivity': 'original r0 roles fixed; 5000 within-role source resamples; each refits calibration',
        'intervals': 'approximate percentile; macro99.1667% for prespecified six comparisons; class95% marginal',
        'not_claimed': [
            'exact bootstrap finite-sample confidence or simultaneous class coverage',
            'four detector cells are four independent datasets',
            'positive CI establishes desired power for future or rare-class experiments',
            'split/bootstrap repetition increases independent source support',
            'coverage gain at differing radius establishes efficiency superiority',
            'all observed coverage deficit is caused by spatial dependence',
            'HCP is a new generic conformal method',
        ],
        'by_kind': by_kind,
    }
    target = OUT/'S1-VERIFIED-SYNTHESIS.json'
    if target.exists():
        raise SystemExit('Refusing to overwrite an existing synthesis; inspect/version explicitly.')
    target.write_text(json.dumps(synthesis, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    lines = [
        '# S1：对原稿公平的逐类校准比较\n',
        '2026-09-28 UTC / 2026-09-27 PDT。六条件主分析与固定角色敏感性均已完成全部 B5000，并由不导入 runner/core 的验证器完成独立验收。结果来自完整回执，不来自运行中的部分 draw。\n',
        '**支持的结论：** 在冻结检测器、匹配、GWD 与逐类校准范围一致时，改为源场景目标的 HCP，在 DIOR 四个检测器上仍有稳定的宏平均场景覆盖改善。该结果支持修正目标权重具有实际必要性；并不证明通用 HCP 理论属于本文，也不证明所有差异来自空间相关。\n',
        '原数据 point 为20个固定划分的均值；主区间包含源重抽与20次重新划分。固定角色敏感性只对原r0角色内重抽，但每次重新校准。两种目标并列，不按效果选择。区间为近似源bootstrap percentile，宏平均使用预设六条件调整后的99.1667%水平。\n',
    ]
    fixed = {r['cell']:r for r in by_kind['fixed-role']}
    for r in by_kind['primary']:
        ci = r['macro_delta_ci']['interval_pp']
        fs = fixed[r['cell']]
        fci = fs['macro_delta_ci']['interval_pp']
        cv = r['point_macro_coverages_percent']
        txt = f"- **{r['cell']}**：原数据宏覆盖 {cv[0]:.3f}% → {cv[1]:.3f}%，差 {r['point_delta_pp']:.3f}pp；"
        txt += f"主区间 [{ci[0]:.3f}, {ci[1]:.3f}]pp。" if ci is not None else f"完整主区间不可用（{r['macro_delta_ci']['missing_draws']}/5000个draw缺必需类别）。"
        txt += f" 固定r0差 {fs['point_delta_pp']:.3f}pp；"
        txt += f"区间 [{fci[0]:.3f}, {fci[1]:.3f}]pp。" if fci is not None else f"完整区间不可用（{fs['macro_delta_ci']['missing_draws']}/5000个draw缺必需类别）。"
        lines.append(txt+'\n')
    lines += [
        '\n**类别异质性要保留。** 船舶不是代表所有类的总体终点；逐类95%区间是边际区间，没有同时保证。\n'
    ]
    for r in by_kind['primary'][:4]:
        ship = next(x for x in r['classes'] if x['class']=='ship')
        ci = ship['delta_ci']['interval_pp']
        lines.append(f"- {r['cell']} 的 ship：差 {ship['point_delta_pp']:.3f}pp，边际95%区间 [{ci[0]:.3f}, {ci[1]:.3f}]pp。\n")
    lines += [
        '\n**DOTA 不是被统计脚本丢弃。** 原始点值仍保留，缺类draw、无穷阈值及其分母均存档；预设协议禁止删掉这些draw后再声称全类别区间。主分析 helicopter 的HCP阈值在约89.993%的内层校准中为无穷；固定r0该比例约99.98%。这些是重抽协议诊断，不是所有真实部署的无穷率。\n',
        '**HRSID 不支持高精度概括。** 只有48个原始TP源；主重抽平均仅30.505个不同源。主/固定角色区间较宽且点值依赖划分。Bootstrap可以描述不确定性，不能补出独立SAR采集。\n',
        '**统计分辨率与功效分开。** DIOR主宏区间宽度约0.66–0.71pp，固定角色约1.50–1.79pp，足以把本次约3pp的正差与零区分；这不是事后证明“80%power”，也不能推广为所有稀有类或漂移实验都有足够效能。六条件极端尾分位只由约21个排序draw定位，已同时保存二项order-statistic Monte Carlo边界与五个1000块结果。\n',
        '**方法效用仍有独立门槛。** 本结果比较覆盖；阈值/半径增大也能提高覆盖。相同覆盖条件下的方向信息、中心半径和集合成本由S3回答；严格一次校准PAC G1/G2的有限性与代价由S5回答。更多方法/seed不替代这些检验。\n',
        '**验收范围。** 全draw的源multiplicity、角色、随机流、缓存/代码/chunk hash已复核；每条件首末outer、指定inner用独立精确有理校准器与字面对象重复重新算q和覆盖。不是每一个q都做第二套全量计算。源ID本身不证明地理iid。\n',
        '**旧结果并存。** 这里是逐类场景宏平均。旧global约20pp、旧混合场景逐类阈值约4–5pp属于不同终点，不能相互替换，也不能用旧CI给本轮终点背书。原有输出与稿件未被覆盖。\n',
        '来源：`s1/LOCAL-S1-COMPLETE.json`、`s1/results/primary-VERIFIED.json`、`s1/results/fixed-role-VERIFIED.json`；精确hash及全部逐类读数见 `S1-VERIFIED-SYNTHESIS.json`。\n'
    ]
    (OUT/'S1-RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps({'status':synthesis['status'],'json':str(target),'sha256':sha(target),'report':str(OUT/'S1-RESULTS.md')},ensure_ascii=False))

if __name__=='__main__': main()
