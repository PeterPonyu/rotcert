#!/usr/bin/env python3
"""Scientific synthesis of the accepted S4 completion run (2026-10-03). Read-only on all S4 roots.

Requires the terminal commit of the formal run (terminal_acceptance). Point estimates come from each cell's base
checkpoint (draw -1, hash-checked through the runner's resolve_checkpoint); intervals and numerical-resolution
statuses come from the run's per-cell INTERVALS.json (independently recomputed inside the run). Writes
S4-SUMMARY.json here and the auxiliary supplement tables plus qa/s4-numbers.json in the manuscript draft.
"""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
import json
import math
import sys
from pathlib import Path

REPO = Path(os.environ.get("ROTCERT_ROOT", Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(REPO / "scripts"))
sys.dont_write_bytecode = True
import s4_multicore as core          # noqa: E402
import s4_multicore_full as full     # noqa: E402

PLAN = core.OUTPUT_ROOT / "completion-20261004e/PLAN.json"
HERE = REPO / "var/s4-completion-20261003"
DRAFT = REPO / "manuscript"
NAMES = {"dior-orcnn-s0": "O-RCNN", "dior-roit-s0": "RoI-Transformer", "dior-rtmdet-s0": "RTMDet", "dior-s2anet-s0": "S2ANet"}
SHIFT_NAMES = {"brightness_070": "Brightness 0.70", "contrast_070": "Contrast 0.70", "gamma_150": "Gamma 1.50",
               "saturation_035": "Saturation 0.35", "blue_cast_115": "Blue cast 1.15"}
HCP, POOLED = 1, 0
PRIMARY_LEVEL, MARGINAL_LEVEL = 0.996875, 0.95
RESOLUTION = ("interval_includes_zero", "negative", "positive", "numerically_unresolved")


def ok(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def pct(x):
    return "NA" if not ok(x) else f"{100 * x:.1f}".replace("-", "$-$")


def build_summary(plan, plan_path, out, cfg, terminal_path=None):
    """v2 (2026-10-04): adds the detection-side and all-ground-truth readouts that the run already stores: recall and
    all-ground-truth success (detected and covered) with their marginal intervals and shift contrasts, TP/GT counts,
    the class-macro share of GT-positive scenes without a true positive, and the normalized center radius."""
    metrics = cfg.METRICS
    m_cov, m_e2e, m_rec = metrics.index("scene_coverage"), metrics.index("all_gt_scene_e2e"), metrics.index("all_gt_scene_recall")
    m_tp, m_gt = metrics.index("tp_count"), metrics.index("gt_count")
    m_zero, m_rad = metrics.index("gt_positive_zero_tp_source_rate"), metrics.index("normalized_center_radius")
    arms = [a["name"] for a in cfg.ARMS]
    summary = dict(schema="S4-scientific-synthesis-v2", plan_sha256=core.sha(plan_path), cells={})
    if terminal_path is not None:
        summary.update(terminal=str(terminal_path), terminal_sha256=core.sha(terminal_path))
    for cell in plan["cells"]:
        base = full.resolve_checkpoint(cell, -1, out)
        iv = json.loads((out / cell["key"] / "INTERVALS.json").read_text())["arrays"]
        rec = dict(model=cell["model"], iou=cell["iou"], arms={}, contrasts={})
        for ai, name in enumerate(arms):
            rec["arms"][name] = {meth: dict(scene_coverage=float(base["macro"][ai, mi, m_cov]),
                                            scene_coverage_interval=iv["macro"].get(f"{ai}/{mi}/{m_cov}"),
                                            all_gt_scene_e2e=float(base["macro"][ai, mi, m_e2e]),
                                            all_gt_scene_e2e_interval=iv["macro"].get(f"{ai}/{mi}/{m_e2e}"),
                                            all_gt_scene_recall=float(base["macro"][ai, mi, m_rec]),
                                            all_gt_scene_recall_interval=iv["macro"].get(f"{ai}/{mi}/{m_rec}"),
                                            tp_count=float(base["values"][ai, mi, :, m_tp].sum()),
                                            gt_count=float(base["values"][ai, mi, :, m_gt].sum()),
                                            gt_positive_zero_tp_source_rate=float(base["macro"][ai, mi, m_zero]),
                                            normalized_center_radius=float(base["macro"][ai, mi, m_rad]))
                                 for meth, mi in (("hcp", HCP), ("pooled", POOLED))}
        for ci, c in enumerate(cfg.CONTRASTS):
            rec["contrasts"][c["name"]] = {meth: dict(point=float(base["macro_contrasts"][ci, mi, m_cov]),
                                                      interval=iv["macro_contrasts"].get(f"{ci}/{mi}/{m_cov}"),
                                                      recall=dict(point=float(base["macro_contrasts"][ci, mi, m_rec]),
                                                                  interval=iv["macro_contrasts"].get(f"{ci}/{mi}/{m_rec}")),
                                                      e2e=dict(point=float(base["macro_contrasts"][ci, mi, m_e2e]),
                                                               interval=iv["macro_contrasts"].get(f"{ci}/{mi}/{m_e2e}")))
                                           for meth, mi in (("hcp", HCP), ("pooled", POOLED))}
        summary["cells"][cell["key"]] = rec
    return summary


def bracket(entry, level, resolution_required=True):
    i = entry.get("interval") or {}
    p = i.get("percentile")
    if i.get("status") != "APPROXIMATE_PERCENTILE" or not p or not all(ok(v) for v in p):
        return "NA"
    if i.get("level") != level:
        raise ValueError(f"interval level {i.get('level')} differs from the declared {level}")
    status = (i.get("numerical_resolution") or {}).get("status")
    if status not in RESOLUTION and (resolution_required or status is not None):
        raise ValueError(f"unknown numerical-resolution status {status!r}")
    mark = "$^{u}$" if status == "numerically_unresolved" else ""
    return f"[{pct(p[0])}, {pct(p[1])}]{mark}"


def detection_table(summary, cfg):
    """Detection-side and all-ground-truth readout at the primary IoU 0.5 (marginal 95% intervals)."""
    lines = [r"\begin{table}[htbp]\centering\footnotesize",
             r"\caption{Detection-side limits of recalibration and all-ground-truth success in the photometric experiment "
             r"on the DIOR-R evaluation scenes at rotated IoU $\ge0.5$, for the four seed-0 detector families. "
             r"Recall: class-macro scene-averaged share of ground-truth objects matched by a detection; change: shifted "
             r"minus clean, in points, with a marginal 95\% percentile interval over 5,000 paired source resamples. "
             r"Success: class-macro scene-averaged share of ground-truth objects that are detected and covered by the GWD "
             r"certificate at $\alpha=0.10$, at the clean thresholds and after same-condition recalibration (HCP). TP: "
             r"matched objects; no TP: class-macro share, among all evaluation scenes, of scenes that contain ground truth of "
             r"the class but no true positive; radius: "
             r"class-macro normalized center radius after recalibration. Every condition re-matches its detections, so "
             r"the true-positive population that conditions the coverage statement changes with the condition.}"
             r"\label{supp:tab:s4-detection}",
             r"\setlength{\tabcolsep}{2.4pt}", r"\begin{tabular}{@{}lrcrrrrr@{}}", r"\toprule",
             r" & & & \multicolumn{2}{c}{Success (\%)} & & & \\", r"\cmidrule(lr){4-5}",
             r"Condition & Recall (\%) & Change [interval] & Clean thr. & Recal. & TP & No TP (\%) & Radius \\", r"\midrule"]
    for i, model in enumerate(cfg.MODELS):
        rec = summary["cells"][full.cell_key(model, .5)]
        clean = rec["arms"]["clean_anchor"]["hcp"]
        if i:
            lines.append(r"\addlinespace[3pt]")
        lines.append(rf"\multicolumn{{8}}{{@{{}}l}}{{{NAMES[model]} ({int(clean['gt_count']):,} ground-truth objects)}} \\")
        lines.append(f"Clean & {pct(clean['all_gt_scene_recall'])} & & {pct(clean['all_gt_scene_e2e'])} & & "
                     f"{int(clean['tp_count']):,} & {pct(clean['gt_positive_zero_tp_source_rate'])} & "
                     f"{clean['normalized_center_radius']:.3f} \\\\")
        for shift in cfg.CONDITIONS[1:]:
            cc = rec["arms"][shift + "__clean_cal"]["hcp"]
            rc = rec["arms"][shift + "__recalibrated"]["hcp"]
            assert cc["gt_count"] == rc["gt_count"] == clean["gt_count"] and cc["tp_count"] == rc["tp_count"]
            drop = rec["contrasts"][shift + "__shift_drop"]["hcp"]["recall"]
            lines.append(f"{SHIFT_NAMES[shift]} & {pct(cc['all_gt_scene_recall'])} & {pct(drop['point'])} "
                         f"{bracket(drop, MARGINAL_LEVEL, resolution_required=False)} & {pct(cc['all_gt_scene_e2e'])} & "
                         f"{pct(rc['all_gt_scene_e2e'])} & {int(cc['tp_count']):,} & "
                         f"{pct(cc['gt_positive_zero_tp_source_rate'])} & {rc['normalized_center_radius']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def render_tables(summary, cfg):
    def table(iou, label, caption):
        lines = [r"\begin{table}[htbp]\centering\small", r"\caption{" + caption + r"}\label{" + label + r"}",
                 r"\setlength{\tabcolsep}{2.4pt}", r"\begin{tabular}{@{}lrcrc@{}}", r"\toprule",
                 r" & \multicolumn{2}{c}{Clean thresholds} & \multicolumn{2}{c}{Recalibrated} \\",
                 r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
                 r"Shift & Cov. & Change [interval] & Cov. & Change [interval] \\", r"\midrule"]
        for i, model in enumerate(cfg.MODELS):
            rec = summary["cells"][full.cell_key(model, iou)]
            anchor = rec["arms"]["clean_anchor"]["hcp"]["scene_coverage"]
            if i:
                lines.append(r"\addlinespace[3pt]")
            lines.append(rf"\multicolumn{{5}}{{@{{}}l}}{{{NAMES[model]} (clean {pct(anchor)}\%)}} \\")
            for shift in cfg.CONDITIONS[1:]:
                cc = rec["arms"][shift + "__clean_cal"]["hcp"]["scene_coverage"]
                rc = rec["arms"][shift + "__recalibrated"]["hcp"]["scene_coverage"]
                drop = rec["contrasts"][shift + "__shift_drop"]["hcp"]
                recov = rec["contrasts"][shift + "__recalibration_recovery"]["hcp"]
                primary = iou == .5 and shift in cfg.CORE_SHIFTS
                star = r"$^{\dagger}$" if primary else ""
                level = PRIMARY_LEVEL if primary else MARGINAL_LEVEL
                lines.append(f"{SHIFT_NAMES[shift]}{star} & {pct(cc)} & {pct(drop['point'])} {bracket(drop, level)} & "
                             f"{pct(rc)} & {pct(recov['point'])} {bracket(recov, level)} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        return "\n".join(lines) + "\n"

    common = (r"Source-uniform HCP class-macro scene coverage (percent) of the GWD certificate at $\alpha=0.10$ on DIOR-R "
              r"evaluation scenes under photometric changes, for the four seed-0 detector families, with thresholds from "
              r"clean calibration scenes or recalibrated on calibration scenes under the same change. Change at clean "
              r"thresholds: shifted minus clean coverage; change after recalibration: recalibrated minus clean-threshold "
              r"coverage under the same shift (percentage points). Intervals: percentile intervals over 5,000 paired source "
              r"resamples of the fixed 5,869/5,869 split, recomputed independently inside the run; NA: a required draw is "
              r"undefined; $^{u}$: Monte Carlo bracket touches zero, so no definitive sign. Coverage is conditional on the "
              r"true positives of each condition; an interval containing zero does not show equivalence, and recall and "
              r"all-ground-truth success are given separately. ")
    t5 = table(.5, "supp:tab:s4-iou05", common + r"Matching at rotated IoU $\ge0.5$. $^{\dagger}$: the 16 prespecified "
               r"primary contrasts (brightness and contrast, both contrast kinds, four detectors) at level 99.6875\%; all "
               r"other intervals are marginal 95\%.")
    t67 = table(.6, "supp:tab:s4-iou06", common + r"Sensitivity analysis with matching at rotated IoU $\ge0.6$; all intervals "
                r"marginal 95\%.") + table(.7, "supp:tab:s4-iou07", common + r"Sensitivity analysis with matching at rotated "
                r"IoU $\ge0.7$; all intervals marginal 95\%.")
    return t5, t67, detection_table(summary, cfg)


def numbers(summary, cfg):
    """Quoted main-text numbers (primary IoU 0.5, HCP)."""
    nums = {}
    for model in cfg.MODELS:
        rec = summary["cells"][full.cell_key(model, .5)]
        nums[model] = {}
        for shift in cfg.CONDITIONS[1:]:
            drop = rec["contrasts"][shift + "__shift_drop"]["hcp"]
            recov = rec["contrasts"][shift + "__recalibration_recovery"]["hcp"]
            nums[model][shift] = dict(drop=drop["point"], drop_interval=(drop["interval"] or {}).get("percentile"),
                                      drop_level=(drop["interval"] or {}).get("level"),
                                      drop_resolution=((drop["interval"] or {}).get("numerical_resolution") or {}).get("status"),
                                      recovery=recov["point"], recovery_interval=(recov["interval"] or {}).get("percentile"),
                                      recovery_resolution=((recov["interval"] or {}).get("numerical_resolution") or {}).get("status"),
                                      recal_cov=rec["arms"][shift + "__recalibrated"]["hcp"]["scene_coverage"],
                                      clean_cal_cov=rec["arms"][shift + "__clean_cal"]["hcp"]["scene_coverage"],
                                      recall=rec["arms"][shift + "__clean_cal"]["hcp"]["all_gt_scene_recall"],
                                      recall_drop=rec["contrasts"][shift + "__shift_drop"]["hcp"]["recall"]["point"],
                                      recall_drop_interval=(rec["contrasts"][shift + "__shift_drop"]["hcp"]["recall"]["interval"] or {}).get("percentile"),
                                      success_clean_thr=rec["arms"][shift + "__clean_cal"]["hcp"]["all_gt_scene_e2e"],
                                      success_recal=rec["arms"][shift + "__recalibrated"]["hcp"]["all_gt_scene_e2e"])
        clean = rec["arms"]["clean_anchor"]["hcp"]
        nums[model]["clean_anchor"] = clean["scene_coverage"]
        nums[model]["clean_recall"] = clean["all_gt_scene_recall"]
        nums[model]["clean_success"] = clean["all_gt_scene_e2e"]
    return nums


def main():
    plan = core.checked_json(PLAN)
    out = Path(plan["output"])
    terminal = full.terminal_acceptance(out)
    cfg = core.modules()["config"]
    summary = build_summary(plan, PLAN, out, cfg, terminal_path=out / "TERMINAL-COMMIT.json")
    (HERE / "S4-SUMMARY.json").write_text(json.dumps(summary, indent=1, sort_keys=True, allow_nan=True) + "\n")
    t5, t67, tdet = render_tables(summary, cfg)
    (DRAFT / "supplement/supp-tab-s4-iou05.tex").write_text(t5)
    (DRAFT / "supplement/supp-tab-s4-iou0607.tex").write_text(t67)
    (DRAFT / "supplement/supp-tab-s4-detection.tex").write_text(tdet)
    (DRAFT / "qa/s4-numbers.json").write_text(json.dumps(numbers(summary, cfg), indent=1, allow_nan=True) + "\n")
    print(json.dumps(dict(terminal_status=terminal.get("status"), cells=len(summary["cells"]))))


if __name__ == "__main__":
    main()
