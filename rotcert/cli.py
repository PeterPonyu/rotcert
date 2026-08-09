"""Command-line entry point: ``rotcert {ingest,match,calibrate,recall,certify,audit,report}``
(design §3.1). Every subcommand emits machine-readable JSON/JSONL; ``--json`` echoes
to stdout in addition to (or instead of) ``-o``.

Backbone-agnostic by construction: ``ingest --detector jsonl`` is the only detector
adapter wired up HERE (the certifier never sees pixels, only boxes+scores+GT, design
§3.2); the mmrotate/RTMDet-R box-side adapter lives in
``orchestration/score_rtmdet.py`` (lazy-imported, GPU-box-only) and simply EMITS the
same canonical detections-JSONL schema this CLI's ``ingest --detector jsonl`` also
accepts -- so every certification command downstream is identical whether the
detections came from a live mmrotate run or a precomputed table.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from rotcert import audit as _audit
from rotcert import certify as _certify
from rotcert import io as _io
from rotcert import matching as _matching
from rotcert import sets as _sets
from rotcert.scores import EXPERIMENTAL_SCORES, SCORES


def _emit(result: Dict[str, Any], out: Optional[str], as_json: bool, summary: str) -> None:
    payload = _io.to_jsonable(result)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    if as_json or not out:
        json.dump(payload, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
    else:
        print(summary)


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


def _cmd_ingest(args: argparse.Namespace) -> int:
    if args.detector != "jsonl":
        print(
            f"error: ingest --detector {args.detector!r} requires the box-side GPU "
            "adapter (orchestration/score_rtmdet.py); this CLI's ingest only wires up "
            "'jsonl' (a precomputed detections table) -- see that script's --help.",
            file=sys.stderr,
        )
        return 1
    raw = _io.load_jsonl(args.data)
    dets = _io.validate_detections(raw)
    dets = _io.populate_scene_ids(dets)
    _io.write_jsonl(args.out, dets)
    print(f"ingest: n={len(dets)} -> {args.out}")
    return 0


# ---------------------------------------------------------------------------
# match
# ---------------------------------------------------------------------------


def _cmd_match(args: argparse.Namespace) -> int:
    dets = _io.populate_scene_ids(_io.validate_detections(_io.load_jsonl(args.dets)))
    gts = _io.populate_scene_ids(_io.validate_gts(_io.load_jsonl(args.gt)))

    dets_by_image: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for d in dets:
        dets_by_image[d["image_id"]].append(d)
    gts_by_image: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for g in gts:
        gts_by_image[g["image_id"]].append(g)

    out_rows: List[Dict[str, Any]] = []
    n_tp = n_fp = n_fn = 0
    for image_id in sorted(set(dets_by_image) | set(gts_by_image)):
        d_list = dets_by_image.get(image_id, [])
        g_list = gts_by_image.get(image_id, [])
        scene_id = (d_list[0] if d_list else g_list[0])["scene_id"]
        if not g_list:
            for d in d_list:
                out_rows.append(
                    {
                        "image_id": image_id, "scene_id": scene_id, "class": d["class"],
                        "pred_obb": d["obb"], "gt_obb": None, "pred_score": d["score"],
                        "iou": None, "match_type": "fp",
                    }
                )
                n_fp += 1
            continue
        if not d_list:
            for g in g_list:
                out_rows.append(
                    {
                        "image_id": image_id, "scene_id": scene_id, "class": g["class"],
                        "pred_obb": None, "gt_obb": g["obb"], "pred_score": None,
                        "iou": None, "match_type": "fn",
                    }
                )
                n_fn += 1
            continue

        m = _matching.greedy_match(
            [{"obb": d["obb"], "score": d["score"], "class": d["class"]} for d in d_list],
            [{"obb": g["obb"], "class": g["class"]} for g in g_list],
            iou_thr=args.iou_thr, iou_metric=args.iou_metric,
        )
        for match in m["matches"]:
            d = d_list[match["det_index"]]
            g = g_list[match["gt_index"]]
            out_rows.append(
                {
                    "image_id": image_id, "scene_id": scene_id, "class": d["class"],
                    "pred_obb": d["obb"], "gt_obb": g["obb"], "pred_score": d["score"],
                    "iou": match["iou"], "match_type": "tp",
                }
            )
            n_tp += 1
        for di in m["unmatched_det_indices"]:
            d = d_list[di]
            out_rows.append(
                {
                    "image_id": image_id, "scene_id": scene_id, "class": d["class"],
                    "pred_obb": d["obb"], "gt_obb": None, "pred_score": d["score"],
                    "iou": None, "match_type": "fp",
                }
            )
            n_fp += 1
        for gi in m["unmatched_gt_indices"]:
            g = g_list[gi]
            out_rows.append(
                {
                    "image_id": image_id, "scene_id": scene_id, "class": g["class"],
                    "pred_obb": None, "gt_obb": g["obb"], "pred_score": None,
                    "iou": None, "match_type": "fn",
                }
            )
            n_fn += 1

    _io.write_jsonl(args.out, out_rows)
    print(f"match: tp={n_tp} fp={n_fp} fn={n_fn} iou_thr={args.iou_thr} iou_metric={args.iou_metric} -> {args.out}")
    return 0


# ---------------------------------------------------------------------------
# calibrate (G1)
# ---------------------------------------------------------------------------


def _cmd_calibrate(args: argparse.Namespace) -> int:
    rows = _io.load_jsonl(args.matched)
    tp_rows = [r for r in rows if r.get("match_type") == "tp"]
    if not tp_rows:
        print("error: calibrate: no matched (match_type='tp') rows found", file=sys.stderr)
        return 1
    matched = [
        {"pred_obb": r["pred_obb"], "gt_obb": r["gt_obb"], "class": r["class"],
         "pred_score": r.get("pred_score")}
        for r in tp_rows
    ]
    cert = _certify.g1_calibrate(
        matched, args.score, alpha=args.alpha,
        mondrian_field="class" if args.mondrian else None,
        scale_norm=args.scale_norm,
    )
    n_strata_ok = len(cert["strata"])
    n_refused = len(cert["refused"])
    summary = f"calibrate: score={args.score} alpha={args.alpha} strata_calibrated={n_strata_ok} refused={n_refused}"
    _emit(cert, args.out, args.json, summary)
    return 0


# ---------------------------------------------------------------------------
# recall (G2)
# ---------------------------------------------------------------------------


def _cmd_recall(args: argparse.Namespace) -> int:
    rows = _io.load_jsonl(args.matched)
    relevant = [r for r in rows if r.get("match_type") in ("tp", "fn")]
    if not relevant:
        print("error: recall: no tp/fn rows found", file=sys.stderr)
        return 1

    by_key: Dict[Any, Dict[str, List[Optional[float]]]] = defaultdict(lambda: defaultdict(list))
    for r in relevant:
        conf = r["pred_score"] if r["match_type"] == "tp" else None
        key = r["class"] if args.mondrian else "__pooled__"
        by_key[key][r["scene_id"]].append(conf)

    if args.mondrian:
        by_class = {cls: list(scenes.values()) for cls, scenes in by_key.items()}
        result = _certify.g2_certify_fnr_mondrian(
            by_class, beta=args.beta, delta=args.delta,
            procedure=args.ltt_procedure, p_value=args.ltt_pvalue,
        )
        summary = f"recall: n_classes={result['n_classes']} certified={result['n_classes_certified']}"
    else:
        scenes = list(next(iter(by_key.values())).values())
        result = _certify.g2_certify_fnr(
            scenes, beta=args.beta, delta=args.delta,
            procedure=args.ltt_procedure, p_value=args.ltt_pvalue,
        )
        summary = f"recall: certified={result['certified']} lambda_star={result.get('lambda_star')}"

    _emit(result, args.out, args.json, summary)
    return 0


# ---------------------------------------------------------------------------
# certify (apply a cert to new detections -> per-box regions)
# ---------------------------------------------------------------------------


def _cmd_certify(args: argparse.Namespace) -> int:
    with open(args.cert, "r", encoding="utf-8") as f:
        cert = json.load(f)
    if cert["score_name"] != "gwd":
        print(
            f"error: certify: per-box region reporting (ball + envelope) is only "
            f"defined for score='gwd', got {cert['score_name']!r}",
            file=sys.stderr,
        )
        return 1
    dets = _io.validate_detections(_io.load_jsonl(args.dets))
    mondrian_field = cert.get("mondrian_field")

    regions: List[Dict[str, Any]] = []
    n_refused = 0
    for d in dets:
        stratum = d["class"] if mondrian_field else None
        stratum_key = str(stratum) if stratum is not None else "None"
        cell = cert["strata"].get(stratum_key) or cert["strata"].get(stratum)
        if cell is None:
            n_refused += 1
            regions.append(
                {"image_id": d["image_id"], "class": d["class"], "obb": d["obb"],
                 "refused": True, "reason": f"stratum {stratum!r} not calibrated (out-of-support or refused)"}
            )
            continue
        q_hat = cell["calibrator"]["q_hat"]
        env = _sets.envelope(np.array(d["obb"]), q_hat)
        regions.append(
            {"image_id": d["image_id"], "class": d["class"], "obb": d["obb"],
             "q_hat": q_hat, "envelope": env, "refused": False}
        )
    _io.write_jsonl(args.out, regions)
    print(f"certify: n={len(regions)} refused={n_refused} -> {args.out}")
    return 0


# ---------------------------------------------------------------------------
# audit (V1/V2 coverage tables for one score; Holm-8 via --holm-cells)
# ---------------------------------------------------------------------------


def _cmd_audit(args: argparse.Namespace) -> int:
    if args.holm_cells:
        with open(args.holm_cells, "r", encoding="utf-8") as f:
            cells = json.load(f)
        holm = _audit.holm8_confirmatory(cells, alpha=args.alpha)
        _emit(holm, args.out, args.json, f"audit (holm): family_size={holm['family_size']}")
        return 0

    rows = _io.load_jsonl(args.matched)
    tp_rows = [r for r in rows if r.get("match_type") == "tp"]
    if not tp_rows:
        print("error: audit: no matched (match_type='tp') rows found", file=sys.stderr)
        return 1
    score = SCORES[args.score]
    from rotcert.scores import BonferroniBoxScore

    matched = [{"pred_obb": r["pred_obb"], "gt_obb": r["gt_obb"], "class": r["class"], "scene_id": r["scene_id"]} for r in tp_rows]
    cert = _certify.g1_calibrate(matched, args.score, alpha=args.alpha, mondrian_field=None)
    calibrator = cert["strata"][None]["calibrator"]

    covered_rows = []
    for m in matched:
        if isinstance(score, BonferroniBoxScore):
            c = score.covers(calibrator, m["pred_obb"], m["gt_obb"])
        else:
            c = score.covers(calibrator, m["pred_obb"], m["gt_obb"])
        w, h, theta = m["pred_obb"][2], m["pred_obb"][3], m["pred_obb"][4]
        st = _audit.classify_theta_stratum(w, h, theta)
        covered_rows.append({"covered": c, "scene_id": m["scene_id"], "theta_stratum": st})

    v1 = _audit.coverage_ci([r["covered"] for r in covered_rows], [r["scene_id"] for r in covered_rows])
    v2 = _audit.stratum_coverage_table(covered_rows)
    result = {"score": args.score, "alpha": args.alpha, "v1_marginal_coverage": v1, "v2_stratum_coverage": v2}
    _emit(result, args.out, args.json, f"audit: v1_coverage={v1['point']:.4f}")
    return 0


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _cmd_report(args: argparse.Namespace) -> int:
    payload: Dict[str, Any] = {}
    if args.audit:
        with open(args.audit, "r", encoding="utf-8") as f:
            payload["audit"] = json.load(f)
    if args.cert:
        with open(args.cert, "r", encoding="utf-8") as f:
            payload["cert"] = json.load(f)
    if not payload:
        print("error: report needs --audit and/or --cert", file=sys.stderr)
        return 1
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".json":
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    else:
        lines = ["# rotcert report", ""]
        if "cert" in payload:
            c = payload["cert"]
            lines += [
                "## G1 certificate", "",
                f"- score: `{c['score_name']}`", f"- alpha: {c['alpha']}",
                f"- strata calibrated: {len(c['strata'])}", f"- strata refused: {len(c['refused'])}", "",
            ]
        if "audit" in payload:
            a = payload["audit"]
            lines += ["## Audit", "", f"```json\n{json.dumps(a, indent=2)}\n```", ""]
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    print(f"wrote {out_path}")
    return 0


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rotcert", description="Angle-aware GWD conformal certification for OBB detectors")
    sub = p.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="adapt raw detections/GT into the canonical schema")
    p_ingest.add_argument("--detector", required=True, choices=["rtmdet-r", "oriented-rcnn", "jsonl"])
    p_ingest.add_argument("--data", required=True, help="path to a raw JSONL table (detector='jsonl')")
    p_ingest.add_argument("--split", default="val")
    p_ingest.add_argument("--angle-convention", default="le90", choices=["le90", "le135", "oc"])
    p_ingest.add_argument("-o", "--out", required=True)
    p_ingest.set_defaults(func=_cmd_ingest)

    p_match = sub.add_parser("match", help="preregistered greedy detection<->GT matching")
    p_match.add_argument("--dets", required=True)
    p_match.add_argument("--gt", required=True)
    p_match.add_argument("--iou-metric", default="rotated", choices=["rotated", "hull"])
    p_match.add_argument("--iou-thr", type=float, default=0.5)
    p_match.add_argument("-o", "--out", required=True)
    p_match.set_defaults(func=_cmd_match)

    p_cal = sub.add_parser("calibrate", help="G1: per-Mondrian-cell split conformal")
    p_cal.add_argument("--matched", required=True)
    p_cal.add_argument(
        "--score", required=True, choices=sorted(SCORES) + sorted(EXPERIMENTAL_SCORES),
        help="score construction; names outside the preregistered six "
             f"({', '.join(sorted(EXPERIMENTAL_SCORES))}) are EXPERIMENTAL opt-ins "
             "(exploratory arm only unless promoted at prereg freeze; "
             "'naive-coord-scaled' needs pred_score on every tp row)",
    )
    p_cal.add_argument("--alpha", type=float, default=0.10)
    p_cal.add_argument("--scale-norm", default=None, choices=[None, "sqrt-area"])
    p_cal.add_argument("--mondrian", action="store_true", help="stratify by class")
    p_cal.add_argument("-o", "--out", default=None)
    p_cal.add_argument("--json", action="store_true")
    p_cal.set_defaults(func=_cmd_calibrate)

    p_recall = sub.add_parser("recall", help="G2: certified rotated-IoU FNR via LTT-HB")
    p_recall.add_argument("--matched", required=True)
    p_recall.add_argument("--risk", default="fnr", choices=["fnr"])
    p_recall.add_argument("--beta", type=float, default=0.20)
    p_recall.add_argument("--delta", type=float, default=0.05)
    p_recall.add_argument("--method", default="ltt-hb", choices=["ltt-hb"])
    p_recall.add_argument("--ltt-procedure", default="bonferroni", choices=["bonferroni", "fixed-sequence"])
    p_recall.add_argument("--ltt-pvalue", default="eb", choices=["eb", "hb"])
    p_recall.add_argument("--mondrian", action="store_true", help="per-class certification + pooled fallback")
    p_recall.add_argument("-o", "--out", default=None)
    p_recall.add_argument("--json", action="store_true")
    p_recall.set_defaults(func=_cmd_recall)

    p_certify = sub.add_parser("certify", help="apply a G1 cert -> per-box GWD-ball + envelope")
    p_certify.add_argument("--cert", required=True)
    p_certify.add_argument("--dets", required=True)
    p_certify.add_argument("-o", "--out", required=True)
    p_certify.set_defaults(func=_cmd_certify)

    p_audit = sub.add_parser("audit", help="V1/V2 coverage tables, or Holm-8 via --holm-cells")
    p_audit.add_argument("--matched", default=None)
    p_audit.add_argument("--score", default="gwd", choices=sorted(SCORES))
    p_audit.add_argument("--alpha", type=float, default=0.10)
    p_audit.add_argument("--holm-cells", default=None, help="JSON list of pre-computed contrast cells (rotcert.audit.set_size_contrast output + detector/dataset/baseline keys)")
    p_audit.add_argument("-o", "--out", default=None)
    p_audit.add_argument("--json", action="store_true")
    p_audit.set_defaults(func=_cmd_audit)

    p_report = sub.add_parser("report", help="compact JSON/Markdown summary")
    p_report.add_argument("--audit", default=None)
    p_report.add_argument("--cert", default=None)
    p_report.add_argument("-o", "--output", required=True)
    p_report.set_defaults(func=_cmd_report)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (_io.SchemaError, _certify.CertifyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
