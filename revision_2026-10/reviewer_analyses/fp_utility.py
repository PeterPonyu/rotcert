#!/usr/bin/env python3
"""False-positive and recall readout at the operating points of the frozen S5 certificate audit (reviewer comments on
practical utility; fulfils the theory-appendix statement that false-positive proportions are reported). For each of the six
S5 cells, on its evaluation role: class-exact one-to-one matching at rotated IoU >= 0.5, retained detections with
confidence >= lambda for lambda in the five-threshold grid (0.05 is the operating threshold of every certificate; higher
thresholds keep the higher-confidence subset of the same matches). Reported: TP, FP and FN counts; the false-positive share
of retained detections, pooled and averaged over evaluation source images with at least one retained detection; the share
of source images with at least one false positive; recall, pooled and averaged over GT-positive source images. All
descriptive; no interval and no certificate. Read-only on frozen trees; writes only below
var/reviewer-gaps-20261004/fp-utility. (2026-10-04.)"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import sys
sys.dont_write_bytecode = True
import hashlib
import json
from pathlib import Path

import numpy as np

R = Path(os.environ.get("ROTCERT_ROOT", Path(__file__).resolve().parents[2]))
S5 = R / "revision_2026-10/execution/s5/results"
OUT = R / "var/reviewer-gaps-20261004/fp-utility"
GRID = (0.05, 0.10, 0.20, 0.30, 0.50)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def readout(c, idx, lam, cls=None):
    ns = len(c["scene_names"])
    tsel = c["tp_conf"] >= lam
    fsel = c["fp_conf"] >= lam
    fnsel = np.ones(len(c["fn_scene"]), dtype=bool)
    if cls is not None:
        tsel &= c["tp_cls"] == cls
        fsel &= c["fp_cls"] == cls
        fnsel &= c["fn_cls"] == cls
    # GT = all matched pairs at the operating threshold (any confidence >= 0.05) plus the stored misses
    gsel = (c["tp_cls"] == cls) if cls is not None else np.ones(len(c["tp_scene"]), dtype=bool)
    tp = np.bincount(c["tp_scene"][tsel], minlength=ns)[idx]
    fp = np.bincount(c["fp_scene"][fsel], minlength=ns)[idx]
    gt = (np.bincount(c["tp_scene"][gsel], minlength=ns) + np.bincount(c["fn_scene"][fnsel], minlength=ns))[idx]
    det = tp + fp
    has_det, has_gt = det > 0, gt > 0
    T, F, Gt = int(tp.sum()), int(fp.sum()), int(gt.sum())
    return dict(lam=lam, TP=T, FP=F, FN=Gt - T, GT=Gt,
                fp_share_pooled=F / (T + F) if T + F else None,
                fp_share_source_mean=float(np.mean(fp[has_det] / det[has_det])) if has_det.any() else None,
                sources_with_detection=int(has_det.sum()), sources_with_fp=int((fp > 0).sum()),
                fp_per_source=float(fp.mean()),
                recall_pooled=T / Gt if Gt else None,
                recall_source_mean=float(np.mean(tp[has_gt] / gt[has_gt])) if has_gt.any() else None,
                gt_sources=int(has_gt.sum()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cells, inputs = [], {}
    for path in sorted(S5.glob("*.json")):
        p = json.loads(path.read_text())
        inp = p["input"]
        for key in ("cache", "roles"):
            assert sha(inp[key]) == inp[key + "_sha256"], (p["cell"], key)
            inputs[inp[key]] = inp[key + "_sha256"]
        inputs[str(path)] = sha(path)
        roles = json.loads(Path(inp["roles"]).read_text())
        with np.load(inp["cache"], allow_pickle=False) as z:
            c = {k: z[k] for k in z.files}
        assert c["tp_conf"].min() >= 0.05 - 1e-12 and c["fp_conf"].min() >= 0.05 - 1e-12
        names = list(map(str, c["scene_names"]))
        lookup = {s: i for i, s in enumerate(names)}
        idx = np.array([lookup[s] for s in roles["eval"]], dtype=int)
        classes = list(map(str, c["class_names"]))
        # cross-check the operating-point counts against the frozen S5 support records
        for k, rec in enumerate(p["support"]["eval"]):
            r = readout(c, idx, 0.05, k)
            assert (r["TP"], r["GT"], r["FP"]) == (rec["TPs"], rec["GTs"], rec["FPs"]), (p["cell"], rec["class"])
        overall = [readout(c, idx, lam) for lam in GRID]
        per_class = [dict(cls=n, **readout(c, idx, 0.05, k)) for k, n in enumerate(classes)]
        shares = [r["fp_share_pooled"] for r in per_class if r["fp_share_pooled"] is not None]
        cells.append(dict(cell=p["cell"], eval_sources=int(len(idx)), classes=len(classes), overall=overall,
                          class_fp_share_pooled_range=[min(shares), max(shares)] if shares else None,
                          class_fp_share_pooled_median=float(np.median(shares)) if shares else None,
                          per_class_operating=per_class,
                          certificates_beta020=p["summary"]["0.2"]))
        o = overall[0]
        print(p["cell"], o["TP"], o["FP"], o["FN"], round(100 * o["fp_share_pooled"], 2), round(100 * o["fp_share_source_mean"], 2),
              round(100 * o["recall_pooled"], 2), flush=True)
    receipt = dict(schema="rotcert-fp-utility-v1", grid=list(GRID), operating_threshold=0.05, matching="class-exact one-to-one, rotated IoU >= 0.5",
                   role="S5 evaluation role", cells=cells, inputs=inputs, script_sha256=sha(__file__),
                   interpretation=("descriptive operating-point readout; false-positive share is FP/(TP+FP) of retained detections, "
                                   "not a false-positive rate over true negatives; no certificate covers false positives"))
    path = OUT / "FP-UTILITY.json"
    path.write_text(json.dumps(receipt, indent=1) + "\n")
    print(path, sha(path))


if __name__ == "__main__":
    main()
