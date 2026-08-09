#!/usr/bin/env python3
"""Extract plot data (CSV) from frozen result JSONs for the rotcert manuscript
figures. Every number here is read verbatim from the frozen records -- no
statistic is recomputed. Run from manuscripts/figures-src/.

Sources:
  - dior_perclass_2026-07-15/results.json
      -> fig1_perclass.csv (per-class OOS coverage, 20 classes x 4 DIOR-R cells)
  - coverage_matched_2026-07-13/results.json
    + coverage_matched_configB_ext_2026-07-15/results.json
      -> fig2_covmatched.csv (per-cell matched-coverage area ratio, R=20 mean
         and [min,max], GWD vs naive-coord and GWD vs hull)
"""
import csv
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")  # tools/rotcert/

CELL_LABELS = {
    "dior_orcnn_fixedgt": "ORCNN",
    "dior_rtmdet": "RTMDet",
    "dior_roi_trans": "RoI-Trans",
    "dior_s2anet": "S2A-Net",
}

# y-axis order: by n_cal_full (ORCNN cell) descending, matching Table
# tab:diorperclass's existing row order in the manuscript.
CLASS_ORDER = [
    "ship", "storagetank", "vehicle", "tenniscourt", "airplane",
    "baseballfield", "windmill", "basketballcourt", "groundtrackfield",
    "harbor", "bridge", "overpass", "expressway-service-area", "chimney",
    "stadium", "expressway-toll-station", "golffield", "airport",
    "trainstation", "dam",
]


def extract_fig1():
    path = os.path.join(ROOT, "dior_perclass_2026-07-15", "results.json")
    with open(path) as f:
        d = json.load(f)
    rows = []
    for cell_key, cell_label in CELL_LABELS.items():
        cell = d["cells"][cell_key]
        for cls in CLASS_ORDER:
            pc = cell["per_class"][cls]
            rows.append({
                "class": cls,
                "class_idx": CLASS_ORDER.index(cls),
                "cell": cell_label,
                "oos_coverage_mean": pc["oos_coverage_mean"],
            })
    out = os.path.join(os.path.dirname(__file__), "fig1_perclass.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["class", "class_idx", "cell", "oos_coverage_mean"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")

    # Per-series CSVs for direct \addplot table use in the pgfplots source
    # (trivial filter of the long-format file above -- same numbers, split by
    # cell so the TikZ source needs no string filtering).
    for cell_label in CELL_LABELS.values():
        sub = [r for r in rows if r["cell"] == cell_label]
        fname = os.path.join(os.path.dirname(__file__), f"fig1_series_{cell_label.replace(' ', '')}.csv")
        with open(fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["class", "class_idx", "oos_coverage_mean"])
            w.writeheader()
            w.writerows({k: r[k] for k in ("class", "class_idx", "oos_coverage_mean")} for r in sub)
        print(f"wrote {fname} ({len(sub)} rows)")

    # Global min/max class coverage across the four cells (each already a
    # frozen per-cell field, min_class_coverage/max_class_coverage; this is
    # just the min/max of four already-computed numbers, not a new statistic)
    # -- used to draw the "~1.3pt band around nominal" reference band.
    global_min = min(d["cells"][k]["min_class_coverage"] for k in CELL_LABELS)
    global_max = max(d["cells"][k]["max_class_coverage"] for k in CELL_LABELS)
    band_out = os.path.join(os.path.dirname(__file__), "fig1_band.csv")
    with open(band_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "value"])
        w.writerow(["nominal_coverage", d["nominal_coverage"]])
        w.writerow(["global_min_class_coverage", global_min])
        w.writerow(["global_max_class_coverage", global_max])
    print(f"wrote {band_out}: nominal={d['nominal_coverage']} min={global_min:.4f} max={global_max:.4f}")


CM_CELL_LABEL = {
    ("rtmdet", "dota"): "RTMDet--DOTA",
    ("orcnn", "dota"): "ORCNN--DOTA",
    ("orcnn", "dior"): "ORCNN--DIOR-R",
    ("rtmdet", "dior"): "RTMDet--DIOR-R",
    ("roi_trans", "dior"): "RoI-Trans--DIOR-R",
    ("s2anet", "dior"): "S2A-Net--DIOR-R",
}
# y-axis order for fig 2, top to bottom (matches Table tab:covmatched order)
CM_ORDER = [
    "RTMDet--DOTA", "ORCNN--DOTA", "ORCNN--DIOR-R", "RTMDet--DIOR-R",
    "RoI-Trans--DIOR-R", "S2A-Net--DIOR-R",
]


def extract_fig2():
    rows = []
    for fname in ["coverage_matched_2026-07-13/results.json",
                  "coverage_matched_configB_ext_2026-07-15/results.json"]:
        path = os.path.join(ROOT, fname)
        with open(path) as f:
            d = json.load(f)
        for c in d["cells"]:
            key = (c["detector"], c["dataset"])
            label = CM_CELL_LABEL[key]
            rows.append({
                "cell": label,
                "cell_idx": CM_ORDER.index(label),
                "baseline": c["baseline"],
                "ratio_matched_mean": c["ratio_matched_r20_mean"],
                "ratio_matched_min": c["ratio_matched_r20_range"][0],
                "ratio_matched_max": c["ratio_matched_r20_range"][1],
            })
    out = os.path.join(os.path.dirname(__file__), "fig2_covmatched.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "cell", "cell_idx", "baseline", "ratio_matched_mean",
            "ratio_matched_min", "ratio_matched_max"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")

    # Per-baseline CSVs for direct \addplot table use (trivial filter of the
    # long-format file above; same numbers).
    for baseline in ["naive-coord", "hull"]:
        sub = [r for r in rows if r["baseline"] == baseline]
        sub.sort(key=lambda r: r["cell_idx"])
        fname = os.path.join(os.path.dirname(__file__),
                              f"fig2_series_{baseline.replace('-', '')}.csv")
        cols = ["cell", "cell_idx", "ratio_matched_mean", "ratio_matched_min", "ratio_matched_max"]
        with open(fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows({k: r[k] for k in cols} for r in sub)
        print(f"wrote {fname} ({len(sub)} rows)")


if __name__ == "__main__":
    extract_fig1()
    extract_fig2()
