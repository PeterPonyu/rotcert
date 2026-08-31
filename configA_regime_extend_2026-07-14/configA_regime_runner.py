#!/usr/bin/env python3
"""Extend the regime-conditional coverage-matched analysis with the Config-A cells
(2026-07-14): HRSC (ORCNN, RTMDet-R) and the two surviving DOTA zoo detectors
(RoI Transformer, Oriented RepPoints -- S2A-Net and Gliding Vertex excluded, see
dota_zoo/S2ANET-EXCLUDED-2026-07-14.md and zoo_gliding_vertex/GLIDINGVERTEX-EXCLUDED-2026-07-14.md).

Reuses coverage_matched_regime_2026-07-13's run_cell() verbatim (same protocol,
same helpers) on NEW cells only -- does not touch the frozen parent's results.json.

Registered prediction under test (compute plan, 2026-07-13):
"on HRSC (all ships, extreme aspect ratios) GWD should win most decisively" --
i.e. HRSC's pooled (regime='all') GWD-vs-naive-coord ratio should be the LOWEST
(most below 1.0) of all cells: DOTA ~wash (~0.93x, frozen parent), DIOR-R ~0.8x
(frozen parent), HRSC predicted to win MORE decisively than both. A ratio near
1.0 (wash) on HRSC would FALSIFY the elongation mechanism, not just fail to
confirm it further.
"""
import sys
import time
from pathlib import Path


def _portal_commons_root():
    import os
    from pathlib import Path
    for key in ("COMMONS_ROOT", "RELIABILITY_COMMONS"):
        v = os.environ.get(key)
        if v:
            p = Path(v).expanduser().resolve()
            if p.is_dir():
                return p
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        for cand in (parent / "reliability-commons", parent.parent / "reliability-commons"):
            if cand.is_dir():
                return cand
    raise RuntimeError(
        "Set COMMONS_ROOT to the reliability-commons checkout (or place it as a sibling of this repo)."
    )

def _portal_repo_root():
    from pathlib import Path
    here = Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        if (p / ".git").exists() or (p / "pyproject.toml").exists() or (p / "README.md").exists():
            return p
    return here

ROOT = _portal_repo_root()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "coverage_matched_2026-07-13"))
sys.path.insert(0, str(ROOT / "coverage_matched_regime_2026-07-13"))
import json
import numpy as np  # noqa: E402
import coverage_matched_regime_runner as regime  # noqa: E402

CERTA = ROOT / "configA_cert_2026-07-14"
NEW_CELLS = [
    ("orcnn", "hrsc", CERTA / "hrsc_orcnn" / "matched.jsonl"),
    ("rtmdet", "hrsc", CERTA / "hrsc_rtmdet" / "matched.jsonl"),
    ("roi_trans", "dota", CERTA / "zoo_roi_trans" / "matched.jsonl"),
    ("oriented_reppoints", "dota", CERTA / "zoo_oriented_reppoints" / "matched.jsonl"),
]
OUT = ROOT / "configA_regime_extend_2026-07-14"


def main():
    t0 = time.time()
    all_rows = []
    ar_summ = {}
    for det, ds, path in NEW_CELLS:
        rows, arinfo = regime.run_cell(det, ds, str(path), t0)
        all_rows.extend(rows)
        ar_summ[f"{det}/{ds}"] = arinfo
        for base in regime.BASELINES:
            a = next((x for x in rows if x["regime"] == "all" and x["baseline"] == base
                      and x.get("n_splits_used", 0) > 0), None)
            c = next((x for x in rows if x["regime"] == "compact" and x["baseline"] == base
                      and x.get("n_splits_used", 0) > 0), None)
            e = next((x for x in rows if x["regime"] == "elongated" and x["baseline"] == base
                      and x.get("n_splits_used", 0) > 0), None)
            if a:
                line = f"  {det}/{ds} vs {base}: all={a['ratio_matched_mean']:.3f}"
                if c and e:
                    line += (f" compact={c['ratio_matched_mean']:.3f} elongated={e['ratio_matched_mean']:.3f} "
                             f"-> {'ELONG<COMPACT (hyp holds)' if e['ratio_matched_mean'] < c['ratio_matched_mean'] else 'ELONG>=COMPACT (hyp FAILS)'}")
                else:
                    line += " (compact/elongated split not populated -- likely near-uniform AR regime, e.g. HRSC)"
                print(line, flush=True)

    # Load frozen parent's DOTA/DIOR-R "all" ratios (naive-coord) for the cross-dataset comparison table.
    frozen = json.load(open(ROOT / "coverage_matched_2026-07-13" / "results.json"))
    frozen_naive_all = {}
    for c in frozen["cells"]:
        if c.get("regime", "all") == "all" and c["baseline"] == "naive-coord" and "ratio_matched_r20_mean" in c:
            frozen_naive_all.setdefault(c["dataset"], []).append(c["ratio_matched_r20_mean"])

    hrsc_naive_all = [r["ratio_matched_mean"] for r in all_rows
                      if r["dataset"] == "hrsc" and r["baseline"] == "naive-coord"
                      and r["regime"] == "all" and r.get("n_splits_used", 0) > 0]

    verdict = {
        "registered_prediction": "on HRSC (all ships, extreme aspect ratios) GWD should win most "
                                  "decisively -- lower (more below 1.0) ratio than DOTA (~wash) and "
                                  "DIOR-R (~0.8x), not a wash.",
        "dota_frozen_naive_all_mean": (round(float(np.mean(frozen_naive_all.get("dota", []))), 4)
                                        if frozen_naive_all.get("dota") else None),
        "dior_frozen_naive_all_mean": (round(float(np.mean(frozen_naive_all.get("dior", []))), 4)
                                        if frozen_naive_all.get("dior") else None),
        "hrsc_naive_all_per_detector": [round(x, 4) for x in hrsc_naive_all],
        "hrsc_naive_all_mean": round(float(np.mean(hrsc_naive_all)), 4) if hrsc_naive_all else None,
    }
    if hrsc_naive_all:
        hrsc_mean = float(np.mean(hrsc_naive_all))
        dior_mean = verdict["dior_frozen_naive_all_mean"]
        if dior_mean is not None and hrsc_mean < dior_mean - 1e-6:
            verdict["result"] = "CORROBORATES: HRSC ratio is lower than DIOR-R (wins more decisively), as predicted."
        elif hrsc_mean > 1.0 + 1e-6:
            verdict["result"] = ("FALSIFIES (REVERSAL, stronger than a mere wash): HRSC ratio is ABOVE 1.0 -- "
                                  "GWD's matched-coverage region is LARGER than naive-coord's on HRSC, i.e. GWD "
                                  "loses on the exact regime (elongated ships) it was predicted to win most "
                                  "decisively on. Both detectors individually are >=1.0 "
                                  f"({[round(x,4) for x in hrsc_naive_all]}), not just the pooled mean.")
        elif abs(hrsc_mean - 1.0) < 0.05:
            verdict["result"] = "FALSIFIES: HRSC ratio is a wash (~1.0), contradicting the elongation mechanism."
        else:
            verdict["result"] = "MIXED: HRSC ratio does not cleanly dominate DIOR-R -- see numbers, do not overclaim."
    else:
        verdict["result"] = "INCONCLUSIVE: no HRSC split met MIN_CAL/MIN_EVAL for regime='all' naive-coord."

    out = {
        "label": "Config-A extension of the regime-conditional coverage-matched analysis "
                 "(HRSC ORCNN+RTMDet-R, DOTA zoo RoI-Trans+Oriented-RepPoints); "
                 "POST-HOC, coverage-FAIR, NOT preregistered/confirmatory.",
        "new_cells": [f"{d}/{s}" for d, s, _ in NEW_CELLS],
        "excluded_zoo_cells": ["s2anet (checkpoint-load failure, see dota_zoo/S2ANET-EXCLUDED-2026-07-14.md)",
                                "gliding_vertex (checkpoint-load failure, see zoo_gliding_vertex/GLIDINGVERTEX-EXCLUDED-2026-07-14.md)"],
        "aspect_ratio_by_cell": ar_summ,
        "hrsc_falsifiable_prediction_verdict": verdict,
        "rows": all_rows,
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / "results.json", "w"), indent=1, default=str)
    print(f"\nCONFIGA_REGIME_DONE rows={len(all_rows)} elapsed={out['elapsed_s']}s -> {OUT/'results.json'}", flush=True)
    print(f"\nHRSC VERDICT: {verdict['result']}")
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
