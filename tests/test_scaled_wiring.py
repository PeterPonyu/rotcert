"""Wiring tests for the EXPERIMENTAL B1-scaled score through g1_calibrate /
g1_coverage / the CLI calibrate command (task #21; the construction itself is
property-tested in test_scores.py). The scaled score is an explicit OPT-IN:
these tests also pin that nothing preregistered changed."""

from __future__ import annotations

import json

import numpy as np
import pytest

from rotcert.certify import CertifyError, g1_calibrate, g1_coverage
from rotcert.cli import main
from rotcert.scores import EXPERIMENTAL_SCORES, SCORES


def _matched_scored(gen_matched_pairs, rng, n, cls="ship", score=None):
    preds, gts = gen_matched_pairs(rng, n)
    scores = rng.uniform(0.3, 0.99, size=n) if score is None else np.full(n, score)
    return [
        {"pred_obb": p.tolist(), "gt_obb": g.tolist(), "class": cls, "pred_score": float(s)}
        for p, g, s in zip(preds, gts, scores)
    ]


def test_registries_unchanged():
    # the preregistered roster is exactly the six constructions; the scaled
    # variant lives ONLY in the opt-in registry
    assert set(SCORES) == {"gwd", "iou", "naive-coord", "hull", "wrapped-coord", "doubled"}
    assert set(EXPERIMENTAL_SCORES) == {"naive-coord-scaled"}


class TestG1CalibrateScaled:
    def test_calibrate_and_coverage_near_target(self, gen_matched_pairs, rng):
        cal = _matched_scored(gen_matched_pairs, rng, 800)
        ev = _matched_scored(gen_matched_pairs, rng, 2000)
        cert = g1_calibrate(cal, "naive-coord-scaled", alpha=0.1)
        assert cert["refused"] == []
        cell = cert["strata"][None]
        assert cell["set_size_cxcy"] is None
        assert cell["set_size_cxcy_median_cal"] > 0.0
        cov = g1_coverage(cert, ev)
        # Bonferroni over 5 coords is conservative: coverage >= target
        assert cov["overall_coverage"] >= 0.9 - 0.03
        assert cov["n"] == 2000

    def test_missing_pred_score_raises(self, gen_matched_pairs, rng):
        cal = _matched_scored(gen_matched_pairs, rng, 50)
        cal[7]["pred_score"] = None
        with pytest.raises(CertifyError, match="pred_score"):
            g1_calibrate(cal, "naive-coord-scaled", alpha=0.1)

    def test_missing_pred_score_on_eval_raises(self, gen_matched_pairs, rng):
        cal = _matched_scored(gen_matched_pairs, rng, 100)
        ev = _matched_scored(gen_matched_pairs, rng, 30)
        del ev[3]["pred_score"]
        cert = g1_calibrate(cal, "naive-coord-scaled", alpha=0.1)
        with pytest.raises(CertifyError, match="pred_score"):
            g1_coverage(cert, ev)

    def test_constant_score_reduces_to_unscaled_decisions(self, gen_matched_pairs, rng):
        # constant confidence across detections -> sigma is a common factor and
        # the scaled per-detection decisions match the unscaled ones exactly
        cal = _matched_scored(gen_matched_pairs, rng, 300, score=0.7)
        ev = _matched_scored(gen_matched_pairs, rng, 500, score=0.7)
        cert_scaled = g1_calibrate(cal, "naive-coord-scaled", alpha=0.1)
        cert_plain = g1_calibrate(cal, "naive-coord", alpha=0.1)
        cov_scaled = g1_coverage(cert_scaled, ev)
        cov_plain = g1_coverage(cert_plain, ev)
        assert cov_scaled["overall_coverage"] == pytest.approx(cov_plain["overall_coverage"])

    def test_mondrian_strata_and_refusal_floor(self, gen_matched_pairs, rng):
        big = _matched_scored(gen_matched_pairs, rng, 200, cls="ship")
        tiny = _matched_scored(gen_matched_pairs, rng, 4, cls="plane")  # 1/(4+1) > 0.1
        cert = g1_calibrate(big + tiny, "naive-coord-scaled", alpha=0.1, mondrian_field="class")
        assert "ship" in cert["strata"]
        assert [r["stratum"] for r in cert["refused"]] == ["plane"]


def test_cli_calibrate_scaled_e2e(tmp_path, gen_matched_pairs, rng):
    rows = []
    for m in _matched_scored(gen_matched_pairs, rng, 120):
        rows.append({**m, "match_type": "tp", "scene_id": "s0", "image_id": "i0"})
    matched_path = tmp_path / "matched.jsonl"
    matched_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "cert.json"
    rc = main(["calibrate", "--matched", str(matched_path), "--score", "naive-coord-scaled",
               "--alpha", "0.1", "--out", str(out)])
    assert rc == 0
    cert = json.loads(out.read_text())
    assert cert["score_name"] == "naive-coord-scaled"
    assert cert["refused"] == []
