"""Full CLI pipeline test via subprocess: ingest -> match -> calibrate -> recall ->
certify -> audit -> report. Synthetic data throughout, no network/GPU/mmrotate."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest


def _run(args: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "rotcert.cli", *args], capture_output=True, text=True
    )


def _gen_dataset(tmp_path: Path, n_scenes: int = 40, dets_per_scene: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    dets, gts = [], []
    for scene_i in range(n_scenes):
        scene_id = f"P{scene_i:04d}"
        n_crop_split = rng.integers(1, 3)
        for crop_i in range(n_crop_split):
            image_id = f"{scene_id}__1024__{crop_i}___0"
            for k in range(dets_per_scene):
                cx, cy = rng.uniform(0, 1000, 2)
                w = rng.uniform(15, 30)
                h = rng.uniform(5, 15)
                theta = rng.uniform(-np.pi / 2, np.pi / 2)
                cls = str(rng.choice(["ship", "harbor", "plane"]))
                gt_obb = [float(cx), float(cy), float(w), float(h), float(theta)]
                gts.append({"image_id": image_id, "scene_id": scene_id, "class": cls, "obb": gt_obb})
                noise = rng.normal(scale=[1.0, 1.0, 1.0, 0.5, 0.05])
                pred_obb = [float(v) for v in (np.array(gt_obb) + noise)]
                pred_obb[2] = max(pred_obb[2], 0.5)
                pred_obb[3] = max(pred_obb[3], 0.5)
                score = float(np.clip(rng.uniform(0.3, 0.99), 0, 1))
                dets.append(
                    {"image_id": image_id, "scene_id": scene_id, "class": cls, "obb": pred_obb, "score": score}
                )
    dets_path = tmp_path / "dets_raw.jsonl"
    gt_path = tmp_path / "gt_raw.jsonl"
    with open(dets_path, "w") as f:
        for d in dets:
            f.write(json.dumps(d) + "\n")
    with open(gt_path, "w") as f:
        for g in gts:
            f.write(json.dumps(g) + "\n")
    return dets_path, gt_path


@pytest.fixture(scope="module")
def pipeline_paths(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("rotcert_e2e")
    dets_raw, gt_raw = _gen_dataset(tmp_path)

    dets_out = tmp_path / "dets.jsonl"
    r = _run(["ingest", "--detector", "jsonl", "--data", str(dets_raw), "-o", str(dets_out)])
    assert r.returncode == 0, r.stderr

    matched_out = tmp_path / "matched.jsonl"
    r = _run(
        ["match", "--dets", str(dets_out), "--gt", str(gt_raw), "--iou-thr", "0.3", "-o", str(matched_out)]
    )
    assert r.returncode == 0, r.stderr

    return {"tmp": tmp_path, "dets": dets_out, "gt": gt_raw, "matched": matched_out}


class TestIngestAndMatch:
    def test_ingest_produces_valid_jsonl(self, pipeline_paths):
        lines = pipeline_paths["dets"].read_text().strip().split("\n")
        assert len(lines) > 0
        rec = json.loads(lines[0])
        assert "scene_id" in rec and rec["scene_id"] is not None

    def test_match_produces_tp_fp_fn(self, pipeline_paths):
        rows = [json.loads(l) for l in pipeline_paths["matched"].read_text().strip().split("\n")]
        types = {r["match_type"] for r in rows}
        assert "tp" in types


class TestCalibrate:
    def test_calibrate_gwd_writes_cert(self, pipeline_paths):
        out = pipeline_paths["tmp"] / "cert_gwd.json"
        r = _run(
            ["calibrate", "--matched", str(pipeline_paths["matched"]), "--score", "gwd", "--alpha", "0.1", "-o", str(out)]
        )
        assert r.returncode == 0, r.stderr
        cert = json.loads(out.read_text())
        assert cert["score_name"] == "gwd"
        assert "None" in cert["strata"] or len(cert["strata"]) >= 1

    def test_calibrate_mondrian_by_class(self, pipeline_paths):
        out = pipeline_paths["tmp"] / "cert_gwd_mondrian.json"
        r = _run(
            [
                "calibrate", "--matched", str(pipeline_paths["matched"]), "--score", "gwd",
                "--alpha", "0.1", "--mondrian", "-o", str(out),
            ]
        )
        assert r.returncode == 0, r.stderr
        cert = json.loads(out.read_text())
        assert cert["mondrian_field"] == "class"

    def test_calibrate_naive_coord_baseline(self, pipeline_paths):
        out = pipeline_paths["tmp"] / "cert_naive.json"
        r = _run(
            ["calibrate", "--matched", str(pipeline_paths["matched"]), "--score", "naive-coord", "--alpha", "0.1", "-o", str(out)]
        )
        assert r.returncode == 0, r.stderr

    def test_calibrate_bad_score_fails(self, pipeline_paths):
        out = pipeline_paths["tmp"] / "cert_bad.json"
        r = _run(["calibrate", "--matched", str(pipeline_paths["matched"]), "--score", "bogus", "--alpha", "0.1", "-o", str(out)])
        assert r.returncode != 0


class TestRecall:
    def test_recall_pooled_runs(self, pipeline_paths):
        out = pipeline_paths["tmp"] / "recall.json"
        r = _run(["recall", "--matched", str(pipeline_paths["matched"]), "--beta", "0.5", "--delta", "0.2", "-o", str(out)])
        assert r.returncode == 0, r.stderr
        res = json.loads(out.read_text())
        assert "certified" in res

    def test_recall_mondrian_runs(self, pipeline_paths):
        out = pipeline_paths["tmp"] / "recall_mondrian.json"
        r = _run(
            ["recall", "--matched", str(pipeline_paths["matched"]), "--beta", "0.5", "--delta", "0.2", "--mondrian", "-o", str(out)]
        )
        assert r.returncode == 0, r.stderr
        res = json.loads(out.read_text())
        assert "pooled_marginal" in res


class TestCertifyApply:
    def test_certify_produces_regions(self, pipeline_paths):
        cert_path = pipeline_paths["tmp"] / "cert_for_apply.json"
        r = _run(
            ["calibrate", "--matched", str(pipeline_paths["matched"]), "--score", "gwd", "--alpha", "0.1", "-o", str(cert_path)]
        )
        assert r.returncode == 0, r.stderr
        regions_out = pipeline_paths["tmp"] / "regions.jsonl"
        r = _run(["certify", "--cert", str(cert_path), "--dets", str(pipeline_paths["dets"]), "-o", str(regions_out)])
        assert r.returncode == 0, r.stderr
        rows = [json.loads(l) for l in regions_out.read_text().strip().split("\n")]
        assert len(rows) > 0
        assert "envelope" in rows[0]

    def test_certify_refuses_for_non_gwd_score(self, pipeline_paths):
        cert_path = pipeline_paths["tmp"] / "cert_naive_for_apply.json"
        r = _run(
            ["calibrate", "--matched", str(pipeline_paths["matched"]), "--score", "naive-coord", "--alpha", "0.1", "-o", str(cert_path)]
        )
        assert r.returncode == 0, r.stderr
        regions_out = pipeline_paths["tmp"] / "regions_bad.jsonl"
        r = _run(["certify", "--cert", str(cert_path), "--dets", str(pipeline_paths["dets"]), "-o", str(regions_out)])
        assert r.returncode != 0


class TestAudit:
    def test_audit_v1_v2_tables(self, pipeline_paths):
        out = pipeline_paths["tmp"] / "audit.json"
        r = _run(["audit", "--matched", str(pipeline_paths["matched"]), "--score", "gwd", "--alpha", "0.1", "-o", str(out)])
        assert r.returncode == 0, r.stderr
        res = json.loads(out.read_text())
        assert "v1_marginal_coverage" in res
        assert "v2_stratum_coverage" in res

    def test_audit_holm_cells(self, pipeline_paths):
        cells_path = pipeline_paths["tmp"] / "holm_cells.json"
        cells = [{"p_value": p, "ratio": 0.7, "detector": "d1", "dataset": "ds1", "baseline": "B1"} for p in [0.001] * 8]
        cells_path.write_text(json.dumps(cells))
        out = pipeline_paths["tmp"] / "holm.json"
        r = _run(["audit", "--holm-cells", str(cells_path), "-o", str(out)])
        assert r.returncode == 0, r.stderr
        res = json.loads(out.read_text())
        assert res["family_size"] == 8


class TestReport:
    def test_report_markdown(self, pipeline_paths):
        cert_path = pipeline_paths["tmp"] / "cert_for_report.json"
        _run(["calibrate", "--matched", str(pipeline_paths["matched"]), "--score", "gwd", "--alpha", "0.1", "-o", str(cert_path)])
        out = pipeline_paths["tmp"] / "report.md"
        r = _run(["report", "--cert", str(cert_path), "-o", str(out)])
        assert r.returncode == 0, r.stderr
        assert out.exists()
        assert "rotcert report" in out.read_text()

    def test_report_needs_input(self, pipeline_paths):
        out = pipeline_paths["tmp"] / "report_empty.md"
        r = _run(["report", "-o", str(out)])
        assert r.returncode != 0


def test_cli_help_runs():
    r = _run(["--help"])
    assert r.returncode == 0
    assert "rotcert" in r.stdout.lower() or "usage" in r.stdout.lower()
