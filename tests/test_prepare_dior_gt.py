"""Smoke tests for orchestration/prepare_dior_gt.py.

No network, no GPU, no mmrotate (conftest convention). Synthetic DIOR-R xmls are
built in-process; the 8-point/polygon path needs cv2 and is skipped where absent.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

# prepare_dior_gt lives in orchestration/ (not an installed package); load it by path.
_ORCH = Path(__file__).resolve().parents[1] / "orchestration" / "prepare_dior_gt.py"
_spec = importlib.util.spec_from_file_location("prepare_dior_gt", _ORCH)
pdg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pdg)


def _write_xml(path: Path, objects_xml: str) -> None:
    path.write_text(
        f"<annotation><folder>DIOR</folder>{objects_xml}</annotation>",
        encoding="utf-8",
    )


def _robndbox5_obj(name="ship", cx=100, cy=120, w=40, h=12, angle=0.3, difficult=0):
    return (
        f"<object><name>{name}</name><difficult>{difficult}</difficult>"
        f"<robndbox><cx>{cx}</cx><cy>{cy}</cy><w>{w}</w><h>{h}</h>"
        f"<angle>{angle}</angle></robndbox></object>"
    )


def _poly8_obj(name="harbor", difficult=0):
    # An axis-aligned 40x12 rectangle as 4 corners (robndbox corner-name form).
    return (
        f"<object><name>{name}</name><difficult>{difficult}</difficult><robndbox>"
        f"<x_left_top>80</x_left_top><y_left_top>60</y_left_top>"
        f"<x_right_top>120</x_right_top><y_right_top>60</y_right_top>"
        f"<x_right_bottom>120</x_right_bottom><y_right_bottom>72</y_right_bottom>"
        f"<x_left_bottom>80</x_left_bottom><y_left_bottom>72</y_left_bottom>"
        f"</robndbox></object>"
    )


def _bndbox_only_obj(name="airplane"):
    # Axis-aligned HBB only -- the WRONG (unknown-for-OBB) schema.
    return (
        f"<object><name>{name}</name><difficult>0</difficult>"
        f"<bndbox><xmin>1</xmin><ymin>2</ymin><xmax>3</xmax><ymax>4</ymax></bndbox></object>"
    )


def test_parse_robndbox5(tmp_path):
    xml = tmp_path / "00001.xml"
    _write_xml(xml, _robndbox5_obj())
    rows = pdg.parse_dior_xml(xml)
    assert len(rows) == 1
    assert rows[0]["kind"] == "robndbox5"
    assert rows[0]["class"] == "ship"


def test_robndbox5_le90_long_edge():
    # h > w on input must be swapped to long-edge (w >= h) by canonicalize_le90.
    obb = pdg.robndbox5_to_le90_obb(cx=10, cy=20, w=12, h=40, angle=0.0)
    assert len(obb) == 5
    cx, cy, w, h, theta = obb
    assert (cx, cy) == (10, 20)
    assert w >= h
    assert -3.15 / 2 <= theta < 3.15 / 2


def test_unknown_schema_refuses_loudly(tmp_path):
    xml = tmp_path / "00002.xml"
    _write_xml(xml, _bndbox_only_obj())
    with pytest.raises(pdg.UnknownDiorSchema):
        pdg.parse_dior_xml(xml)


def test_difficult_policy_counts(tmp_path):
    xml = tmp_path / "00003.xml"
    _write_xml(xml, _robndbox5_obj(difficult=0) + _robndbox5_obj(name="bridge", difficult=1))
    out = tmp_path / "gt_drop.jsonl"
    rc = pdg.main(["--annfiles-dir", str(tmp_path), "--difficult-policy", "drop", "-o", str(out)])
    assert rc == 0
    prov = json.loads(Path(str(out) + ".provenance.json").read_text())
    assert prov["n_difficult_seen"] == 1
    assert prov["n_difficult_dropped"] == 1
    assert prov["n_gt_records"] == 1  # the difficult=1 instance dropped


def test_main_smoke_robndbox5(tmp_path):
    _write_xml(tmp_path / "00010.xml", _robndbox5_obj(name="ship"))
    _write_xml(tmp_path / "00011.xml", _robndbox5_obj(name="harbor", w=30, h=8, angle=-0.9))
    out = tmp_path / "dior_test_gt.jsonl"
    rc = pdg.main(["--annfiles-dir", str(tmp_path), "-o", str(out)])
    assert rc == 0
    recs = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(recs) == 2
    for r in recs:
        assert set(r) >= {"image_id", "class", "obb", "gt_id"}
        assert len(r["obb"]) == 5
        assert r["obb"][2] >= r["obb"][3]  # w >= h (le90 long-edge)
    prov = json.loads(Path(str(out) + ".provenance.json").read_text())
    assert prov["n_gt_records"] == 2
    assert prov["n_degenerate_skipped"] == 0
    assert prov["n_robndbox5"] == 2


def test_imageset_filter(tmp_path):
    _write_xml(tmp_path / "keep_me.xml", _robndbox5_obj())
    _write_xml(tmp_path / "drop_me.xml", _robndbox5_obj(name="bridge"))
    iset = tmp_path / "test.txt"
    iset.write_text("keep_me\n", encoding="utf-8")
    out = tmp_path / "gt.jsonl"
    rc = pdg.main(["--annfiles-dir", str(tmp_path), "--imageset-file", str(iset), "-o", str(out)])
    assert rc == 0
    recs = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(recs) == 1
    assert recs[0]["image_id"] == "keep_me"


def test_poly8_path(tmp_path):
    pytest.importorskip("cv2")
    xml = tmp_path / "00020.xml"
    _write_xml(xml, _poly8_obj(name="harbor"))
    rows = pdg.parse_dior_xml(xml)
    assert rows[0]["kind"] == "poly8"
    out = tmp_path / "gt_poly.jsonl"
    rc = pdg.main(["--annfiles-dir", str(tmp_path), "-o", str(out)])
    assert rc == 0
    rec = json.loads(out.read_text().splitlines()[0])
    assert rec["class"] == "harbor"
    assert rec["obb"][2] >= rec["obb"][3]  # 40x12 rect -> w >= h
