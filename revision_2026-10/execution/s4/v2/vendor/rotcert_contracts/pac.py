"""Explicit fixed-lambda S5 contracts. Provenance claims are auditable assumptions.

These functions check role disjointness, fixed class dictionary, declared iid law,
operating point and finite family budgets. They cannot infer independence or
absence of training leakage from source-ID strings; every output says so.
"""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from typing import Mapping

import numpy as np

from .exact import integer, probability, rational, score_groups
from .bounds import empirical_bernstein, tolerance_rank

FIXED_LAMBDA = Fraction(1, 20)
FIXED_Q_GRID = (0., 1., 2., 4., 8., 16., 32., 64., 128., 256., 512., math.inf)
FAMILY_DELTA = Fraction(1, 40)
DISCLAIMER = "valid only under declared iid/design/training provenance; source IDs alone do not prove independence"


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _ids(ids, name):
    ids = tuple(ids)
    if any(not isinstance(x, str) or not x for x in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{name} must contain distinct nonempty source IDs")
    return ids


@dataclass(frozen=True)
class Provenance:
    protocol_sha256: str
    source_population: str
    sampling_unit: str
    detector_sha256: str
    matching_sha256: str
    preprocessing_sha256: str
    source_roles_sha256: str
    class_dictionary: tuple[str, ...]
    calibration_ids: tuple[str, ...]
    training_ids: tuple[str, ...]
    design_ids: tuple[str, ...]
    operating_lambda: Fraction
    candidate_origin: str = "fixed_before_validation"
    declared_iid_sources: bool = True
    declared_training_model_selection_independent: bool = True
    representative_seed: int = 20260928

    def validate(self, observed_ids):
        for name in ("protocol_sha256", "detector_sha256", "matching_sha256", "preprocessing_sha256", "source_roles_sha256"):
            value = getattr(self, name)
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"{name} must be a lowercase SHA256")
        if not self.source_population or not self.sampling_unit:
            raise ValueError("population and source sampling unit are required")
        classes = _ids(self.class_dictionary, "class_dictionary")
        if not classes:
            raise ValueError("fixed class dictionary cannot be empty")
        calibration = set(_ids(self.calibration_ids, "calibration_ids"))
        training = set(_ids(self.training_ids, "training_ids"))
        design = set(_ids(self.design_ids, "design_ids"))
        if calibration & (training | design):
            raise ValueError("validation sources overlap training/design sources")
        if set(_ids(observed_ids, "observed_ids")) != calibration:
            raise ValueError("records must cover the complete declared calibration role, including empty sources")
        if rational(self.operating_lambda) != FIXED_LAMBDA:
            raise ValueError("S5 is frozen at exact lambda=1/20; no post-selection reuse")
        if self.candidate_origin != "fixed_before_validation":
            raise ValueError("this S5 implementation only accepts the frozen exogenous candidates")
        if self.declared_iid_sources is not True or self.declared_training_model_selection_independent is not True:
            raise ValueError("iid and training/model-selection independence must be declared")
        integer(self.representative_seed, "representative_seed")

    def common_record(self):
        return {k: getattr(self, k) for k in ("protocol_sha256", "source_population", "sampling_unit",
                "detector_sha256", "matching_sha256", "preprocessing_sha256", "source_roles_sha256")} | {
                "class_dictionary": list(self.class_dictionary), "operating_lambda_exact": str(FIXED_LAMBDA),
                "declarations": DISCLAIMER}


def _base(p, class_id, route, target, data_hash):
    if class_id not in p.class_dictionary:
        raise ValueError("class is outside the frozen dictionary")
    eta = FAMILY_DELTA / len(p.class_dictionary)
    return {"route": route, "class_id": class_id, "target": target,
            "probability_kind": "PAC_report_and_bad", "family_delta_exact": str(FAMILY_DELTA),
            "class_delta_exact": str(eta), "family_size": len(p.class_dictionary),
            "common_provenance": p.common_record(), "calibration_ids": sorted(p.calibration_ids),
            "data_sha256": data_hash, "assumptions": DISCLAIMER}, eta


def _finish(record):
    return record | {"certificate_sha256": _digest(record)}


def _score_records(records: Mapping[str, object], p: Provenance):
    p.validate(records)
    result = {key: score_groups([records[key]])[0] for key in sorted(records)}
    encoded = {key: values.tolist() for key, values in result.items()}
    return result, _digest(encoded)


def representative_pac_g1(records, class_id, provenance: Provenance, alpha="0.1"):
    """One exactly uniform PRNG integer per TP-positive source, sorted ID order.

    Record indices and complete PCG64 state; the frozen seed is not evidence of
    independence. The theorem's auxiliary randomization assumption is explicit.
    No representative is drawn for an empty scene and empty scenes stay recorded.
    """
    alpha = probability(alpha, "alpha")
    if alpha != Fraction(1, 10):
        raise ValueError("S5 alpha is frozen at 1/10")
    groups, data_hash = _score_records(records, provenance)
    record, eta = _base(provenance, class_id, "representative_tolerance", "scene_uniform_TP_G1", data_hash)
    class_index = provenance.class_dictionary.index(class_id)
    seed = [provenance.representative_seed, 5, class_index]
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
    initial_state = rng.bit_generator.state
    selected = []
    scores = []
    for source, values in groups.items():
        if len(values):
            index = int(rng.integers(len(values)))
            scores.append(float(values[index]))
            selected.append({"source_id": source, "m": len(values), "index": index, "score": float(values[index])})
    rank = tolerance_rank(len(scores), alpha, eta)
    q = None if rank.rank is None else float(np.partition(scores, rank.rank-1)[rank.rank-1])
    record.update({"reported": q is not None, "status": "finite_certificate" if q is not None else "no_finite_certificate",
                   "q": q, "mathematical_fallback": "infinity", "risk_limit_exact": str(alpha),
                   "n_source": len(groups), "n_TP_positive": len(scores), "empty_source_count": len(groups)-len(scores),
                   "rank": rank.record(), "representatives": selected,
                   "rng": {"seed_sequence": seed, "bit_generator": "PCG64", "initial_state": initial_state,
                           "final_state": rng.bit_generator.state,
                           "assumption": "frozen auxiliary seed was chosen independently of scene scores"}})
    return _finish(record)


def fixed_sequence_g1(records, class_id, provenance: Provenance, alpha="0.1", q_grid=FIXED_Q_GRID):
    alpha = probability(alpha, "alpha")
    if alpha != Fraction(1, 10) or tuple(q_grid) != FIXED_Q_GRID:
        raise ValueError("S5 alpha and q grid must equal the frozen design")
    groups, data_hash = _score_records(records, provenance)
    record, eta = _base(provenance, class_id, "EB_fixed_sequence", "scene_uniform_TP_G1", data_hash)
    nonempty = [g for g in groups.values() if len(g)]
    trace, chosen, chosen_q = [], None, None
    # The entire descending order is frozen before observing validation losses.
    # Stop on the first failure; do not skip it and inspect later candidates.
    for q in reversed(q_grid):
        losses = [Fraction(int(np.count_nonzero(g > q)), len(g)) for g in nonempty]
        bound = empirical_bernstein(losses, eta)
        passed = bound.passes(alpha)
        entry = {"q": "infinity" if math.isinf(q) else q, "bound": bound.record(), "passed": passed}
        trace.append(entry)
        if not passed:
            break
        chosen, chosen_q = entry, entry["q"]
    reported = chosen is not None
    status = ("no_certificate" if not reported else
              "vacuous_all_space_certificate" if chosen_q == "infinity" else "finite_certificate")
    record.update({"reported": reported, "status": status, "q": chosen_q,
                   "risk_limit_exact": str(alpha), "trace": trace,
                   "n_source": len(groups), "n_TP_positive": len(nonempty),
                   "empty_source_count": len(groups)-len(nonempty),
                   "selection": "largest-to-smallest; stop at first failure; no per-q Bonferroni",
                   "route_selection": "reported separately; no data-adaptive choice across PAC routes"})
    return _finish(record)


def fixed_lambda_g2(records, class_id, provenance: Provenance, beta="0.2"):
    """records map source ID -> (H, m) at the SAME frozen matching/lambda.

    Include all GT-positive sources, including m=0. H=0 sources are recorded,
    excluded from this conditional target, and never assigned zero miss loss.
    """
    beta = probability(beta, "beta")
    if beta not in (Fraction(1, 10), Fraction(1, 5)):
        raise ValueError("S5 beta must be the declared 1/10 or 1/5")
    provenance.validate(records)
    losses, serialized = [], {}
    for source in sorted(records):
        H, m = records[source]
        H, m = integer(H, "H"), integer(m, "m")
        if m > H:
            raise ValueError("TP count cannot exceed GT count")
        serialized[source] = [H, m]
        if H:
            losses.append(Fraction(H-m, H))
    record, eta = _base(provenance, class_id, "EB_fixed_lambda_G2", "GT_positive_scene_FNR_G2", _digest(serialized))
    bound = empirical_bernstein(losses, eta)
    passed = bound.passes(beta)
    record.update({"reported": passed, "status": "finite_certificate" if passed else "no_certificate",
                   "risk_limit_exact": str(beta), "bound": bound.record(), "n_source": len(records),
                   "n_GT_positive": len(losses), "empty_GT_source_count": len(records)-len(losses)})
    return _finish(record)


def compose_pac(g1, g2):
    """Conservative sufficient integration route: disjoint calibration samples.

    Other shared-validation constructions may be valid but are deliberately not
    accepted without a separate simultaneous-selection proof and implementation.
    """
    for item in (g1, g2):
        content = {k: v for k, v in item.items() if k != "certificate_sha256"}
        if _digest(content) != item.get("certificate_sha256"):
            raise ValueError("component certificate content hash mismatch")
        if item.get("probability_kind") != "PAC_report_and_bad":
            raise ValueError("marginal G1 cannot be promoted to PAC composition")
    if g1["target"] != "scene_uniform_TP_G1" or g2["target"] != "GT_positive_scene_FNR_G2":
        raise ValueError("composition targets/sampling denominators do not match")
    if g1["route"] not in ("representative_tolerance", "EB_fixed_sequence") or g2["route"] != "EB_fixed_lambda_G2":
        raise ValueError("unsupported component route")
    if g1["common_provenance"] != g2["common_provenance"] or g1["class_id"] != g2["class_id"]:
        raise ValueError("population/class/detector/matching/lambda/protocol provenance mismatch")
    if set(g1["calibration_ids"]) & set(g2["calibration_ids"]):
        raise ValueError("this integration route requires disjoint G1/G2 calibration samples")
    alpha, beta = Fraction(g1["risk_limit_exact"]), Fraction(g2["risk_limit_exact"])
    delta = Fraction(g1["class_delta_exact"]) + Fraction(g2["class_delta_exact"])
    family_delta = Fraction(g1["family_delta_exact"]) + Fraction(g2["family_delta_exact"])
    reported = g1["reported"] and g2["reported"]
    result = {"reported": reported, "status": "modular_PAC_certificate" if reported else "no_joint_certificate",
              "class_id": g1["class_id"], "E2E_risk_limit_exact": str(min(Fraction(1), alpha+beta)),
              "class_failure_budget_exact": str(delta), "class_family_failure_budget_exact": str(family_delta),
              "quantifier": "P(report_G1 AND report_G2 AND r_L > min(1,alpha+beta)) <= delta1+delta2",
              "not_claimed": ["conditional-on-report confidence", "beta+(1-beta)*alpha", "simultaneous across PAC routes"],
              "components": [g1["certificate_sha256"], g2["certificate_sha256"]],
              "G1_useful_finite": g1["status"] == "finite_certificate", "assumptions": DISCLAIMER}
    return _finish(result)


def fixed_sequence_e2e(records, class_id, provenance: Provenance, eps="0.3", q_grid=FIXED_Q_GRID):
    """Independent S5 companion family, delta=.05 across the class dictionary.

    records map each source to (H, matched_scores); unmatched GT contribute one
    failure each for every q, including infinity. Full confidence filtering and
    class-exact matching must already use the provenance's fixed lambda. No
    composition or best-route selector inherits this separate family's budget.
    """
    epsilon = probability(eps, "eps")
    if epsilon not in (Fraction(3,10), Fraction(1,5)) or tuple(q_grid) != FIXED_Q_GRID:
        raise ValueError("E2E uses frozen eps in {3/10,1/5} and the fixed S5 q grid")
    provenance.validate(records)
    if class_id not in provenance.class_dictionary:
        raise ValueError("class is outside the frozen dictionary")
    sources, serialized = [], {}
    for source in sorted(records):
        H, scores = records[source]
        H = integer(H, "H")
        scores = score_groups([scores])[0]
        if len(scores) > H:
            raise ValueError("matched TP count exceeds GT count")
        serialized[source] = [H, scores.tolist()]
        if H:
            sources.append((H,scores))
    eta = Fraction(1,20) / len(provenance.class_dictionary)
    trace, chosen = [], None
    for q in reversed(q_grid):
        losses = [Fraction(H-int(np.count_nonzero(scores <= q)),H) for H,scores in sources]
        bound = empirical_bernstein(losses, eta)
        passed = bound.passes(epsilon)
        entry = {"q":"infinity" if math.isinf(q) else q, "bound":bound.record(), "passed":passed}
        trace.append(entry)
        if not passed:
            break
        chosen = entry["q"]
    status = ("no_certificate" if chosen is None else "all_space_localization_with_recall_certificate"
              if chosen == "infinity" else "finite_certificate")
    return _finish({"route":"direct_E2E_fixed_sequence", "class_id":class_id,
                    "target":"GT_positive_scene_E2E", "probability_kind":"PAC_report_and_bad",
                    "family_delta_exact":"1/20", "class_delta_exact":str(eta),
                    "family_size":len(provenance.class_dictionary),
                    "common_provenance":provenance.common_record(),
                    "calibration_ids":sorted(provenance.calibration_ids),
                    "data_sha256":_digest(serialized), "reported":chosen is not None,
                    "status":status, "q":chosen, "risk_limit_exact":str(epsilon),
                    "n_source":len(records), "n_GT_positive":len(sources),
                    "empty_GT_source_count":len(records)-len(sources), "trace":trace,
                    "quantifier":"P(report AND population_E2E_risk>eps) <= delta_E/class_count",
                    "family_separation":"independent .05 family; no .95 simultaneous statement across routes",
                    "assumptions":DISCLAIMER})
