#!/usr/bin/env python3
"""v3 (2026-10-04, after the second independent review): identical computation and seed to v2; the receipt is
strict JSON (non-finite numbers written as the strings "inf", "-inf" or "nan"). v2 docstring follows.

Second semi-synthetic simulation of the G2 recall certificate (extends g2_simulation.py; its per-class validity rows
stay in G2-SIMULATION.json and are summarized here, not recomputed).

Adds what the first run did not establish:
  1. Joint family validity. Whole G2-calibration scenes are resampled with their full class x threshold miss-rate vectors,
     so class co-occurrence and random per-class support are kept. Two populations per cell: the empirical one (mixed
     nulls and alternatives) and an all-null boundary (one all-miss pseudo-scene per class, weighted so that each class's
     risk at the fixed threshold equals beta). A family error is any certificate issued for a (class, threshold) whose
     population risk is at least beta. Rules: the paper's EB rule at one fixed threshold (eta = delta/K) and on the five-
     threshold grid (eta = delta/(5K)); the one-sided Hoeffding rule; and the union rule that issues if EB or Hoeffding,
     each at eta/2, passes (valid by the union bound).
  2. Planning without a size cap. The fixed-moment plug-in size n* is computed up to 1e9 sources, so every class with an
     observed mean below beta has a denominator at every multiplier; no-plan reasons are recorded.
  3. Pilot -> plan -> validation. A pilot of the current size gives a plug-in size; a fresh sample of that size (and of 1.5
     times it) is drawn and the certificate evaluated, so the realized certification probability of following the plan is
     measured, with no-plan pilots counted as failures. Pilot/population size ratios are pooled over classes with no-plan
     as +inf, and the hardest classes are listed.
  4. A prespecified-rule comparison of EB, Hoeffding and the union rule on the frozen calibration moments.

Populations are the frozen per-scene miss rates of each cell's G2 calibration role. Read-only on frozen trees; writes only
G2-SIMULATION-v2.json below var/reviewer-gaps-20261004/g2-sim. (2026-10-04.)"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import sys
sys.dont_write_bytecode = True
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist

R = Path(os.environ.get("ROTCERT_ROOT", Path(__file__).resolve().parents[2]))
E = R / "revision_2026-10/execution"
OUT = R / "var/reviewer-gaps-20261004/g2-sim"
V1 = OUT / "G2-SIMULATION.json"
DELTA = 1 / 40
GRID = (0.05, 0.10, 0.20, 0.30, 0.50)
BETAS = (0.2, 0.1)
SEED = 20261005
M_JOINT, M_POWER, M_PILOT, CHUNK = 20000, 4000, 2000, 1000
PLAN_CAP = 10 ** 9
MULTS = (0.5, 1, 1.5, 2, 3, 4)
RULES = ("EB", "Hoeffding", "union")


def _strict(x):
    """Recursively replace non-finite floats by strings so the receipt is strict JSON."""
    if isinstance(x, dict):
        return {k: _strict(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_strict(v) for v in x]
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return x if math.isfinite(x) else ("nan" if math.isnan(x) else ("inf" if x > 0 else "-inf"))
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def eb(mean, var, n, eta):
    n = np.asarray(n, dtype=float)
    ell = math.log(2 / eta)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = mean + np.sqrt(2 * np.maximum(var, 0.0) * ell / n) + 7 * ell / (3 * (n - 1))
    return np.where(n >= 2, np.minimum(1.0, u), 1.0)


def hoeffding(mean, n, eta):
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = mean + np.sqrt(math.log(1 / eta) / (2 * n))
    return np.where(n >= 2, np.minimum(1.0, u), 1.0)


def passes(rule, mean, var, n, eta, beta):
    if rule == "EB":
        return eb(mean, var, n, eta) < beta
    if rule == "Hoeffding":
        return hoeffding(mean, n, eta) < beta
    return (eb(mean, var, n, eta / 2) < beta) | (hoeffding(mean, n, eta / 2) < beta)


def plugin_n(rule, mean, var, eta, beta, cap=PLAN_CAP):
    """Smallest n >= 2 at which the rule passes with these moments held fixed; NaN if none up to cap."""
    mean, var = np.atleast_1d(mean).astype(float), np.atleast_1d(var).astype(float)
    ok = lambda n: passes(rule, mean, var, n, eta, beta)
    out = np.full(mean.shape, np.nan)
    feasible = (mean < beta) & ok(np.full(mean.shape, float(cap)))
    lo, hi = np.full(mean.shape, 1.0), np.full(mean.shape, float(cap))
    for _ in range(80):
        mid = np.floor((lo + hi) / 2)
        good = ok(np.maximum(mid, 2.0))
        hi = np.where(good & feasible, mid, hi)
        lo = np.where(good & feasible, lo, mid)
        if np.all(hi - lo <= 1):
            break
    out[feasible] = np.maximum(hi[feasible], 2.0)
    return out


def cp_upper(k, m, level=0.95):
    return 1.0 if k >= m else float(beta_dist.ppf(level, k + 1, m - k))


def load(path):
    p = json.loads(path.read_text())
    inp = p["input"]
    for key in ("cache", "roles"):
        assert sha(inp[key]) == inp[key + "_sha256"], (p["cell"], key)
    roles = json.loads(Path(inp["roles"]).read_text())
    with np.load(inp["cache"], allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    names = list(map(str, data["scene_names"]))
    lookup = {s: i for i, s in enumerate(names)}
    idx = np.array([lookup[s] for s in roles["G2_cal"]], dtype=int)
    classes = list(map(str, data["class_names"]))
    assert classes == [x["class"] for x in p["classes"]]
    S, K, G = len(idx), len(classes), len(GRID)
    has = np.zeros((S, K), dtype=bool)
    loss = np.zeros((S, K, G))
    frozen = {}
    for k in range(K):
        sel = data["tp_cls"] == k
        fn = np.bincount(data["fn_scene"][data["fn_cls"] == k], minlength=len(names))
        gt = (np.bincount(data["tp_scene"][sel], minlength=len(names)) + fn)[idx]
        has[:, k] = gt > 0
        for j, lam in enumerate(GRID):
            tp = np.bincount(data["tp_scene"][sel & (data["tp_conf"] >= lam)], minlength=len(names))[idx]
            loss[:, k, j] = np.where(gt > 0, 1.0 - tp / np.maximum(gt, 1), 0.0)
        assert np.all(np.diff(loss[has[:, k], k, :], axis=1) >= -1e-15)     # miss rate is nondecreasing in the threshold
        for b in BETAS:
            g2 = p["classes"][k]["G2"][str(b)]
            bound = g2["bound"]
            x = loss[has[:, k], k, 0]
            assert len(x) == bound["n"]
            if len(x) >= 2:
                assert abs(x.mean() - bound["mean"]) < 1e-12 and abs(x.var(ddof=1) - bound["variance_ddof1"]) < 1e-12
            frozen[(k, b)] = dict(reported=bool(g2["reported"]), n=int(bound["n"]),
                                  mean=None if len(x) < 1 else float(x.mean()),
                                  var=None if len(x) < 2 else float(x.var(ddof=1)))
    ins = {inp["cache"]: inp["cache_sha256"], inp["roles"]: inp["roles_sha256"], str(path): sha(path)}
    return p["cell"], classes, has, loss, frozen, ins


def joint_population(has, loss, beta, boundary):
    """Scene population (rows) with weights; boundary adds one all-miss pseudo-scene per class below beta."""
    S, K, G = loss.shape
    w = np.full(S, 1.0 / S)
    H, L = has.astype(float), loss.copy()
    if boundary:
        A = (w[:, None] * H * loss[:, :, 0]).sum(0)
        B = (w[:, None] * H).sum(0)
        atoms = np.where(B > 0, np.maximum(beta * B - A, 0.0) / (1 - beta), 0.0)
        add = np.flatnonzero(atoms > 0)
        H = np.vstack([H, np.eye(K)[add]])
        extra = np.zeros((len(add), K, G))
        extra[np.arange(len(add)), add, :] = 1.0
        L = np.concatenate([L, extra])
        w = np.append(w, atoms[add])
    w = w / w.sum()
    Bk = w @ H
    risk = np.einsum("s,sk,skg->kg", w, H, L) / np.where(Bk > 0, Bk, np.nan)[:, None]
    return H, L, w, risk


def joint_validity(rng, has, loss, K, beta, boundary, mult):
    H, L, w, risk = joint_population(has, loss, beta, boundary)
    if boundary:
        assert np.all(np.isnan(risk) | (risk >= beta - 1e-12))
    # Boundary: every pair is null by construction (risk == beta up to rounding at the fixed threshold). Empirical: a pair
    # is null when its population risk is at least beta; an absent class (NaN) is never certifiable either way.
    null = np.ones_like(risk, dtype=bool) if boundary else ~(risk < beta)
    n_total = int(mult * has.shape[0])
    G = len(GRID)
    Lf = (H[:, :, None] * L).reshape(len(w), K * G)
    L2 = Lf ** 2
    events = {f"{r}-{s}": 0 for r in RULES for s in ("single", "grid")}
    true_certs = {f"{r}-{s}": 0 for r in RULES for s in ("single", "grid")}
    done = 0
    while done < M_JOINT:
        m = min(CHUNK, M_JOINT - done)
        counts = rng.multinomial(n_total, w, size=m).astype(float)
        n = counts @ H                                       # m x K
        s1 = (counts @ Lf).reshape(m, K, G)
        s2 = (counts @ L2).reshape(m, K, G)
        nn = n[:, :, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = np.where(nn > 0, s1 / nn, 0.0)
            var = np.where(nn > 1, (s2 - nn * mean ** 2) / (nn - 1), 0.0)
        for rule in RULES:
            for scope, eta, cols in (("single", DELTA / K, slice(0, 1)), ("grid", DELTA / (K * G), slice(0, G))):
                ok = passes(rule, mean[:, :, cols], var[:, :, cols], nn, eta, beta)
                key = f"{rule}-{scope}"
                events[key] += int(np.any(ok & null[None, :, cols], axis=(1, 2)).sum())
                true_certs[key] += int((ok & ~null[None, :, cols]).sum())
        done += m
    return dict(boundary=boundary, mult=mult, n_scenes=n_total, m=M_JOINT,
                null_pairs_single=int(null[:, 0].sum()), null_pairs_grid=int(null.sum()),
                family_errors=events, family_cp95_upper={k: cp_upper(v, M_JOINT) for k, v in events.items()},
                mean_true_certificates={k: v / M_JOINT for k, v in true_certs.items()})


def sample_moments(rng, values, probs, n, size):
    counts = rng.multinomial(int(n), probs, size=size)
    mean = (counts @ values) / n
    var = (counts @ values ** 2 - n * mean ** 2) / (n - 1)
    return mean, var


def class_planning(rng, x, eta, beta):
    """Fixed-moment planning, pilot accuracy and pilot -> plan -> validation, for EB and Hoeffding."""
    v, c = np.unique(x, return_counts=True)
    probs = c / c.sum()
    mu, var = float(x.mean()), float(x.var(ddof=1))
    row = dict(n_current=int(len(x)), mean=mu, var=var)
    for rule in ("EB", "Hoeffding"):
        r = {}
        m0, v0 = sample_moments(rng, v, probs, len(x), M_POWER)
        r["p_current"] = float(np.mean(passes(rule, m0, v0, len(x), eta, beta)))
        nstar = plugin_n(rule, mu, var, eta, beta)[0]
        r["plugin_n"] = None if np.isnan(nstar) else int(nstar)
        r["no_plan_reason"] = None if not np.isnan(nstar) else ("mean_at_or_above_beta" if mu >= beta else "above_cap")
        if not np.isnan(nstar):
            for mult in MULTS:
                n = max(2, int(math.ceil(mult * nstar)))
                mm, vv = sample_moments(rng, v, probs, n, M_POWER)
                r[f"p_x{mult}"] = float(np.mean(passes(rule, mm, vv, n, eta, beta)))
            pm, pv = sample_moments(rng, v, probs, len(x), M_PILOT)
            nhat = plugin_n(rule, pm, pv, eta, beta)
            reason_mean = int(np.sum(np.isnan(nhat) & (pm >= beta)))
            ratio = np.where(np.isnan(nhat), np.inf, nhat / nstar)
            r["pilot"] = dict(m=M_PILOT, no_plan=int(np.isnan(nhat).sum()), no_plan_fraction=float(np.mean(np.isnan(nhat))),
                              no_plan_mean_at_or_above_beta=reason_mean,
                              no_plan_above_cap=int(np.isnan(nhat).sum()) - reason_mean,
                              ratio_quantiles_inf_no_plan={str(q): float(np.quantile(ratio, q, method="inverted_cdf")) for q in (.1, .25, .5, .75, .9)},
                              finite_ratio_median=float(np.median(ratio[np.isfinite(ratio)])) if np.isfinite(ratio).any() else None,
                              ratio_below_half=int(np.sum(ratio < 0.5)), ratio_half_to_two=int(np.sum((ratio >= 0.5) & (ratio <= 2))),
                              ratio_above_two_finite=int(np.sum(np.isfinite(ratio) & (ratio > 2))))
            for factor in (1.0, 1.5):
                issued = 0
                for nh in nhat[~np.isnan(nhat)]:
                    n = max(2, int(math.ceil(factor * nh)))
                    mm, vv = sample_moments(rng, v, probs, n, 1)
                    issued += int(passes(rule, mm, vv, n, eta, beta)[0])
                r["pilot"][f"realized_x{factor}"] = issued / M_PILOT
                r["pilot"][f"realized_x{factor}_given_plan"] = issued / max(1, int((~np.isnan(nhat)).sum()))
            r["pilot_ratios"] = ratio                         # kept in memory for pooling only
        row[rule] = r
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    t0 = time.process_time()
    v1 = json.loads(V1.read_text())
    inputs, cells = {str(V1): sha(V1)}, []
    for path in sorted((E / "s5/results").glob("*.json")):
        cell, classes, has, loss, frozen, ins = load(path)
        inputs.update(ins)
        cells.append((cell, classes, has, loss, frozen))
        print(cell, "scenes", has.shape[0], "classes", len(classes), flush=True)
    # 4. prespecified-rule comparison on the frozen moments
    observed = []
    for cell, classes, has, loss, frozen in cells:
        K = len(classes)
        for b in BETAS:
            for k, name in enumerate(classes):
                f = frozen[(k, b)]
                row = dict(cell=cell, cls=name, beta=b, n=f["n"], mean=f["mean"], var=f["var"], frozen_EB_reported=f["reported"])
                for rule in RULES:
                    row[rule] = bool(f["n"] >= 2 and passes(rule, f["mean"], f["var"], f["n"], DELTA / K, b))
                assert row["EB"] == f["reported"], (cell, name, b)
                observed.append(row)
    obs_summary = {str(b): {rule: sum(r[rule] for r in observed if r["beta"] == b) for rule in RULES} for b in BETAS}
    obs_summary["differing_beta0.2"] = [dict(cell=r["cell"], cls=r["cls"], n=r["n"], mean=r["mean"], EB=r["EB"],
                                             Hoeffding=r["Hoeffding"], union=r["union"])
                                        for r in observed if r["beta"] == 0.2 and len({r["EB"], r["Hoeffding"], r["union"]}) > 1]
    print("observed", json.dumps(obs_summary), flush=True)
    # 1. joint family validity
    joint = []
    for cell, classes, has, loss, frozen in cells:
        for b in BETAS:
            for boundary in (True, False):
                for mult in (1, 4):
                    res = joint_validity(rng, has, loss, len(classes), b, boundary, mult)
                    res.update(cell=cell, beta=b)
                    joint.append(res)
                    print("joint", cell, b, boundary, mult, res["family_errors"], flush=True)
    # 2-3. planning
    planning, pooled = [], {(b, rule): [] for b in BETAS for rule in ("EB", "Hoeffding")}
    for cell, classes, has, loss, frozen in cells:
        K = len(classes)
        for b in BETAS:
            for k, name in enumerate(classes):
                x = loss[has[:, k], k, 0]
                if len(x) < 2:
                    continue
                row = class_planning(rng, x, DELTA / K, b)
                row.update(cell=cell, cls=name, beta=b, eta=DELTA / K)
                for rule in ("EB", "Hoeffding"):
                    ratios = row[rule].pop("pilot_ratios", None)
                    if ratios is not None:
                        pooled[(b, rule)].append(ratios)
                planning.append(row)
        print("planning", cell, flush=True)

    def across(b, rule, key, sub=None):
        xs = [(r[rule][sub][key] if sub else r[rule][key]) for r in planning
              if r["beta"] == b and r[rule]["plugin_n"] is not None and (r[rule].get(sub) if sub else True)]
        xs = [x for x in xs if x is not None]
        if not xs:
            return dict(n=0)
        a = np.asarray(xs, dtype=float)
        return dict(n=len(a), median=float(np.median(a)), q10=float(np.quantile(a, .1)), min=float(a.min()),
                    below_half=int(np.sum(a < .5)))

    plan_summary = {}
    for b in BETAS:
        for rule in ("EB", "Hoeffding"):
            rows = [r for r in planning if r["beta"] == b]
            with_plan = [r for r in rows if r[rule]["plugin_n"] is not None]
            allr = np.concatenate(pooled[(b, rule)]) if pooled[(b, rule)] else np.array([])
            hardest = sorted(with_plan, key=lambda r: -r[rule]["pilot"]["no_plan"])[:4]
            worst_q90 = sorted(with_plan, key=lambda r: -r[rule]["pilot"]["ratio_quantiles_inf_no_plan"]["0.9"])[:3]
            plan_summary[f"{rule}-{b}"] = dict(
                class_rows=len(rows), classes_with_plan=len(with_plan),
                no_plan_reasons={k: sum(r[rule]["no_plan_reason"] == k for r in rows) for k in ("mean_at_or_above_beta", "above_cap")},
                expected_certificates_now=float(sum(r[rule]["p_current"] for r in rows)),
                power={f"x{m}": across(b, rule, f"p_x{m}") for m in MULTS},
                realized={f: across(b, rule, f"realized_x{f}", "pilot") for f in ("1.0", "1.5")},
                realized_given_plan={f: across(b, rule, f"realized_x{f}_given_plan", "pilot") for f in ("1.0", "1.5")},
                pooled_pilots=dict(m=int(allr.size), no_plan_fraction=float(np.mean(~np.isfinite(allr))) if allr.size else None,
                                   quantiles={str(q): float(np.quantile(allr, q, method="inverted_cdf")) for q in (.1, .25, .5, .75, .9)} if allr.size else None,
                                   below_half=float(np.mean(allr < .5)) if allr.size else None,
                                   half_to_two=float(np.mean((allr >= .5) & (allr <= 2))) if allr.size else None,
                                   above_two=float(np.mean(allr > 2)) if allr.size else None),
                per_class_no_plan_fraction=dict(across(b, rule, "no_plan_fraction", "pilot"),
                                                max=max((r[rule]["pilot"]["no_plan_fraction"] for r in with_plan), default=None)),
                hardest_by_no_plan=[dict(cell=r["cell"], cls=r["cls"], n_current=r["n_current"], mean=r["mean"],
                                         plugin_n=r[rule]["plugin_n"], no_plan_fraction=r[rule]["pilot"]["no_plan"] / M_PILOT,
                                         finite_ratio_median=r[rule]["pilot"]["finite_ratio_median"],
                                         realized_x1=r[rule]["pilot"]["realized_x1.0"], realized_x15=r[rule]["pilot"]["realized_x1.5"])
                                    for r in hardest],
                largest_q90=[dict(cell=r["cell"], cls=r["cls"], q90=r[rule]["pilot"]["ratio_quantiles_inf_no_plan"]["0.9"],
                                  plugin_n=r[rule]["plugin_n"]) for r in worst_q90])
    # per-class validity of the first run, summarized with Monte Carlo bounds (not recomputed)
    cls_valid = [r for r in v1["validity"] if r["cls"] != "__family__"]
    worst = max(cls_valid, key=lambda r: r["bad"])
    v1_summary = dict(scenarios=len(cls_valid), draws=sum(r["m"] for r in cls_valid), bad=sum(r["bad"] for r in cls_valid),
                      bad_by_mult={str(m): sum(r["bad"] for r in cls_valid if r["mult"] == m) for m in (1, 2, 4, 8)},
                      worst=dict(cell=worst["cell"], cls=worst["cls"], beta=worst["beta"], mult=worst["mult"], bad=worst["bad"],
                                 m=worst["m"], cp95_upper=cp_upper(worst["bad"], worst["m"]), eta=worst["eta"]),
                      grid_within_class=dict(scenarios=len(v1["grid_rows"]), draws=sum(r["m"] for r in v1["grid_rows"]),
                                             bad=sum(r["bad"] for r in v1["grid_rows"]),
                                             cp95_upper_per_scenario=cp_upper(0, v1["grid_rows"][0]["m"])))
    jsum = {}
    for key in (f"{r}-{s}" for r in RULES for s in ("single", "grid")):
        for boundary in (True, False):
            rows = [j for j in joint if j["boundary"] == boundary]
            errs = [j["family_errors"][key] for j in rows]
            worst_j = max(rows, key=lambda j: j["family_errors"][key])
            jsum[f"{key}-{'boundary' if boundary else 'empirical'}"] = dict(
                scenarios=len(rows), draws=len(rows) * M_JOINT, family_errors=int(sum(errs)),
                worst=dict(cell=worst_j["cell"], beta=worst_j["beta"], mult=worst_j["mult"], errors=worst_j["family_errors"][key],
                           cp95_upper=worst_j["family_cp95_upper"][key]),
                cp95_upper_if_zero=cp_upper(0, M_JOINT), delta=DELTA)
    summary = dict(observed=obs_summary, joint=jsum, planning=plan_summary, per_class_v1=v1_summary)
    receipt = dict(schema="rotcert-g2-simulation-v3", seed=SEED, delta=DELTA, grid=list(GRID), betas=list(BETAS),
                   reps=dict(joint=M_JOINT, power=M_POWER, pilot=M_PILOT), plan_cap=PLAN_CAP, multipliers=list(MULTS),
                   rules=dict(EB="mean + sqrt(2 v ln(2/eta)/n) + 7 ln(2/eta)/(3(n-1)) < beta (paper rule)",
                              Hoeffding="mean + sqrt(ln(1/eta)/(2n)) < beta",
                              union="EB at eta/2 or Hoeffding at eta/2 (union bound)"),
                   inputs=inputs, script_sha256=sha(__file__), process_cpu_seconds=time.process_time() - t0,
                   summary=summary, observed=observed, joint=joint, planning=planning,
                   interpretation=("semi-synthetic: populations are the frozen G2 calibration samples; no new certificate; "
                                   "the joint rows resample whole scenes and keep class co-occurrence"))
    path = OUT / "G2-SIMULATION-v3.json"
    if path.exists():
        raise SystemExit("receipt exists")
    path.write_text(json.dumps(_strict(receipt), indent=1, allow_nan=False) + "\n")
    print(path, sha(path))


if __name__ == "__main__":
    main()
