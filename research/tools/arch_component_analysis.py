"""Architectural-component statistical analysis over the NAS campaign.

Extracts which computation-graph primitives (and combinations) correlate with
better AR-curriculum / binding / induction probe scores, under heavy single-seed
probe noise. All program_results metrics are treated as noisy single draws;
the rerun JSONs are the only variance anchors.

Run:  python -m research.tools.arch_component_analysis
Outputs JSON tables + a markdown report under research/reports/arch_component_analysis_2026-05-23/.
"""

from __future__ import annotations

import json
import sqlite3
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression

ROOT = Path("/home/tim/Projects/LLM")
DB = ROOT / "research/runs.db"
HYDRA = ROOT / "research/reports/hydra_eval_2026-05-22"
OUT = ROOT / "research/reports/arch_component_analysis_2026-05-23"
OUT.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(20260523)
N_BOOT = 1000

# DB target columns -> short name. Ordered most-trusted-N last is irrelevant; we
# annotate N in output. binding_range / binding_multislot have ~no DB coverage.
TARGETS = {
    "ar_curriculum": "ar_curriculum_auc_pair_final",
    "binding_intermediate": "binding_intermediate_auc",
    "induction_intermediate": "induction_intermediate_auc",
    "binding_curriculum": "binding_curriculum_auc",
    "binding_screening": "binding_screening_auc",
    "induction_screening": "induction_screening_auc",
}
# Controls for partial correlation: model_dim (from graph_json), graph_depth
# (~n_blocks), n_train_steps (~training tokens), param_count (overall scale).
CONTROL_COLS = ["graph_depth", "n_train_steps", "param_count"]
MIN_PRESENT = 20  # per-group minimum for a primitive to be testable in a subset


def _op_names(graph_json: str) -> set[str]:
    try:
        g = json.loads(graph_json)
    except (json.JSONDecodeError, TypeError):
        return set()
    nodes = g.get("nodes", {})
    it = nodes.values() if isinstance(nodes, dict) else nodes
    ops = set()
    for n in it:
        name = n.get("op_name") or n.get("op")
        if name and name not in ("input", "output"):
            ops.add(name)
    return ops


def _model_dim(graph_json: str) -> float:
    try:
        return float(json.loads(graph_json).get("model_dim") or np.nan)
    except (json.JSONDecodeError, TypeError, ValueError):
        return np.nan


def load() -> dict:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    cols = (
        ["result_id", "graph_fingerprint", "graph_json"]
        + CONTROL_COLS
        + list(TARGETS.values())
    )
    # program_results_compat (= graph_runs LEFT JOIN graphs) is canonical post-Phase-5b;
    # the legacy program_results table is not updated by probe backfills and returns
    # stale pre-leak-fix binding. See research/notes/adjacent_token_merge_leak_2026-05-23.md.
    rows = con.execute(
        f"SELECT {','.join(cols)} FROM program_results_compat"
    ).fetchall()
    con.close()

    recs = []
    for r in rows:
        d = dict(zip(cols, r))
        ops = _op_names(d["graph_json"])
        if not ops:
            continue
        d["ops"] = ops
        d["model_dim"] = _model_dim(d["graph_json"])
        recs.append(d)
    return {"recs": recs, "all_ops": sorted({o for d in recs for o in d["ops"]})}


def _finite(x):
    x = np.asarray(x, float)
    return x[np.isfinite(x)]


def _bootstrap_r(present: np.ndarray, y: np.ndarray, n=N_BOOT):
    """Bootstrap CI for Pearson r between a binary indicator and y."""
    idx = np.arange(len(y))
    rs = np.empty(n)
    for i in range(n):
        s = RNG.choice(idx, len(idx), replace=True)
        p, yy = present[s], y[s]
        if p.std() == 0 or yy.std() == 0:
            rs[i] = np.nan
            continue
        rs[i] = np.corrcoef(p, yy)[0, 1]
    rs = rs[np.isfinite(rs)]
    if rs.size < n * 0.5:
        return np.nan, np.nan
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def _residualize(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return y - LinearRegression().fit(X, y).predict(X)


def per_primitive(data: dict) -> dict:
    recs, all_ops = data["recs"], data["all_ops"]
    out = {}
    for tname, col in TARGETS.items():
        sub = [d for d in recs if d.get(col) is not None and np.isfinite(d[col])]
        if len(sub) < 40:
            continue
        y = np.array([d[col] for d in sub], float)
        # build control matrix; rows with any missing control dropped for partial only
        ctrl_raw = np.array(
            [
                [d["model_dim"], d["graph_depth"], d["n_train_steps"], d["param_count"]]
                for d in sub
            ],
            float,
        )
        # log-scale the magnitude controls
        ctrl = ctrl_raw.copy()
        for j in (0, 2, 3):  # model_dim, n_train_steps, param_count
            ctrl[:, j] = np.log10(np.clip(ctrl[:, j], 1, None))
        ctrl_ok = np.all(np.isfinite(ctrl), axis=1)

        results = []
        for op in all_ops:
            present = np.array([op in d["ops"] for d in sub], float)
            n_pre, n_abs = int(present.sum()), int((1 - present).sum())
            if n_pre < MIN_PRESENT or n_abs < MIN_PRESENT:
                continue
            yp, ya = y[present == 1], y[present == 0]
            delta = float(yp.mean() - ya.mean())
            psd = np.sqrt(
                ((n_pre - 1) * yp.var(ddof=1) + (n_abs - 1) * ya.var(ddof=1))
                / (n_pre + n_abs - 2)
            )
            cohen_d = float(delta / psd) if psd > 0 else np.nan
            r = float(np.corrcoef(present, y)[0, 1])
            lo, hi = _bootstrap_r(present, y)
            # partial correlation on the control-complete subset
            pr = prlo = prhi = np.nan
            if ctrl_ok.sum() > 40 and present[ctrl_ok].std() > 0:
                Xc = ctrl[ctrl_ok]
                ry = _residualize(Xc, y[ctrl_ok])
                rp = _residualize(Xc, present[ctrl_ok])
                if rp.std() > 0 and ry.std() > 0:
                    pr = float(np.corrcoef(rp, ry)[0, 1])
                    prlo, prhi = _bootstrap_r(rp, ry)
            sig = bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0) == (hi > 0))
            results.append(
                {
                    "op": op,
                    "n_present": n_pre,
                    "n_absent": n_abs,
                    "mean_present": float(yp.mean()),
                    "mean_absent": float(ya.mean()),
                    "delta": delta,
                    "cohen_d": cohen_d,
                    "pearson_r": r,
                    "r_ci": [lo, hi],
                    "r_sig_excl_0": sig,
                    "partial_r": pr,
                    "partial_ci": [prlo, prhi],
                }
            )
        results.sort(
            key=lambda d: -abs(d["pearson_r"]) if np.isfinite(d["pearson_r"]) else 0
        )
        out[tname] = {"n_rows": len(sub), "results": results}
    return out


def combos(data: dict) -> dict:
    """2-op and 3-op lift vs top tercile of each metric."""
    recs = data["recs"]
    out = {}
    for tname, col in TARGETS.items():
        sub = [d for d in recs if d.get(col) is not None and np.isfinite(d[col])]
        if len(sub) < 60:
            continue
        y = np.array([d[col] for d in sub], float)
        hi_thr = np.percentile(y, 66.7)
        overall = y.mean()
        # candidate ops: present in >=15% of subset, <=85%
        cnt = {}
        for d in sub:
            for o in d["ops"]:
                cnt[o] = cnt.get(o, 0) + 1
        cand = [o for o, c in cnt.items() if 0.10 * len(sub) <= c <= 0.90 * len(sub)]
        rows = []
        for k in (2, 3):
            for combo in combinations(sorted(cand), k):
                mask = np.array([all(o in d["ops"] for o in combo) for d in sub], bool)
                supp = int(mask.sum())
                if supp < 20:
                    continue
                yc = y[mask]
                frac_hi_combo = float((yc >= hi_thr).mean())
                frac_hi_all = float((y >= hi_thr).mean())
                rows.append(
                    {
                        "combo": list(combo),
                        "k": k,
                        "support": supp,
                        "mean_metric": float(yc.mean()),
                        "overall_mean": float(overall),
                        "delta": float(yc.mean() - overall),
                        "frac_in_top_tercile": frac_hi_combo,
                        "tercile_lift": float(frac_hi_combo / frac_hi_all)
                        if frac_hi_all
                        else np.nan,
                    }
                )
        rows.sort(key=lambda d: -d["delta"])
        out[tname] = {
            "n_rows": len(sub),
            "top_positive": rows[:12],
            "top_negative": rows[-8:],
        }
    return out


def variance_budget() -> dict:
    """Estimate probe-noise std per metric from rerun anchors and compare to
    the between-architecture spread in the DB."""
    anchors = {}

    # induction_intermediate: 3 good reruns (median-of-3 internal seeds each)
    ind = json.load(open(HYDRA / "n4_induction_reruns.json"))
    good = [r["auc"] for r in ind["reruns"]]
    anchors["induction_intermediate"] = {
        "probe_std_good_mode": float(np.std(good, ddof=1)),
        "good_mode_mean": float(np.mean(good)),
        "catastrophic_draw": ind["original_run_auc"],
        "catastrophic_gap": float(np.mean(good) - ind["original_run_auc"]),
        "note": "good-mode std tiny; bimodal catastrophic-failure tail of ~0.42",
    }

    # ar_validation rank: per-seed scores within one 3-seed call
    av = json.load(open(HYDRA / "n2_ar_val_7seed.json"))["result"]
    seed_scores = [s["score"] for s in json.loads(av["ar_validation_seed_scores_json"])]
    per_seed_std = float(np.std(seed_scores, ddof=1))
    anchors["ar_validation_rank"] = {
        "per_seed_std": per_seed_std,
        "sem_3seed_call": per_seed_std / np.sqrt(3),
        "between_call_means": [1.88, 3.00],
        "between_call_gap": 1.12,
        "note": "single rank_score std ~1.0; 3-seed-call mean still swings 1.88<->3.00",
    }

    # binding_multislot held_slot + binding_range from 7-call file
    bm = json.load(open(HYDRA / "n2_binding_multiseed.json"))
    ms = [s["held_slot"] for s in bm["multislot"]["per_seed"]]
    br = [s["auc"] for s in bm["binding_range"]["per_seed"]]
    anchors["binding_multislot_held_slot"] = {
        "probe_std": float(np.std(ms, ddof=1)),
        "mean": float(np.mean(ms)),
        "cv": float(np.std(ms, ddof=1) / np.mean(ms)),
        "low_draw": 0.0495,
        "low_draw_sigma": float((np.mean(ms) - 0.0495) / np.std(ms, ddof=1)),
    }
    anchors["binding_range"] = {
        "probe_std": float(np.std(br, ddof=1)),
        "mean": float(np.mean(br)),
        "cv": float(np.std(br, ddof=1) / np.mean(br)),
        "note": "lowest-variance probe in the suite",
    }

    return {"anchors": anchors}


def db_spread(data: dict) -> dict:
    out = {}
    for tname, col in TARGETS.items():
        y = _finite([d[col] for d in data["recs"] if d.get(col) is not None])
        if y.size < 40:
            continue
        out[tname] = {
            "n": int(y.size),
            "mean": float(y.mean()),
            "std": float(y.std(ddof=1)),
            "p10": float(np.percentile(y, 10)),
            "p90": float(np.percentile(y, 90)),
            "iqr": float(np.percentile(y, 75) - np.percentile(y, 25)),
        }
    return out


def main():
    data = load()
    print(
        f"loaded {len(data['recs'])} graph rows | {len(data['all_ops'])} distinct ops"
    )
    pp = per_primitive(data)
    cb = combos(data)
    vb = variance_budget()
    sp = db_spread(data)

    json.dump(pp, open(OUT / "per_primitive.json", "w"), indent=1)
    json.dump(cb, open(OUT / "combinations.json", "w"), indent=1)
    json.dump(
        {"variance": vb, "db_spread": sp},
        open(OUT / "variance_budget.json", "w"),
        indent=1,
    )

    # compact console summary
    for tname, blk in pp.items():
        print(f"\n## {tname}  (n={blk['n_rows']})")
        top = [r for r in blk["results"] if r["r_sig_excl_0"]][:10]
        for r in top:
            print(
                f"  {r['op']:24s} r={r['pearson_r']:+.3f} "
                f"CI[{r['r_ci'][0]:+.3f},{r['r_ci'][1]:+.3f}] "
                f"pr={r['partial_r']:+.3f} d={r['cohen_d']:+.2f} "
                f"n+={r['n_present']} Δ={r['delta']:+.4f}"
            )
    print("\n== variance anchors ==")
    for k, v in vb["anchors"].items():
        print(f"  {k}: {json.dumps(v)}")
    print("\n== db single-seed spread ==")
    for k, v in sp.items():
        print(f"  {k}: std={v['std']:.4f} mean={v['mean']:.4f} n={v['n']}")
    print(f"\nartifacts -> {OUT}")


if __name__ == "__main__":
    main()
