#!/usr/bin/env python
"""Validate the CLOSED-BOOK measured descriptors against induction labels — the decisive test.

Does a name-free, label-free, random-init MEASURED descriptor predict trained induction, and does
it beat (a) #params/op-count, (b) the strongest EXISTING measured profile column
(profile_grad_exploding_op_count, ROC≈0.83 — incidentally a mixing-op detector), and (c) the
declared-catalog rep (graph_semantic_features = the "cheating" rep)?  And — the real prize — does it
rank the novel STDP winner `e656938e359ada50` HIGH from mechanism alone, where the label-fit GBM
gave 0.026?

Headline metric is SINGLE-descriptor ROC with NO fitting (closed-book). A small OOF GBM on the 8
measured descriptors is reported only to compare representations head-to-head.

Usage::
    python -m research.tools.measured_descriptor_eval [--n-neg 1500] [--n-seeds 2] [--with-winner]
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from research.tools.induction_predictor_foundation import _fit_gbm
from research.tools.measured_descriptors import (
    DESCRIPTOR_NAMES,
    MeasuredDescriptorExtractor,
)

logger = logging.getLogger(__name__)
_RUNS_DB = "research/runs.db"
_META_DB = "research/meta_analysis.db"
_THR = 0.35
_STDP_WINNER = "e656938e359ada50"  # pragma: allowlist secret
_EXISTING = [
    "profile_grad_exploding_op_count",
    "profile_max_jacobian_condition_num",
    "profile_pair_max_lipschitz_estimate",
]


# allowlisted label columns (avoids SQL injection; values are fixed capability metrics)
_LABEL_COLS = (
    "induction_screening_auc",
    "nano_induction_nearest_max_accuracy",
    "ar_gate_score",
    "ar_curriculum_auc_pair_final",
)


def _corpus(
    runs_db: str, label_col: str, thr: float, n_neg: int, seed: int = 0
) -> List[Tuple[str, str, float]]:
    """All positives (label>thr) + a random sample of negatives; each with graph_json."""
    if (
        label_col not in _LABEL_COLS
    ):  # allowlist makes the interpolation below injection-safe
        raise ValueError(f"label_col must be one of {_LABEL_COLS}")
    con = sqlite3.connect(runs_db)
    query = (
        f"SELECT g.graph_fingerprint, g.graph_json, r.{label_col} "
        "FROM graphs g JOIN graph_runs r ON g.graph_fingerprint=r.graph_fingerprint "
        f"WHERE g.graph_json_is_placeholder=0 AND r.{label_col} IS NOT NULL"
    )
    rows = con.execute(query).fetchall()  # nosec B608  # nosemgrep: python-sql-string-formatting
    con.close()
    # dedup by fingerprint (mean label), keep one graph_json
    by_fp: Dict[str, Tuple[str, List[float]]] = {}
    for fp, gj, auc in rows:
        _, accs = by_fp.setdefault(str(fp), (gj, []))
        accs.append(float(auc))
    items = [(fp, gj, float(np.mean(a))) for fp, (gj, a) in by_fp.items()]
    pos = [it for it in items if it[2] > thr]
    neg = [it for it in items if it[2] <= thr]
    rng = np.random.default_rng(seed)
    if len(neg) > n_neg:
        neg = [neg[i] for i in rng.choice(len(neg), n_neg, replace=False)]
    return pos + neg


def _existing_baselines(
    runs_db: str, meta_db: str, fps: List[str]
) -> Dict[str, np.ndarray]:
    """Pull pre-existing MEASURED profile columns + op_count for the same fingerprints."""
    con = sqlite3.connect(meta_db)
    con.execute(f"ATTACH '{runs_db}' AS runs")  # nosec B608  # nosemgrep: python-sql-string-formatting
    cols = _EXISTING
    out: Dict[str, List[float]] = {c: [] for c in cols}
    out["op_count"] = []
    fp_set = {fp: i for i, fp in enumerate(fps)}
    got = {c: np.full(len(fps), np.nan) for c in cols}
    got["op_count"] = np.full(len(fps), np.nan)
    q = f"""SELECT gp.graph_fingerprint, {",".join("gp." + c for c in cols)},
                   r.graph_n_ops
            FROM graph_profile_observations gp
            JOIN runs.graph_runs r ON gp.graph_fingerprint=r.graph_fingerprint"""
    for row in con.execute(q):
        fp = str(row[0])
        if fp not in fp_set:
            continue
        i = fp_set[fp]
        for j, c in enumerate(cols):
            if row[1 + j] is not None:
                got[c][i] = float(row[1 + j])
        if row[-1] is not None:
            got["op_count"][i] = float(row[-1])
    con.close()
    return got


def _single_feature_table(
    feats: Dict[str, np.ndarray], y: np.ndarray, thr: float
) -> Dict[str, Dict[str, Any]]:
    """Spearman + ROC(label>thr) of each feature, no fitting (closed-book)."""
    pos = (y > thr).astype(int)
    table: Dict[str, Dict[str, Any]] = {}
    for name, x in feats.items():
        m = np.isfinite(x)
        if m.sum() < 50 or len(np.unique(x[m])) < 3:
            table[name] = {"roc": None, "spearman": None, "n": int(m.sum())}
            continue
        pm = pos[m]
        rho = float(spearmanr(x[m], y[m])[0])
        roc = float(roc_auc_score(pm, x[m])) if 0 < pm.sum() < len(pm) else None
        # direction-agnostic ROC (a descriptor may be inversely related)
        if roc is not None and roc < 0.5:
            roc = 1.0 - roc
        table[name] = {
            "roc": round(roc, 4) if roc else None,
            "spearman": round(float(rho), 4) if np.isfinite(rho) else None,
            "n": int(m.sum()),
        }
    return table


def _oof_gbm_roc(
    X: np.ndarray, y: np.ndarray, thr: float, seed: int = 42
) -> Optional[float]:
    """OOF GBM ranking ROC for a representation (for head-to-head rep comparison only)."""
    ok = np.all(np.isfinite(X), axis=1)
    X, y = X[ok], y[ok]
    pos = (y > thr).astype(int)
    if pos.sum() < 5 or pos.sum() == len(pos):
        return None
    oof = np.zeros(len(y))
    for fit_ix, hold_ix in KFold(5, shuffle=True, random_state=seed).split(X):
        oof[hold_ix] = _fit_gbm(X[fit_ix], y[fit_ix]).predict(X[hold_ix])
    return round(float(roc_auc_score(pos, oof)), 4)


def _reach_operating_point(
    reach: np.ndarray, y: np.ndarray, thr: float
) -> Dict[str, Any]:
    """Re-fit the rescue prefilter τ: recall of capable kept / incapable pruned at long_range_reach
    >= τ. The live gate (screening_measured_rescue._DEFAULT_TAU=0.01) was set on induction; this lets
    each axis confirm or move its operating point from the SAME probed arrays (no extra GPU pass)."""
    m = np.isfinite(reach)
    reach, y = reach[m], y[m]
    pos = y > thr
    neg = ~pos
    out: Dict[str, Any] = {}
    for tau in (0.005, 0.01, 0.02, 0.05, 0.1):
        keep = reach >= tau
        out[f"tau_{tau}"] = {
            "capable_kept": round(float(keep[pos].mean()), 4) if pos.any() else None,
            "incapable_pruned": round(float((~keep[neg]).mean()), 4)
            if neg.any()
            else None,
        }
    return out


def _winner_descriptors(
    runs_db: str, ext: MeasuredDescriptorExtractor
) -> Optional[Dict[str, float]]:
    from research.tools.probe_novel_candidates import _collect_pool

    for c in _collect_pool(runs_db, 600, 12000, 3_000_000):
        if c["fingerprint"] == _STDP_WINNER:
            import json as _json

            gj = _json.dumps(c["graph"].to_dict())
            return ext.descriptors(gj)
    return None


def run(
    runs_db: str,
    meta_db: str,
    label_col: str,
    thr: float,
    n_neg: int,
    n_seeds: int,
    with_winner: bool,
) -> Dict[str, Any]:
    ext = MeasuredDescriptorExtractor(n_seeds=n_seeds)
    items = _corpus(runs_db, label_col, thr, n_neg)
    t0 = time.time()
    fps: List[str] = []
    rows: List[Dict[str, float]] = []
    labels: List[float] = []
    n_fail = 0
    for i, (fp, gj, auc) in enumerate(items):
        d = ext.descriptors(gj)
        if d is None:
            n_fail += 1
            continue
        fps.append(fp)
        rows.append(d)
        labels.append(auc)
        if (i + 1) % 200 == 0:
            logger.info("  %d/%d probed (%.1fs)", i + 1, len(items), time.time() - t0)
    y = np.array(labels)
    measured = {n: np.array([r[n] for r in rows]) for n in DESCRIPTOR_NAMES}
    baselines = _existing_baselines(runs_db, meta_db, fps)

    feats = {**measured, **baselines}
    out: Dict[str, Any] = {
        "device": ext.device,
        "label_col": label_col,
        "threshold": thr,
        "n_probed": len(fps),
        "n_failed": n_fail,
        "n_pos_gt_thr": int((y > thr).sum()),
        "elapsed_s": round(time.time() - t0, 1),
        "single_feature_roc": _single_feature_table(feats, y, thr),
        "long_range_reach_operating_point": _reach_operating_point(
            measured["long_range_reach"], y, thr
        ),
        "oof_gbm_roc": {
            "measured_8": _oof_gbm_roc(
                np.column_stack([measured[n] for n in DESCRIPTOR_NAMES]), y, thr
            ),
        },
    }
    if with_winner:
        wd = _winner_descriptors(runs_db, ext)
        if wd is not None:
            pct = {}
            for n in DESCRIPTOR_NAMES:
                dist = measured[n][np.isfinite(measured[n])]
                pct[n] = round(float((dist < wd[n]).mean()), 4)
            out["stdp_winner"] = {
                "fingerprint": _STDP_WINNER,
                "descriptors": {k: round(v, 4) for k, v in wd.items()},
                "percentile_vs_corpus": pct,
            }
        else:
            out["stdp_winner"] = {"available": False}
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=_RUNS_DB)
    p.add_argument("--meta", default=_META_DB)
    p.add_argument(
        "--label-col", default="induction_screening_auc", choices=_LABEL_COLS
    )
    p.add_argument("--thr", type=float, default=_THR)
    p.add_argument("--n-neg", type=int, default=1500)
    p.add_argument("--n-seeds", type=int, default=2)
    p.add_argument("--with-winner", action="store_true")
    p.add_argument("--out", default="research/reports/measured_descriptor_eval.json")
    args = p.parse_args()
    report = run(
        args.db,
        args.meta,
        args.label_col,
        args.thr,
        args.n_neg,
        args.n_seeds,
        args.with_winner,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
