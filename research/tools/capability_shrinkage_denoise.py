#!/usr/bin/env python
"""Does hierarchical-shrinkage denoising of noisy capability labels help?

The induction/binding/ar capability probes are single-seed fine-tune runs whose
numbers swing with the seed (documented). Noisy per-graph labels (a) attenuate the
cross-axis correlations and (b) cap how well a predictor can rank capability. This
script tests whether **partial pooling** (empirical-Bayes shrinkage of each graph's
value toward its architectural-cluster mean) recovers signal.

Two experiments, over a sweep of the assumed seed-noise fraction f
(f=0 keeps raw labels; f=1 fully pools each graph to its cluster mean):

  1. Cross-axis correlation sharpening — induction/binding/ar Spearman, raw vs shrunk.
  2. Induction predictor — GBM (fingerprint + op-presence) trained on the *shrunk*
     induction target, scored forward-in-time. Cluster stats are estimated on TRAIN
     ONLY (no leakage); the headline metric is ROC-AUC against the RAW held-out label.

Cluster = template family (between-template variance share of capability is
0.34-0.59, so there is real cluster signal to borrow). Two-level shrinkage:
cluster means are themselves count-shrunk toward the grand mean (handles small /
unseen templates gracefully).

Usage::

    python -m research.tools.capability_shrinkage_denoise
    python -m research.tools.capability_shrinkage_denoise --thr 0.35 --out my.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from research.defaults import RUNS_DB
from research.tools.induction_predictor_foundation import (
    _build_corpus,
    _fit_gbm,
    _op_presence_features,
    _ranking_metrics,
    _spearman,
)

logger = logging.getLogger(__name__)

_NONE_CLUSTER = "__no_template__"
_F_SWEEP: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)


def _template_map(db_path: str) -> Dict[str, str]:
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            "SELECT graph_fingerprint, template_name FROM program_graph_features "
            "WHERE template_name IS NOT NULL"
        ).fetchall()
    finally:
        con.close()
    return {str(fp): str(t) for fp, t in rows}


def _capability_map(db_path: str, col: str) -> Dict[str, float]:
    con = sqlite3.connect(db_path)
    try:
        query = (
            f"SELECT graph_fingerprint, {col} FROM graph_runs WHERE {col} IS NOT NULL"  # nosec B608  # nosemgrep: python-sql-string-formatting
        )
        rows = con.execute(query).fetchall()
    finally:
        con.close()
    acc: Dict[str, List[float]] = defaultdict(list)
    for fp, v in rows:
        acc[str(fp)].append(float(v))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def _shrink(
    values: np.ndarray,
    clusters: List[str],
    train_mask: np.ndarray,
    f: float,
    prior_strength: float = 5.0,
) -> np.ndarray:
    """Two-level empirical-Bayes shrinkage of per-graph values toward cluster means.

    Cluster means and the grand mean are estimated on ``train_mask`` rows only.
    Level 1: cluster mean count-shrunk toward grand mean (weight n/(n+prior)).
    Level 2: each graph shrunk toward its (shrunk) cluster mean by fraction ``f``.
    """
    vals_tr = values[train_mask]
    grand = float(vals_tr.mean()) if vals_tr.size else float(values.mean())
    sums: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    for i, c in enumerate(clusters):
        if train_mask[i]:
            sums[c] += float(values[i])
            counts[c] += 1
    cluster_mu: Dict[str, float] = {}
    for c, n in counts.items():
        raw_mu = sums[c] / n
        w = n / (n + prior_strength)
        cluster_mu[c] = grand + w * (raw_mu - grand)
    out = np.empty_like(values, dtype=np.float64)
    for i, c in enumerate(clusters):
        mu = cluster_mu.get(c, grand)
        out[i] = mu + (1.0 - f) * (values[i] - mu)
    return out


def _correlation_sharpening(db_path: str) -> Dict[str, Any]:
    """Pairwise Spearman among induction/binding/ar, raw vs shrunk.

    Cluster means are estimated over each metric's FULL coverage (not the sparse
    pairwise overlap), then correlated on the intersection — otherwise tiny
    per-cluster counts on a thin overlap make the shrinkage degenerate.
    """
    tmpl = _template_map(db_path)
    cols = {
        "induction": "induction_intermediate_auc",
        "binding": "binding_intermediate_auc",
        "ar": "ar_curriculum_auc_pair_final",
    }
    raw: Dict[str, Dict[str, float]] = {}
    shrunk: Dict[str, Dict[str, Dict[float, float]]] = {}
    for name, col in cols.items():
        cap = _capability_map(db_path, col)
        fps = sorted(cap)
        clusters = [tmpl.get(fp, _NONE_CLUSTER) for fp in fps]
        vals = np.array([cap[fp] for fp in fps])
        all_mask = np.ones(len(fps), dtype=bool)
        per_f = {f: _shrink(vals, clusters, all_mask, f) for f in _F_SWEEP}
        raw[name] = cap
        shrunk[name] = {
            fp: {f: float(per_f[f][i]) for f in _F_SWEEP} for i, fp in enumerate(fps)
        }
    out: Dict[str, Any] = {}
    for a, b in [("induction", "binding"), ("induction", "ar"), ("binding", "ar")]:
        keys = sorted(set(raw[a]) & set(raw[b]))
        if len(keys) < 10:
            continue
        row: Dict[str, Any] = {"n": len(keys)}
        for f in _F_SWEEP:
            xa = np.array([shrunk[a][k][f] for k in keys])
            xb = np.array([shrunk[b][k][f] for k in keys])
            rho, _ = _spearman(xa, xb)
            row[f"f_{f:g}"] = round(rho, 3)
        out[f"{a}_vs_{b}"] = row
    return out


def _predictor_sweep(db_path: str, thr: float) -> Dict[str, Any]:
    """GBM on shrunk induction target; ROC forward-in-time vs RAW (and shrunk) label."""
    X, _, ind, _, fps = _build_corpus(db_path)
    n = len(X)
    cut = int(n * 0.8)
    train_mask = np.zeros(n, dtype=bool)
    train_mask[:cut] = True

    tmpl = _template_map(db_path)
    clusters = [tmpl.get(fp, _NONE_CLUSTER) for fp in fps]
    n_with_tmpl = sum(1 for c in clusters if c != _NONE_CLUSTER)

    X_ops, _ = _op_presence_features(db_path, fps)
    Xfull = np.hstack([X, X_ops])
    ind_raw_test = ind[cut:]

    results: Dict[str, Any] = {}
    for f in _F_SWEEP:
        ind_shrunk = _shrink(ind, clusters, train_mask, f)
        gbm = _fit_gbm(Xfull[:cut], ind_shrunk[:cut])
        pred_test = np.asarray(gbm.predict(Xfull[cut:]), dtype=np.float64)
        vs_raw = _ranking_metrics(pred_test, ind_raw_test, thr)
        vs_shrunk = _ranking_metrics(pred_test, ind_shrunk[cut:], thr)
        results[f"f_{f:g}"] = {
            "roc_vs_raw_label": round(vs_raw["roc_auc_gt_thr"] or 0.0, 4),
            "spearman_vs_raw_label": round(vs_raw["spearman_rho"], 4),
            "roc_vs_shrunk_label_ceiling": round(vs_shrunk["roc_auc_gt_thr"] or 0.0, 4),
        }
    return {
        "n_total": n,
        "n_train": cut,
        "n_test": n - cut,
        "n_features": Xfull.shape[1],
        "n_graphs_with_template": n_with_tmpl,
        "headline": "roc_vs_raw_label (f=0 is the raw-target baseline)",
        "by_noise_fraction": results,
    }


def run(db_path: str, thr: float) -> Dict[str, Any]:
    return {
        "induction_threshold": thr,
        "noise_fraction_sweep": list(_F_SWEEP),
        "cluster": "template_family",
        "experiment_1_correlation_sharpening": _correlation_sharpening(db_path),
        "experiment_2_predictor_on_shrunk_target": _predictor_sweep(db_path, thr),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(RUNS_DB))
    parser.add_argument("--thr", type=float, default=0.35)
    parser.add_argument(
        "--out", default="research/reports/capability_shrinkage_denoise.json"
    )
    args = parser.parse_args()

    report = run(args.db, args.thr)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
