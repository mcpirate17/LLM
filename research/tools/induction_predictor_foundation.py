#!/usr/bin/env python
"""Foundation check for retargeting the investigation predictor.

The deployed investigation predictor (``predictor_ridge.py``) is an 18-feature
linear Ridge regressing onto ``loss_ratio`` (a perplexity ratio).  Two
structural handicaps were diagnosed:

  1. WRONG TARGET — loss_ratio is ~orthogonal to capability
     (global spearman(loss_ratio, induction) = -0.21), yet selection now runs
     on capability (induction/binding/ar), not perplexity.
  2. WRONG MODEL CLASS — a linear fit on 18 fingerprint scalars.

Before anyone redesigns the production model, this script validates the
FOUNDATION, forward-in-time, on the same deduped corpus the production
predictors train on:

  A. baseline-as-deployed : Ridge -> loss_ratio, ranked by -loss_ratio
  B. same features, fixed target : Ridge -> induction
  C. fixed target + nonlinear : GBM -> induction

All three are scored by how well their ranking recovers the held-out
induction capability label (the thing selection actually cares about), using a
forward-in-time 80/20 split.  Because the capability label is zero-inflated
(~98% near zero), we report both Spearman (ranking) and ROC-AUC for the
rare-positive event induction_auc > 0.35.

This is a measurement only: it does NOT touch any production artifact.

Usage::

    python -m research.tools.induction_predictor_foundation
    python -m research.tools.induction_predictor_foundation --thr 0.35 --out my.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from research.defaults import RUNS_DB
from research.scientist.intelligence.ml_corpus import (
    load_deduped_predictor_training_rows,
)
from research.scientist.intelligence.predictor_ridge import (
    _FINGERPRINT_KEYS,
    _extract_features,
)

logger = logging.getLogger(__name__)

_FEATURE_NAMES: List[str] = [*_FINGERPRINT_KEYS, "novelty_score", "structural_novelty"]


def _spearman(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """spearmanr returning explicit (rho, pvalue) floats (scipy stubs are loose)."""
    res = spearmanr(a, b)
    return float(res[0]), float(res[1])  # type: ignore[index]


def _fingerprint_timestamps(db_path: str) -> Dict[str, float]:
    """Map canonical graph fingerprint -> latest run timestamp (epoch seconds)."""
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            "SELECT graph_fingerprint, MAX(timestamp) FROM graph_runs "
            "WHERE graph_fingerprint IS NOT NULL AND timestamp IS NOT NULL "
            "GROUP BY graph_fingerprint"
        ).fetchall()
    finally:
        con.close()
    out: Dict[str, float] = {}
    for fp, ts in rows:
        try:
            out[str(fp)] = float(ts)
        except (TypeError, ValueError):
            continue
    return out


def _build_corpus(
    db_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Return (X[n,18], loss_ratio[n], induction[n], timestamp[n], fps[n]).

    Rows are sorted by timestamp (oldest first) so a prefix split is temporal.
    Only rows that have BOTH an induction capability label and a join-able
    timestamp are kept.  Fails loud if the fingerprint join is degenerate.
    """
    rows = load_deduped_predictor_training_rows(db_path, validate=False)
    ts_map = _fingerprint_timestamps(db_path)

    X_list: List[np.ndarray] = []
    lr_list: List[float] = []
    ind_list: List[float] = []
    ts_list: List[float] = []
    fp_list: List[str] = []
    n_no_ind = n_no_ts = n_bad_feat = 0

    for row in rows:
        ind = row.get("induction_screening_auc_500")
        if ind is None or not np.isfinite(float(ind)):
            n_no_ind += 1
            continue
        fp = str(row.get("canonical_fingerprint") or "")
        ts = ts_map.get(fp)
        if ts is None:
            n_no_ts += 1
            continue
        nov = row.get("novelty_score")
        snov = row.get("structural_novelty")
        feats = _extract_features(
            row.get("fingerprint_json"),
            float(nov) if nov is not None else 0.0,
            float(snov) if snov is not None else 0.0,
        )
        if feats is None:
            n_bad_feat += 1
            continue
        target = row.get("target_loss_ratio")
        lr = float(target) if target is not None else float("nan")
        if not np.isfinite(lr) or lr < 0.0 or lr > 2.0:
            continue
        X_list.append(feats)
        lr_list.append(lr)
        ind_list.append(float(ind))
        ts_list.append(float(ts))
        fp_list.append(fp)

    if len(X_list) < 100:
        raise SystemExit(
            f"insufficient joined rows: {len(X_list)} "
            f"(no_induction={n_no_ind}, no_timestamp={n_no_ts}, bad_feat={n_bad_feat})"
        )
    logger.info(
        "corpus: kept=%d dropped(no_ind=%d,no_ts=%d,bad_feat=%d)",
        len(X_list),
        n_no_ind,
        n_no_ts,
        n_bad_feat,
    )
    order = np.argsort(np.asarray(ts_list))
    X = np.asarray(X_list, dtype=np.float64)[order]
    lr = np.asarray(lr_list, dtype=np.float64)[order]
    ind = np.asarray(ind_list, dtype=np.float64)[order]
    ts = np.asarray(ts_list, dtype=np.float64)[order]
    fps = [fp_list[i] for i in order]
    return X, lr, ind, ts, fps


def _op_presence_features(
    db_path: str,
    fps: List[str],
    min_graph_count: int = 20,
    vocab: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Binary op-presence matrix aligned to ``fps``.

    If ``vocab`` (raw op names) is given it is used verbatim — required at score
    time so the feature layout matches the trained model. Otherwise the vocab is
    derived from ops appearing in >= ``min_graph_count`` of the given graphs.
    """
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            "SELECT graph_fingerprint, op_name FROM program_graph_ops "
            "WHERE graph_fingerprint IS NOT NULL AND op_name IS NOT NULL"
        ).fetchall()
    finally:
        con.close()
    fp_ops: Dict[str, set[str]] = {}
    for fp, op in rows:
        fp_ops.setdefault(str(fp), set()).add(str(op))
    if vocab is None:
        corpus_fps = set(fps)
        op_counts: Dict[str, int] = {}
        for fp, ops in fp_ops.items():
            if fp in corpus_fps:
                for op in ops:
                    op_counts[op] = op_counts.get(op, 0) + 1
        vocab = sorted(op for op, c in op_counts.items() if c >= min_graph_count)
    idx = {op: j for j, op in enumerate(vocab)}
    mat = np.zeros((len(fps), len(vocab)), dtype=np.float64)
    for i, fp in enumerate(fps):
        for op in fp_ops.get(fp, ()):  # type: ignore[union-attr]
            j = idx.get(op)
            if j is not None:
                mat[i, j] = 1.0
    return mat, [f"op_{op}" for op in vocab]


def _sibling_probe_features(
    db_path: str, fps: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """Cheap sibling-probe features aligned to ``fps`` (NaN where unmeasured).

    Excludes the induction probe (the prediction target) to avoid leakage.
    """
    names = ["binding_screening_auc", "ar_legacy_auc"]
    con = sqlite3.connect(db_path)
    maps: List[Dict[str, float]] = []
    try:
        for col in names:
            rows = con.execute(
                f"SELECT graph_fingerprint, AVG({col}) FROM graph_runs "  # nosec B608  # nosemgrep: python-sql-string-formatting
                f"WHERE {col} IS NOT NULL GROUP BY graph_fingerprint"
            ).fetchall()
            maps.append({str(fp): float(v) for fp, v in rows if v is not None})
    finally:
        con.close()
    mat = np.full((len(fps), len(names)), np.nan, dtype=np.float64)
    for i, fp in enumerate(fps):
        for j, m in enumerate(maps):
            if fp in m:
                mat[i, j] = m[fp]
    return mat, names


def _ridge_fit(
    X: np.ndarray, y: np.ndarray, alpha: float = 1.0
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    Xn = (X - mean) / std
    n_feat = Xn.shape[1]
    w = np.linalg.solve(Xn.T @ Xn + alpha * np.eye(n_feat), Xn.T @ y)
    bias = float(np.mean(y - Xn @ w))
    return w, bias, mean, std


def _ridge_predict(
    X: np.ndarray, w: np.ndarray, bias: float, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    return ((X - mean) / std) @ w + bias


def _ranking_metrics(
    score: np.ndarray, induction: np.ndarray, thr: float
) -> Dict[str, Any]:
    rho, p = _spearman(score, induction)
    pos = (induction > thr).astype(int)
    out: Dict[str, Any] = {
        "spearman_rho": rho if np.isfinite(rho) else 0.0,
        "spearman_p": p if np.isfinite(p) else 1.0,
        "n_pos_gt_thr": int(pos.sum()),
        "prevalence_gt_thr": float(pos.mean()),
    }
    if 0 < pos.sum() < len(pos):
        out["roc_auc_gt_thr"] = float(roc_auc_score(pos, score))
    else:
        out["roc_auc_gt_thr"] = None
    return out


def _fit_gbm(Xtr: np.ndarray, ytr: np.ndarray) -> Any:
    """Single source of the LightGBM regressor config used across experiments."""
    import lightgbm as lgb

    gbm = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
    )
    gbm.fit(Xtr, ytr)
    return gbm


def _eval_gbm(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    indte: np.ndarray,
    thr: float,
    feat_names: List[str],
) -> Dict[str, Any]:
    """Train a LightGBM regressor on induction and score the held-out slice."""
    gbm = _fit_gbm(Xtr, ytr)
    pred = np.asarray(gbm.predict(Xte), dtype=np.float64)
    out = _ranking_metrics(pred, indte, thr)
    out["n_features"] = int(Xtr.shape[1])
    out["feature_importance"] = {
        name: int(imp)
        for name, imp in sorted(
            zip(feat_names, gbm.feature_importances_),
            key=lambda kv: -kv[1],
        )[:10]
    }
    return out


def run(db_path: str, thr: float, alpha: float) -> Dict[str, Any]:
    X, lr, ind, _, fps = _build_corpus(db_path)
    n = len(X)
    cut = int(n * 0.8)
    Xtr, Xte = X[:cut], X[cut:]
    lrtr = lr[:cut]
    indtr, indte = ind[:cut], ind[cut:]

    # Foundation: per-feature forward Spearman with induction (test slice).
    feat_corr: Dict[str, float] = {}
    for j, name in enumerate(_FEATURE_NAMES):
        if np.std(Xte[:, j]) < 1e-12:
            feat_corr[name] = 0.0
            continue
        r, _ = _spearman(Xte[:, j], indte)
        feat_corr[name] = r if np.isfinite(r) else 0.0
    top_feats = sorted(feat_corr.items(), key=lambda kv: -abs(kv[1]))[:8]

    results: Dict[str, Any] = {}

    # A. baseline-as-deployed: Ridge -> loss_ratio, rank by -loss_ratio.
    w, b, m, s = _ridge_fit(Xtr, lrtr, alpha)
    pred_lr = _ridge_predict(Xte, w, b, m, s)
    results["A_ridge_loss_ratio_as_deployed"] = _ranking_metrics(-pred_lr, indte, thr)

    # B. same features, fixed target: Ridge -> induction.
    w, b, m, s = _ridge_fit(Xtr, indtr, alpha)
    pred_b = _ridge_predict(Xte, w, b, m, s)
    results["B_ridge_induction"] = _ranking_metrics(pred_b, indte, thr)

    # C. fixed target + nonlinear, 18 fingerprint features.
    results["C_gbm_induction_fingerprint"] = _eval_gbm(
        Xtr, indtr, Xte, indte, thr, _FEATURE_NAMES
    )

    # D1. + op-presence features (pure architecture, no probes).
    X_ops, op_names = _op_presence_features(db_path, fps)
    Xd1 = np.hstack([X, X_ops])
    names_d1 = [*_FEATURE_NAMES, *op_names]
    results["D1_gbm_induction_fingerprint_plus_ops"] = _eval_gbm(
        Xd1[:cut], indtr, Xd1[cut:], indte, thr, names_d1
    )

    # D2. + cheap sibling probes (binding/ar; never the induction target).
    X_probes, probe_names = _sibling_probe_features(db_path, fps)
    Xd2 = np.hstack([Xd1, X_probes])
    names_d2 = [*names_d1, *probe_names]
    results["D2_gbm_induction_plus_ops_plus_probes"] = _eval_gbm(
        Xd2[:cut], indtr, Xd2[cut:], indte, thr, names_d2
    )

    return {
        "n_total": n,
        "n_train": cut,
        "n_test": n - cut,
        "split": "temporal_80_20",
        "induction_threshold": thr,
        "n_op_features": len(op_names),
        "train_induction_mean": float(indtr.mean()),
        "test_induction_mean": float(indte.mean()),
        "foundation_top_feature_spearman_with_induction": dict(top_feats),
        "models": results,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(RUNS_DB))
    parser.add_argument("--thr", type=float, default=0.35)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--out",
        default="research/reports/induction_predictor_foundation.json",
    )
    args = parser.parse_args()

    report = run(args.db, args.thr, args.alpha)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
