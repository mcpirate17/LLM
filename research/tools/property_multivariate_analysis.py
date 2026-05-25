#!/usr/bin/env python
"""Proper multivariate analysis of the property -> capability relationship.

Analysis FIRST (the step skipped earlier), with the classical tools for many correlated predictors:

  1. PCA  — unsupervised structure: intrinsic dimensionality of the property space, top loadings.
  2. PLS  — supervised latent regression (properties -> induction): cross-validated Q^2 vs #components
            (the principled "how many dimensions" answer), X/Y variance explained, component loadings
            (which property combinations covary with capability). PLS handles correlated, high-dim
            predictors by projection — it does NOT overfit with more features the way the GBM did.
  3. CART — a single interpretable decision tree (the actual property-threshold rules) + temporal ROC.

Compares PLS and the tree on the temporal holdout vs the GBM reference (0.89). Uses the v2 semantic
features (backfilled). Recursive partitioning for two capability axes (induction + AR) is wired but
needs the AR labels (Colab pending) — induction is analyzed in full here.

Usage::  python -m research.tools.property_multivariate_analysis
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Dict, List

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from research.defaults import RUNS_DB
from research.tools.backfill_graph_semantics import load_semantic_features
from research.tools.induction_predictor_foundation import _build_corpus

logger = logging.getLogger(__name__)


def _pca_block(Xz: np.ndarray, names: List[str]) -> Dict[str, Any]:
    pca = PCA().fit(Xz)
    cum = np.cumsum(pca.explained_variance_ratio_)
    load1 = sorted(
        zip(names, np.asarray(pca.components_)[0]), key=lambda kv: -abs(kv[1])
    )[:8]
    return {
        "n_features": Xz.shape[1],
        "n_components_for_90pct_var": int(np.searchsorted(cum, 0.90) + 1),
        "n_components_for_95pct_var": int(np.searchsorted(cum, 0.95) + 1),
        "top_pc1_loadings": {n: round(float(w), 3) for n, w in load1},
    }


def _pls_block(Xz_tr, y_tr, Xz_te, cap_te, names, max_k) -> Dict[str, Any]:
    # Cross-validated Q^2 (predictive R^2) vs #latent components — does it plateau, not degrade?
    kf = KFold(5, shuffle=True, random_state=42)
    q2_curve = []
    for k in range(1, max_k + 1):
        errs, tots = [], []
        for tr_i, va_i in kf.split(Xz_tr):
            pls = PLSRegression(n_components=k).fit(Xz_tr[tr_i], y_tr[tr_i])
            pred = pls.predict(Xz_tr[va_i]).ravel()
            errs.append(((y_tr[va_i] - pred) ** 2).sum())
            tots.append(((y_tr[va_i] - y_tr[tr_i].mean()) ** 2).sum())
        q2 = 1.0 - sum(errs) / max(sum(tots), 1e-9)
        q2_curve.append({"k": k, "q2": round(float(q2), 4)})
    best = max(q2_curve, key=lambda r: r["q2"])
    pls = PLSRegression(n_components=best["k"]).fit(Xz_tr, y_tr)
    pred_te = pls.predict(Xz_te).ravel()
    roc = (
        float(roc_auc_score(cap_te, pred_te))
        if 0 < cap_te.sum() < len(cap_te)
        else None
    )
    load1 = sorted(
        zip(names, np.asarray(pls.x_weights_)[:, 0]), key=lambda kv: -abs(kv[1])
    )[:10]
    return {
        "q2_vs_components": q2_curve,
        "best_n_components": best["k"],
        "best_cv_q2": best["q2"],
        "temporal_roc_at_best_k": round(roc, 4) if roc else None,
        "component1_top_loadings": {n: round(float(w), 3) for n, w in load1},
    }


def _tree_block(Xtr, cap_tr, Xte, cap_te, names, max_depth) -> Dict[str, Any]:
    tree = DecisionTreeClassifier(
        max_depth=max_depth, min_samples_leaf=20, random_state=42
    )
    tree.fit(Xtr, cap_tr)
    proba = tree.predict_proba(Xte)[:, 1]
    roc = (
        float(roc_auc_score(cap_te, proba)) if 0 < cap_te.sum() < len(cap_te) else None
    )
    imp = sorted(zip(names, tree.feature_importances_), key=lambda kv: -kv[1])[:8]
    return {
        "temporal_roc": round(roc, 4) if roc else None,
        "top_features": {n: round(float(w), 3) for n, w in imp if w > 0},
        "rules": export_text(
            tree, feature_names=list(names), max_depth=max_depth
        ).split("\n"),
    }


def run(db_path: str, thr: float, max_pls_k: int, tree_depth: int) -> Dict[str, Any]:
    _, _, ind_all, _, fps_all = _build_corpus(db_path)
    X, names, present = load_semantic_features(fps_all)
    pos = {fp: i for i, fp in enumerate(fps_all)}
    ind = ind_all[np.array([pos[fp] for fp in present])]
    n = len(present)
    cut = int(n * 0.8)
    scaler = StandardScaler().fit(X[:cut])
    Xz = np.asarray(scaler.transform(X))
    cap = (ind > thr).astype(int)

    return {
        "n_graphs": n,
        "induction_threshold": thr,
        "gbm_temporal_roc_reference": 0.89,
        "pca": _pca_block(Xz[:cut], names),
        "pls": _pls_block(Xz[:cut], ind[:cut], Xz[cut:], cap[cut:], names, max_pls_k),
        "decision_tree": _tree_block(
            X[:cut], cap[:cut], X[cut:], cap[cut:], names, tree_depth
        ),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=str(RUNS_DB))
    p.add_argument("--thr", type=float, default=0.35)
    p.add_argument("--max-pls-k", type=int, default=25)
    p.add_argument("--tree-depth", type=int, default=4)
    args = p.parse_args()
    print(
        json.dumps(
            run(args.db, args.thr, args.max_pls_k, args.tree_depth),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
