#!/usr/bin/env python
"""Recursive partitioning + PLS2 for TWO capability axes (induction & AR/reasoning).

The "recursive partition for two" step: do induction (retrieval) and ar_curriculum (reasoning) have
DIFFERENT property signatures? Uses the two classical multi-target tools:

  - PLS2 (multi-target PLS): one latent decomposition of properties onto BOTH targets; reports the
    per-axis variance explained and which properties load toward induction vs toward AR.
  - multi-output recursive partitioning (CART): an interpretable tree predicting both axes; per-axis
    feature importances + cross-validated ROC, showing whether the splits differ by axis.

Operates on graphs that have semantic features AND both labels (induction_screening_auc,
ar_curriculum_auc_pair_final). Standardized properties; 5-fold CV ROC per axis.

Usage::  python -m research.tools.recursive_partition_capability
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from research.defaults import RUNS_DB
from research.tools.backfill_graph_semantics import load_semantic_features


def _both_axis_corpus(db_path: str):
    con = sqlite3.connect(db_path)
    ind: Dict[str, List[float]] = defaultdict(list)
    ar: Dict[str, List[float]] = defaultdict(list)
    for fp, v in con.execute(
        "SELECT graph_fingerprint, induction_screening_auc FROM graph_runs "
        "WHERE induction_screening_auc IS NOT NULL"
    ):
        ind[str(fp)].append(float(v))
    for fp, v in con.execute(
        "SELECT graph_fingerprint, ar_curriculum_auc_pair_final FROM graph_runs "
        "WHERE ar_curriculum_auc_pair_final IS NOT NULL"
    ):
        ar[str(fp)].append(float(v))
    con.close()
    fps = sorted(set(ind) & set(ar))
    X, names, present = load_semantic_features(fps)
    y_ind = np.array([np.mean(ind[fp]) for fp in present])
    y_ar = np.array([np.mean(ar[fp]) for fp in present])
    return X, names, np.column_stack([y_ind, y_ar]), present


def _axis_loadings(names, weights, top=8):
    return {
        n: round(float(w), 3)
        for n, w in sorted(zip(names, weights), key=lambda kv: -abs(kv[1]))[:top]
    }


def run(db_path: str, ind_thr: float, ar_thr: float, tree_depth: int) -> Dict[str, Any]:
    X, names, Y, present = _both_axis_corpus(db_path)
    Xz = np.asarray(StandardScaler().fit_transform(X))
    cap = np.column_stack([Y[:, 0] > ind_thr, Y[:, 1] > ar_thr]).astype(int)

    # PLS2: shared latent decomposition onto both axes.
    pls = PLSRegression(n_components=8).fit(Xz, Y)
    xw = np.asarray(pls.x_weights_)
    yld = np.asarray(
        pls.y_loadings_
    )  # [2 axes, n_comp]: which axis each component serves
    pls_block = {
        "n_components": 8,
        "comp1_axis_loadings": {
            "induction": round(float(yld[0, 0]), 3),
            "ar": round(float(yld[1, 0]), 3),
        },
        "comp2_axis_loadings": {
            "induction": round(float(yld[0, 1]), 3),
            "ar": round(float(yld[1, 1]), 3),
        },
        "component1_top_properties": _axis_loadings(names, xw[:, 0]),
        "component2_top_properties": _axis_loadings(names, xw[:, 1]),
    }

    # Multi-output recursive partitioning, with per-axis CV ROC + importances.
    tree = DecisionTreeRegressor(
        max_depth=tree_depth, min_samples_leaf=20, random_state=42
    )
    oof = cross_val_predict(tree, X, Y, cv=5)
    tree.fit(X, Y)
    axes = {}
    for j, axis in enumerate(("induction", "ar_curriculum")):
        c = cap[:, j]
        roc = float(roc_auc_score(c, oof[:, j])) if 0 < c.sum() < len(c) else None
        # per-axis importance: refit single-output tree for clean importances
        t1 = DecisionTreeRegressor(
            max_depth=tree_depth, min_samples_leaf=20, random_state=42
        ).fit(X, Y[:, j])
        imp = sorted(zip(names, t1.feature_importances_), key=lambda kv: -kv[1])[:6]
        axes[axis] = {
            "n_capable": int(c.sum()),
            "cv_roc": round(roc, 4) if roc else None,
            "top_features": {n: round(float(w), 3) for n, w in imp if w > 0},
        }
    return {
        "n_graphs": len(present),
        "induction_threshold": ind_thr,
        "ar_threshold": ar_thr,
        "pls2": pls_block,
        "recursive_partition_per_axis": axes,
        "note": "Different top features per axis => induction and AR need distinct property signatures.",
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=str(RUNS_DB))
    p.add_argument("--ind-thr", type=float, default=0.35)
    p.add_argument("--ar-thr", type=float, default=0.5)
    p.add_argument("--tree-depth", type=int, default=4)
    args = p.parse_args()
    print(
        json.dumps(
            run(args.db, args.ind_thr, args.ar_thr, args.tree_depth),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
