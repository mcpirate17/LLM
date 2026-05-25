#!/usr/bin/env python
"""How many properties is right? Sweep #features vs forward-generalization (temporal ROC).

Answers "is N properties enough or do we need more?" empirically. Rank the backfilled
semantic features by GBM gain importance (on the temporal-train split), then evaluate
temporal-holdout ROC using the top-k features for a sweep of k. The curve shows whether
generalization keeps improving with more properties or peaks and then degrades (overfit) —
i.e. whether the answer is "add more" or "select a subset".

Usage::  python -m research.tools.feature_count_sweep
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Dict, List

import numpy as np

from research.defaults import RUNS_DB
from research.tools.backfill_graph_semantics import load_semantic_features
from research.tools.capability_shrinkage_denoise import (
    _NONE_CLUSTER,
    _shrink,
    _template_map,
)
from research.tools.induction_predictor_foundation import (
    _build_corpus,
    _fit_gbm,
    _ranking_metrics,
)

logger = logging.getLogger(__name__)


def run(db_path: str, thr: float, shrink_f: float, ks: List[int]) -> Dict[str, Any]:
    _, _, ind_all, _, fps_all = _build_corpus(db_path)
    X, names, present = load_semantic_features(fps_all)
    pos = {fp: i for i, fp in enumerate(fps_all)}
    ind = ind_all[np.array([pos[fp] for fp in present])]
    tmpl = _template_map(db_path)
    clusters = [tmpl.get(fp, _NONE_CLUSTER) for fp in present]

    n = len(present)
    cut = int(n * 0.8)
    train_mask = np.zeros(n, dtype=bool)
    train_mask[:cut] = True
    y = _shrink(ind, clusters, train_mask, shrink_f)

    # Rank features by gain importance from a model trained on the temporal-train split.
    full = _fit_gbm(X[:cut], y[:cut])
    order = np.argsort(-np.asarray(full.feature_importances_, dtype=np.float64))
    ranked = [names[i] for i in order]

    sweep = []
    for k in ks:
        k = min(k, X.shape[1])
        cols = order[:k]
        model = _fit_gbm(X[:cut][:, cols], y[:cut])
        pred = np.asarray(model.predict(X[cut:][:, cols]), dtype=np.float64)
        m = _ranking_metrics(pred, ind[cut:], thr)
        sweep.append(
            {
                "k_features": int(k),
                "temporal_roc": round(m["roc_auc_gt_thr"] or 0.0, 4),
                "temporal_spearman": round(m["spearman_rho"], 4),
            }
        )
    best = max(sweep, key=lambda r: r["temporal_roc"])
    return {
        "n_graphs": n,
        "total_features": X.shape[1],
        "top_15_features_by_importance": ranked[:15],
        "sweep": sweep,
        "best_k": best["k_features"],
        "best_temporal_roc": best["temporal_roc"],
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=str(RUNS_DB))
    p.add_argument("--thr", type=float, default=0.35)
    p.add_argument("--shrink", type=float, default=0.75)
    p.add_argument("--ks", default="5,10,15,20,30,40,60,80,113")
    args = p.parse_args()
    ks = [int(x) for x in args.ks.split(",")]
    print(json.dumps(run(args.db, args.thr, args.shrink, ks), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
