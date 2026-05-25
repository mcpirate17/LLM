#!/usr/bin/env python
"""Head-to-head: NAME-based vs MATH+STRUCTURE features for predicting induction.

Tests the core thesis — representing ops by what they COMPUTE (graph_semantic_features,
backfilled) generalizes to novel ops better than opaque op-name one-hots. On the labeled
induction corpus, trains a GBM on each representation and compares:
  - out_of_fold (random 5-fold) — ranking within the known distribution.
  - temporal (first 80% train / last 20% future) — forward generalization = the novelty proxy.

If math+structure lifts the TEMPORAL ROC over names, it generalizes to new architectures better.

Usage::  python -m research.tools.compare_feature_representations
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Dict, List

import numpy as np

from research.defaults import RUNS_DB
from research.tools.backfill_graph_semantics import load_semantic_features
from research.tools.capability_screener import _static_matrix
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


def _eval_split(
    X: np.ndarray, ind: np.ndarray, clusters: List[str], rest, held, thr, shrink_f
):
    train_mask = np.zeros(len(ind), dtype=bool)
    train_mask[rest] = True
    y = _shrink(ind, clusters, train_mask, shrink_f)
    model = _fit_gbm(X[rest], y[rest])
    pred = np.asarray(model.predict(X[held]), dtype=np.float64)
    return _ranking_metrics(pred, ind[held], thr)


def _oof(X, ind, clusters, thr, shrink_f, k=5):
    rng = np.random.default_rng(42)
    folds = np.array_split(rng.permutation(len(ind)), k)
    oof = np.zeros(len(ind), dtype=np.float64)
    for f in range(k):
        held = folds[f]
        rest = np.concatenate([folds[j] for j in range(k) if j != f])
        tmask = np.zeros(len(ind), dtype=bool)
        tmask[rest] = True
        y = _shrink(ind, clusters, tmask, shrink_f)
        oof[held] = np.asarray(
            _fit_gbm(X[rest], y[rest]).predict(X[held]), dtype=np.float64
        )
    return _ranking_metrics(oof, ind, thr)


def run(db_path: str, thr: float, shrink_f: float) -> Dict[str, Any]:
    _, _, ind_all, _, fps_all = _build_corpus(db_path)
    # Intersect corpus with graphs that have backfilled semantic features.
    Xsem, _, present = load_semantic_features(fps_all)
    pos = {fp: i for i, fp in enumerate(fps_all)}
    idx = np.array([pos[fp] for fp in present])
    ind = ind_all[idx]
    fps = present
    # Name-based features for the SAME graphs (op-presence + op_count + pair_count).
    Xname, _, _ = _static_matrix(db_path, fps)
    tmpl = _template_map(db_path)
    clusters = [tmpl.get(fp, _NONE_CLUSTER) for fp in fps]

    n = len(fps)
    cut = int(n * 0.8)
    rest, held = np.arange(cut), np.arange(cut, n)
    out: Dict[str, Any] = {
        "n_graphs": n,
        "n_name_features": Xname.shape[1],
        "n_semantic_features": Xsem.shape[1],
        "induction_threshold": thr,
    }
    for label, X in (("name_based", Xname), ("math_structure", Xsem)):
        oof_m = _oof(X, ind, clusters, thr, shrink_f)
        tmp_m = _eval_split(X, ind, clusters, rest, held, thr, shrink_f)
        out[label] = {
            "out_of_fold_roc": round(oof_m["roc_auc_gt_thr"] or 0.0, 4),
            "out_of_fold_spearman": round(oof_m["spearman_rho"], 4),
            "temporal_roc": round(tmp_m["roc_auc_gt_thr"] or 0.0, 4),
            "temporal_spearman": round(tmp_m["spearman_rho"], 4),
        }
    out["temporal_roc_delta_semantic_minus_name"] = round(
        out["math_structure"]["temporal_roc"] - out["name_based"]["temporal_roc"], 4
    )
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=str(RUNS_DB))
    p.add_argument("--thr", type=float, default=0.35)
    p.add_argument("--shrink", type=float, default=0.75)
    args = p.parse_args()
    print(json.dumps(run(args.db, args.thr, args.shrink), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
