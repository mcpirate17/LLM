#!/usr/bin/env python
"""Can we predict the UNKNOWN? Leave-one-op-out extrapolation test.

The real question behind the discovery program: can a model predict the capability of an
op it has NEVER trained on? Supervised labels alone cannot — a name one-hot for an unseen
op is always zero. But if ops are represented by WHAT THEY COMPUTE (math + structure), a
never-seen op still has a full property vector, and the model can borrow from
mathematically-similar ops it HAS seen.

Test: for each target op, hold out every graph containing it (simulate the op being
unknown), train on the rest, predict the held-out graphs. Compare name-based vs
math+structure features by how well each ranks the actually-capable held-out graphs.
If math+structure >> names on held-out novel ops, the property space extrapolates.

Usage::  python -m research.tools.leave_op_out_test --ops stdp_attention,tropical_attention
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from collections import defaultdict
from typing import Any, Dict, List, Set

import numpy as np

from research.defaults import RUNS_DB
from research.tools.backfill_graph_semantics import load_semantic_features
from research.tools.capability_screener import _static_matrix
from research.tools.capability_shrinkage_denoise import (
    _NONE_CLUSTER,
    _shrink,
    _template_map,
)
from research.tools.induction_predictor_foundation import _build_corpus, _fit_gbm

logger = logging.getLogger(__name__)


def _op_presence(db_path: str, fps: List[str]) -> Dict[str, Set[str]]:
    con = sqlite3.connect(db_path)
    fp_set = set(fps)
    out: Dict[str, Set[str]] = defaultdict(set)
    for fp, op in con.execute(
        "SELECT graph_fingerprint, op_name FROM program_graph_ops"
    ):
        if str(fp) in fp_set:
            out[str(fp)].add(str(op))
    con.close()
    return out


def _capable_recall_at_k(pred: np.ndarray, capable: np.ndarray, k: int) -> float:
    """Fraction of truly-capable held-out graphs found in the top-k predicted."""
    if capable.sum() == 0:
        return float("nan")
    top = set(np.argsort(-pred)[:k].tolist())
    return float(sum(1 for i in np.where(capable)[0] if i in top) / capable.sum())


def run(db_path: str, ops: List[str], thr: float, shrink_f: float) -> Dict[str, Any]:
    _, _, ind_all, _, fps_all = _build_corpus(db_path)
    Xsem, _, present = load_semantic_features(fps_all)
    pos = {fp: i for i, fp in enumerate(fps_all)}
    idx = np.array([pos[fp] for fp in present])
    ind = ind_all[idx]
    fps = present
    Xname, _, _ = _static_matrix(db_path, fps)
    tmpl = _template_map(db_path)
    clusters = [tmpl.get(fp, _NONE_CLUSTER) for fp in fps]
    presence = _op_presence(db_path, fps)
    n = len(fps)

    results: Dict[str, Any] = {}
    for op in ops:
        held = np.array(
            [i for i, fp in enumerate(fps) if op in presence.get(fp, set())]
        )
        if len(held) < 8:
            results[op] = {"error": f"only {len(held)} held-out graphs"}
            continue
        train = np.array([i for i in range(n) if i not in set(held.tolist())])
        capable = ind[held] > thr
        train_mask = np.zeros(n, dtype=bool)
        train_mask[train] = True
        y = _shrink(ind, clusters, train_mask, shrink_f)
        row: Dict[str, Any] = {
            "n_held_out": int(len(held)),
            "n_capable_held_out": int(capable.sum()),
            "held_out_max_induction": round(float(ind[held].max()), 4),
        }
        for label, X in (("name_based", Xname), ("math_structure", Xsem)):
            model = _fit_gbm(X[train], y[train])
            pred = np.asarray(model.predict(X[held]), dtype=np.float64)
            from sklearn.metrics import roc_auc_score

            roc = (
                round(float(roc_auc_score(capable, pred)), 4)
                if 0 < capable.sum() < len(capable)
                else None
            )
            row[label] = {
                "roc": roc,
                "recall_at_10pct": round(
                    _capable_recall_at_k(pred, capable, max(len(held) // 10, 1)), 3
                ),
                "top_pred_on_capable": round(float(pred[capable].max()), 4)
                if capable.sum()
                else None,
                "mean_pred_capable": round(float(pred[capable].mean()), 4)
                if capable.sum()
                else None,
                "mean_pred_incapable": round(float(pred[~capable].mean()), 4)
                if (~capable).sum()
                else None,
            }
        results[op] = row
    return {"n_corpus": n, "threshold": thr, "leave_op_out": results}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=str(RUNS_DB))
    p.add_argument(
        "--ops",
        default="stdp_attention,tropical_attention,ultrametric_attention,clifford_attention,gated_progressive_attention",
    )
    p.add_argument("--thr", type=float, default=0.35)
    p.add_argument("--shrink", type=float, default=0.75)
    args = p.parse_args()
    report = run(
        args.db,
        [o.strip() for o in args.ops.split(",") if o.strip()],
        args.thr,
        args.shrink,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
