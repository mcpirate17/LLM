#!/usr/bin/env python
"""The flywheel proof: do MINTED probe labels let us predict held-out novel architectures?

score_probed_with_reps.py showed historical-only models (names OR math) cannot predict the
probed unknown-good — no good novel exemplar exists in the historical labels. This tests the
fix: mint real labels by probing novel candidates, add them to training, and check whether the
model can now predict HELD-OUT novel graphs it still never trained on. Two axes:

  - condition: historical_only  vs  historical + minted_train
  - representation: op-names  vs  math+structure (graph_semantic_features)

If (historical+minted) >> (historical_only) on minted-held, minting labels enables novel-region
prediction. If math+structure's gain > names' gain, rich properties transfer each minted label
further (one probed stdp teaches a spiking/stateful neighborhood). That is the framework.

Consumes minted_labels.json (from probe_novel_candidates batch). Regenerates graphs to featurize.

Usage::  python -m research.tools.flywheel_transfer_demo
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from research.defaults import RUNS_DB
from research.tools.backfill_graph_semantics import load_semantic_features
from research.tools.capability_screener import _static_matrix, featurize_op_sets
from research.tools.capability_shrinkage_denoise import (
    _NONE_CLUSTER,
    _shrink,
    _template_map,
)
from research.tools.graph_semantic_features import GraphSemanticExtractor
from research.tools.induction_predictor_foundation import _build_corpus, _fit_gbm
from research.tools.probe_novel_candidates import _collect_pool

logger = logging.getLogger(__name__)


def _corr(a: np.ndarray, b: np.ndarray):
    if len(a) > 2 and a.std() > 0 and b.std() > 0:
        return round(float(np.corrcoef(a, b)[0, 1]), 3)
    return None


def _roc(pred: np.ndarray, capable: np.ndarray):
    from sklearn.metrics import roc_auc_score

    if 0 < capable.sum() < len(capable):
        return round(float(roc_auc_score(capable, pred)), 3)
    return None


def _historical(db_path: str):
    _, _, ind_all, _, fps_all = _build_corpus(db_path)
    Xsem, sem_names, present = load_semantic_features(fps_all)
    pos = {fp: i for i, fp in enumerate(fps_all)}
    ind = ind_all[np.array([pos[fp] for fp in present])]
    Xname, _, name_vocab = _static_matrix(db_path, present)
    tmpl = _template_map(db_path)
    clusters = [tmpl.get(fp, _NONE_CLUSTER) for fp in present]
    return Xname, Xsem, ind, clusters, name_vocab, sem_names


def _mint_features(
    db_path: str, minted: Dict[str, float], name_vocab: List[str], sem_names: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    cand = _collect_pool(db_path, pool=600, max_attempts=12000, seed0=3_000_000)
    by = {c["fingerprint"]: c for c in cand}
    ext = GraphSemanticExtractor(db_path)
    Xn, Xs, y, fps = [], [], [], []
    for fp, actual in minted.items():
        c = by.get(fp)
        if c is None:
            continue
        Xn.append(
            featurize_op_sets(
                [set(c["ops"])], [c["op_count"]], [c["pair_count"]], name_vocab
            )[0]
        )
        feats = ext.features(c["graph"].to_dict()["nodes"])
        Xs.append([feats.get(n, 0.0) for n in sem_names])
        y.append(float(actual))
        fps.append(fp)
    return np.array(Xn), np.array(Xs), np.array(y), fps


def run(
    db_path: str, minted_file: str, thr: float, shrink_f: float, seed: int
) -> Dict[str, Any]:
    minted = {
        r["fingerprint"]: r["actual_induction_auc"]
        for r in json.loads(Path(minted_file).read_text())["results"]
        if r.get("actual_induction_auc") is not None
    }
    Xn_h, Xs_h, ind_h, clusters_h, name_vocab, sem_names = _historical(db_path)
    Xn_m, Xs_m, y_m, _ = _mint_features(db_path, minted, name_vocab, sem_names)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y_m))
    cut = int(len(order) * 0.5)
    tr_m, te_m = order[:cut], order[cut:]
    capable = y_m[te_m] > thr

    # shrink historical target once (full historical as train cluster basis)
    y_h = _shrink(ind_h, clusters_h, np.ones(len(ind_h), dtype=bool), shrink_f)

    out: Dict[str, Any] = {
        "n_minted": len(y_m),
        "n_minted_capable": int((y_m > thr).sum()),
        "n_held": len(te_m),
        "n_held_capable": int(capable.sum()),
        "minted_max_induction": round(float(y_m.max()), 4) if len(y_m) else None,
    }
    for rep, Xh, Xm in (("name", Xn_h, Xn_m), ("math_structure", Xs_h, Xs_m)):
        # historical only
        m_hist = _fit_gbm(Xh, y_h)
        p_hist = np.asarray(m_hist.predict(Xm[te_m]), dtype=np.float64)
        # historical + minted-train
        Xaug = np.vstack([Xh, Xm[tr_m]])
        yaug = np.concatenate([y_h, y_m[tr_m]])
        m_aug = _fit_gbm(Xaug, yaug)
        p_aug = np.asarray(m_aug.predict(Xm[te_m]), dtype=np.float64)
        out[rep] = {
            "historical_only": {
                "corr": _corr(p_hist, y_m[te_m]),
                "roc": _roc(p_hist, capable),
            },
            "historical_plus_minted": {
                "corr": _corr(p_aug, y_m[te_m]),
                "roc": _roc(p_aug, capable),
            },
        }
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=str(RUNS_DB))
    p.add_argument("--minted", default="research/reports/minted_labels.json")
    p.add_argument("--thr", type=float, default=0.35)
    p.add_argument("--shrink", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    print(
        json.dumps(
            run(args.db, args.minted, args.thr, args.shrink, args.seed),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
