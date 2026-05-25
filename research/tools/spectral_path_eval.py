#!/usr/bin/env python
"""Does spectral-path abstraction add predictive signal beyond the 113-feature v2 rep?

Plugs the ``spath_*`` block (spectral_path_features.py) into the EXACT backtest machinery the
deployed oracle uses, per capability axis:

  - incremental value: LightGBM ranking ROC@thr (OOF 5-fold + temporal 80/20) for
    {base 113, spath only, base ⊕ spath}. A positive base→base+spath delta is the green light to
    merge the block into graph_semantic_features.py (bump FEATURE_VERSION + backfill).
  - novelty: is the confirmed STDP winner `e656938e359ada50` flagged OOD (kNN-distance percentile)
    in spath-space — i.e. does the spectral-path rep see it as novel where labels can't predict it?

Reuses the oracle corpus (_axis_corpus) and the shared GBM/metrics helpers — no reimplementation.

Usage::  python -m research.tools.spectral_path_eval [--db runs.db] [--meta meta_analysis.db]
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.model_selection import KFold

from research.defaults import RUNS_DB
from research.tools.induction_predictor_foundation import (
    _fingerprint_timestamps,
    _fit_gbm,
    _ranking_metrics,
)
from research.tools.novelty_scorer import NoveltyScorer
from research.tools.pls_partition_oracle import AXES, _axis_corpus
from research.tools.spectral_path_features import (
    FEATURE_VERSION,
    SpectralPathExtractor,
)

logger = logging.getLogger(__name__)
# confirmed novel STDP-attention winner, induction 0.894 (not a secret)
_STDP_WINNER = "e656938e359ada50"  # pragma: allowlist secret


def _spath_matrix(
    db_path: str, meta_db: str, fps: List[str]
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """spath feature matrix aligned to ``fps``; returns (X, names, keep_mask)."""
    ext = SpectralPathExtractor(db_path, meta_db)
    con = sqlite3.connect(db_path)
    gj: Dict[str, Any] = {}
    qm = ",".join("?" * len(fps))
    for fp, blob in con.execute(
        f"SELECT graph_fingerprint, graph_json FROM graphs WHERE graph_fingerprint IN ({qm})",  # nosec B608
        fps,
    ):
        try:
            gj[str(fp)] = json.loads(blob)["nodes"]
        except Exception:
            pass
    con.close()
    feats: Dict[str, Dict[str, float]] = {}
    for fp in fps:
        if fp in gj:
            feats[fp] = ext.features(gj[fp])
    if not feats:
        return np.zeros((len(fps), 0)), [], np.zeros(len(fps), dtype=bool)
    names = sorted(next(iter(feats.values())).keys())
    keep = np.array([fp in feats for fp in fps], dtype=bool)
    X = np.array(
        [[feats[fp].get(n, 0.0) for n in names] for fp in fps if fp in feats],
        dtype=np.float64,
    )
    return X, names, keep


def _oof_scores(X: np.ndarray, y: np.ndarray, seed: int = 42) -> np.ndarray:
    """Out-of-fold GBM predictions (5-fold), so every row is scored by a model that didn't see it."""
    oof = np.zeros(len(y))
    for fit_ix, hold_ix in KFold(n_splits=5, shuffle=True, random_state=seed).split(X):
        oof[hold_ix] = _fit_gbm(X[fit_ix], y[fit_ix]).predict(X[hold_ix])
    return oof


def _temporal_roc(X: np.ndarray, y: np.ndarray, thr: float) -> Any:
    """Forward-in-time 80/20 (corpus is oldest-first): train past, score future."""
    cut = int(len(y) * 0.8)
    score = _fit_gbm(X[:cut], y[:cut]).predict(X[cut:])
    return _ranking_metrics(score, y[cut:], thr)["roc_auc_gt_thr"]


def _eval_axis(
    db_path: str,
    meta_db: str,
    axis: str,
    thr: float,
    winner_nodes: Any = None,
) -> Dict[str, Any]:
    ts = _fingerprint_timestamps(db_path)
    corpus = _axis_corpus(db_path, AXES[axis][0], ts)
    Xsp, sp_names, keep = _spath_matrix(db_path, meta_db, corpus.fps)
    Xb = corpus.X[keep]
    y = corpus.y[keep]
    Xcat = np.hstack([Xb, Xsp])
    out: Dict[str, Any] = {
        "axis": axis,
        "threshold": thr,
        "n": int(len(y)),
        "n_capable": int((y > thr).sum()),
        "n_base_feats": Xb.shape[1],
        "n_spath_feats": Xsp.shape[1],
    }
    for tag, X in (("base", Xb), ("spath", Xsp), ("base+spath", Xcat)):
        oof = _ranking_metrics(_oof_scores(X, y), y, thr)
        out[tag] = {
            "oof_roc": _round(oof["roc_auc_gt_thr"]),
            "oof_rho": _round(oof["spearman_rho"]),
            "temporal_roc": _round(_temporal_roc(X, y, thr)),
        }
    out["delta_oof_roc_base_to_cat"] = _round(
        _sub(out["base+spath"]["oof_roc"], out["base"]["oof_roc"])
    )
    out["delta_temporal_roc_base_to_cat"] = _round(
        _sub(out["base+spath"]["temporal_roc"], out["base"]["temporal_roc"])
    )
    out["spath_novelty"] = _novelty_check(db_path, meta_db, Xsp, sp_names, winner_nodes)
    return out


def _novelty_check(
    db_path: str,
    meta_db: str,
    Xsp: np.ndarray,
    names: List[str],
    winner_nodes: Any,
) -> Dict[str, Any]:
    """Percentile of the STDP winner's kNN-distance novelty in spath-space (OOD ⇒ → probe)."""
    if Xsp.shape[1] == 0:
        return {"available": False}
    scorer = NoveltyScorer(Xsp, names, k=10)
    in_dist_p = float(np.median([scorer.percentile(v) for v in scorer.novelty(Xsp)]))
    base = {"fingerprint": _STDP_WINNER, "in_dist_median_pctile": _round(in_dist_p)}
    if (
        winner_nodes is None
    ):  # winner is a generated candidate; pass --with-winner to reconstruct
        return {
            **base,
            "available": False,
            "note": "run --with-winner to reconstruct the graph",
        }
    v = SpectralPathExtractor(db_path, meta_db).features(winner_nodes)
    x = np.array([[v.get(n, 0.0) for n in names]], dtype=np.float64)
    nov = float(scorer.novelty(x)[0])
    return {
        **base,
        "available": True,
        "novelty": _round(nov),
        "percentile": _round(scorer.percentile(nov)),
    }


def _load_winner_nodes(db_path: str) -> Any:
    """Reconstruct the STDP winner's node dict by regenerating the probe pool (oracle's mechanism)."""
    from research.tools.probe_novel_candidates import _collect_pool

    for c in _collect_pool(db_path, 600, 12000, 3_000_000):
        if c["fingerprint"] == _STDP_WINNER:
            return c["graph"].to_dict()["nodes"]
    return None


def _round(v: Any) -> Any:
    return round(float(v), 4) if isinstance(v, (int, float)) and v is not None else v


def _sub(a: Any, b: Any) -> Any:
    return (
        (a - b) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
    )


def run(db_path: str, meta_db: str, with_winner: bool = False) -> Dict[str, Any]:
    winner = _load_winner_nodes(db_path) if with_winner else None
    return {
        "spath_version": FEATURE_VERSION,
        "axes": {
            axis: _eval_axis(db_path, meta_db, axis, thr, winner)
            for axis, (_, thr) in AXES.items()
        },
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=str(RUNS_DB))
    p.add_argument("--meta", default="research/meta_analysis.db")
    p.add_argument(
        "--with-winner",
        action="store_true",
        help="reconstruct the STDP winner via the probe pool to score its spath-novelty",
    )
    args = p.parse_args()
    print(
        json.dumps(run(args.db, args.meta, args.with_winner), indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
