#!/usr/bin/env python
"""Standalone capability screener — rank graphs by predicted induction, at scale.

NOT wired into the pipeline. Trains a GBM to predict (cluster-shrunk) induction
capability and persists it so millions of *un-probed* candidate graphs can be
triaged cheaply, before spending any probe compute.

The decisive design constraint for scale: which features are FREE (parseable from
the graph definition, no forward pass) vs. which need the graph actually run.
  - FREE / static : op-presence + op_count + pair_count   (program_graph_{ops,features})
  - NOT free      : the fingerprint metrics (jacobian_*, cka_*, ...) need a forward pass.
`train` reports held-out ROC for BOTH so we know the cost of going fingerprint-free;
the persisted screener is the static one (the only thing that scales to millions).

Target denoising: induction labels are seed-noisy, so the training target is shrunk
toward its template-family mean (see capability_shrinkage_denoise.py). Cluster stats
are estimated train-only for the reported metric; the persisted model is refit on the
full corpus.

Usage::

    python -m research.tools.capability_screener train --db research/runs.db
    python -m research.tools.capability_screener train --shrink 0.75 --persist-tier static
    python -m research.tools.capability_screener score --limit 100000 --top 500
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from research.defaults import RUNS_DB
from research.tools.capability_shrinkage_denoise import (
    _NONE_CLUSTER,
    _shrink,
    _template_map,
)
from research.tools.induction_predictor_foundation import (
    _FEATURE_NAMES,
    _build_corpus,
    _fit_gbm,
    _op_presence_features,
    _ranking_metrics,
)

logger = logging.getLogger(__name__)

_STATE_DIR = Path("research/runtime/capability_screener")
_MODEL_PATH = _STATE_DIR / "screener_model.txt"
_META_PATH = _STATE_DIR / "screener_meta.json"
_SCORE_CHUNK = 50_000


def _meta_features(db_path: str, fps: List[str]) -> Tuple[np.ndarray, List[str]]:
    """op_count + pair_count per graph (free, from program_graph_features)."""
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            "SELECT graph_fingerprint, op_count, pair_count FROM program_graph_features "
            "WHERE op_count IS NOT NULL"
        ).fetchall()
    finally:
        con.close()
    m = {str(fp): (float(oc or 0.0), float(pc or 0.0)) for fp, oc, pc in rows}
    mat = np.zeros((len(fps), 2), dtype=np.float64)
    for i, fp in enumerate(fps):
        oc, pc = m.get(fp, (0.0, 0.0))
        mat[i, 0], mat[i, 1] = oc, pc
    return mat, ["op_count", "pair_count"]


def _static_matrix(
    db_path: str, fps: List[str], op_vocab: List[str] | None = None
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Free static features: op-presence + op_count + pair_count. Returns (X, names, vocab)."""
    X_ops, op_names = _op_presence_features(db_path, fps, vocab=op_vocab)
    X_meta, meta_names = _meta_features(db_path, fps)
    vocab = [n[len("op_") :] for n in op_names]
    return np.hstack([X_ops, X_meta]), [*op_names, *meta_names], vocab


def load_screener() -> Tuple[Any, Dict[str, Any]]:
    """Load the persisted booster + metadata. Fails loud if it needs a forward pass."""
    import lightgbm as lgb

    meta = json.loads(_META_PATH.read_text())
    if meta.get("uses_fingerprint"):
        raise SystemExit(
            "persisted screener needs fingerprint features (forward pass) — retrain "
            "with --persist-tier static to score un-probed graphs."
        )
    return lgb.Booster(model_file=str(_MODEL_PATH)), meta


def featurize_op_sets(
    op_sets: List[set],
    op_counts: List[int],
    pair_counts: List[int],
    op_vocab: List[str],
) -> np.ndarray:
    """In-memory static features for graphs NOT in the DB (the scale path).

    Layout must match `_static_matrix`: op-presence over ``op_vocab`` then
    [op_count, pair_count].
    """
    idx = {op: j for j, op in enumerate(op_vocab)}
    X = np.zeros((len(op_sets), len(op_vocab) + 2), dtype=np.float64)
    for i, ops in enumerate(op_sets):
        for op in ops:
            j = idx.get(op)
            if j is not None:
                X[i, j] = 1.0
        X[i, -2] = float(op_counts[i])
        X[i, -1] = float(pair_counts[i])
    return X


def score_op_sets(
    op_sets: List[set], op_counts: List[int], pair_counts: List[int]
) -> np.ndarray:
    """Predict capability for graphs given only their op-sets (no DB, no forward pass)."""
    booster, meta = load_screener()
    X = featurize_op_sets(op_sets, op_counts, pair_counts, list(meta["op_vocab"]))
    return np.asarray(booster.predict(X), dtype=np.float64)


def train(
    db_path: str, thr: float, shrink_f: float, persist_tier: str
) -> Dict[str, Any]:
    X_fp, _, ind, _, fps = _build_corpus(db_path)
    n = len(X_fp)
    cut = int(n * 0.8)
    train_mask = np.zeros(n, dtype=bool)
    train_mask[:cut] = True

    tmpl = _template_map(db_path)
    clusters = [tmpl.get(fp, _NONE_CLUSTER) for fp in fps]
    ind_shrunk = _shrink(ind, clusters, train_mask, shrink_f)

    X_static, static_names, op_vocab = _static_matrix(db_path, fps)
    X_full = np.hstack([X_static, X_fp])

    tiers = {
        "static_free": (X_static, static_names),
        "full_needs_forward_pass": (X_full, [*static_names, *_FEATURE_NAMES]),
    }
    metrics: Dict[str, Any] = {}
    for tname, (Xall, _) in tiers.items():
        gbm = _fit_gbm(Xall[:cut], ind_shrunk[:cut])
        pred = np.asarray(gbm.predict(Xall[cut:]), dtype=np.float64)
        m = _ranking_metrics(pred, ind[cut:], thr)  # scored vs RAW held-out label
        metrics[tname] = {
            "n_features": int(Xall.shape[1]),
            "roc_vs_raw": round(m["roc_auc_gt_thr"] or 0.0, 4),
            "spearman_vs_raw": round(m["spearman_rho"], 4),
        }

    # Persist the chosen tier, refit on the FULL corpus (shrink over all data).
    ind_shrunk_full = _shrink(ind, clusters, np.ones(n, dtype=bool), shrink_f)
    X_persist = X_static if persist_tier == "static" else X_full
    final = _fit_gbm(X_persist, ind_shrunk_full)
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    final.booster_.save_model(str(_MODEL_PATH))
    meta = {
        "persist_tier": persist_tier,
        "feature_layout": "op_presence + op_count + pair_count"
        + ("" if persist_tier == "static" else " + fingerprint(18)"),
        "op_vocab": op_vocab,
        "uses_fingerprint": persist_tier != "static",
        "induction_threshold": thr,
        "shrink_f": shrink_f,
        "n_train_total": n,
        "holdout_metrics": metrics,
    }
    _META_PATH.write_text(json.dumps(meta, indent=2, sort_keys=True))
    logger.info("saved screener -> %s", _MODEL_PATH)
    return {
        "split": "temporal_80_20",
        "n_total": n,
        "tiers": metrics,
        "persisted": persist_tier,
    }


def score(db_path: str, limit: int, top_k: int, out: str) -> Dict[str, Any]:
    import lightgbm as lgb

    meta = json.loads(_META_PATH.read_text())
    if meta.get("uses_fingerprint"):
        raise SystemExit(
            "persisted screener needs fingerprint features (forward pass) — not a "
            "scale screener. Retrain with --persist-tier static."
        )
    booster = lgb.Booster(model_file=str(_MODEL_PATH))
    op_vocab = list(meta["op_vocab"])

    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            "SELECT DISTINCT graph_fingerprint FROM program_graph_features "
            "WHERE op_count IS NOT NULL LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        con.close()
    fps = [str(r[0]) for r in rows]
    if not fps:
        raise SystemExit("no graphs to score")

    t0 = time.time()
    preds = np.empty(len(fps), dtype=np.float64)
    for start in range(0, len(fps), _SCORE_CHUNK):
        chunk = fps[start : start + _SCORE_CHUNK]
        X, _, _ = _static_matrix(db_path, chunk, op_vocab=op_vocab)
        preds[start : start + len(chunk)] = np.asarray(
            booster.predict(X), dtype=np.float64
        )
    elapsed = time.time() - t0

    order = np.argsort(-preds)
    top = [
        {"graph_fingerprint": fps[i], "score": round(float(preds[i]), 4)}
        for i in order[:top_k]
    ]
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"top": top, "n_scored": len(fps)}, indent=2))
    return {
        "n_scored": len(fps),
        "elapsed_s": round(elapsed, 2),
        "graphs_per_sec": int(len(fps) / elapsed) if elapsed > 0 else None,
        "score_p50": round(float(np.percentile(preds, 50)), 4),
        "score_p99": round(float(np.percentile(preds, 99)), 4),
        "top_written": out_path.as_posix(),
        "top_preview": top[:5],
    }


def backtest(db_path: str, thr: float, shrink_f: float, k: int = 5) -> Dict[str, Any]:
    """Backpredict every labeled graph and compare to its real induction label.

    in_sample: the persisted model scoring graphs it was trained on (optimistic).
    out_of_fold: k-fold — each graph predicted by a model that never saw it (honest).
    Plus a calibration table (predicted decile vs actual capable-rate).
    """
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

    _, _, ind, _, fps = _build_corpus(db_path)
    n = len(fps)
    X, _, _ = _static_matrix(db_path, fps)
    tmpl = _template_map(db_path)
    clusters = [tmpl.get(fp, _NONE_CLUSTER) for fp in fps]

    report: Dict[str, Any] = {
        "n_labeled_graphs": n,
        "threshold": thr,
        "shrink_f": shrink_f,
    }
    try:
        booster, meta = load_screener()
        Xp, _, _ = _static_matrix(db_path, fps, op_vocab=list(meta["op_vocab"]))
        pred_in = np.asarray(booster.predict(Xp), dtype=np.float64)
        report["in_sample_persisted_model"] = _ranking_metrics(pred_in, ind, thr)
    except SystemExit as exc:
        report["in_sample_persisted_model"] = {"error": str(exc)}

    rng = np.random.default_rng(42)
    folds = np.array_split(rng.permutation(n), k)
    oof = np.zeros(n, dtype=np.float64)
    for f in range(k):
        held = folds[f]
        rest = np.concatenate([folds[j] for j in range(k) if j != f])
        train_mask = np.zeros(n, dtype=bool)
        train_mask[rest] = True
        y = _shrink(ind, clusters, train_mask, shrink_f)
        model = _fit_gbm(X[rest], y[rest])
        oof[held] = np.asarray(model.predict(X[held]), dtype=np.float64)
    report["out_of_fold_honest"] = _ranking_metrics(oof, ind, thr)

    order = np.argsort(oof)
    report["calibration_deciles"] = [
        {
            "decile": i,
            "pred_mean": round(float(oof[b].mean()), 4),
            "actual_capable_rate": round(float((ind[b] > thr).mean()), 4),
            "actual_induction_mean": round(float(ind[b].mean()), 4),
        }
        for i, b in enumerate(np.array_split(order, 10))
    ]
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "score", "backtest"])
    parser.add_argument("--db", default=str(RUNS_DB))
    parser.add_argument("--thr", type=float, default=0.35)
    parser.add_argument(
        "--shrink", type=float, default=0.75, help="seed-noise shrink fraction"
    )
    parser.add_argument("--persist-tier", choices=["static", "full"], default="static")
    parser.add_argument("--limit", type=int, default=100_000)
    parser.add_argument("--top", type=int, default=500)
    parser.add_argument("--out", default="research/reports/capability_screen_topk.json")
    args = parser.parse_args()

    if args.mode == "train":
        report = train(args.db, args.thr, args.shrink, args.persist_tier)
    elif args.mode == "backtest":
        report = backtest(args.db, args.thr, args.shrink)
    else:
        report = score(args.db, args.limit, args.top, args.out)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
