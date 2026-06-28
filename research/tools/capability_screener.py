#!/usr/bin/env python
"""Capability screener — rank graphs by predicted capability at scale, OOD-honest.

Triages millions of *un-probed* candidate graphs cheaply, before spending any probe
compute, using only FREE static features (parseable from the graph definition, no
forward pass): op-presence + op_count + pair_count (program_graph_{ops,features}).

Decisive design choices (see tasks/induction_corpus_handoff.md):

  - NO provenance/completeness gate on the training labels. A label is a label: the
    corpus is EVERY fingerprint that carries an axis label (``_capability_map``), not
    the ~3k full-experiment rows the deduped predictor corpus is capped at. This pulls
    the induction corpus from 3143 → ~17.9k op-reachable labels.
  - OOD generalization is the objective, never in-distribution holdout. Model selection
    and the headline metric use leave-template-family-out (hold out whole families =
    novel-region proxy) + forward-temporal — NOT random/in-dist ROC (that rewards
    regress-to-familiar and collapses off-manifold, the documented STDP failure).
  - ALL predictor types compete per axis (GBM / ExtraTrees recursive-partition /
    PLS→tree / PLS→GBM / PLS→Bayesian) and the OOD-best is self-selected and persisted.
    The shared zoo lives in ``pls_partition_oracle`` — reused here on op-presence
    features so there is exactly one model zoo.

Two axes ship as independent screeners (rank-distinct labels):
  - induction              → research/runtime/capability_screener/        (thr 0.35)
  - nano_induction_nearest → research/runtime/capability_screener_nano/   (thr 0.50)

Usage::

    python -m research.tools.capability_screener train --axis induction
    python -m research.tools.capability_screener train --axis nano_induction_nearest
    python -m research.tools.capability_screener backtest --axis induction
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
    _capability_map,
    _shrink,
    _template_map,
)
from research.tools.induction_predictor_foundation import (
    _fingerprint_timestamps,
    _op_presence_features,
)

# NOTE: the shared model zoo lives in research.tools.pls_partition_oracle, which pulls
# novelty_scorer -> probe_novel_candidates -> THIS module. Import the zoo lazily inside
# train/backtest/_label_corpus to keep the scale-scoring API (load_screener /
# featurize_op_sets, used by probe_novel_candidates) free of that cycle.

logger = logging.getLogger(__name__)

_STATE_DIR = Path("research/runtime/capability_screener")
_NANO_STATE_DIR = Path("research/runtime/capability_screener_nano")
_SCORE_CHUNK = 50_000

# axis -> (graph_runs label column, capable threshold, persisted state dir).
_AXES: Dict[str, Tuple[str, float, Path]] = {
    "induction": ("induction_screening_auc", 0.35, _STATE_DIR),
    "nano_induction_nearest": (
        "nano_induction_nearest_max_accuracy",
        0.5,
        _NANO_STATE_DIR,
    ),
}

# Confirmed novel OOD winners (NOT in the labeled corpus — their op-sets are scored
# externally). The deployed in-dist screener anti-correlated on these (predicted
# 0.13–0.29 vs real induction 0.44–0.89): the rebuilt OOD-trained screener must rank
# them above the corpus median or it has the same regress-to-familiar pathology.
_OOD_SANITY_WINNERS: Tuple[Tuple[str, List[str], float], ...] = (
    (
        "e656938e359ada50",  # pragma: allowlist secret  (graph fingerprint, not a secret)
        [
            "add",
            "conv1d_seq",
            "gated_linear",
            "layernorm",
            "lif_neuron",
            "linear_proj",
            "rmsnorm",
            "rwkv_channel",
            "softmax_attention",
            "stdp_attention",
            "swiglu_mlp",
        ],
        0.894,
    ),
    (
        "684ab3df7765207c",  # pragma: allowlist secret  (graph fingerprint, not a secret)
        [
            "add",
            "layernorm",
            "lif_neuron",
            "linear_proj",
            "rmsnorm",
            "softmax_attention",
            "spectral_filter",
            "stdp_attention",
        ],
        0.742,
    ),
    (
        "04a1b6b05745bf8d",  # pragma: allowlist secret  (graph fingerprint, not a secret)
        [
            "add",
            "chebyshev_spectral_mix",
            "entmax_attention",
            "lif_neuron",
            "linear_proj",
            "outer_product",
            "rmsnorm",
            "sigmoid",
            "stdp_attention",
        ],
        0.438,
    ),
)


def _model_path(state_dir: Path) -> Path:
    return state_dir / "screener_model.joblib"


def _meta_path(state_dir: Path) -> Path:
    return state_dir / "screener_meta.json"


def _legacy_model_path(state_dir: Path) -> Path:
    return state_dir / "screener_model.txt"


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


def _label_corpus(db_path: str, axis_col: str) -> Tuple[Any, List[str]]:
    """Op-presence + meta features over ALL fps with an axis label — NO gating.

    Reuses ``_capability_map`` (every labeled fingerprint, mean over runs),
    ``_fingerprint_timestamps`` (temporal order), ``_template_map`` (family clusters)
    and ``_static_matrix`` (free features). Returns an ``AxisCorpus`` plus the op vocab
    needed at score time so the feature layout matches.
    """
    from research.tools.pls_partition_oracle import AxisCorpus

    cap = _capability_map(db_path, axis_col)
    if not cap:
        raise SystemExit(f"no labels for {axis_col}")
    ts_map = _fingerprint_timestamps(db_path)
    tmpl = _template_map(db_path)
    fps = sorted(
        cap, key=lambda fp: ts_map.get(fp, 0.0)
    )  # oldest first → temporal split
    X_static, names, vocab = _static_matrix(db_path, fps)
    y = np.array([cap[fp] for fp in fps], dtype=np.float64)
    clusters = [tmpl.get(fp, _NONE_CLUSTER) for fp in fps]
    return AxisCorpus(X_static, names, fps, y, clusters), vocab


def load_screener(state_dir: Path = _STATE_DIR) -> Tuple[Any, Dict[str, Any]]:
    """Load the persisted model + metadata. Model exposes ``.predict(X)`` (joblib bundle,
    any sklearn/PLS/GBM kind) with a legacy LightGBM-booster fallback. Fails loud if the
    persisted model needs a forward pass."""
    meta = json.loads(_meta_path(state_dir).read_text())
    if meta.get("uses_fingerprint"):
        raise SystemExit(
            "persisted screener needs fingerprint features (forward pass) — retrain "
            "to score un-probed graphs."
        )
    jp = _model_path(state_dir)
    if jp.exists():
        import joblib

        return joblib.load(jp), meta
    import lightgbm as lgb

    return lgb.Booster(model_file=str(_legacy_model_path(state_dir))), meta


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


def _novel_winner_check(
    model: Any, vocab: List[str], corpus_pred: np.ndarray
) -> Dict[str, Any]:
    """Score the confirmed external novel winners; report their percentile vs the corpus.

    pair_count is unknown for off-DB graphs → approximated by op_count (a minor feature
    relative to op-presence). The go/no-go is whether the OOD-trained model ranks these
    above the corpus median (the in-dist screener did not)."""
    ops_list = [set(o) for _, o, _ in _OOD_SANITY_WINNERS]
    counts = [len(o) for o in ops_list]
    X = featurize_op_sets(ops_list, counts, counts, vocab)
    preds = np.asarray(model.predict(X), dtype=np.float64)
    median = float(np.median(corpus_pred))
    winners = [
        {
            "fingerprint": fp,
            "real_induction": real,
            "predicted": round(float(p), 4),
            "corpus_percentile": round(float((corpus_pred < p).mean()), 3),
            "above_median": bool(p > median),
        }
        for (fp, _, real), p in zip(_OOD_SANITY_WINNERS, preds)
    ]
    return {
        "corpus_median_pred": round(median, 4),
        "winners": winners,
        "all_above_median": all(w["above_median"] for w in winners),
    }


def train(
    db_path: str, axis: str, shrink_f: float, params: Dict[str, int]
) -> Dict[str, Any]:
    """Build the ungated op-presence corpus, OOD-self-select among all model kinds, persist."""
    from research.tools.pls_partition_oracle import (
        _fit_axis_model,
        _KINDS,
        _leave_family_out_ab,
        _temporal_ab,
    )

    axis_col, thr, state_dir = _AXES[axis]
    cp, vocab = _label_corpus(db_path, axis_col)
    n = len(cp.fps)
    n_cap = int((cp.y > thr).sum())
    if not (0 < n_cap < n):
        raise SystemExit(f"axis {axis}: degenerate labels (n={n}, capable={n_cap})")

    lfo = _leave_family_out_ab(cp, thr, shrink_f, params)
    temporal = _temporal_ab(cp, thr, shrink_f, params)
    best = max(
        _KINDS, key=lambda k: lfo[k]["roc"] or 0.0
    )  # OOD metric drives selection

    ys = _shrink(cp.y, cp.clusters, np.ones(n, dtype=bool), shrink_f)
    model = _fit_axis_model(best, cp.X, ys, cp.names, params)
    corpus_pred = np.asarray(model.predict(cp.X), dtype=np.float64)
    winner_check = _novel_winner_check(model, vocab, corpus_pred)

    state_dir.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(model, _model_path(state_dir))
    legacy = _legacy_model_path(state_dir)
    if legacy.exists():
        legacy.unlink()  # stale lgb-booster format from the capped-corpus era
    meta = {
        "axis": axis,
        "axis_col": axis_col,
        "model_kind": best,
        "feature_layout": "op_presence + op_count + pair_count",
        "op_vocab": vocab,
        "uses_fingerprint": False,
        "induction_threshold": thr,
        "shrink_f": shrink_f,
        "n_train_total": n,
        "n_capable": n_cap,
        "ood_selection_metric": "leave_template_family_out_roc",
        "leave_family_out_roc": {k: lfo[k]["roc"] for k in _KINDS},
        "temporal_roc": {k: temporal[k]["roc"] for k in _KINDS},
        "novel_winner_check": winner_check,
    }
    _meta_path(state_dir).write_text(json.dumps(meta, indent=2, sort_keys=True))
    logger.info("saved %s screener (%s) -> %s", axis, best, _model_path(state_dir))
    return {
        "axis": axis,
        "n_total": n,
        "n_capable": n_cap,
        "selected_kind": best,
        "leave_family_out_roc": meta["leave_family_out_roc"],
        "temporal_roc": meta["temporal_roc"],
        "novel_winner_check": winner_check,
        "persisted": str(_model_path(state_dir)),
    }


def backtest(
    db_path: str, axis: str, shrink_f: float, params: Dict[str, int]
) -> Dict[str, Any]:
    """OOD-honest A/B over all model kinds: leave-template-family-out + forward-temporal."""
    from research.tools.pls_partition_oracle import (
        _leave_family_out_ab,
        _temporal_ab,
    )

    axis_col, thr, _ = _AXES[axis]
    cp, _ = _label_corpus(db_path, axis_col)
    return {
        "axis": axis,
        "n_total": len(cp.fps),
        "n_capable": int((cp.y > thr).sum()),
        "n_features": int(cp.X.shape[1]),
        "threshold": thr,
        "note": "ROC scored vs RAW held-out label; OOD-honest. NOT in-dist holdout.",
        "leave_family_out": _leave_family_out_ab(cp, thr, shrink_f, params),
        "forward_temporal": _temporal_ab(cp, thr, shrink_f, params),
    }


def score(db_path: str, axis: str, limit: int, top_k: int, out: str) -> Dict[str, Any]:
    _, _, state_dir = _AXES[axis]
    model, meta = load_screener(state_dir)
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
            model.predict(X), dtype=np.float64
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
        "axis": axis,
        "n_scored": len(fps),
        "elapsed_s": round(elapsed, 2),
        "graphs_per_sec": int(len(fps) / elapsed) if elapsed > 0 else None,
        "score_p50": round(float(np.percentile(preds, 50)), 4),
        "score_p99": round(float(np.percentile(preds, 99)), 4),
        "top_written": out_path.as_posix(),
        "top_preview": top[:5],
    }


def _params(args: argparse.Namespace) -> Dict[str, int]:
    return {
        "n_components": args.pls_components,
        "tree_depth": args.tree_depth,
        "min_leaf": args.min_leaf,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "score", "backtest"])
    parser.add_argument("--db", default=str(RUNS_DB))
    parser.add_argument("--axis", choices=list(_AXES), default="induction")
    parser.add_argument(
        "--shrink", type=float, default=0.75, help="seed-noise shrink fraction"
    )
    parser.add_argument("--pls-components", type=int, default=20)
    parser.add_argument("--tree-depth", type=int, default=4)
    parser.add_argument("--min-leaf", type=int, default=20)
    parser.add_argument("--limit", type=int, default=100_000)
    parser.add_argument("--top", type=int, default=500)
    parser.add_argument("--out", default="research/reports/capability_screen_topk.json")
    args = parser.parse_args()

    if args.mode == "train":
        report = train(args.db, args.axis, args.shrink, _params(args))
    elif args.mode == "backtest":
        report = backtest(args.db, args.axis, args.shrink, _params(args))
    else:
        report = score(args.db, args.axis, args.limit, args.top, args.out)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
