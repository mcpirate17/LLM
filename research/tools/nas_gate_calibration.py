#!/usr/bin/env python
"""OOD gate-calibration: which oracle axis EARNS a hard gate, which stays a soft signal.

A hard reject-gate is only justified if rejecting on it almost never discards a true
winner. This tool measures that, per (gate-axis G, target-axis T), on leave-family-out
OOD folds (each row scored by a model that never saw its template family — the closest
in-corpus proxy for a novel design).

For every (G, T) pair it reports the full confusion-matrix suite at two operating points:

  - deployed: G's prediction thresholded at its deployed label threshold (what the
              cascade does today).
  - recall95: the prediction threshold that RETAINS >=95% of T's true winners — the
              "keep the cream" operating point. Reports how much junk that prunes.

Decision rule (the gate must EARN it): G qualifies as a hard gate for T iff, at the
recall95 point, it still prunes a meaningful fraction (>=`min_prune`) with retained
enrichment >1 and ROC(G,T) >= `min_roc`. Otherwise G is a SOFT rank signal only.

The diagonal (G==T) is each axis's self-gate quality; the off-diagonal exposes
axis-mismatch (e.g. gate on ar_gate while you actually want induction).

Read-only. No model is retrained or persisted; this reuses the oracle's own
leave-family-out CV (`pls_partition_oracle._fit_predict`) and corpus helpers.

Usage::
    python -m research.tools.nas_gate_calibration
    python -m research.tools.nas_gate_calibration --target-recall 0.95 --k 5 \
        --out research/reports/nas_gate_calibration.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from research.defaults import RUNS_DB
from research.tools.backfill_graph_semantics import load_semantic_features
from research.tools.capability_shrinkage_denoise import (
    _NONE_CLUSTER,
    _capability_map,
    _shrink,
    _template_map,
)
from research.tools.pls_partition_oracle import AXES, _META_PATH, _fit_predict

logger = logging.getLogger(__name__)

_DEFAULT_OUT = "research/reports/nas_gate_calibration.json"
_DEFAULT_PARAMS = {"n_components": 20, "tree_depth": 4, "min_leaf": 20}


def _load_oracle_meta() -> Dict[str, Any]:
    """Deployed thresholds + per-axis selected model kind (so calibration matches prod)."""
    path = Path(_META_PATH)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _selected_kind(meta: Dict[str, Any], axis: str) -> str:
    sel = (meta.get("selected_per_axis") or {}).get(axis) or {}
    return str(sel.get("kind", "gbm"))


def _oof_predictions(
    X: np.ndarray,
    y: np.ndarray,
    clusters: List[str],
    names: List[str],
    kind: str,
    params: Dict[str, int],
    shrink_f: float,
    k: int,
) -> np.ndarray:
    """Leave-family-out OOF predictions: each row scored by a model blind to its family."""
    n = len(y)
    groups = np.asarray(clusters)
    n_groups = len(set(clusters))
    splits = min(k, n_groups)
    if splits < 2:
        return np.full(n, np.nan)
    gkf = GroupKFold(n_splits=splits)
    oof = np.full(n, np.nan, dtype=np.float64)
    for rest, held in gkf.split(X, y, groups):
        mask = np.zeros(n, dtype=bool)
        mask[rest] = True
        ys = _shrink(y, clusters, mask, shrink_f)  # train-family stats only
        oof[held] = _fit_predict(kind, X[rest], ys[rest], X[held], names, **params)
    return oof


def _confusion(
    pred: np.ndarray, capable: np.ndarray, pred_thr: float
) -> Dict[str, Any]:
    """Confusion-matrix suite for the gate `pred >= pred_thr` against `capable`."""
    keep = pred >= pred_thr  # gate PASSES (retains) the candidate
    tp = int((keep & capable).sum())
    fp = int((keep & ~capable).sum())
    fn = int((~keep & capable).sum())  # rejected winners — the cost we must avoid
    tn = int((~keep & ~capable).sum())
    n = tp + fp + fn + tn
    n_cap = tp + fn
    base = n_cap / n if n else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0  # precision of retained
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    recall = tp / n_cap if n_cap else 0.0  # winners RETAINED (sensitivity)
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prune = (fn + tn) / n if n else 0.0  # fraction rejected
    return {
        "pred_thr": round(float(pred_thr), 6),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "base_rate": round(base, 5),
        "ppv_precision_retained": round(ppv, 4),
        "npv": round(npv, 4),
        "recall_winners_retained": round(recall, 4),
        "winners_lost_frac": round(1.0 - recall, 4),  # the cream-loss metric
        "specificity": round(spec, 4),
        "false_omission_rate": round(1.0 - npv, 4),  # winners among rejected
        "prune_rate": round(prune, 4),
        "enrichment_retained": round(ppv / base, 3) if base > 0 else None,
        "lr_plus": round(recall / (1.0 - spec), 3) if spec < 1.0 else None,
        "lr_minus": round((1.0 - recall) / spec, 3) if spec > 0 else None,
    }


def _recall_constrained_thr(
    pred: np.ndarray, capable: np.ndarray, target_recall: float
) -> float:
    """Highest pred threshold that still retains >= target_recall of the winners.

    Higher threshold ⇒ more pruning, so we want the most aggressive prune that keeps the
    cream. The winners' predicted scores: keep the lower (1-target_recall) quantile out.
    """
    pos = pred[capable]
    if pos.size == 0:
        return float("-inf")
    # to retain >= target_recall of winners, threshold must be <= the
    # (1-target_recall) quantile of winner predictions.
    q = float(np.quantile(pos, max(0.0, 1.0 - target_recall)))
    return q


def _thresholds(meta: Dict[str, Any]) -> Dict[str, float]:
    thr = {ax: AXES[ax][1] for ax in AXES}
    if meta.get("thresholds"):
        thr.update({k: float(v) for k, v in meta["thresholds"].items()})
    return thr


def _all_axis_oof(
    cap: Dict[str, Dict[str, float]],
    X_all: np.ndarray,
    names: List[str],
    present: List[str],
    clusters_all: List[str],
    meta: Dict[str, Any],
    params: Dict[str, int],
    shrink_f: float,
    k: int,
) -> Dict[str, Dict[str, float]]:
    """Leave-family-out OOF prediction per gate-axis over its labeled rows."""
    row = {fp: i for i, fp in enumerate(present)}
    oof_pred: Dict[str, Dict[str, float]] = {}
    for g in AXES:
        g_fps = [fp for fp in present if fp in cap[g]]
        if len(g_fps) < 100:
            logger.info("axis %s: only %d labeled rows — skipped", g, len(g_fps))
            continue
        idx = np.array([row[fp] for fp in g_fps])
        yg = np.array([cap[g][fp] for fp in g_fps], dtype=np.float64)
        cg = [clusters_all[i] for i in idx]
        kind = _selected_kind(meta, g)
        pred = _oof_predictions(X_all[idx], yg, cg, names, kind, params, shrink_f, k)
        oof_pred[g] = {fp: float(p) for fp, p in zip(g_fps, pred) if np.isfinite(p)}
        logger.info(
            "axis %s: OOF predictions for %d rows (kind=%s)", g, len(oof_pred[g]), kind
        )
    return oof_pred


def _evaluate_pairs(
    oof_pred: Dict[str, Dict[str, float]],
    cap: Dict[str, Dict[str, float]],
    thresholds: Dict[str, float],
    target_recall: float,
) -> Dict[str, Any]:
    """Confusion suite for every (gate-axis G, target-axis T) at deployed + recall95."""
    pairs: Dict[str, Any] = {}
    for g in oof_pred:
        for t in AXES:
            common = [fp for fp in oof_pred[g] if fp in cap[t]]
            if len(common) < 100:
                continue
            pred = np.array([oof_pred[g][fp] for fp in common])
            yt = np.array([cap[t][fp] for fp in common], dtype=np.float64)
            capable = yt > thresholds[t]
            n_cap = int(capable.sum())
            if not (0 < n_cap < len(capable)):
                continue
            r95_thr = _recall_constrained_thr(pred, capable, target_recall)
            pairs[f"{g}__gate_for__{t}"] = {
                "gate_axis": g,
                "target_axis": t,
                "n": len(common),
                "n_capable": n_cap,
                "roc_auc": round(float(roc_auc_score(capable.astype(int), pred)), 4),
                "at_deployed_threshold": _confusion(pred, capable, thresholds[g]),
                "at_recall95_threshold": _confusion(pred, capable, r95_thr),
            }
    return pairs


def _rank_gates(
    pairs: Dict[str, Any],
    oof_pred: Dict[str, Dict[str, float]],
    min_prune: float,
    min_roc: float,
) -> Dict[str, Any]:
    """Per target axis, rank candidate gates; flag those that EARN a hard gate."""
    recs: Dict[str, Any] = {}
    for t in AXES:
        cands: List[Dict[str, Any]] = []
        for g in oof_pred:
            pr = pairs.get(f"{g}__gate_for__{t}")
            if not pr:
                continue
            r95 = pr["at_recall95_threshold"]
            enr = r95.get("enrichment_retained")
            qualifies = (
                r95["prune_rate"] >= min_prune
                and enr is not None
                and enr > 1.0
                and pr["roc_auc"] >= min_roc
            )
            cands.append(
                {
                    "gate_axis": g,
                    "roc_auc": pr["roc_auc"],
                    "recall95_prune_rate": r95["prune_rate"],
                    "recall95_enrichment": enr,
                    "deployed_winners_lost_frac": pr["at_deployed_threshold"][
                        "winners_lost_frac"
                    ],
                    "qualifies_as_hard_gate": bool(qualifies),
                }
            )
        cands.sort(
            key=lambda c: (c["qualifies_as_hard_gate"], c["recall95_prune_rate"]),
            reverse=True,
        )
        recs[t] = cands
    return recs


def calibrate(
    db_path: str,
    feat_db: str,
    target_recall: float,
    shrink_f: float,
    k: int,
    min_prune: float,
    min_roc: float,
) -> Dict[str, Any]:
    meta = _load_oracle_meta()
    params = dict(meta.get("pls_params") or _DEFAULT_PARAMS)
    thresholds = _thresholds(meta)
    tmpl = _template_map(db_path)
    cap = {ax: _capability_map(db_path, AXES[ax][0]) for ax in AXES}

    # One shared feature matrix over the union of labeled fingerprints.
    universe = sorted({fp for m in cap.values() for fp in m if fp in tmpl})
    X_all, names, present = load_semantic_features(universe, meta_db=feat_db)
    clusters_all = [tmpl.get(fp, _NONE_CLUSTER) for fp in present]
    logger.info(
        "calibration corpus: %d labeled fps, %d with features, %d feature dims",
        len(universe),
        len(present),
        X_all.shape[1] if X_all.size else 0,
    )

    oof_pred = _all_axis_oof(
        cap, X_all, names, present, clusters_all, meta, params, shrink_f, k
    )
    pairs = _evaluate_pairs(oof_pred, cap, thresholds, target_recall)
    return {
        "corpus": {"labeled_fps": len(universe), "with_features": len(present)},
        "target_recall": target_recall,
        "shrink_f": shrink_f,
        "k_family_folds": k,
        "deployed_thresholds": thresholds,
        "decision_rule": {
            "min_recall_winners_retained": target_recall,
            "min_prune_rate": min_prune,
            "min_roc": min_roc,
            "rule": "hard-gate iff at recall95 op-point: prune>=min_prune AND "
            "enrichment_retained>1 AND roc>=min_roc; else SOFT signal only.",
        },
        "pairs": pairs,
        "gate_recommendations": _rank_gates(pairs, oof_pred, min_prune, min_roc),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=str(RUNS_DB))
    p.add_argument(
        "--feat-db",
        default="research/db_backups/meta_analysis_pre_ar_gate_report_apply_20260531_173038.db",
        help="DB holding the graph_semantic_features table (live meta DB has lost it; "
        "the oracle was trained on this backup).",
    )
    p.add_argument("--target-recall", type=float, default=0.95)
    p.add_argument("--shrink-f", type=float, default=0.5)
    p.add_argument("--k", type=int, default=5, help="leave-family-out folds")
    p.add_argument("--min-prune", type=float, default=0.30)
    p.add_argument("--min-roc", type=float, default=0.70)
    p.add_argument("--out", default=_DEFAULT_OUT)
    args = p.parse_args()

    report = calibrate(
        args.db,
        args.feat_db,
        args.target_recall,
        args.shrink_f,
        args.k,
        args.min_prune,
        args.min_roc,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))

    # Concise stdout summary: each target axis and the gates that qualify.
    print(
        f"\n=== NAS gate calibration (leave-family-out OOD, recall>={args.target_recall}) ==="
    )
    print(f"corpus: {report['corpus']}")
    for t, cands in report["gate_recommendations"].items():
        print(f"\nTARGET = {t}:")
        print(
            f"  {'gate_axis':<24} {'roc':>6} {'r95_prune':>10} {'r95_enrich':>11} "
            f"{'deployed_lost':>14} {'HARD?':>6}"
        )
        for c in cands:
            print(
                f"  {c['gate_axis']:<24} {c['roc_auc']:>6.3f} "
                f"{c['recall95_prune_rate']:>10.3f} "
                f"{str(c['recall95_enrichment']):>11} "
                f"{c['deployed_winners_lost_frac']:>14.3f} "
                f"{'YES' if c['qualifies_as_hard_gate'] else 'no':>6}"
            )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
