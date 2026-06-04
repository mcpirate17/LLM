#!/usr/bin/env python
"""Does the MEASURED capability score actually pick the cream? — OOD validation (Phase 2.2).

The cascade funnel now ranks survivors by `measured_descriptors.capability_score` (a label-free
composite read off the graph's actual computation), on the thesis that it generalizes to novel archs
where the declared-feature oracle is anti-predictive (fab audit r=−0.55; STDP winner at 99.8th
measured-pctile). This tool checks that thesis on the labeled corpus, and tunes the score weights.

For a stratified sample of graphs that have a real `graph_json` AND a capability label, it probes the
measured descriptors and reports:
  - ROC(capability_score, capable) with the CURRENT weights — does the composite discriminate?
  - per-descriptor single-feature ROC — which signals carry it.
  - a logistic fit (descriptors → capable, CV ROC) whose coefficients are a SUGGESTED weight vector.
  - percentile of named novel winners within the sample (sanity).

The measured score is label-free (no fit), so this is inherently OOD — there is no train/test leak;
the logistic fit is only a suggestion for `_CAPABILITY_WEIGHTS`, validated by CV ROC.

Read-only. Usage::
    python -m research.tools.nas_funnel_ood_eval --target induction --sample 600
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

from research.defaults import RUNS_DB
from research.tools.capability_shrinkage_denoise import _capability_map
from research.tools.measured_descriptors import (
    _CAPABILITY_WEIGHTS,
    _LIP_STABLE,
    MeasuredDescriptorExtractor,
    capability_score_from_descriptors,
)

logger = logging.getLogger(__name__)

# Composition inputs of capability_score, in fixed order (matches _CAPABILITY_WEIGHTS keys).
_SCORE_TERMS = (
    "long_range_reach",
    "content_match_gating",
    "content_dependence",
    "causality_violation",
    "instability",
)
_TARGETS: Dict[str, Tuple[str, float]] = {
    "induction": ("induction_screening_auc", 0.35),
    "nano": ("nano_induction_nearest_max_accuracy", 0.5),
}
# Known novel winners present in runs.db (sanity percentiles). These are graph
# fingerprints, not secrets.
_NOVEL_WINNERS = (
    "7fd270feea70ef44",  # pragma: allowlist secret
    "818545a24795febd",  # pragma: allowlist secret
    "94560e64147ba8df",  # pragma: allowlist secret
)


def _graph_json_map(db: str, fps: List[str]) -> Dict[str, str]:
    """fp -> real graph_json for the given fingerprints (placeholder graphs excluded)."""
    con = sqlite3.connect(db)
    try:
        out: Dict[str, str] = {}
        for i in range(0, len(fps), 900):  # sqlite param limit
            chunk = fps[i : i + 900]
            q = (
                "SELECT graph_fingerprint, graph_json FROM graphs "
                "WHERE graph_json_is_placeholder=0 AND graph_fingerprint IN (%s)"
                % ",".join("?" * len(chunk))
            )
            for fp, gj in con.execute(q, chunk):
                out[str(fp)] = gj
        return out
    finally:
        con.close()


def _sample(
    cap: Dict[str, float], have_json: set, thr: float, n: int, seed: int
) -> Tuple[List[str], np.ndarray]:
    """Stratified sample: all positives (capped) + random negatives → balanced enough for ROC."""
    rng = np.random.default_rng(seed)
    pos = [fp for fp, v in cap.items() if fp in have_json and v > thr]
    neg = [fp for fp, v in cap.items() if fp in have_json and v <= thr]
    n_pos = min(len(pos), max(n // 2, 1))
    n_neg = min(len(neg), n - n_pos)
    sel_pos = list(rng.choice(pos, n_pos, replace=False)) if pos else []
    sel_neg = list(rng.choice(neg, n_neg, replace=False)) if neg else []
    fps = sel_pos + sel_neg
    y = np.array([1] * len(sel_pos) + [0] * len(sel_neg), dtype=int)
    return fps, y


def _descriptor_terms(d: Dict[str, float]) -> List[float]:
    """The 5 capability_score inputs in `_SCORE_TERMS` order (instability = lipschitz blow-up)."""
    instability = max(0.0, float(d.get("measured_lipschitz", 0.0)) - _LIP_STABLE)
    return [
        float(d.get("long_range_reach", 0.0)),
        float(d.get("content_match_gating", 0.0)),
        float(d.get("content_dependence", 0.0)),
        float(d.get("causality_violation", 0.0)),
        instability,
    ]


def _roc(score: np.ndarray, y: np.ndarray) -> Optional[float]:
    return float(roc_auc_score(y, score)) if 0 < y.sum() < len(y) else None


def _probe_sample(
    mdx: MeasuredDescriptorExtractor,
    gj_map: Dict[str, str],
    fps: List[str],
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]:
    """Probe each sampled graph → (term matrix X, aligned labels, raw descriptor dicts)."""
    rows: List[List[float]] = []
    keep_y: List[int] = []
    descs: List[Dict[str, float]] = []
    for fp, label in zip(fps, y):
        try:
            d = mdx.descriptors(gj_map[fp])
        except Exception:  # noqa: BLE001
            d = None
        if d is None:
            continue
        rows.append(_descriptor_terms(d))
        keep_y.append(int(label))
        descs.append(d)
    return np.array(rows, dtype=np.float64), np.array(keep_y, dtype=int), descs


def evaluate(
    db: str, target: str, sample: int, seed: int, device: Optional[str]
) -> Dict[str, Any]:
    col, thr = _TARGETS[target]
    cap = _capability_map(db, col)
    gj_map = _graph_json_map(db, list(cap))
    fps, y = _sample(cap, set(gj_map), thr, sample, seed)
    logger.info(
        "sampled %d graphs (%d positive) for target=%s", len(fps), int(y.sum()), target
    )
    mdx = MeasuredDescriptorExtractor(device=device, n_seeds=1)
    X, yk, descs = _probe_sample(mdx, gj_map, fps, y)
    if len(yk) < 30 or not (0 < yk.sum() < len(yk)):
        return {"error": "insufficient probeable labeled sample", "n_probed": len(yk)}

    cap_score = np.array([capability_score_from_descriptors(d) for d in descs])
    per_term = {term: _roc(X[:, i], yk) for i, term in enumerate(_SCORE_TERMS)}
    # logistic fit: descriptors -> capable; coefficients suggest a weight vector.
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    cv = cross_val_score(clf, X, yk, cv=5, scoring="roc_auc")
    clf.fit(X, yk)
    suggested = {t: round(float(c), 4) for t, c in zip(_SCORE_TERMS, clf.coef_[0])}

    # named novel-winner percentiles within this sample's capability scores.
    winners: Dict[str, Any] = {}
    for fp in _NOVEL_WINNERS:
        if fp in gj_map:
            d = mdx.descriptors(gj_map[fp])
            if d is not None:
                sc = capability_score_from_descriptors(d)
                pct = float((cap_score < sc).mean())
                winners[fp[:12]] = {
                    "capability_score": round(sc, 4),
                    "pctile": round(pct, 3),
                }

    return {
        "target": target,
        "threshold": thr,
        "n_probed": int(len(yk)),
        "n_positive": int(yk.sum()),
        "current_weights": dict(_CAPABILITY_WEIGHTS),
        "capability_score_roc": _roc(cap_score, yk),
        "per_descriptor_roc": per_term,
        "logistic_cv_roc_mean": round(float(cv.mean()), 4),
        "logistic_cv_roc_std": round(float(cv.std()), 4),
        "suggested_weights_from_logistic": suggested,
        "novel_winner_percentiles": winners,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=str(RUNS_DB))
    p.add_argument("--target", choices=tuple(_TARGETS), default="induction")
    p.add_argument("--sample", type=int, default=600)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="research/reports/nas_funnel_ood_eval.json")
    args = p.parse_args()
    report = evaluate(args.db, args.target, args.sample, args.seed, args.device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
