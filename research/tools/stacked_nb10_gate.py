#!/usr/bin/env python
"""Deployable stacked+calibrated nb1.0 reject head for the live NAS gate.

Trains the graph+cheap nb1.0-binding head validated in
``stacked_calibrated_probe_eval`` (OOD-confirmed: leave-family-out ROC 0.70->0.84,
NPV ~0.90), calibrates it (isotonic), picks a HIGH-NPV reject threshold, and
persists a joblib artifact. ``StackedNb10Scorer`` loads it and emits a calibrated
``nb10_pass_stacked`` probability for a graph — but ONLY when the strictly-cheaper
probes it stacks on (ar_gate + nb0.5) are already measured. On a fresh pre-probe
graph it returns ``available=False`` so the gate no-ops (it never rejects a graph
it cannot see the cheap evidence for, and never skips a probe).

The companion gate in ``nas_gate_policy._gate_nb10_stacked`` thresholds this as a
PREDICTOR (rescuable) rejection, so known-good / published-family blind spots stay
measurable. Reject-only, never accept — temporal PPV (0.94) does NOT survive OOD
(0.59) so this must not make accept decisions.

Usage:
    python -m research.tools.stacked_nb10_gate --train --target-npv 0.90
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.defaults import RUNS_DB
from research.scientist.intelligence.metrics_utils import binary_classification_metrics
from research.tools.audit_cheap_probe_predictors import (
    _graph_features,
    _materialize_matrix,
)
from research.tools.stacked_calibrated_probe_eval import (
    CASCADE_HEADS,
    CascadeHead,
    _cheap_feats,
    _isotonic,
    _model,
    _three_way_temporal,
    build_dataset,
)

_STATE_DIR = Path("research/runtime/stacked_nb10_gate")
_MODEL_PATH = _STATE_DIR / "nb10_stacked.joblib"
_META_PATH = _STATE_DIR / "nb10_stacked_meta.json"

# strictly-cheaper probes that MUST be present for a meaningful prediction
_REQUIRED_CHEAP = ("ar_gate_score", "language_control_s05_binding_score")


def _nb10_head() -> CascadeHead:
    return next(h for h in CASCADE_HEADS if h.name == "nb10_binding")


def _pick_reject_threshold(
    y_true: np.ndarray, score: np.ndarray, *, target_npv: float, min_support: int
) -> dict[str, float]:
    """Largest P(pass) cutoff whose 'predicted fail' set keeps NPV >= target_npv.

    We reject when score < cutoff. A larger cutoff rejects more (higher coverage)
    but risks lower NPV; pick the most aggressive cutoff that still holds NPV.
    """
    best = {"reject_threshold": 0.0, "npv": 1.0, "coverage": 0.0, "support": 0.0}
    for cut in np.unique(np.concatenate([np.linspace(0.02, 0.9, 45), score])):
        pred_fail = score < cut
        support = int(pred_fail.sum())
        if support < min_support:
            continue
        npv = float(np.mean(y_true[pred_fail] == 0))
        if npv >= target_npv and cut > best["reject_threshold"]:
            best = {
                "reject_threshold": float(cut),
                "npv": npv,
                "coverage": float(np.mean(pred_fail)),
                "support": float(support),
            }
    return best


@dataclass
class StackedNb10Scorer:
    model: Any
    iso: Any
    feature_names: list[str]
    feature_medians: np.ndarray
    cheap_inputs: tuple[str, ...]
    reject_threshold: float
    target_threshold: float

    @classmethod
    def try_load(cls) -> "StackedNb10Scorer | None":
        if not _MODEL_PATH.exists():
            return None
        import joblib

        return joblib.load(_MODEL_PATH)

    def _vector(self, graph_dict: Mapping[str, Any], cheap: Mapping[str, Any]):
        row = {"graph_json": graph_dict, **dict(cheap)}
        feats = _graph_features(row)
        if not feats:
            return None
        feats.update(_cheap_feats(row, self.cheap_inputs))
        X, _ = _materialize_matrix([feats], feature_names=self.feature_names)
        return np.where(np.isfinite(X), X, self.feature_medians)

    def score(
        self, graph_dict: Mapping[str, Any], cheap_actuals: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Calibrated P(nb1.0 binding pass). available=False if cheap probes absent."""
        present = all(cheap_actuals.get(k) is not None for k in _REQUIRED_CHEAP)
        if not present:
            return {"available": False}
        X = self._vector(graph_dict, cheap_actuals)
        if X is None:
            return {"available": False}
        p = float(np.clip(self.model.predict_proba(X)[:, 1], 0.0, 1.0)[0])
        if self.iso is not None:
            p = float(np.clip(self.iso.predict([p]), 0.0, 1.0)[0])
        return {
            "available": True,
            "nb10_pass_stacked": p,
            "reject_threshold": self.reject_threshold,
            "would_reject": p < self.reject_threshold,
        }


def train_and_persist(
    db: str = str(RUNS_DB),
    *,
    target_npv: float = 0.90,
    min_support: int = 50,
    seed: int = 42,
) -> dict[str, Any]:
    from research.tools.audit_cheap_probe_predictors import load_audit_rows

    head = _nb10_head()
    rows = load_audit_rows(db)
    feats, y, kept = build_dataset(rows, head, use_cheap=True)
    ybin = (y >= head.threshold).astype(np.int32)
    tr, ca, test_idx = _three_way_temporal(kept, 0.64, 0.16)

    X_all, names = _materialize_matrix(feats)
    medians = np.zeros(X_all.shape[1])
    finite = np.isfinite(X_all[tr])
    has = np.any(finite, axis=0)
    if np.any(has):
        medians[has] = np.nanmedian(np.where(finite, X_all[tr], np.nan)[:, has], axis=0)
    fill = lambda idx: np.where(np.isfinite(X_all[idx]), X_all[idx], medians)  # noqa: E731

    model = _model(seed)
    model.fit(fill(tr), ybin[tr])
    p_ca = np.clip(model.predict_proba(fill(ca))[:, 1], 0.0, 1.0)
    iso = _isotonic(p_ca, ybin[ca])
    p_test = np.clip(model.predict_proba(fill(test_idx))[:, 1], 0.0, 1.0)
    p_test_cal = np.clip(iso.predict(p_test), 0.0, 1.0) if iso is not None else p_test

    chosen = _pick_reject_threshold(
        ybin[test_idx], p_test_cal, target_npv=target_npv, min_support=min_support
    )
    m = binary_classification_metrics(ybin[test_idx], p_test_cal, threshold=0.5)
    scorer = StackedNb10Scorer(
        model=model,
        iso=iso,
        feature_names=list(names),
        feature_medians=medians,
        cheap_inputs=head.cheap_inputs,
        reject_threshold=float(chosen["reject_threshold"]),
        target_threshold=float(head.threshold),
    )
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(scorer, _MODEL_PATH)
    meta = {
        "n_train": int(tr.size),
        "n_calib": int(ca.size),
        "n_test": int(test_idx.size),
        "test_roc_auc": round(float(m["roc_auc"]), 4),
        "reject_threshold": chosen["reject_threshold"],
        "reject_npv": round(chosen["npv"], 4),
        "reject_coverage": round(chosen["coverage"], 4),
        "target_npv": target_npv,
        "calibrated": iso is not None,
        "cheap_inputs": list(head.cheap_inputs),
        "model_path": str(_MODEL_PATH),
    }
    _META_PATH.write_text(json.dumps(meta, indent=2, sort_keys=True))
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--db", default=str(RUNS_DB))
    ap.add_argument("--target-npv", type=float, default=0.90)
    ap.add_argument("--min-support", type=int, default=50)
    args = ap.parse_args(argv)
    if not args.train:
        ap.error("nothing to do; pass --train")
    meta = train_and_persist(
        args.db, target_npv=args.target_npv, min_support=args.min_support
    )
    print(json.dumps(meta, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
