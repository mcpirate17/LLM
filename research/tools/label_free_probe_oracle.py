"""Runtime adapter for the label-free probe oracle.

The persisted ``pls_partition_oracle`` predicts measured probe axes from graph
property features.  Generated candidates are represented by graph semantics.  The
AR axis is exposed as a hard no-go gate; non-AR axes are exposed as rank signals,
without S1 pass/fail, validation loss, leaderboard score, or other outcome labels
at scoring time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np

from research.defaults import RUNS_DB

logger = logging.getLogger(__name__)

AR_GATE_AXIS = "ar_gate"
DEFAULT_PROBE_AXES = (AR_GATE_AXIS, "nano_induction_nearest")
DEFAULT_RANK_AXES = ("nano_induction_nearest", "induction", "ar_curriculum")


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def probe_axis_score(
    predicted: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    *,
    axes: Iterable[str] = DEFAULT_PROBE_AXES,
) -> tuple[float, Dict[str, Dict[str, float]]]:
    """Normalize AR/nano probe predictions by their trained thresholds."""
    details: Dict[str, Dict[str, float]] = {}
    ratios: list[float] = []
    for axis in axes:
        pred = _finite_float(predicted.get(axis))
        thr = _finite_float(thresholds.get(axis))
        if pred is None or thr is None or thr <= 0.0:
            continue
        ratio = pred / thr
        ratios.append(ratio)
        details[axis] = {
            "predicted": round(pred, 6),
            "threshold": round(thr, 6),
            "ratio": round(ratio, 6),
        }
    return (max(ratios) if ratios else 0.0), details


def probe_axis_gate(
    predicted: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    *,
    axis: str = AR_GATE_AXIS,
) -> Dict[str, Any]:
    """Return a hard gate decision for one probe axis."""
    pred = _finite_float(predicted.get(axis))
    thr = _finite_float(thresholds.get(axis))
    passed = pred is not None and thr is not None and thr > 0.0 and pred >= thr
    detail: Dict[str, Any] = {
        "axis": axis,
        "passed": bool(passed),
    }
    if pred is not None:
        detail["predicted"] = round(pred, 6)
    if thr is not None:
        detail["threshold"] = round(thr, 6)
    if pred is not None and thr is not None and thr > 0.0:
        detail["ratio"] = round(pred / thr, 6)
    return detail


def probe_any_axis_gate(
    predicted: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    *,
    axes: Iterable[str] = DEFAULT_RANK_AXES,
) -> Dict[str, Any]:
    """Require at least one downstream capability axis to clear its threshold."""
    _, details = probe_axis_score(predicted, thresholds, axes=axes)
    passed_axes = [
        axis
        for axis, detail in details.items()
        if float(detail.get("ratio", 0.0)) >= 1.0
    ]
    best_axis = None
    best_ratio = 0.0
    for axis, detail in details.items():
        ratio = float(detail.get("ratio", 0.0))
        if ratio > best_ratio:
            best_axis = axis
            best_ratio = ratio
    return {
        "passed": bool(passed_axes),
        "axes": tuple(axes),
        "passed_axes": passed_axes,
        "best_axis": best_axis,
        "best_ratio": round(best_ratio, 6),
    }


@dataclass
class LabelFreeProbeOracleScorer:
    """Scores graph dicts with the persisted probe-axis oracle."""

    oracle: Any
    extractor: Any
    thresholds: Dict[str, float]
    source: str = "pls_partition_oracle"

    @classmethod
    def load(
        cls,
        *,
        runs_db: str = str(RUNS_DB),
        meta_db: str = "research/meta_analysis.db",
    ) -> "LabelFreeProbeOracleScorer":
        from research.tools.graph_semantic_features import GraphSemanticExtractor
        from research.tools.pls_partition_oracle import AxisOracle, _META_PATH

        oracle = AxisOracle.load()
        thresholds = dict(getattr(oracle, "thresholds", {}) or {})
        if not all(axis in thresholds for axis in DEFAULT_PROBE_AXES):
            meta_path = Path(_META_PATH)
            raise RuntimeError(
                "persisted probe oracle is missing AR/nano axes"
                + (f" ({meta_path})" if meta_path.exists() else "")
            )
        return cls(
            oracle=oracle,
            extractor=GraphSemanticExtractor(runs_db, meta_db),
            thresholds={k: float(v) for k, v in thresholds.items()},
        )

    @classmethod
    def try_load(
        cls,
        *,
        runs_db: str = str(RUNS_DB),
        meta_db: str = "research/meta_analysis.db",
    ) -> "LabelFreeProbeOracleScorer | None":
        try:
            return cls.load(runs_db=runs_db, meta_db=meta_db)
        except Exception as exc:  # noqa: BLE001
            logger.info("label-free probe oracle unavailable: %s", exc)
            return None

    def score_nodes(self, nodes: Dict[str, Any] | list[Any]) -> Dict[str, Any]:
        features = self.extractor.features(nodes)
        decision = self.oracle.evaluate_features(features)
        predicted = dict(decision.get("predicted") or {})
        score, details = probe_axis_score(predicted, self.thresholds)
        rank_score, rank_details = probe_axis_score(
            predicted,
            self.thresholds,
            axes=DEFAULT_RANK_AXES,
        )
        gate = probe_axis_gate(predicted, self.thresholds)
        downstream_gate = probe_any_axis_gate(
            predicted,
            self.thresholds,
            axes=DEFAULT_RANK_AXES,
        )
        return {
            "label_free_probe_gate": gate,
            "label_free_probe_gate_pass": bool(gate["passed"]),
            "label_free_probe_downstream_gate": downstream_gate,
            "label_free_probe_downstream_gate_pass": bool(downstream_gate["passed"]),
            "label_free_probe_rank_score": round(float(rank_score), 6),
            "label_free_probe_rank_axes": rank_details,
            "label_free_probe_score": round(float(score), 6),
            "label_free_probe_axes": details,
            "label_free_probe_predictions": predicted,
            "label_free_probe_recommendation": decision.get("recommendation"),
            "label_free_probe_novelty_pctile": decision.get("novelty_pctile"),
            "label_free_probe_source": self.source,
        }

    def score_graph_dict(self, graph_dict: Dict[str, Any]) -> Dict[str, Any]:
        nodes = graph_dict.get("nodes", graph_dict)
        return self.score_nodes(nodes)
