"""Best-effort NAS pipeline screening for fab proposal specs.

The NAS pipeline scores graph candidates, while component_fab emits single
component specs. This adapter builds a small proxy graph from codegen-relevant
axes and asks the existing CPU/NAS oracle to score that proxy. It is a cheap
screen only; Tier-2 remains the downstream evidence gate.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from component_fab.proposer.measured_screen import (
    LONG_RANGE_THRESHOLD,
    MAX_CAUSALITY_VIOLATION,
    measured_screen_for_spec,
)
from component_fab.proposer.spec_generator import ProposalSpec

logger = logging.getLogger(__name__)
_REPO = Path(__file__).resolve().parents[2]
_ORACLE_META = (
    _REPO / "research" / "runtime" / "pls_partition_oracle" / "oracle_meta.json"
)
_PREDICTOR_REPORT = (
    _REPO / "research" / "runtime" / "learning" / "predictor_metrics_report.json"
)


@dataclass(frozen=True, slots=True)
class NasScreenResult:
    proposal_id: str
    available: bool
    gate_pass: bool
    downstream_gate_pass: bool
    rank_score: float
    source: str
    reason: str = ""
    calibration: dict[str, Any] | None = None
    raw: dict[str, Any] | None = None


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def nas_calibration_context() -> dict[str, Any]:
    """Return PPV/NPV/ROC context for interpreting NAS screen decisions."""

    oracle_meta = _load_json(_ORACLE_META)
    predictor = _load_json(_PREDICTOR_REPORT)
    graph_predictor = predictor.get("graph_predictor") or {}
    selected_metrics = (
        graph_predictor.get("saved_runtime_artifact_evaluation", {}).get(
            "selected_metrics"
        )
        or graph_predictor.get("val_metrics_selected_threshold")
        or graph_predictor.get("derived_val_classification_metrics")
        or {}
    )
    temporal_metrics = (
        graph_predictor.get("temporal_holdout_evaluation", {}).get("selected_metrics")
        or {}
    )
    axes = {}
    for axis, row in (oracle_meta.get("selected_per_axis") or {}).items():
        if not isinstance(row, Mapping):
            continue
        axes[str(axis)] = {
            "kind": row.get("kind"),
            "leave_family_out_roc": row.get("leave_family_out_roc"),
        }
    return {
        "oracle_axes": axes,
        "oracle_thresholds": oracle_meta.get("thresholds") or {},
        "graph_predictor_selected": {
            key: selected_metrics.get(key)
            for key in (
                "roc_auc",
                "precision_ppv",
                "npv",
                "recall_tpr_sensitivity",
                "specificity_tnr",
                "threshold",
            )
            if key in selected_metrics
        },
        "graph_predictor_temporal": {
            key: temporal_metrics.get(key)
            for key in (
                "roc_auc",
                "precision_ppv",
                "npv",
                "recall_tpr_sensitivity",
                "specificity_tnr",
                "threshold",
            )
            if key in temporal_metrics
        },
    }


def _measured_calibration() -> dict[str, Any]:
    """Operating-point context for the measured-descriptor binding screen.

    The 2026-06-03 audit retired the oracle proxy-graph gate (anti-predictive,
    OOD on the 3-op stub). This screen instead reads the position-Jacobian of the
    REAL fab module. ``nas_calibration_context`` is still exposed for the
    oracle's own PPV/NPV/ROC, but the gate decision here is the measured one.
    """

    return {
        "screen": "measured_descriptors",
        "gate_axis": "long_range_reach",
        "gate_threshold": LONG_RANGE_THRESHOLD,
        "operating_point": "long_range_reach>=0.01 keeps 99.3% of "
        "induction-capable graphs, prunes ~55% of incapable (n=1102)",
        "note": "filters non-binders (MLP-class long_range_reach~0); not a "
        "fine-grained ranker within one architecture family",
        "oracle_reference": nas_calibration_context(),
    }


def score_spec_with_nas(
    spec: ProposalSpec, scorer: Any | None = None
) -> NasScreenResult:
    """Screen one spec on the MEASURED graph properties of its real module.

    ``scorer`` is an optional ``MeasuredDescriptorExtractor`` reused across specs.
    Replaces the former NAS oracle proxy-graph path (anti-predictive, see
    ``measured_screen``); same ``NasScreenResult`` contract so callers are
    unchanged.
    """

    ms = measured_screen_for_spec(spec, extractor=scorer)
    return NasScreenResult(
        proposal_id=spec.proposal_id,
        available=ms.available,
        gate_pass=ms.binds_likely,
        downstream_gate_pass=ms.causality_violation <= MAX_CAUSALITY_VIOLATION,
        rank_score=ms.rank_score,
        source="measured_descriptors" if ms.available else "unavailable",
        reason=ms.reason,
        calibration=_measured_calibration(),
        raw=ms.descriptors,
    )


def score_specs_with_nas(
    specs: Iterable[ProposalSpec],
    *,
    enabled: bool = True,
) -> dict[str, NasScreenResult]:
    if not enabled:
        return {}
    extractor: Any | None = None
    try:
        from research.tools.measured_descriptors import MeasuredDescriptorExtractor

        extractor = MeasuredDescriptorExtractor(n_seeds=2)
    except Exception as exc:  # noqa: BLE001
        logger.debug("measured-descriptor extractor unavailable: %s", exc)
    return {
        spec.proposal_id: score_spec_with_nas(spec, scorer=extractor) for spec in specs
    }


def nas_score_multiplier(result: NasScreenResult | None) -> float:
    if result is None or not result.available:
        return 1.0
    if not result.gate_pass:
        return 0.55
    if not result.downstream_gate_pass:
        return 0.70
    if result.rank_score >= 1.25:
        return 1.08
    if result.rank_score >= 1.0:
        return 1.03
    return 1.0
