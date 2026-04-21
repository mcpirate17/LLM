from __future__ import annotations

from typing import Any, Dict

from ..ml_influence_policy import (
    build_ml_influence_policy,
    load_predictor_metrics_report,
)
from ..runner._types import RunConfig


def build_ml_influence_status() -> Dict[str, Any]:
    config = RunConfig()
    report = load_predictor_metrics_report()
    policy = build_ml_influence_policy(config=config, report=report)
    components = dict(policy.get("components") or {})
    ensemble_section = report.get("ensemble_calibrated") or {}
    temporal_f1_threshold = (
        (
            (ensemble_section.get("temporal_holdout_evaluation") or {}).get(
                "operating_points"
            )
            or {}
        )
        .get("f1", {})
        .get("threshold")
    )
    saved_threshold = (
        ensemble_section.get("saved_runtime_artifact_evaluation") or {}
    ).get("selected_threshold")
    explicit_floor = float(config.screening_ensemble_p_pass_floor or 0.0)
    deprecated_floor = float(config.gbm_gate_threshold or 0.0)
    if explicit_floor > 0.0:
        resolved_floor = explicit_floor
        resolved_floor_source = "config.screening_ensemble_p_pass_floor"
    elif deprecated_floor > 0.0:
        resolved_floor = deprecated_floor
        resolved_floor_source = "config.gbm_gate_threshold"
    elif temporal_f1_threshold is not None:
        resolved_floor = float(temporal_f1_threshold)
        resolved_floor_source = (
            "ensemble.temporal_holdout_evaluation.operating_points.f1.threshold"
        )
    else:
        resolved_floor = float(saved_threshold or 0.0)
        resolved_floor_source = (
            "ensemble.saved_runtime_artifact_evaluation.selected_threshold"
        )
    notes = {
        "screening_ensemble": (
            "Ranks screening candidates and can hard-skip low ensemble-calibrated "
            "P(pass_s1) graphs when enabled. planning_score is a ranking blend, "
            "not a calibrated probability."
        ),
        "gbm_gate": "Primary graph-structure gate inside the screening ensemble.",
        "graph_predictor": (
            "Topology-aware sidecar used by the screening ensemble for pass, rank, "
            "loss, and induction hints."
        ),
        "learned_candidate_weights": (
            "Biases graph generation before compile/eval using notebook-derived weights."
        ),
        "screening_signal_weights": (
            "Biases op/template/motif weighting from screening outcomes and synergy "
            "signals during grammar construction."
        ),
        "learned_grammar_weights": "Biases grammar category weights from attribution analytics.",
        "investigation_predictor": (
            "Filters investigation-stage leaderboard candidates by predicted loss_ratio; "
            "does not influence graph generation."
        ),
    }
    for name, status in components.items():
        status["enabled_by_default"] = bool(status.get("requested"))
        status["active_in_search"] = (
            bool(status.get("allowed")) and name != "investigation_predictor"
        )
        status["note"] = notes.get(name, "")

    return {
        "defaults": {
            "gbm_prescreener_enabled": bool(config.gbm_prescreener_enabled),
            "screening_ensemble_p_pass_floor": float(
                config.screening_ensemble_p_pass_floor
            ),
            "gbm_gate_threshold": float(config.gbm_gate_threshold),
            "resolved_screening_ensemble_p_pass_floor": float(resolved_floor),
            "resolved_screening_ensemble_p_pass_floor_source": resolved_floor_source,
            "use_learned_candidate_weights": bool(config.use_learned_candidate_weights),
            "use_screening_signal_weights": bool(config.use_screening_signal_weights),
            "use_learned_grammar_weights": bool(config.use_learned_grammar_weights),
            "investigation_predictor_enabled": bool(
                config.investigation_predictor_enabled
            ),
            "allow_unproven_ml_influence": bool(config.allow_unproven_ml_influence),
        },
        "active_generation_influencers": list(
            policy.get("active_generation_influencers") or []
        ),
        "components": components,
    }
