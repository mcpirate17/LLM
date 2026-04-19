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
    notes = {
        "screening_ensemble": (
            "Ranks screening candidates and can hard-skip low P(pass_s1) graphs "
            "when enabled."
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
