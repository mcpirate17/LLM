from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .runner._types import RunConfig

_PREDICTOR_REPORT_PATH = Path("research/runtime/learning/predictor_metrics_report.json")

# Lowered from 0.85/0.70 to reflect F1-optimal operating point.
# Old high_precision threshold gave PPV≥0.70 but recall ~0.50 — starving
# downstream models of labeled data.  At F1 threshold PPV is ~0.50 (1.7×
# lift over 30% prevalence), which is usable for screening, and recall
# jumps to ~0.80.  AUC 0.80 matches the honest full-split evaluation.
_SCREENING_ENSEMBLE_MIN_AUC = 0.80
_SCREENING_ENSEMBLE_MIN_PPV = 0.45
_INVESTIGATION_PREDICTOR_MIN_SPEARMAN = 0.25
_INVESTIGATION_PREDICTOR_MIN_N_TEST = 100


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def load_predictor_metrics_report(
    path: str | Path = _PREDICTOR_REPORT_PATH,
) -> Dict[str, Any]:
    report_path = Path(path)
    if not report_path.exists():
        return {}
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _quality_tier_from_auc(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value >= 0.85:
        return "strong"
    if value >= 0.75:
        return "usable"
    return "weak"


def build_ml_influence_policy(
    config: RunConfig | None = None,
    report: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    cfg = config or RunConfig()
    metrics = report if report is not None else load_predictor_metrics_report()
    allow_unproven = bool(getattr(cfg, "allow_unproven_ml_influence", False))

    ensemble_metrics = (metrics.get("ensemble_calibrated") or {}).get(
        "val_metrics_selected_threshold"
    ) or {}
    gbm_metrics = (metrics.get("gbm_gate") or {}).get(
        "val_metrics_selected_threshold"
    ) or {}
    graph_metrics = (metrics.get("graph_predictor") or {}).get(
        "derived_val_classification_metrics"
    ) or {}
    investigation_metrics = metrics.get("investigation_predictor") or {}

    ensemble_auc = _float_or_none(ensemble_metrics.get("roc_auc"))
    ensemble_ppv = _float_or_none(ensemble_metrics.get("precision_ppv"))
    investigation_spearman = _float_or_none(investigation_metrics.get("spearman_rho"))
    investigation_n_test = int(investigation_metrics.get("n_test") or 0)

    screening_proven = bool(
        ensemble_auc is not None
        and ensemble_ppv is not None
        and ensemble_auc >= _SCREENING_ENSEMBLE_MIN_AUC
        and ensemble_ppv >= _SCREENING_ENSEMBLE_MIN_PPV
    )
    investigation_proven = bool(
        investigation_spearman is not None
        and investigation_spearman >= _INVESTIGATION_PREDICTOR_MIN_SPEARMAN
        and investigation_n_test >= _INVESTIGATION_PREDICTOR_MIN_N_TEST
    )

    components = {
        "screening_ensemble": {
            "requested": bool(cfg.gbm_prescreener_enabled),
            "proven": screening_proven,
            "allowed": bool(
                cfg.gbm_prescreener_enabled and (screening_proven or allow_unproven)
            ),
            "quality_tier": _quality_tier_from_auc(ensemble_auc),
            "roc_auc": ensemble_auc,
            "precision_ppv": ensemble_ppv,
            "reason": (
                "meets_screening_thresholds"
                if screening_proven
                else "heldout_metrics_below_threshold"
            ),
        },
        "gbm_gate": {
            "quality_tier": _quality_tier_from_auc(
                _float_or_none(gbm_metrics.get("roc_auc"))
            ),
            "roc_auc": _float_or_none(gbm_metrics.get("roc_auc")),
        },
        "graph_predictor": {
            "quality_tier": _quality_tier_from_auc(
                _float_or_none(graph_metrics.get("roc_auc"))
            ),
            "roc_auc": _float_or_none(graph_metrics.get("roc_auc")),
        },
        "learned_candidate_weights": {
            "requested": bool(cfg.use_learned_candidate_weights),
            "proven": False,
            "allowed": bool(cfg.use_learned_candidate_weights and allow_unproven),
            "quality_tier": "unproven",
            "reason": "requires_manual_override_until_validated",
        },
        "screening_signal_weights": {
            "requested": bool(cfg.use_screening_signal_weights),
            "proven": False,
            "allowed": bool(cfg.use_screening_signal_weights and allow_unproven),
            "quality_tier": "unproven",
            "reason": "requires_manual_override_until_validated",
        },
        "learned_grammar_weights": {
            "requested": bool(cfg.use_learned_grammar_weights),
            "proven": False,
            "allowed": bool(cfg.use_learned_grammar_weights and allow_unproven),
            "quality_tier": "unproven",
            "reason": "requires_manual_override_until_validated",
        },
        "investigation_predictor": {
            "requested": bool(cfg.investigation_predictor_enabled),
            "proven": investigation_proven,
            "allowed": bool(
                cfg.investigation_predictor_enabled
                and (investigation_proven or allow_unproven)
            ),
            "quality_tier": ("usable" if investigation_proven else "monitor_only"),
            "spearman_rho": investigation_spearman,
            "n_test": investigation_n_test,
            "reason": (
                "meets_investigation_thresholds"
                if investigation_proven
                else "heldout_metrics_below_threshold"
            ),
        },
    }
    return {
        "allow_unproven_ml_influence": allow_unproven,
        "components": components,
        "active_generation_influencers": [
            name
            for name in (
                "screening_ensemble",
                "learned_candidate_weights",
                "screening_signal_weights",
                "learned_grammar_weights",
            )
            if components[name]["allowed"]
        ],
    }


def component_is_allowed(
    component_name: str,
    config: RunConfig,
    report: Dict[str, Any] | None = None,
) -> bool:
    policy = build_ml_influence_policy(config=config, report=report)
    return bool((policy.get("components") or {}).get(component_name, {}).get("allowed"))
