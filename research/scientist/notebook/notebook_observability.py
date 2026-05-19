"""Mixin for LabNotebook — split from notebook_misc."""

from __future__ import annotations

import json
import math
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from .graph_artifacts import resolve_graph_json_value
from ._notebook_misc_shared import (
    _cached_extract_observability_metadata,
    _ObservabilityAccumulator,
    _capability_signal_count,
    _reference_metric_baselines,
    _reference_beating_metrics,
    _template_label_from_evidence,
    _summarize_template_stat,
    _empty_template_stat,
    _append_language_control_metrics,
    _empty_language_control_metrics,
    _summarize_language_control_metrics,
    _discover_template_names,
    _TEMPLATE_DEF_RE,
)
from ..json_utils import fast_loads as _json_loads


_TEMPLATE_OBSERVABILITY_PROCESS_CACHE: Dict[
    tuple[str, int], tuple[tuple[Any, ...], float, Dict[str, Any]]
] = {}


def clear_template_observability_process_cache() -> None:
    """Clear cross-notebook template observability cache after writes."""
    _TEMPLATE_OBSERVABILITY_PROCESS_CACHE.clear()


def _bounded_prior(value: float, *, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _template_prior_weights(
    template_rows: list,
    *,
    min_support: int,
) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    evidence_bonuses = {
        "established": 0.25,
        "building": 0.10,
        "sparse": 0.0,
        "insufficient": -0.05,
    }
    for row in template_rows:
        support = int(row.get("n_used") or 0)
        name = str(row.get("name") or "").strip()
        if support < min_support or not name:
            continue
        loss_ratio = row.get("avg_validation_loss_ratio")
        if loss_ratio is None:
            loss_ratio = row.get("avg_loss_ratio")
        loss_term = (
            0.0
            if loss_ratio is None
            else _bounded_prior(1.1 - float(loss_ratio), lo=0.0, hi=1.0)
        )
        evidence_bonus = evidence_bonuses.get(str(row.get("evidence_level") or ""), 0.0)
        weight = (
            0.35
            + (1.45 * float(row.get("s1_rate") or 0.0))
            + (0.45 * loss_term)
            + evidence_bonus
        )
        weights[name] = round(_bounded_prior(weight, lo=0.2, hi=3.5), 4)
    return weights


def _motif_prior_weights(
    motif_rows: list,
    *,
    min_support: int,
    toxic_reason_tokens: tuple[str, ...],
) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for row in motif_rows:
        support = int(row.get("n_used") or 0)
        name = str(row.get("name") or "").strip()
        if support < min_support or not name:
            continue
        loss_ratio = row.get("avg_loss_ratio")
        loss_term = (
            0.0
            if loss_ratio is None
            else _bounded_prior(1.05 - float(loss_ratio), lo=0.0, hi=1.0)
        )
        failure_reason = str(row.get("top_failure_reason") or "").lower()
        toxic_penalty = (
            0.2 if any(t in failure_reason for t in toxic_reason_tokens) else 0.0
        )
        weight = (
            0.25
            + (1.55 * float(row.get("s1_rate") or 0.0))
            + (0.35 * loss_term)
            - toxic_penalty
        )
        weights[name] = round(_bounded_prior(weight, lo=0.15, hi=3.0), 4)
    return weights


def _slot_generation_priors(
    slot_rows: list,
    *,
    min_support: int,
    toxic_reason_tokens: tuple[str, ...],
) -> tuple[Dict[str, Dict[str, float]], Dict[str, List[str]]]:
    slot_multipliers: Dict[str, Dict[str, float]] = {}
    slot_denylist: Dict[str, List[str]] = {}
    for row in slot_rows:
        support = int(row.get("n_used") or 0)
        motif_name = str(row.get("top_selected_motif") or "").strip()
        slot_key = str(row.get("slot_key") or "").strip()
        if support < min_support or not slot_key or not motif_name:
            continue
        s1_rate = float(row.get("s1_rate") or 0.0)
        loss_ratio = row.get("avg_loss_ratio")
        loss_term = (
            0.0
            if loss_ratio is None
            else _bounded_prior(1.05 - float(loss_ratio), lo=0.0, hi=1.0)
        )
        slot_map = slot_multipliers.setdefault(slot_key, {})
        if s1_rate >= 0.55:
            slot_map[motif_name] = round(
                _bounded_prior(
                    1.05 + (0.65 * s1_rate) + (0.20 * loss_term),
                    lo=1.0,
                    hi=2.5,
                ),
                4,
            )
        elif s1_rate <= 0.18:
            slot_map[motif_name] = round(
                _bounded_prior(0.15 + (0.85 * s1_rate), lo=0.1, hi=0.55),
                4,
            )
            failure_reason = str(row.get("top_failure_reason") or "").lower()
            if any(t in failure_reason for t in toxic_reason_tokens):
                slot_denylist.setdefault(slot_key, []).append(motif_name)
    return slot_multipliers, slot_denylist


def _finite_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_tuple(value: Any) -> tuple[str, ...]:
    try:
        return tuple(
            str(item) for item in (_json_loads(value) or []) if item is not None
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        return ()


def _json_slot_tuple(value: Any) -> tuple[Dict[str, Any], ...]:
    try:
        loaded = _json_loads(value) or []
    except (json.JSONDecodeError, TypeError, ValueError):
        return ()
    return tuple(item for item in loaded if isinstance(item, dict))


def _observability_metric_values(row: Any) -> Dict[str, Any]:
    return {
        "loss_ratio": row["loss_ratio"],
        "validation_loss_ratio": row["validation_loss_ratio"],
        "discovery_loss_ratio": row["discovery_loss_ratio"],
        "novelty_score": row["novelty_score"],
        "novelty_confidence": row["novelty_confidence"],
        "induction_screening_auc": row["induction_screening_auc"],
        "binding_screening_auc": (
            row["binding_curriculum_auc"]
            if row["binding_curriculum_auc"] is not None
            else row["binding_screening_auc"]
        ),
        "binding_screening_composite": row["binding_screening_composite"],
        "ar_legacy_auc": row["ar_legacy_auc"],
        "hellaswag_acc": row["hellaswag_acc"],
        "blimp_overall_accuracy": row["blimp_overall_accuracy"],
        "composite_score": row["composite_score"],
        "induction_intermediate_auc": row["induction_intermediate_auc"],
        "binding_intermediate_auc": row["binding_intermediate_auc"],
        "ar_curriculum_auc_pair_final": row["ar_curriculum_auc_pair_final"],
        "ar_curriculum_s0_retention": row["ar_curriculum_s0_retention"],
        "ar_curriculum_max_passing_stage": row["ar_curriculum_max_passing_stage"],
        "fp_jacobian_effective_rank": row["fp_jacobian_effective_rank"],
        "fp_sensitivity_uniformity": row["fp_sensitivity_uniformity"],
        "fp_jacobian_erf_density": row["fp_jacobian_erf_density"],
        "fp_id_collapse_rate": row["fp_id_collapse_rate"],
        "fp_id_collapse_rate_normalized": row["fp_id_collapse_rate_normalized"],
        "fp_jacobian_erf_decay_slope": row["fp_jacobian_erf_decay_slope"],
        "fp_jacobian_erf_first_norm": row["fp_jacobian_erf_first_norm"],
        "fp_jacobian_erf_last_norm": row["fp_jacobian_erf_last_norm"],
        "fp_logit_margin_velocity": row["fp_logit_margin_velocity"],
        "fp_logit_margin_delta": row["fp_logit_margin_delta"],
        "fp_jacobian_erf_variance_log": row["fp_jacobian_erf_variance_log"],
        "fp_jacobian_spectral_norm_log": row["fp_jacobian_spectral_norm_log"],
        "fp_icld_velocity": row["fp_icld_velocity"],
        "fp_icld_delta_loss": row["fp_icld_delta_loss"],
        "screening_hellaswag_correct": row["screening_hellaswag_correct"],
        "screening_hellaswag_total": row["screening_hellaswag_total"],
        "screening_wikitext_status": row["screening_wikitext_status"],
    }


def _language_control_values(row: Any) -> Dict[str, Any]:
    return {
        "language_control_s05_sentence_assoc_score": row[
            "language_control_s05_sentence_assoc_score"
        ],
        "language_control_s05_binding_order_acc": row[
            "language_control_s05_binding_order_acc"
        ],
        "language_control_s05_binding_score": row["language_control_s05_binding_score"],
        "language_control_s10_sentence_assoc_score": row[
            "language_control_s10_sentence_assoc_score"
        ],
        "language_control_s10_binding_order_acc": row[
            "language_control_s10_binding_order_acc"
        ],
        "language_control_s10_binding_score": row["language_control_s10_binding_score"],
        "language_control_investigation_sentence_assoc_score": row[
            "language_control_investigation_sentence_assoc_score"
        ],
        "language_control_investigation_binding_order_acc": row[
            "language_control_investigation_binding_order_acc"
        ],
        "language_control_investigation_binding_score": row[
            "language_control_investigation_binding_score"
        ],
    }


def _failure_root_cause(row: Any) -> str:
    failure_details = {}
    raw_failure = row["failure_details_json"]
    if raw_failure:
        try:
            failure_details = (
                _json_loads(raw_failure)
                if isinstance(raw_failure, str)
                else raw_failure
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            failure_details = {}
    return (
        failure_details.get("root_cause_code")
        or row["error_type"]
        or row["stage_at_death"]
        or "unknown"
    )


def _empty_observability_accumulator() -> _ObservabilityAccumulator:
    return _ObservabilityAccumulator(
        template_stats={},
        motif_stats={},
        slot_stats={},
        experiment_buckets={},
        loss_values=[],
        validation_losses=[],
        discovery_losses=[],
        motifs_per_graph=[],
        templates_per_graph=[],
    )


def _experiment_bucket(acc: _ObservabilityAccumulator, row: Any) -> Dict[str, Any]:
    experiment_id = str(row["experiment_id"] or "")
    bucket = acc.experiment_buckets.setdefault(
        experiment_id or f"exp_{len(acc.experiment_buckets)}",
        {
            "experiment_id": experiment_id or None,
            "timestamp": float(row["timestamp"] or 0.0),
            "templates": {},
            "slots": {},
            "training_losses": [],
            "validation_losses": [],
            "discovery_losses": [],
        },
    )
    bucket["timestamp"] = max(
        float(bucket.get("timestamp") or 0.0),
        float(row["timestamp"] or 0.0),
    )
    return bucket


def _append_global_losses(
    acc: _ObservabilityAccumulator, bucket: Dict[str, Any], metrics: Dict[str, Any]
) -> None:
    for metric_name, acc_values, bucket_key in (
        ("loss_ratio", acc.loss_values, "training_losses"),
        ("validation_loss_ratio", acc.validation_losses, "validation_losses"),
        ("discovery_loss_ratio", acc.discovery_losses, "discovery_losses"),
    ):
        value = _finite_float(metrics.get(metric_name))
        if value is not None:
            acc_values.append(value)
            bucket[bucket_key].append(value)


_TEMPLATE_METRIC_APPENDS = (
    ("validation_loss_ratio", "validation_losses"),
    ("discovery_loss_ratio", "discovery_losses"),
    ("novelty_score", "novelties"),
    ("novelty_confidence", "novelty_confidences"),
    ("induction_screening_auc", "induction_screening_aucs"),
    ("binding_screening_auc", "binding_screening_aucs"),
    ("binding_screening_composite", "binding_screening_composites"),
    ("ar_legacy_auc", "ar_legacy_aucs"),
    ("hellaswag_acc", "hellaswag_accs"),
    ("blimp_overall_accuracy", "blimp_overall_accuracies"),
    ("composite_score", "composite_scores"),
    ("induction_intermediate_auc", "induction_intermediate_aucs"),
    ("binding_intermediate_auc", "binding_intermediate_aucs"),
    ("ar_curriculum_auc_pair_final", "ar_curriculum_aucs"),
    ("ar_curriculum_s0_retention", "ar_curriculum_retentions"),
    ("fp_jacobian_erf_density", "erf_densities"),
    ("fp_id_collapse_rate", "id_collapse_rates"),
    ("fp_id_collapse_rate_normalized", "id_collapse_rate_normalizeds"),
    ("fp_jacobian_erf_decay_slope", "erf_decay_slopes"),
    ("fp_jacobian_erf_first_norm", "erf_first_norms"),
    ("fp_jacobian_erf_last_norm", "erf_last_norms"),
    ("fp_logit_margin_velocity", "logit_margin_velocities"),
    ("fp_logit_margin_delta", "logit_margin_deltas"),
    ("fp_jacobian_erf_variance_log", "erf_variance_logs"),
    ("fp_jacobian_spectral_norm_log", "spec_norm_logs"),
    ("fp_icld_velocity", "icld_velocities"),
    ("fp_icld_delta_loss", "icld_delta_losses"),
    ("fp_jacobian_effective_rank", "jacobian_effective_ranks"),
    ("fp_sensitivity_uniformity", "sensitivity_uniformities"),
)


_SLOT_METRIC_APPENDS = (
    ("composite_score", "composite_scores"),
    ("induction_screening_auc", "induction_screening_aucs"),
    ("induction_intermediate_auc", "induction_intermediate_aucs"),
    ("binding_screening_auc", "binding_screening_aucs"),
    ("binding_screening_composite", "binding_screening_composites"),
    ("binding_intermediate_auc", "binding_intermediate_aucs"),
    ("ar_legacy_auc", "ar_legacy_aucs"),
    ("ar_curriculum_auc_pair_final", "ar_curriculum_aucs"),
    ("ar_curriculum_s0_retention", "ar_curriculum_retentions"),
)


def _append_finite_metrics(
    stat: Dict[str, Any], metrics: Dict[str, Any], mappings: tuple[tuple[str, str], ...]
) -> None:
    for metric_name, stat_key in mappings:
        value = _finite_float(metrics.get(metric_name))
        if value is not None:
            stat[stat_key].append(value)


def _update_template_fast_lane(stat: Dict[str, Any], row: Any) -> None:
    if not row["routing_fast_lane_applied"]:
        return
    stat["routing_fast_lane_runs"] += 1
    if row["routing_fast_lane_status"] == "ok":
        stat["routing_fast_lane_ok"] += 1
    lane_score = _finite_float(row["routing_fast_lane_score"])
    lane_improvement = _finite_float(row["routing_fast_lane_ppl_improvement"])
    lane_slope = _finite_float(row["routing_fast_lane_slope"])
    slope_consistent = bool(row["routing_fast_lane_slope_consistent"])
    if lane_score is not None:
        stat["routing_fast_lane_scores"].append(lane_score)
    if lane_improvement is not None:
        stat["routing_fast_lane_improvements"].append(lane_improvement)
    if lane_slope is not None:
        stat["routing_fast_lane_slopes"].append(lane_slope)
    if (lane_improvement is not None and lane_improvement < 0.98) or (
        lane_slope is not None and lane_slope > 0 and slope_consistent
    ):
        stat["routing_fast_lane_positive"] += 1


def _update_template_stat(
    stat: Dict[str, Any],
    row: Any,
    metrics: Dict[str, Any],
    language_control_values: Dict[str, Any],
    root_cause: str,
) -> None:
    stat["n_used"] += 1
    stat["n_stage0"] += 1 if row["stage0_passed"] else 0
    stat["n_stage05"] += 1 if row["stage05_passed"] else 0
    stat["n_stage1"] += 1 if row["stage1_passed"] else 0
    fingerprint = str(row["graph_fingerprint"] or "").strip()
    if fingerprint:
        stat["fingerprints"].add(fingerprint)
        if row["stage1_passed"]:
            stat["stage1_fingerprints"].add(fingerprint)
    loss_ratio = _finite_float(metrics.get("loss_ratio"))
    if loss_ratio is not None:
        stat["losses"].append(loss_ratio)
        if row["stage1_passed"]:
            stat["stage1_losses"].append(loss_ratio)
    _append_finite_metrics(stat, metrics, _TEMPLATE_METRIC_APPENDS)
    if metrics.get("ar_curriculum_max_passing_stage") is not None:
        stat["ar_curriculum_max_passes"].append(
            int(metrics["ar_curriculum_max_passing_stage"])
        )
    _append_language_control_metrics(stat, language_control_values)
    _update_template_screening_metrics(stat, metrics)
    _update_template_fast_lane(stat, row)
    if not row["stage1_passed"]:
        stat["failure_reasons"][root_cause] = (
            stat["failure_reasons"].get(root_cause, 0) + 1
        )


def _update_template_screening_metrics(
    stat: Dict[str, Any], metrics: Dict[str, Any]
) -> None:
    hs_correct = metrics.get("screening_hellaswag_correct")
    hs_total = metrics.get("screening_hellaswag_total")
    if hs_correct is not None and hs_total is not None and hs_total:
        stat["screening_hellaswag_accs"].append(
            float(hs_correct) / max(float(hs_total), 1.0)
        )
    status = metrics.get("screening_wikitext_status")
    if status is not None:
        stat["screening_wikitext_runs"] += 1
        if str(status) == "ok":
            stat["screening_wikitext_ok"] += 1


def _record_template_observations(
    acc: _ObservabilityAccumulator,
    bucket: Dict[str, Any],
    templates: tuple[str, ...],
    row: Any,
    metrics: Dict[str, Any],
    language_control_values: Dict[str, Any],
    root_cause: str,
    slot_counts: Dict[str, int],
) -> None:
    loss_ratio = _finite_float(metrics.get("loss_ratio"))
    for template in templates:
        name = str(template)
        stat = acc.template_stats.setdefault(
            name,
            _empty_template_stat(name=name, slot_count=slot_counts.get(name, 0)),
        )
        _update_template_stat(stat, row, metrics, language_control_values, root_cause)
        trend = bucket["templates"].setdefault(name, {"n": 0, "s1": 0, "losses": []})
        _update_trend_item(trend, bool(row["stage1_passed"]), loss_ratio)


def _update_motif_stat(
    stat: Dict[str, Any], row: Any, loss_ratio: Optional[float], root_cause: str
) -> None:
    stat["n_used"] += 1
    stat["n_stage1"] += 1 if row["stage1_passed"] else 0
    if loss_ratio is not None:
        stat["losses"].append(loss_ratio)
    if not row["stage1_passed"]:
        stat["failure_reasons"][root_cause] = (
            stat["failure_reasons"].get(root_cause, 0) + 1
        )


def _record_motif_observations(
    acc: _ObservabilityAccumulator,
    motifs: tuple[str, ...],
    row: Any,
    loss_ratio: Optional[float],
    root_cause: str,
) -> None:
    for motif in motifs:
        name = str(motif)
        stat = acc.motif_stats.setdefault(
            name,
            {
                "name": name,
                "n_used": 0,
                "n_stage1": 0,
                "losses": [],
                "failure_reasons": {},
            },
        )
        _update_motif_stat(stat, row, loss_ratio, root_cause)


def _slot_key(slot: Dict[str, Any]) -> str:
    return str(
        slot.get("slot_key")
        or f"{slot.get('template_name', 'unknown')}.slot{slot.get('slot_index', 0)}"
    )


def _empty_slot_stat(slot: Dict[str, Any], slot_key: str) -> Dict[str, Any]:
    return {
        "slot_key": slot_key,
        "template_name": str(slot.get("template_name") or "unknown"),
        "slot_index": int(slot.get("slot_index") or 0),
        "slot_classes": list(slot.get("slot_classes") or []),
        "n_used": 0,
        "n_stage1": 0,
        "losses": [],
        "composite_scores": [],
        "induction_screening_aucs": [],
        "induction_intermediate_aucs": [],
        "binding_screening_aucs": [],
        "binding_screening_composites": [],
        "binding_intermediate_aucs": [],
        "ar_legacy_aucs": [],
        "language_control_metrics": _empty_language_control_metrics(),
        "ar_curriculum_aucs": [],
        "ar_curriculum_retentions": [],
        "ar_curriculum_max_passes": [],
        "failure_reasons": {},
        "selected_motifs": {},
    }


def _update_slot_stat(
    stat: Dict[str, Any],
    slot: Dict[str, Any],
    row: Any,
    metrics: Dict[str, Any],
    language_control_values: Dict[str, Any],
    root_cause: str,
) -> None:
    stat["n_used"] += 1
    stat["n_stage1"] += 1 if row["stage1_passed"] else 0
    loss_ratio = _finite_float(metrics.get("loss_ratio"))
    if loss_ratio is not None:
        stat["losses"].append(loss_ratio)
    _append_finite_metrics(stat, metrics, _SLOT_METRIC_APPENDS)
    if metrics.get("ar_curriculum_max_passing_stage") is not None:
        stat["ar_curriculum_max_passes"].append(
            int(metrics["ar_curriculum_max_passing_stage"])
        )
    _append_language_control_metrics(stat, language_control_values)
    motif_name = slot.get("selected_motif")
    if motif_name:
        motif_key = str(motif_name)
        stat["selected_motifs"][motif_key] = (
            stat["selected_motifs"].get(motif_key, 0) + 1
        )
    if not row["stage1_passed"]:
        stat["failure_reasons"][root_cause] = (
            stat["failure_reasons"].get(root_cause, 0) + 1
        )


def _record_slot_observations(
    acc: _ObservabilityAccumulator,
    bucket: Dict[str, Any],
    slot_usage: tuple[Dict[str, Any], ...],
    row: Any,
    metrics: Dict[str, Any],
    language_control_values: Dict[str, Any],
    root_cause: str,
) -> None:
    loss_ratio = _finite_float(metrics.get("loss_ratio"))
    for slot in slot_usage:
        slot_key = _slot_key(slot)
        stat = acc.slot_stats.setdefault(slot_key, _empty_slot_stat(slot, slot_key))
        _update_slot_stat(stat, slot, row, metrics, language_control_values, root_cause)
        trend = bucket["slots"].setdefault(slot_key, {"n": 0, "s1": 0, "losses": []})
        _update_trend_item(trend, bool(row["stage1_passed"]), loss_ratio)


def _update_trend_item(
    trend: Dict[str, Any], stage1_passed: bool, loss_ratio: Optional[float]
) -> None:
    trend["n"] += 1
    trend["s1"] += 1 if stage1_passed else 0
    if loss_ratio is not None:
        trend["losses"].append(loss_ratio)


def _mean_or_none(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _observability_loss_distribution(
    owner: Any, acc: _ObservabilityAccumulator
) -> Dict[str, Dict[str, Optional[float]]]:
    return {
        "training": {
            "median": owner._percentile(acc.loss_values, 0.5),
            "p25": owner._percentile(acc.loss_values, 0.25),
            "p75": owner._percentile(acc.loss_values, 0.75),
        },
        "validation": {
            "median": owner._percentile(acc.validation_losses, 0.5),
            "p25": owner._percentile(acc.validation_losses, 0.25),
            "p75": owner._percentile(acc.validation_losses, 0.75),
        },
        "discovery": {
            "median": owner._percentile(acc.discovery_losses, 0.5),
            "p25": owner._percentile(acc.discovery_losses, 0.25),
            "p75": owner._percentile(acc.discovery_losses, 0.75),
        },
    }


def _prepare_template_rows(
    acc: _ObservabilityAccumulator, slot_counts: Dict[str, int]
) -> tuple[
    frozenset[str], list[Dict[str, Any]], list[Dict[str, Any]], list[Dict[str, Any]]
]:
    active_template_names = frozenset(_discover_template_names())
    template_stats = dict(acc.template_stats)
    for name in active_template_names:
        template_stats.setdefault(
            str(name),
            _empty_template_stat(
                name=str(name), slot_count=slot_counts.get(str(name), 0)
            ),
        )
    template_rows = [_summarize_template_stat(stat) for stat in template_stats.values()]
    reference_baselines = _reference_metric_baselines(template_rows)
    for row in template_rows:
        row["capability_signal_count"] = _capability_signal_count(row)
        row["reference_beating_metrics"] = _reference_beating_metrics(
            row, reference_baselines
        )
        row["structural_category"] = _template_label_from_evidence(
            row, reference_baselines
        )
    active_rows = [row for row in template_rows if row["name"] in active_template_names]
    inactive_rows = [
        row
        for row in template_rows
        if row["name"] not in active_template_names and row["n_used"] > 0
    ]
    return active_template_names, template_rows, active_rows, inactive_rows


def _template_sort_loss(row: Dict[str, Any]) -> float:
    return (
        row["avg_validation_loss_ratio"]
        if row["avg_validation_loss_ratio"] is not None
        else row["avg_loss_ratio"]
        if row["avg_loss_ratio"] is not None
        else 999.0
    )


def _rank_template_rows(
    active_rows: list[Dict[str, Any]], inactive_rows: list[Dict[str, Any]], limit: int
) -> Dict[str, list[Dict[str, Any]]]:
    evidence_rank = {"insufficient": 0, "sparse": 1, "building": 2, "established": 3}
    return {
        "top_templates": sorted(
            active_rows,
            key=lambda row: (
                -(row["s1_rate"] or 0.0),
                _template_sort_loss(row),
                -(row["n_used"] or 0),
            ),
        )[:limit],
        "struggling_templates": sorted(
            [row for row in active_rows if row["n_used"] >= 3],
            key=lambda row: (
                row["s1_rate"] or 0.0,
                _template_sort_loss(row),
                -(row["n_used"] or 0),
            ),
        )[:limit],
        "all_templates": sorted(
            active_rows,
            key=lambda row: (
                evidence_rank.get(str(row.get("evidence_level") or ""), 0),
                row["s1_rate"] if row["s1_rate"] is not None else -1.0,
                -(row["n_used"] or 0),
                row["name"],
            ),
        ),
        "inactive_templates": sorted(
            inactive_rows, key=lambda row: (-(row["n_used"] or 0), row["name"])
        ),
        "low_loss_template_families": sorted(
            [row for row in active_rows if row.get("repeated_low_loss_family")],
            key=lambda row: (
                -(row.get("repeated_low_loss_count") or 0),
                row["best_loss_ratio"] if row["best_loss_ratio"] is not None else 999.0,
                -(row["n_used"] or 0),
            ),
        )[:limit],
    }


def _summarize_motif_rows(
    acc: _ObservabilityAccumulator, limit: int
) -> list[Dict[str, Any]]:
    rows = []
    for stat in acc.motif_stats.values():
        reasons = stat["failure_reasons"]
        rows.append(
            {
                "name": stat["name"],
                "n_used": stat["n_used"],
                "s1_rate": stat["n_stage1"] / max(stat["n_used"], 1),
                "avg_loss_ratio": _mean_or_none(stat["losses"]),
                "top_failure_reason": (
                    max(reasons.items(), key=lambda item: item[1])[0]
                    if reasons
                    else None
                ),
            }
        )
    return sorted(
        [row for row in rows if row["n_used"] >= 2],
        key=lambda row: (-(row["n_used"] or 0), row["avg_loss_ratio"] or 999.0),
    )[:limit]


_SLOT_AVG_FIELDS = (
    ("avg_composite_score", "composite_scores"),
    ("avg_loss_ratio", "losses"),
    ("avg_induction_screening_auc", "induction_screening_aucs"),
    ("avg_induction_intermediate_auc", "induction_intermediate_aucs"),
    ("avg_binding_screening_auc", "binding_screening_aucs"),
    ("avg_binding_screening_composite", "binding_screening_composites"),
    ("avg_binding_intermediate_auc", "binding_intermediate_aucs"),
    ("avg_ar_legacy_auc", "ar_legacy_aucs"),
    ("avg_ar_curriculum_auc_pair_final", "ar_curriculum_aucs"),
    ("avg_ar_curriculum_s0_retention", "ar_curriculum_retentions"),
    ("avg_ar_curriculum_max_passing_stage", "ar_curriculum_max_passes"),
)


def _summarize_slot_stat(stat: Dict[str, Any]) -> Dict[str, Any]:
    reasons = stat["failure_reasons"]
    selected = stat["selected_motifs"]
    row = {
        "slot_key": stat["slot_key"],
        "template_name": stat["template_name"],
        "slot_index": stat["slot_index"],
        "slot_classes": stat["slot_classes"],
        "n_used": stat["n_used"],
        "s1_rate": stat["n_stage1"] / max(stat["n_used"], 1),
        "n_ar_curriculum": len(stat.get("ar_curriculum_aucs") or []),
        **_summarize_language_control_metrics(stat),
        "top_failure_reason": (
            max(reasons.items(), key=lambda item: item[1])[0] if reasons else None
        ),
        "top_selected_motif": (
            max(selected.items(), key=lambda item: item[1])[0] if selected else None
        ),
    }
    row.update(
        {
            output_name: _mean_or_none(stat.get(stat_name) or [])
            for output_name, stat_name in _SLOT_AVG_FIELDS
        }
    )
    return row


def _slot_row_is_active(
    row: Dict[str, Any],
    active_template_names: frozenset[str],
    slot_counts: Dict[str, int],
) -> bool:
    expected = int(slot_counts.get(row["template_name"], 0) or 0)
    return row["template_name"] in active_template_names and (
        expected <= 0 or row["slot_index"] < expected
    )


def _summarize_slot_rows(
    acc: _ObservabilityAccumulator,
    active_template_names: frozenset[str],
    slot_counts: Dict[str, int],
    limit: int,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    all_rows = sorted(
        [
            row
            for row in (_summarize_slot_stat(stat) for stat in acc.slot_stats.values())
            if _slot_row_is_active(row, active_template_names, slot_counts)
        ],
        key=lambda row: (row["template_name"], row["slot_index"], row["slot_key"]),
    )
    visible_rows = sorted(
        [row for row in all_rows if row["n_used"] >= 2],
        key=lambda row: (
            row["s1_rate"] if row["s1_rate"] is not None else 1.0,
            row["avg_loss_ratio"] if row["avg_loss_ratio"] is not None else 999.0,
            -(row["n_used"] or 0),
        ),
    )[:limit]
    return all_rows, visible_rows


def _build_observability_recommendations(
    ranked: Dict[str, list[Dict[str, Any]]],
    active_template_rows: list[Dict[str, Any]],
    slot_rows: list[Dict[str, Any]],
    loss_distribution: Dict[str, Dict[str, Optional[float]]],
) -> List[str]:
    recommendations: List[str] = []
    _append_template_recommendations(recommendations, ranked, loss_distribution)
    _append_slot_recommendations(recommendations, slot_rows)
    _append_signal_recommendations(
        recommendations,
        ranked["struggling_templates"],
        ranked["all_templates"],
        active_template_rows,
        ranked["low_loss_template_families"],
    )
    return recommendations[:6]


def _append_template_recommendations(
    recommendations: List[str],
    ranked: Dict[str, list[Dict[str, Any]]],
    loss_distribution: Dict[str, Dict[str, Optional[float]]],
) -> None:
    weak = next(
        (row for row in ranked["struggling_templates"] if (row["s1_rate"] or 0) < 0.15),
        None,
    )
    if weak:
        recommendations.append(
            f"{weak['name']} is over-sampled relative to quality: S1 {(weak['s1_rate'] * 100):.1f}% over {weak['n_used']} runs. Reduce weight or harden motifs for {weak['top_failure_reason'] or 'unknown failures'}."
        )
    slot_heavy = next(
        (
            row
            for row in ranked["struggling_templates"]
            if (row.get("slot_count") or 0) >= 3 and (row.get("s1_rate") or 0) < 0.25
        ),
        None,
    )
    if slot_heavy:
        recommendations.append(
            f"Slot-heavy template {slot_heavy['name']} underperforms with {slot_heavy['slot_count']} inferred motif slots. Tighten slot compatibility checks or narrow allowed motifs."
        )
    val_median = loss_distribution["validation"]["median"]
    train_median = loss_distribution["training"]["median"]
    if (
        val_median is not None
        and train_median is not None
        and val_median > train_median * 1.15
    ):
        recommendations.append(
            f"Validation loss ratio median ({val_median:.3f}) is materially worse than training median ({train_median:.3f}). Improve generalization gates or reduce brittle template/motif combinations."
        )
    best = ranked["top_templates"][0] if ranked["top_templates"] else None
    if best:
        best_loss = (
            f"{best['avg_loss_ratio']:.3f}"
            if best["avg_loss_ratio"] is not None
            else "n/a"
        )
        recommendations.append(
            f"Exploit {best['name']} more aggressively: S1 {(best['s1_rate'] * 100):.1f}% with avg loss {best_loss}."
        )


def _append_slot_recommendations(
    recommendations: List[str], slot_rows: list[Dict[str, Any]]
) -> None:
    weak_slot = next((row for row in slot_rows if (row["s1_rate"] or 0) < 0.15), None)
    if weak_slot:
        recommendations.append(
            f"Weak slot {weak_slot['slot_key']} is collapsing candidate quality: S1 {(weak_slot['s1_rate'] * 100):.1f}% with motif {weak_slot['top_selected_motif'] or 'none'} and failures dominated by {weak_slot['top_failure_reason'] or 'unknown'}."
        )


def _append_signal_recommendations(
    recommendations: List[str],
    struggling_templates: list[Dict[str, Any]],
    all_templates: list[Dict[str, Any]],
    active_template_rows: list[Dict[str, Any]],
    low_loss_template_families: list[Dict[str, Any]],
) -> None:
    routing_reprieve = _find_routing_reprieve(struggling_templates)
    sparse_attention = _find_sparse_attention_template(all_templates)
    induction_gap = _find_induction_gap(active_template_rows)
    repeated_low_loss = _find_repeated_low_loss_family(low_loss_template_families)
    if routing_reprieve is not None:
        recommendations.append(
            f"{routing_reprieve['name']} looks under-credited by short S1: fast lane positive on {(routing_reprieve['routing_fast_lane_positive_rate'] * 100):.1f}% of routing probes across {routing_reprieve['routing_fast_lane_runs']} runs. Treat it as a slow starter, not a dead template."
        )
    if sparse_attention is not None:
        recommendations.append(
            f"{sparse_attention['name']} is still data-sparse. Continue randomized weighting/backfills before trusting its rank or slot guidance."
        )
    if induction_gap is not None:
        recommendations.append(
            f"{induction_gap['name']} survives screening but still shows weak induction signal. Keep it in data-building mode, not champion mode."
        )
    if repeated_low_loss is not None:
        recommendations.append(
            f"{repeated_low_loss['name']} is a repeated low-loss family: {repeated_low_loss['repeated_low_loss_count']} S1 survivors at loss_ratio <= 0.45. Track it separately from benchmark-champion templates."
        )


def _find_routing_reprieve(rows: list[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return next(
        (
            row
            for row in rows
            if (row.get("routing_fast_lane_runs") or 0) >= 3
            and (row.get("routing_fast_lane_positive_rate") or 0) >= 0.5
        ),
        None,
    )


def _find_sparse_attention_template(
    rows: list[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    return next(
        (
            row
            for row in rows
            if row["name"].startswith("attn_")
            and str(row.get("evidence_level")) in {"insufficient", "sparse"}
        ),
        None,
    )


def _find_induction_gap(rows: list[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return next(
        (
            row
            for row in rows
            if (row.get("avg_induction_screening_auc") or 0.0) < 0.02
            and (row.get("n_used") or 0) >= 5
            and (row.get("s1_rate") or 0.0) >= 0.2
        ),
        None,
    )


def _find_repeated_low_loss_family(
    rows: list[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    return next(
        (row for row in rows if (row.get("repeated_low_loss_count") or 0) >= 3),
        None,
    )


def _trend_points_for_key(
    buckets: list[Dict[str, Any]], group_key: str, item_key: str
) -> list[Dict[str, Any]]:
    points = []
    for bucket in buckets:
        item = bucket[group_key].get(item_key)
        if not item or not item["n"]:
            continue
        points.append(
            {
                "timestamp": bucket["timestamp"],
                "experiment_id": bucket.get("experiment_id"),
                "s1_rate": item["s1"] / max(item["n"], 1),
                "avg_loss_ratio": _mean_or_none(item["losses"]),
            }
        )
    return points


def _build_observability_trends(
    owner: Any,
    acc: _ObservabilityAccumulator,
    top_templates: list[Dict[str, Any]],
    slot_rows: list[Dict[str, Any]],
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], list[Dict[str, Any]]]:
    buckets = sorted(
        acc.experiment_buckets.values(),
        key=lambda item: float(item.get("timestamp") or 0.0),
    )[-20:]
    template_trends = [
        {"name": name, "points": points}
        for name in [row["name"] for row in top_templates[:3]]
        if (points := _trend_points_for_key(buckets, "templates", name))
    ]
    slot_trends = [
        {"slot_key": key, "points": points}
        for key in [row["slot_key"] for row in slot_rows[:3]]
        if (points := _trend_points_for_key(buckets, "slots", key))
    ]
    loss_trends = [
        {
            "timestamp": bucket["timestamp"],
            "experiment_id": bucket.get("experiment_id"),
            "training_median": owner._percentile(bucket["training_losses"], 0.5),
            "validation_median": owner._percentile(bucket["validation_losses"], 0.5),
            "discovery_median": owner._percentile(bucket["discovery_losses"], 0.5),
        }
        for bucket in buckets
    ]
    return template_trends, slot_trends, loss_trends


def _observability_summary(
    acc: _ObservabilityAccumulator,
    active_template_rows: list[Dict[str, Any]],
    template_rows: list[Dict[str, Any]],
    inactive_template_rows: list[Dict[str, Any]],
    motif_rows: list[Dict[str, Any]],
    low_loss_template_families: list[Dict[str, Any]],
    slot_counts: Dict[str, int],
    active_template_names: frozenset[str],
) -> Dict[str, Any]:
    return {
        "avg_templates_per_graph": _mean_or_zero(acc.templates_per_graph),
        "avg_motifs_per_graph": _mean_or_zero(acc.motifs_per_graph),
        "templates_tracked": len(active_template_rows),
        "templates_observed_total": len(template_rows),
        "motifs_tracked": len(motif_rows),
        "zero_slot_templates": sorted(
            [
                name
                for name, count in slot_counts.items()
                if count == 0 and name in active_template_names
            ]
        )[:10],
        "inactive_templates_tracked": len(inactive_template_rows),
        "inactive_template_names": sorted(
            row["name"] for row in inactive_template_rows
        )[:10],
        "insufficient_templates": _count_evidence(active_template_rows, "insufficient"),
        "sparse_templates": _count_evidence(active_template_rows, "sparse"),
        "established_templates": _count_evidence(active_template_rows, "established"),
        "routing_fast_lane_templates": sum(
            1
            for row in active_template_rows
            if (row.get("routing_fast_lane_runs") or 0) > 0
        ),
        "routing_fast_lane_runs": sum(
            int(row.get("routing_fast_lane_runs") or 0) for row in active_template_rows
        ),
        "routing_fast_lane_positive_templates": sum(
            1
            for row in active_template_rows
            if (row.get("routing_fast_lane_positive_rate") or 0) >= 0.5
            and (row.get("routing_fast_lane_runs") or 0) >= 3
        ),
        "repeated_low_loss_templates": [
            row["name"] for row in low_loss_template_families
        ],
    }


def _mean_or_zero(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _count_evidence(rows: list[Dict[str, Any]], level: str) -> int:
    return sum(1 for row in rows if str(row.get("evidence_level")) == level)


class _ObservabilityMixin:
    """Template observability + slot statistics."""

    __slots__ = ()
    _DASHBOARD_SUMMARY_TTL_S = 2.0
    _TEMPLATE_OBSERVABILITY_TTL_S = 10.0

    @staticmethod
    def _percentile(values: List[float], pct: float) -> Optional[float]:
        clean = sorted(v for v in values if v is not None and math.isfinite(v))
        if not clean:
            return None
        if len(clean) == 1:
            return float(clean[0])
        pos = max(0.0, min(1.0, pct)) * (len(clean) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return float(clean[lo])
        frac = pos - lo
        return float(clean[lo] + (clean[hi] - clean[lo]) * frac)

    @staticmethod
    @lru_cache(maxsize=1)
    def _infer_template_slot_counts() -> Dict[str, int]:
        counts: Dict[str, int] = {}
        structural_overrides = {
            # 1 motif slot (norm_wrap) + 3 semantic router slots tracked explicitly
            "hybrid_sparse_triplet_router": 4,
            # 1 motif slot (norm_wrap) + 8 structural routing slots
            "multiscale_difficulty_router": 9,
            # 1 motif slot (norm_wrap) + 6 structural routing slots
            "multiscale_rich_lane_router": 7,
            # 1 motif slot (norm_wrap) + 10 structural stem/lane/merge slots
            "intelligent_multilane_router": 11,
            # 1 norm slot + role slots for trunk/controller/write/read/merge/mix/stabilize
            "typed_slot_memory_block": 7,
            # 1 norm slot + trunk/controller/retrieval/merge/mix/stabilize
            "sparse_relation_graph_block": 6,
            # 1 norm slot + trunk/controller/write/read/merge/mix/stabilize
            "token_program_interpreter_block": 7,
            # codex capability-first templates emit explicit structural slots only
            "codex_ssm_retention_block": 4,
            "codex_ssm_delta_memory_block": 2,
        }
        template_dir = Path(__file__).resolve().parents[2] / "synthesis"
        template_files = sorted(template_dir.glob("_templates*.py"))
        for template_file in template_files:
            try:
                source = template_file.read_text(encoding="utf-8")
            except OSError:
                continue
            matches = list(_TEMPLATE_DEF_RE.finditer(source))
            for idx, match in enumerate(matches):
                name = match.group(1)
                start = match.start()
                end = (
                    matches[idx + 1].start() if idx + 1 < len(matches) else len(source)
                )
                body = source[start:end]
                counts[name.removeprefix("tpl_")] = body.count(
                    "_pick_compatible_motif("
                ) + body.count("_pick_compatible_motif_from_classes(")
        counts.update(structural_overrides)
        return counts

    def get_template_slot_observability(self, limit: int = 8) -> Dict[str, Any]:
        now = time.time()
        cached = self._template_observability_cache.get(limit)
        if cached is not None and now < float(
            self._template_observability_cache_expires_at or 0.0
        ):
            return dict(cached)

        process_cache_key, process_signature = self._template_process_cache_key(limit)
        process_hit = self._get_template_process_cache_hit(
            process_cache_key,
            process_signature,
            now,
        )
        if process_hit is not None:
            self._cache_template_observability(limit, process_hit, now)
            return process_hit

        self.flush_writes()
        if not self._read_only:
            self._ensure_graph_features()
        rows = self._fetch_template_observability_rows()
        slot_counts = self._infer_template_slot_counts()
        if not rows:
            result = self._empty_template_observability_result(slot_counts, limit)
            self._cache_template_observability(limit, result, now)
            return result

        acc = self._accumulate_observability_stats(rows, slot_counts)
        result = self._assemble_observability_result(acc, slot_counts, limit)
        self._cache_template_observability(limit, result, now)
        self._cache_template_process_result(
            process_cache_key,
            process_signature,
            result,
            now,
        )
        return result

    def _template_process_cache_key(
        self, limit: int
    ) -> tuple[tuple[str, int] | None, tuple[Any, ...] | None]:
        if getattr(self, "_is_memory", False):
            return None, None
        cache_key = (str(getattr(self, "db_path", "")), int(limit))
        signature_row = self.conn.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM program_results_compat) AS pr_count,
                (SELECT MAX(timestamp) FROM program_results_compat) AS pr_max_ts,
                (SELECT COUNT(*) FROM program_graph_features) AS gf_count,
                (SELECT MAX(rowid) FROM program_graph_features) AS gf_max_rowid
            """
        ).fetchone()
        return cache_key, tuple(signature_row) if signature_row else None

    def _get_template_process_cache_hit(
        self,
        cache_key: tuple[str, int] | None,
        signature: tuple[Any, ...] | None,
        now: float,
    ) -> Dict[str, Any] | None:
        if cache_key is None or signature is None:
            return None
        process_cached = _TEMPLATE_OBSERVABILITY_PROCESS_CACHE.get(cache_key)
        if process_cached is None:
            return None
        if process_cached[0] == signature and now < process_cached[1]:
            return dict(process_cached[2])
        return None

    def _cache_template_process_result(
        self,
        cache_key: tuple[str, int] | None,
        signature: tuple[Any, ...] | None,
        result: Dict[str, Any],
        now: float,
    ) -> None:
        if cache_key is None or signature is None:
            return
        _TEMPLATE_OBSERVABILITY_PROCESS_CACHE[cache_key] = (
            signature,
            now + self._TEMPLATE_OBSERVABILITY_TTL_S,
            dict(result),
        )

    def _cache_template_observability(
        self,
        limit: int,
        result: Dict[str, Any],
        now: float,
    ) -> None:
        self._template_observability_cache[limit] = dict(result)
        self._template_observability_cache_expires_at = (
            now + self._TEMPLATE_OBSERVABILITY_TTL_S
        )

    def _empty_template_observability_result(
        self,
        slot_counts: Dict[str, int],
        limit: int,
    ) -> Dict[str, Any]:
        return self._assemble_observability_result(
            _ObservabilityAccumulator(
                template_stats={},
                motif_stats={},
                slot_stats={},
                experiment_buckets={},
                loss_values=[],
                validation_losses=[],
                discovery_losses=[],
                motifs_per_graph=[],
                templates_per_graph=[],
            ),
            slot_counts,
            limit,
        )

    def _fetch_template_observability_rows(self) -> list:
        return self.conn.execute(
            """
            SELECT
                pr.experiment_id, pr.timestamp, pr.graph_fingerprint,
                gf.templates_json, gf.motifs_json, gf.slot_usage_json,
                pr.stage0_passed, pr.stage05_passed, pr.stage1_passed,
                pr.loss_ratio, pr.discovery_loss_ratio, pr.validation_loss_ratio,
                pr.novelty_score, pr.novelty_confidence,
                pr.error_type, pr.stage_at_death, pr.failure_details_json,
                pr.induction_screening_auc, pr.binding_screening_auc, pr.binding_curriculum_auc,
                pr.binding_screening_composite,
                pr.ar_legacy_auc, pr.hellaswag_acc, pr.blimp_overall_accuracy,
                l.composite_score,
                pr.induction_intermediate_auc,
                pr.binding_intermediate_auc,
                pr.ar_curriculum_auc_pair_final,
                pr.ar_curriculum_s0_retention,
                pr.ar_curriculum_max_passing_stage,
                pr.language_control_s05_sentence_assoc_score,
                pr.language_control_s05_binding_order_acc,
                pr.language_control_s05_binding_score,
                pr.language_control_s10_sentence_assoc_score,
                pr.language_control_s10_binding_order_acc,
                pr.language_control_s10_binding_score,
                pr.language_control_investigation_sentence_assoc_score,
                pr.language_control_investigation_binding_order_acc,
                pr.language_control_investigation_binding_score,
                pr.fp_jacobian_effective_rank,
                pr.fp_sensitivity_uniformity,
                pr.fp_jacobian_erf_density,
                pr.fp_id_collapse_rate,
                pr.fp_id_collapse_rate_normalized,
                pr.fp_jacobian_erf_decay_slope,
                pr.fp_jacobian_erf_first_norm,
                pr.fp_jacobian_erf_last_norm,
                pr.fp_logit_margin_velocity,
                pr.fp_logit_margin_delta,
                pr.fp_jacobian_erf_variance,
                CASE WHEN pr.fp_jacobian_erf_variance IS NOT NULL
                     THEN log(abs(pr.fp_jacobian_erf_variance) + 0.000000001)
                     ELSE NULL
                END AS fp_jacobian_erf_variance_log,
                CASE WHEN pr.fp_jacobian_spectral_norm IS NOT NULL
                     THEN log(abs(pr.fp_jacobian_spectral_norm) + 0.000000001)
                     ELSE NULL
                END AS fp_jacobian_spectral_norm_log,
                pr.fp_icld_velocity,
                pr.fp_icld_delta_loss,
                pr.screening_hellaswag_correct, pr.screening_hellaswag_total,
                pr.screening_wikitext_status,
                pr.routing_fast_lane_applied, pr.routing_fast_lane_status,
                pr.routing_fast_lane_score,
                pr.routing_fast_lane_ppl_improvement,
                pr.routing_fast_lane_slope,
                pr.routing_fast_lane_slope_consistent
            FROM program_results_compat pr
            JOIN program_graph_features gf ON gf.result_id = pr.result_id
            LEFT JOIN leaderboard l ON l.result_id = pr.result_id
            """
        ).fetchall()

    def get_generation_observability_priors(
        self,
        *,
        max_rows: int = 48,
        min_support: int = 4,
    ) -> Dict[str, Any]:
        """Translate observability telemetry into live generation priors.

        This is intentionally conservative: it boosts repeated winners,
        downweights weak motifs/templates, and only applies slot-specific
        priors when a slot has enough support to be meaningfully directional.
        """
        obs = self.get_template_slot_observability(limit=max_rows)
        template_rows = obs.get("all_templates") or []
        motif_rows = obs.get("motif_slots") or []
        slot_rows = obs.get("all_slots") or obs.get("slot_observability") or []

        toxic_reason_tokens = (
            "compilation",
            "nan",
            "overflow",
            "shape",
            "invalid",
            "causality",
            "oom",
        )

        template_weights = _template_prior_weights(
            template_rows,
            min_support=min_support,
        )
        motif_weights = _motif_prior_weights(
            motif_rows,
            min_support=min_support,
            toxic_reason_tokens=toxic_reason_tokens,
        )
        slot_multipliers, slot_denylist = _slot_generation_priors(
            slot_rows,
            min_support=min_support,
            toxic_reason_tokens=toxic_reason_tokens,
        )

        return {
            "template_weights": template_weights,
            "motif_weights": motif_weights,
            "slot_multipliers": slot_multipliers,
            "slot_denylist": {
                key: sorted(set(values)) for key, values in slot_denylist.items()
            },
            "metadata": {
                "min_support": int(min_support),
                "template_count": len(template_weights),
                "motif_count": len(motif_weights),
                "slot_count": len(slot_multipliers),
            },
        }

    def _accumulate_observability_stats(
        self, rows: list, slot_counts: Dict[str, int]
    ) -> _ObservabilityAccumulator:
        """Parse rows and accumulate per-template/motif/slot statistics."""
        return self._accumulate_observability_stats_impl(rows, slot_counts)

    def _accumulate_observability_stats_impl(
        self, rows: list, slot_counts: Dict[str, int]
    ) -> _ObservabilityAccumulator:
        """Parse rows and accumulate per-template/motif/slot statistics."""
        acc = _empty_observability_accumulator()
        for row in rows:
            if row["templates_json"] is not None:
                templates = _json_tuple(row["templates_json"])
                motifs = _json_tuple(row["motifs_json"])
                slot_usage = _json_slot_tuple(row["slot_usage_json"])
            else:
                graph_json = resolve_graph_json_value(
                    self.conn,
                    self.db_path,
                    row.get("graph_json"),
                )
                templates, motifs, slot_usage = _cached_extract_observability_metadata(
                    graph_json
                )
            bucket = _experiment_bucket(acc, row)
            metrics = _observability_metric_values(row)
            language_values = _language_control_values(row)
            root_cause = _failure_root_cause(row)
            loss_ratio = _finite_float(metrics.get("loss_ratio"))
            acc.motifs_per_graph.append(float(len(motifs)))
            acc.templates_per_graph.append(float(len(templates)))
            _append_global_losses(acc, bucket, metrics)
            _record_template_observations(
                acc,
                bucket,
                templates,
                row,
                metrics,
                language_values,
                root_cause,
                slot_counts,
            )
            _record_motif_observations(acc, motifs, row, loss_ratio, root_cause)
            _record_slot_observations(
                acc, bucket, slot_usage, row, metrics, language_values, root_cause
            )
        return acc

    def _assemble_observability_result(
        self,
        acc: _ObservabilityAccumulator,
        slot_counts: Dict[str, int],
        limit: int,
    ) -> Dict[str, Any]:
        """Sort, rank, and assemble the final observability result dict."""
        return self._assemble_observability_result_impl(acc, slot_counts, limit)

    def _assemble_observability_result_impl(
        self,
        acc: _ObservabilityAccumulator,
        slot_counts: Dict[str, int],
        limit: int,
    ) -> Dict[str, Any]:
        """Sort, rank, and assemble the final observability result dict."""
        (
            active_template_names,
            template_rows,
            active_template_rows,
            inactive_template_rows,
        ) = _prepare_template_rows(acc, slot_counts)
        ranked = _rank_template_rows(
            active_template_rows, inactive_template_rows, limit
        )
        motif_rows = _summarize_motif_rows(acc, limit)
        all_slot_rows, slot_rows = _summarize_slot_rows(
            acc, active_template_names, slot_counts, limit
        )
        loss_distribution = _observability_loss_distribution(self, acc)
        recommendations = _build_observability_recommendations(
            ranked, active_template_rows, slot_rows, loss_distribution
        )
        template_trends, slot_trends, loss_trends = _build_observability_trends(
            self, acc, ranked["top_templates"], slot_rows
        )
        return {
            "top_templates": ranked["top_templates"],
            "struggling_templates": ranked["struggling_templates"],
            "all_templates": ranked["all_templates"],
            "low_loss_template_families": ranked["low_loss_template_families"],
            "inactive_templates": ranked["inactive_templates"],
            "all_slots": all_slot_rows,
            "motif_slots": motif_rows,
            "slot_observability": slot_rows,
            "loss_distribution": loss_distribution,
            "template_trends": template_trends,
            "slot_trends": slot_trends,
            "loss_trends": loss_trends,
            "recommendations": recommendations,
            "summary": _observability_summary(
                acc,
                active_template_rows,
                template_rows,
                inactive_template_rows,
                motif_rows,
                ranked["low_loss_template_families"],
                slot_counts,
                active_template_names,
            ),
        }

    # ── Training Curves ──
