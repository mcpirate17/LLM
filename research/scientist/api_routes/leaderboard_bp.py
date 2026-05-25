"""leaderboard API route registration."""

from __future__ import annotations

import json
import logging
import math
import statistics
from typing import Any, Dict, List
from flask import jsonify, request
from ..json_utils import json_safe as _json_safe
from ..leaderboard_dashboard_fields import (
    CHAMPION_DASHBOARD_FIELDS,
    INTERMEDIATE_SCREEN_DASHBOARD_FIELDS,
)
from ..leaderboard_scoring import composite_score_ceiling, get_scoring_version
from ..leaderboard_rescore import rescore_leaderboard
from ..naming import annotate_display_names
from ..trust_policy import is_trusted_entry, sql_trusted_clause
from .deps import ApiRouteContext
from ._utils import register_notebook_routes, with_notebook_context
from ._strategy_recommendations import (
    annotate_qkv_usage,
    attach_long_context_breakdown,
    capability_quality_for_entry,
    compute_cross_run_stability,
    infer_tier_for_program,
    count_discovery_tiers,
    promotion_evidence_for_entry,
)
from ._strategy_report import parse_bool_query
from ._fingerprint_failures import (
    FINGERPRINT_STATUS_FIELDS,
    NON_FAILURE_STATUSES,
    attach_fingerprint_failure_metadata,
)

logger = logging.getLogger(__name__)


def _default_cross_run_stability() -> Dict[str, Any]:
    return {
        "trend": "unknown",
        "seen_runs": 0,
        "latest_rank": None,
        "previous_rank": None,
        "rank_delta": None,
    }


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _log_abs_metric(value: Any) -> float | None:
    numeric = _to_float(value)
    if numeric is None:
        return None
    return math.log10(abs(numeric) + 0.000000001)


def _attach_derived_fingerprint_metrics(entries: List[Dict[str, Any]]) -> None:
    for entry in entries:
        if entry.get("fp_jacobian_erf_variance_log") is None:
            entry["fp_jacobian_erf_variance_log"] = _log_abs_metric(
                entry.get("fp_jacobian_erf_variance")
            )
        if entry.get("fp_jacobian_spectral_norm_log") is None:
            entry["fp_jacobian_spectral_norm_log"] = _log_abs_metric(
                entry.get("fp_jacobian_spectral_norm")
                if entry.get("fp_jacobian_spectral_norm") is not None
                else entry.get("jacobian_spectral_norm")
            )


def _semantic_warning_for_entry(entry: Dict[str, Any]) -> Dict[str, Any] | None:
    cohort = str(entry.get("result_cohort") or "").strip().lower()
    if cohort != "backfill":
        return None

    validation_loss_ratio = _to_float(entry.get("validation_loss_ratio"))
    if validation_loss_ratio is None or validation_loss_ratio >= 0.1:
        return None

    wikitext_perplexity = _to_float(entry.get("wikitext_perplexity"))
    tinystories_perplexity = _to_float(entry.get("tinystories_perplexity"))
    hellaswag_acc = _to_float(entry.get("hellaswag_acc"))

    evidence: List[str] = []
    if wikitext_perplexity is not None and wikitext_perplexity > 500.0:
        evidence.append(f"WikiText perplexity {wikitext_perplexity:.2f}")
    if tinystories_perplexity is not None and tinystories_perplexity > 500.0:
        evidence.append(f"TinyStories perplexity {tinystories_perplexity:.2f}")
    if hellaswag_acc is not None and hellaswag_acc < 0.2:
        evidence.append(f"HellaSwag {hellaswag_acc:.2%}")

    if not evidence:
        return None

    return {
        "code": "backfill_metric_mismatch",
        "severity": "warning",
        "label": "Backfill mismatch",
        "message": (
            "Backfill row has a very low validation-style loss ratio but poor "
            "real-token quality, so these metrics should not be read as "
            "candidate-grade evidence."
        ),
        "evidence": evidence,
    }


def _current_discovery_tier(entry: Dict[str, Any]) -> str:
    tier = str(entry.get("tier") or "screening").strip().lower()
    if tier == "validation" and not bool(entry.get("validation_passed")):
        return "validation_pending"
    return tier or "screening"


FAILED_DISCOVERY_TIERS = {
    "screened_out",
    "investigation_failed",
    "investigation_fingerprint_incomplete",
    "validation_failed",
    "failed",
    "rejected",
}


def _discovery_tier_rank(entry: Dict[str, Any]) -> int:
    tier = _current_discovery_tier(entry)
    if tier in {
        "validation",
        "validation_pending",
        "validation_failed",
        "breakthrough",
    }:
        return 3
    if tier in {
        "investigation",
        "investigation_failed",
        "investigation_fingerprint_incomplete",
    }:
        return 2
    return 1


def _discovery_timestamp(entry: Dict[str, Any]) -> float:
    try:
        return float(entry.get("timestamp") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _discovery_preference(entry: Dict[str, Any]) -> tuple:
    cohort = str(entry.get("result_cohort") or "").strip().lower()
    trust = str(entry.get("trust_label") or "").strip().lower()
    comparability = str(entry.get("comparability_label") or "").strip().lower()
    model_source = str(entry.get("model_source") or "").strip().lower()
    return (
        cohort != "backfill",
        trust == "candidate_grade",
        comparability == "candidate_comparable",
        model_source == "exact_graph_replay",
        bool(entry.get("entry_id")),
        _discovery_tier_rank(entry),
        float(entry.get("composite_score") or 0.0),
        _discovery_timestamp(entry),
    )


def _dedupe_discovery_entries_by_fingerprint(
    entries: List[Dict[str, Any]], *, limit: int
) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: Dict[str, int] = {}
    for entry in entries:
        fp = str(entry.get("graph_fingerprint") or "").strip().lower()
        key = f"fp:{fp}" if fp else f"rid:{entry.get('result_id')}"
        if key in seen:
            idx = seen[key]
            if _discovery_preference(entry) > _discovery_preference(deduped[idx]):
                deduped[idx] = entry
            continue
        seen[key] = len(deduped)
        deduped.append(entry)
    return deduped[:limit]


def _search_discoveries(
    nb,
    *,
    query: str,
    tier: str | None,
    limit: int,
    trusted_only: bool = True,
    include_references: bool = False,
) -> List[Dict[str, Any]]:
    """Search leaderboard + raw stage1 survivors across the full notebook."""
    q = str(query or "").strip()
    if not q:
        return []

    wildcard = f"%{q}%"
    prefix = f"{q}%"
    sql = """
        SELECT
            pr.*,
            l.entry_id,
            l.tier AS leaderboard_tier,
            l.composite_score,
            l.screening_loss_ratio AS lb_screening_loss_ratio,
            l.screening_novelty,
            l.screening_passed,
            l.investigation_loss_ratio AS lb_investigation_loss_ratio,
            l.investigation_robustness,
            l.investigation_passed,
            l.validation_loss_ratio AS lb_validation_loss_ratio,
            l.validation_baseline_ratio AS lb_validation_baseline_ratio,
            l.validation_passed,
            l.induction_intermediate_auc AS lb_induction_intermediate_auc,
            l.induction_intermediate_max_gap_acc AS lb_induction_intermediate_max_gap_acc,
            l.induction_intermediate_protocol_version AS lb_induction_intermediate_protocol_version,
            l.binding_intermediate_auc AS lb_binding_intermediate_auc,
            l.binding_intermediate_max_distance_acc AS lb_binding_intermediate_max_distance_acc,
            l.binding_intermediate_protocol_version AS lb_binding_intermediate_protocol_version,
            l.discovery_loss_ratio AS leaderboard_discovery_loss_ratio,
            l.is_reference,
            l.reference_name,
            l.model_source AS leaderboard_model_source,
            l.architecture_desc AS leaderboard_architecture_desc,
            l.timestamp AS leaderboard_timestamp
        FROM program_results_compat pr
        LEFT JOIN leaderboard l ON l.result_id = pr.result_id
        WHERE COALESCE(pr.stage1_passed, 0) = 1
          AND (
                LOWER(COALESCE(pr.graph_fingerprint, '')) LIKE LOWER(?)
             OR LOWER(COALESCE(pr.result_id, '')) LIKE LOWER(?)
             OR LOWER(COALESCE(pr.model_source, '')) LIKE LOWER(?)
             OR LOWER(COALESCE(l.reference_name, '')) LIKE LOWER(?)
             OR LOWER(COALESCE(l.architecture_desc, '')) LIKE LOWER(?)
          )
    """
    if trusted_only:
        sql += f" AND {sql_trusted_clause(table_alias='pr')}"
    sql += """
        ORDER BY
            CASE
                WHEN LOWER(COALESCE(pr.graph_fingerprint, '')) = LOWER(?) THEN 0
                WHEN LOWER(COALESCE(pr.graph_fingerprint, '')) LIKE LOWER(?) THEN 1
                WHEN LOWER(COALESCE(pr.result_id, '')) = LOWER(?) THEN 2
                WHEN LOWER(COALESCE(pr.result_id, '')) LIKE LOWER(?) THEN 3
                ELSE 4
            END,
            COALESCE(l.composite_score, 0) DESC,
            COALESCE(l.timestamp, pr.timestamp) DESC
        LIMIT ?
    """
    rows = nb.conn.execute(
        sql,
        (
            wildcard,
            wildcard,
            wildcard,
            wildcard,
            wildcard,
            q,
            prefix,
            q,
            prefix,
            max(limit * 8, 200),
        ),
    ).fetchall()

    entries: List[Dict[str, Any]] = []
    for row in rows:
        entry = dict(row)
        entry["tier"] = entry.get("leaderboard_tier") or infer_tier_for_program(
            nb, entry
        )
        entry["architecture_desc"] = (
            entry.get("leaderboard_architecture_desc")
            or entry.get("architecture_desc")
            or entry.get("graph_fingerprint")
        )
        entry["model_source"] = entry.get("leaderboard_model_source") or entry.get(
            "model_source"
        )
        if entry.get("lb_screening_loss_ratio") is not None:
            entry["screening_loss_ratio"] = entry.get("lb_screening_loss_ratio")
        if entry.get("lb_investigation_loss_ratio") is not None:
            entry["investigation_loss_ratio"] = entry.get("lb_investigation_loss_ratio")
        if entry.get("lb_validation_loss_ratio") is not None:
            entry["validation_loss_ratio"] = entry.get("lb_validation_loss_ratio")
        if entry.get("lb_validation_baseline_ratio") is not None:
            entry["validation_baseline_ratio"] = entry.get(
                "lb_validation_baseline_ratio"
            )
        for metric_key in (
            "induction_intermediate_auc",
            "induction_intermediate_max_gap_acc",
            "induction_intermediate_protocol_version",
            "binding_intermediate_auc",
            "binding_intermediate_max_distance_acc",
            "binding_intermediate_protocol_version",
        ):
            lb_key = f"lb_{metric_key}"
            if entry.get(metric_key) is None and entry.get(lb_key) is not None:
                entry[metric_key] = entry.get(lb_key)
        if (
            entry.get("discovery_loss_ratio") is None
            and entry.get("leaderboard_discovery_loss_ratio") is not None
        ):
            entry["discovery_loss_ratio"] = entry.get(
                "leaderboard_discovery_loss_ratio"
            )
        entry["timestamp"] = entry.get("leaderboard_timestamp") or entry.get(
            "timestamp"
        )
        entry["architecture_family"] = nb._classify_architecture_family(
            graph_json=entry.get("graph_json"),
            routing_mode=entry.get("routing_mode"),
        )
        if tier:
            normalized_tier = str(tier).strip().lower()
            current_tier = _current_discovery_tier(entry)
            if normalized_tier == "failed":
                if current_tier not in FAILED_DISCOVERY_TIERS:
                    continue
            elif current_tier != normalized_tier:
                continue
        if not include_references and entry.get("is_reference"):
            continue
        if trusted_only and not is_trusted_entry(entry):
            continue
        entries.append(entry)

    annotated = nb._attach_canonical_program_scores(entries)
    return _dedupe_discovery_entries_by_fingerprint(annotated, limit=limit)


def _matches_discovery_query(entry: Dict[str, Any], query: str) -> bool:
    q = str(query or "").strip().lower()
    if not q:
        return True
    haystacks = (
        entry.get("display_name"),
        entry.get("reference_name"),
        entry.get("architecture_desc"),
        entry.get("architecture_family"),
        entry.get("graph_fingerprint"),
        entry.get("result_id"),
    )
    return any(str(value or "").lower().find(q) >= 0 for value in haystacks)


def _entry_has_promotion_path(entry: dict) -> bool:
    """Heuristic filter for candidates that still have a credible path forward."""
    if entry.get("is_reference"):
        return True
    if entry.get("is_pinned"):
        return True

    tier = str(entry.get("tier") or "screening").strip().lower()
    if tier == "screened_out":
        return False
    if tier in {"validation", "breakthrough"}:
        return True

    stage1_passed = entry.get("stage1_passed")
    if stage1_passed is not None and not bool(stage1_passed):
        return False

    # NOTE: novelty_valid_for_promotion is informational — it should never
    # block promotion.  Missing or heuristic novelty is a data quality flag,
    # not a disqualifying gate.

    composite = float(entry.get("composite_score") or 0.0)
    screening_loss = entry.get("screening_loss_ratio")
    investigation_loss = entry.get("investigation_loss_ratio")
    validation_loss = entry.get("validation_loss_ratio")

    if validation_loss is not None and float(validation_loss) < 1.0:
        return True
    if investigation_loss is not None and float(investigation_loss) < 1.0:
        return True
    if screening_loss is not None and float(screening_loss) < 1.0 and composite > 0.0:
        return True
    if tier == "investigation" and composite >= 0.25:
        return True
    if tier == "screening" and composite >= 0.75:
        return True
    return False


def _compact_leaderboard_entry(entry: dict) -> dict:
    return {
        "entry_id": entry.get("entry_id"),
        "result_id": entry.get("result_id"),
        "tier": entry.get("tier"),
        "composite_score": entry.get("composite_score"),
        "score_breakdown": entry.get("score_breakdown") or {},
        "capability_quality": entry.get("capability_quality"),
        "semantic_warning": entry.get("semantic_warning"),
        "semantic_warning_count": entry.get("semantic_warning_count"),
        "fingerprint_failed": entry.get("fingerprint_failed"),
        "fingerprint_failure_count": entry.get("fingerprint_failure_count"),
        "fingerprint_failure_summary": entry.get("fingerprint_failure_summary"),
        "promotion_evidence": entry.get("promotion_evidence"),
        "loss_ratio": entry.get("loss_ratio"),
        "screening_loss_ratio": entry.get("screening_loss_ratio"),
        "screening_novelty": entry.get("screening_novelty"),
        "investigation_loss_ratio": entry.get("investigation_loss_ratio"),
        "investigation_robustness": entry.get("investigation_robustness"),
        "investigation_passed": entry.get("investigation_passed"),
        "validation_loss_ratio": entry.get("validation_loss_ratio"),
        "validation_baseline_ratio": entry.get("validation_baseline_ratio"),
        "validation_multi_seed_std": entry.get("validation_multi_seed_std"),
        "validation_passed": entry.get("validation_passed"),
        "discovery_loss_ratio": entry.get("discovery_loss_ratio"),
        "novelty_score": entry.get("novelty_score"),
        "novelty_confidence": entry.get("novelty_confidence"),
        "novelty_valid_for_promotion": entry.get("novelty_valid_for_promotion"),
        "param_count": entry.get("param_count"),
        "graph_n_params_estimate": entry.get("graph_n_params_estimate"),
        "throughput_tok_s": entry.get("throughput_tok_s"),
        "forward_time_ms": entry.get("forward_time_ms"),
        "flops_forward": entry.get("flops_forward"),
        "flops_per_param": entry.get("flops_per_param"),
        "peak_memory_mb": entry.get("peak_memory_mb"),
        "sample_efficiency": entry.get("sample_efficiency"),
        "architecture_family": entry.get("architecture_family"),
        "graph_fingerprint": entry.get("graph_fingerprint"),
        "routing_mode": entry.get("routing_mode"),
        "stage1_passed": entry.get("stage1_passed"),
        "is_reference": entry.get("is_reference"),
        "is_pinned": entry.get("is_pinned"),
        "model_source": entry.get("model_source"),
        "reference_name": entry.get("reference_name"),
        "timestamp": entry.get("timestamp"),
        "tags": entry.get("tags"),
        # Scaling & efficiency
        "scaling_param_efficiency": entry.get("scaling_param_efficiency"),
        "scaling_gate_passed": entry.get("scaling_gate_passed"),
        # Routing & sparsity
        "routing_savings_ratio": entry.get("routing_savings_ratio"),
        "routing_utilization_entropy": entry.get("routing_utilization_entropy"),
        "n_routing_ops": entry.get("n_routing_ops"),
        "n_sparse_ops": entry.get("n_sparse_ops"),
        "compression_ratio": entry.get("compression_ratio"),
        "ncd_score": entry.get("ncd_score"),
        "depth_savings_ratio": entry.get("depth_savings_ratio"),
        "recursion_savings_ratio": entry.get("recursion_savings_ratio"),
        "activation_sparsity_score": entry.get("activation_sparsity_score"),
        # Robustness
        "fp_jacobian_spectral_norm": entry.get("fp_jacobian_spectral_norm"),
        "fp_jacobian_effective_rank": entry.get("fp_jacobian_effective_rank"),
        "fp_sensitivity_uniformity": entry.get("fp_sensitivity_uniformity"),
        "fp_spec_norm_status": entry.get("fp_spec_norm_status"),
        "fp_jacobian_erf_density": entry.get("fp_jacobian_erf_density"),
        "fp_id_collapse_rate": entry.get("fp_id_collapse_rate"),
        "fp_id_collapse_rate_normalized": entry.get("fp_id_collapse_rate_normalized"),
        "fp_jacobian_erf_decay_slope": entry.get("fp_jacobian_erf_decay_slope"),
        "fp_jacobian_erf_first_norm": entry.get("fp_jacobian_erf_first_norm"),
        "fp_jacobian_erf_last_norm": entry.get("fp_jacobian_erf_last_norm"),
        "fp_logit_margin_velocity": entry.get("fp_logit_margin_velocity"),
        "fp_logit_margin_initial": entry.get("fp_logit_margin_initial"),
        "fp_logit_margin_final": entry.get("fp_logit_margin_final"),
        "fp_logit_margin_delta": entry.get("fp_logit_margin_delta"),
        "fp_jacobian_erf_variance": entry.get("fp_jacobian_erf_variance"),
        "fp_jacobian_erf_status": entry.get("fp_jacobian_erf_status"),
        "fp_jacobian_erf_variance_log": entry.get("fp_jacobian_erf_variance_log"),
        "fp_jacobian_spectral_norm_log": entry.get("fp_jacobian_spectral_norm_log"),
        "fp_icld_velocity": entry.get("fp_icld_velocity"),
        "fp_icld_early_loss": entry.get("fp_icld_early_loss"),
        "fp_icld_late_loss": entry.get("fp_icld_late_loss"),
        "fp_icld_delta_loss": entry.get("fp_icld_delta_loss"),
        "fp_icld_status": entry.get("fp_icld_status"),
        "fp_id_collapse_status": entry.get("fp_id_collapse_status"),
        "fp_logit_margin_status": entry.get("fp_logit_margin_status"),
        "robustness_noise_score": entry.get("robustness_noise_score"),
        "quant_int8_retention": entry.get("quant_int8_retention"),
        "robustness_long_ctx_score": entry.get("robustness_long_ctx_score"),
        "robustness_long_ctx_scaling_score": entry.get(
            "robustness_long_ctx_scaling_score"
        ),
        "robustness_long_ctx_assoc_score": entry.get("robustness_long_ctx_assoc_score"),
        "robustness_long_ctx_multi_hop_score": entry.get(
            "robustness_long_ctx_multi_hop_score"
        ),
        "robustness_long_ctx_passkey_score": entry.get(
            "robustness_long_ctx_passkey_score"
        ),
        "max_viable_seq_len": entry.get("max_viable_seq_len"),
        # Real-token eval fields (needed by StabilityQualityQuadrant)
        "wikitext_perplexity": entry.get("wikitext_perplexity"),
        "wikitext_ppl": entry.get("wikitext_ppl"),
        "wikitext_score": entry.get("wikitext_score"),
        "screening_wikitext_metric_version": entry.get(
            "screening_wikitext_metric_version"
        ),
        "tokenizer_mode": entry.get("tokenizer_mode"),
        "corpus_path": entry.get("corpus_path"),
        "evaluation_protocol_version": entry.get("evaluation_protocol_version"),
        "peak_ppl": entry.get("peak_ppl"),
        "robustness_grade": entry.get("robustness_grade"),
        "evaluation_stage": entry.get("evaluation_stage"),
        "steps_to_divergence": entry.get("steps_to_divergence"),
        "loss_improvement_rate": entry.get("loss_improvement_rate"),
        "baseline_loss_ratio": entry.get("baseline_loss_ratio"),
        # HellaSwag commonsense reasoning
        "hellaswag_acc": entry.get("hellaswag_acc"),
        "hellaswag_metric_version": entry.get("hellaswag_metric_version"),
        "hellaswag_tokenizer_mode": entry.get("hellaswag_tokenizer_mode"),
        "hellaswag_tiktoken_encoding": entry.get("hellaswag_tiktoken_encoding"),
        # Binding probes
        "ar_legacy_auc": entry.get("ar_legacy_auc"),
        "ar_legacy_final_acc": entry.get("ar_legacy_final_acc"),
        "ar_legacy_timed_out": bool(entry.get("ar_legacy_timed_out"))
        if entry.get("ar_legacy_timed_out") is not None
        else None,
        "ar_legacy_above_chance": bool(entry.get("ar_legacy_above_chance"))
        if entry.get("ar_legacy_above_chance") is not None
        else None,
        "induction_screening_auc": entry.get("induction_screening_auc"),
        "binding_screening_auc": entry.get("binding_screening_auc"),
        "binding_screening_composite": entry.get("binding_screening_composite"),
        "local_only": entry.get("local_only"),
        **{field: entry.get(field) for field in CHAMPION_DASHBOARD_FIELDS},
        # AR Gate investigation probe
        "ar_gate_metric_version": entry.get("ar_gate_metric_version"),
        "ar_gate_in_dist_pair_acc": entry.get("ar_gate_in_dist_pair_acc"),
        "ar_gate_in_dist_class_acc": entry.get("ar_gate_in_dist_class_acc"),
        "ar_gate_held_pair_acc": entry.get("ar_gate_held_pair_acc"),
        "ar_gate_held_class_acc": entry.get("ar_gate_held_class_acc"),
        "ar_gate_score": entry.get("ar_gate_score"),
        "ar_gate_status": entry.get("ar_gate_status"),
        "ar_gate_elapsed_ms": entry.get("ar_gate_elapsed_ms"),
        "ar_gate_train_steps_done": entry.get("ar_gate_train_steps_done"),
        # v2 investigation-tier probes
        "induction_intermediate_auc": entry.get("induction_intermediate_auc"),
        "induction_intermediate_max_gap_acc": entry.get(
            "induction_intermediate_max_gap_acc"
        ),
        "induction_intermediate_protocol_version": entry.get(
            "induction_intermediate_protocol_version"
        ),
        "binding_intermediate_auc": entry.get("binding_intermediate_auc"),
        "binding_intermediate_max_distance_acc": entry.get(
            "binding_intermediate_max_distance_acc"
        ),
        "binding_intermediate_protocol_version": entry.get(
            "binding_intermediate_protocol_version"
        ),
        **{field: entry.get(field) for field in INTERMEDIATE_SCREEN_DASHBOARD_FIELDS},
        # Language-control probe ladder (v14)
        "language_control_metric_version": entry.get("language_control_metric_version"),
        "language_control_s05_sentence_assoc_score": entry.get(
            "language_control_s05_sentence_assoc_score"
        ),
        "language_control_s05_binding_order_acc": entry.get(
            "language_control_s05_binding_order_acc"
        ),
        "language_control_s05_binding_score": entry.get(
            "language_control_s05_binding_score"
        ),
        "language_control_s10_sentence_assoc_score": entry.get(
            "language_control_s10_sentence_assoc_score"
        ),
        "language_control_s10_binding_order_acc": entry.get(
            "language_control_s10_binding_order_acc"
        ),
        "language_control_s10_binding_score": entry.get(
            "language_control_s10_binding_score"
        ),
        "language_control_investigation_sentence_assoc_score": entry.get(
            "language_control_investigation_sentence_assoc_score"
        ),
        "language_control_investigation_binding_order_acc": entry.get(
            "language_control_investigation_binding_order_acc"
        ),
        "language_control_investigation_binding_score": entry.get(
            "language_control_investigation_binding_score"
        ),
        # BLiMP linguistic minimal pairs
        "blimp_overall_accuracy": entry.get("blimp_overall_accuracy"),
        "blimp_n_subtasks": entry.get("blimp_n_subtasks"),
        "blimp_status": entry.get("blimp_status"),
        "tinystories_perplexity": entry.get("tinystories_perplexity"),
        "tinystories_score": entry.get("tinystories_score"),
        "diagnostic_score": entry.get("diagnostic_score"),
        "cross_task_score": entry.get("cross_task_score"),
    }


def _attach_dashboard_entry_metadata(entries: List[Dict[str, Any]]) -> None:
    if not entries:
        return
    attach_fingerprint_failure_metadata(entries)
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry["capability_quality"] = capability_quality_for_entry(entry)
        entry["promotion_evidence"] = promotion_evidence_for_entry(entry)
        semantic_warning = _semantic_warning_for_entry(entry)
        entry["semantic_warning"] = semantic_warning
        entry["semantic_warning_count"] = 1 if semantic_warning else 0
        if not isinstance(entry.get("score_breakdown"), dict):
            entry["score_breakdown"] = {}


def _apply_arch_spec_metrics(entries: List[Dict[str, Any]]) -> None:
    for entry in entries:
        spec_json = entry.get("_arch_spec_json")
        if not spec_json:
            continue
        try:
            spec = json.loads(spec_json) if isinstance(spec_json, str) else spec_json
        except (ValueError, TypeError, json.JSONDecodeError):
            continue
        if not isinstance(spec, dict):
            continue
        if spec.get("gap_nats") is not None:
            entry["gap_vs_gpt2"] = float(spec["gap_nats"])
        if (
            spec.get("improvement_rate") is not None
            and entry.get("loss_improvement_rate") is None
        ):
            entry["loss_improvement_rate"] = float(spec["improvement_rate"])


def _annotate_capability_quality(entries: List[Dict[str, Any]]) -> None:
    for entry in entries:
        entry["capability_quality"] = capability_quality_for_entry(entry)


def _apply_cross_run_stability(
    nb,
    entries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    stability = compute_cross_run_stability(nb, entries)
    stability_by_result = {
        candidate.get("result_id"): candidate
        for candidate in stability.get("candidates", [])
        if candidate.get("result_id")
    }
    default_stability = _default_cross_run_stability()
    for entry in entries:
        entry["cross_run_stability"] = stability_by_result.get(
            entry.get("result_id"),
            default_stability.copy(),
        )
    return stability


def _attach_literature_to_entries(nb, entries: List[Dict[str, Any]]) -> None:
    """Batch-attach literature provenance (lit_family / lit_match_type / lit_ref)
    to leaderboard rows from the graphs table. One query for the whole page;
    defensive — older DBs without the columns simply get no badge."""
    if not entries:
        return
    conn = getattr(nb, "conn", None)
    if conn is None:
        return
    fps = {
        str(e.get("graph_fingerprint") or "").strip()
        for e in entries
        if isinstance(e, dict)
    }
    fps.discard("")
    if not fps:
        return
    try:
        placeholders = ",".join("?" * len(fps))
        rows = conn.execute(  # nosemgrep: python-sql-string-formatting
            f"SELECT graph_fingerprint, lit_family, lit_match_type, lit_ref "  # nosec B608
            f"FROM graphs WHERE graph_fingerprint IN ({placeholders})",
            tuple(fps),
        ).fetchall()
    except Exception:
        return  # lit_* columns absent
    by_fp = {r[0]: r for r in rows}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        row = by_fp.get(str(entry.get("graph_fingerprint") or "").strip())
        if row and row[1]:
            entry["lit_family"] = row[1]
            entry["lit_match_type"] = row[2]
            entry["lit_ref"] = row[3]


def _enrich_ranked_entries(
    nb,
    entries: List[Dict[str, Any]],
    *,
    analytics,
) -> Dict[str, Any]:
    attach_long_context_breakdown(nb, entries)
    stability = _apply_cross_run_stability(nb, entries)
    annotate_qkv_usage(entries, analytics)
    _apply_arch_spec_metrics(entries)
    _annotate_capability_quality(entries)
    _attach_literature_to_entries(nb, entries)
    return stability


def _discovery_references(nb, search_query: str) -> List[Dict[str, Any]]:
    references = nb.get_references()
    annotate_display_names(references)
    if not search_query:
        return references
    return [
        entry for entry in references if _matches_discovery_query(entry, search_query)
    ]


def _strip_heavy_program_fields(programs: List[Dict[str, Any]]) -> None:
    for program in programs:
        program.pop("graph_json", None)
        program.pop("_graph_json", None)
        program.pop("loss_curve", None)


def _classify_and_name_program_entries(nb, programs: List[Dict[str, Any]]) -> None:
    for program in programs:
        program["architecture_family"] = nb._classify_architecture_family(
            graph_json=program.get("graph_json"),
            routing_mode=program.get("routing_mode"),
        )
        program["tier"] = infer_tier_for_program(nb, program)
    annotate_display_names(programs)
    _strip_heavy_program_fields(programs)
    _annotate_capability_quality(programs)
    _attach_dashboard_entry_metadata(programs)


def _discoveries_payload(
    *,
    entries: List[Dict[str, Any]],
    references: List[Dict[str, Any]],
    tier_counts: Dict[str, Any],
    trusted_only: bool,
    view: str,
    **extra: Any,
) -> Dict[str, Any]:
    payload = {
        "entries": _json_safe(entries),
        "references": _json_safe(references),
        "total": len(entries),
        "counts": tier_counts,
        "tier_counts": tier_counts,
        "trusted_only": trusted_only,
        "view": view,
    }
    payload.update(extra)
    return payload


def _leaderboard_score_scale(nb) -> Dict[str, Any]:
    rows = nb.conn.execute(
        """
        SELECT composite_score
        FROM leaderboard
        WHERE composite_score IS NOT NULL
          AND COALESCE(is_reference, 0) = 0
        ORDER BY composite_score ASC
        """
    ).fetchall()
    scores = [float(row["composite_score"]) for row in rows]
    max_possible = composite_score_ceiling(get_scoring_version())
    if not scores:
        return {
            "min": 0.0,
            "p25": 0.0,
            "average": 0.0,
            "max_observed": 0.0,
            "max_possible": max_possible,
            "count": 0,
            "min_source": "db_p25",
            "max_source": "scoring_ceiling",
        }
    p25_index = int((len(scores) - 1) * 0.25)
    return {
        "min": scores[0],
        "p25": scores[p25_index],
        "average": statistics.fmean(scores),
        "max_observed": scores[-1],
        "max_possible": max_possible,
        "count": len(scores),
        "min_source": "db_p25",
        "max_source": "scoring_ceiling",
    }


def _all_discoveries_payload(
    nb,
    *,
    limit: int,
    trusted_only: bool,
    references: List[Dict[str, Any]],
    tier_counts: Dict[str, Any],
    analytics,
) -> Dict[str, Any]:
    programs = nb.get_top_programs(
        limit,
        sort_by="loss_ratio",
        trusted_only=trusted_only,
    )
    attach_long_context_breakdown(nb, programs)
    _attach_derived_fingerprint_metrics(programs)
    annotate_qkv_usage(programs, analytics)
    _classify_and_name_program_entries(nb, programs)
    return _discoveries_payload(
        entries=programs,
        references=references,
        tier_counts=tier_counts,
        trusted_only=trusted_only,
        view="all",
    )


def _program_graph_rows(
    nb,
    *,
    view: str,
    limit: int,
    include_failed: bool,
    search_query: str,
) -> List[Dict[str, Any]]:
    unranked_only = view == "backlog"
    fingerprint_failed_only = view == "fingerprint_failed"
    capped_limit = max(min(int(limit), 5000), 1)
    where = ["TRIM(COALESCE(pr.graph_fingerprint, '')) <> ''"]
    params: List[Any] = []
    if unranked_only:
        where.append("l.entry_id IS NULL")
    if fingerprint_failed_only:
        non_failure_placeholders = ",".join("?" for _ in NON_FAILURE_STATUSES)
        status_clauses = [
            (
                f"(TRIM(COALESCE(pr.{field}, '')) <> '' "
                f"AND LOWER(TRIM(COALESCE(pr.{field}, ''))) "
                f"NOT IN ({non_failure_placeholders}))"
            )
            for field, _label in FINGERPRINT_STATUS_FIELDS
        ]
        where.append(
            "("
            "COALESCE(l.tier, '') = 'investigation_fingerprint_incomplete'"
            f" OR {' OR '.join(status_clauses)}"
            ")"
        )
        for _field, _label in FINGERPRINT_STATUS_FIELDS:
            params.extend(sorted(NON_FAILURE_STATUSES))
    if not include_failed:
        where.append("COALESCE(pr.stage1_passed, 0) = 1")
    if search_query:
        wildcard = f"%{search_query}%"
        where.append(
            "("
            "LOWER(COALESCE(pr.graph_fingerprint, '')) LIKE LOWER(?)"
            " OR LOWER(COALESCE(pr.result_id, '')) LIKE LOWER(?)"
            " OR LOWER(COALESCE(pr.model_source, '')) LIKE LOWER(?)"
            ")"
        )
        params.extend([wildcard, wildcard, wildcard])
    sql = f"""
        SELECT pr.*, l.entry_id AS leaderboard_entry_id
        FROM program_results_compat pr
        LEFT JOIN leaderboard l ON l.result_id = pr.result_id
        WHERE {" AND ".join(where)}
        ORDER BY pr.timestamp DESC
        LIMIT ?
    """
    params.append(capped_limit)
    rows = nb.conn.execute(sql, tuple(params)).fetchall()
    return nb._attach_canonical_program_scores([dict(row) for row in rows])


def _promote_leaderboard_entry_ids(programs: List[Dict[str, Any]]) -> None:
    for program in programs:
        if program.get("leaderboard_entry_id") and not program.get("entry_id"):
            program["entry_id"] = program["leaderboard_entry_id"]


def _attach_metric_completeness(programs: List[Dict[str, Any]]) -> None:
    completeness_fields = (
        "rapid_screening_passed",
        "wikitext_perplexity",
        "hellaswag_acc",
        "induction_intermediate_auc",
        "binding_intermediate_auc",
        "discovery_loss_ratio",
        "validation_loss_ratio",
    )
    for program in programs:
        missing = [field for field in completeness_fields if program.get(field) is None]
        program["missing_metrics"] = missing
        program["missing_metrics_count"] = len(missing)
        program["completeness_ratio"] = 1.0 - len(missing) / len(completeness_fields)


def _program_graph_discoveries_payload(
    nb,
    *,
    view: str,
    limit: int,
    include_failed: bool,
    search_query: str,
    references: List[Dict[str, Any]],
    tier_counts: Dict[str, Any],
    analytics,
) -> Dict[str, Any]:
    programs = _program_graph_rows(
        nb,
        view=view,
        limit=limit,
        include_failed=include_failed,
        search_query=search_query,
    )
    _promote_leaderboard_entry_ids(programs)
    _attach_metric_completeness(programs)
    attach_long_context_breakdown(nb, programs)
    _attach_derived_fingerprint_metrics(programs)
    annotate_qkv_usage(programs, analytics)
    _classify_and_name_program_entries(nb, programs)
    return _discoveries_payload(
        entries=programs,
        references=references,
        tier_counts=tier_counts,
        trusted_only=False,
        view=view,
        include_failed=include_failed,
    )


def _ranked_discovery_entries(
    nb,
    *,
    tier: str | None,
    limit: int,
    sort_by: str,
    search_query: str,
    search_scope: str,
    trusted_only: bool,
) -> List[Dict[str, Any]]:
    if search_query and search_scope == "all":
        return _search_discoveries(
            nb,
            query=search_query,
            tier=tier,
            limit=limit,
            trusted_only=trusted_only,
            include_references=False,
        )
    return nb.get_leaderboard(
        tier=tier,
        limit=limit,
        sort_by=sort_by,
        include_references=False,
        trusted_only=trusted_only,
        tier_match_mode="current",
    )


def _ranked_discoveries_payload(
    nb,
    *,
    tier: str | None,
    limit: int,
    sort_by: str,
    search_query: str,
    search_scope: str,
    trusted_only: bool,
    references: List[Dict[str, Any]],
    tier_counts: Dict[str, Any],
    analytics,
) -> Dict[str, Any]:
    entries = _ranked_discovery_entries(
        nb,
        tier=tier,
        limit=limit,
        sort_by=sort_by,
        search_query=search_query,
        search_scope=search_scope,
        trusted_only=trusted_only,
    )
    stability = _enrich_ranked_entries(nb, entries, analytics=analytics)
    _attach_derived_fingerprint_metrics(entries)
    _attach_dashboard_entry_metadata(entries)
    annotate_display_names(entries)
    return _discoveries_payload(
        entries=entries,
        references=references,
        tier_counts=tier_counts,
        trusted_only=trusted_only,
        view="ranked",
        cross_run_stability_summary=stability.get("summary", {}),
        cross_run_stability_window=stability.get("window_size", 0),
        search={"query": search_query, "scope": search_scope},
    )


def register_leaderboard_routes(app, context: ApiRouteContext):
    notebook_path = context.notebook_path
    wnb = with_notebook_context(notebook_path)

    def api_leaderboard(nb=None):
        """Get leaderboard entries, optionally filtered by tier."""
        tier = request.args.get("tier")
        limit = request.args.get("limit", 50, type=int)
        sort_by = request.args.get("sort", "composite_score")
        quality = str(request.args.get("quality") or "").strip().lower()
        include_references = str(
            request.args.get("include_references", "1")
        ).strip().lower() not in {"0", "false", "no"}
        trusted_only = parse_bool_query(request.args.get("trusted_only"), default=True)
        compact = str(request.args.get("compact", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
        }
        from ..analytics import ExperimentAnalytics

        analytics = None if compact else ExperimentAnalytics(nb)
        base_limit = limit if quality != "promotable" else max(limit * 4, 100)
        entries = nb.get_leaderboard(
            tier=tier,
            limit=base_limit,
            sort_by=sort_by,
            include_references=include_references,
            trusted_only=trusted_only,
        )
        if quality == "promotable":
            entries = [entry for entry in entries if _entry_has_promotion_path(entry)]
            entries = entries[:limit]
        _attach_dashboard_entry_metadata(entries)
        if not compact:
            stability = _enrich_ranked_entries(
                nb,
                entries,
                analytics=analytics,
            )
        else:
            entries = [_compact_leaderboard_entry(entry) for entry in entries]
            stability = {"summary": {}, "window_size": 0}
        tiers = {}
        for entry in entries:
            t = entry.get("tier", "screening")
            if t not in tiers:
                tiers[t] = []
            tiers[t].append(entry)
        return jsonify(
            {
                "entries": entries,
                "by_tier": tiers,
                "total": len(entries),
                "compact": compact,
                "quality": quality or "all",
                "trusted_only": trusted_only,
                "cross_run_stability_summary": stability.get("summary", {}),
                "cross_run_stability_window": stability.get("window_size", 0),
            }
        )

    def api_leaderboard_update_status(nb=None):
        body = request.get_json(silent=True) or {}
        tier = str(body.get("tier") or "").strip().lower()
        entry_id = str(body.get("entry_id") or "").strip()
        result_id = str(body.get("result_id") or "").strip()

        valid_tiers = {
            "screening",
            "screened_out",
            "investigation",
            "investigation_failed",
            "investigation_fingerprint_incomplete",
            "validation",
            "validation_failed",
            "breakthrough",
        }
        if tier not in valid_tiers:
            return jsonify(
                {
                    "error": (
                        "tier must be one of screening, screened_out, "
                        "investigation, investigation_failed, "
                        "investigation_fingerprint_incomplete, validation, "
                        "validation_failed, breakthrough"
                    )
                }
            ), 400
        if not entry_id and not result_id:
            return jsonify({"error": "entry_id or result_id is required"}), 400

        row = None
        if entry_id:
            row = nb.conn.execute(
                "SELECT entry_id, result_id, tier FROM leaderboard WHERE entry_id = ?",
                (entry_id,),
            ).fetchone()
        if row is None and result_id:
            row = nb.conn.execute(
                "SELECT entry_id, result_id, tier FROM leaderboard WHERE result_id = ?",
                (result_id,),
            ).fetchone()
        if row is None:
            return jsonify({"error": "Leaderboard entry not found"}), 404

        resolved_entry_id = row["entry_id"]
        nb.promote_to_tier(resolved_entry_id, tier)

        updated = nb.conn.execute(
            "SELECT entry_id, result_id, tier, timestamp FROM leaderboard WHERE entry_id = ?",
            (resolved_entry_id,),
        ).fetchone()

        return jsonify(
            {
                "success": True,
                "entry": dict(updated)
                if updated
                else {"entry_id": resolved_entry_id, "tier": tier},
            }
        )

    def api_leaderboard_pin(nb=None):
        body = request.get_json(silent=True) or {}
        entry_id = str(body.get("entry_id") or "").strip()
        result_id = str(body.get("result_id") or "").strip()
        pinned = bool(body.get("pinned", False))

        if not entry_id and not result_id:
            return jsonify({"error": "entry_id or result_id is required"}), 400

        resolved_entry_id = entry_id
        if not resolved_entry_id and result_id:
            row = nb.conn.execute(
                "SELECT entry_id FROM leaderboard WHERE result_id = ?",
                (result_id,),
            ).fetchone()
            if row:
                resolved_entry_id = row["entry_id"]
        if not resolved_entry_id:
            return jsonify({"error": "Leaderboard entry not found"}), 404

        nb.set_leaderboard_pin(resolved_entry_id, pinned)
        return jsonify(
            {"success": True, "entry_id": resolved_entry_id, "pinned": pinned}
        )

    def api_leaderboard_rescore(nb=None):
        body = request.get_json(silent=True) or {}
        result_ids = body.get("result_ids") or []
        if isinstance(result_ids, str):
            result_ids = [result_ids]
        if not isinstance(result_ids, list):
            return jsonify({"error": "result_ids must be a list of strings"}), 400

        only_stale_raw = body.get("only_stale", False)
        only_stale = (
            parse_bool_query(only_stale_raw, default=False)
            if isinstance(only_stale_raw, str)
            else bool(only_stale_raw)
        )
        normalized_ids = [
            str(result_id).strip() for result_id in result_ids if str(result_id).strip()
        ]
        total, changed = rescore_leaderboard(
            nb,
            result_ids=normalized_ids or None,
            only_stale=only_stale,
            reason="api_leaderboard_rescore",
        )
        return jsonify(
            {
                "success": True,
                "total": total,
                "changed": changed,
                "only_stale": only_stale,
                "result_ids": normalized_ids,
            }
        )

    def api_leaderboard_queue_rerun_preview(nb=None):
        """Dry-run preview of the auto rerun rule.

        Computes which fingerprints are within striking distance of the
        rank-N composite boundary using the per-tier CV plumbing.  Does
        NOT enqueue anything — caller chooses to apply via the
        sibling POST endpoint.

        Query / body params:
            top_n        int   boundary rank (default 15)
            n            int   reruns per fingerprint (default 2; cosmetic
                               here, only used in the preview output)
            n_runs_cap   int   exclude fps already at this many runs (default 4)
        """
        from research.tools import queue_rerun as qr

        if request.method == "POST":
            body = request.get_json(silent=True) or {}
        else:
            body = request.args
        try:
            top_n = int(body.get("top_n") or 15)
            n_per_fp = int(body.get("n") or 2)
            n_cap = int(body.get("n_runs_cap") or qr.N_RUNS_CAP_DEFAULT)
        except (TypeError, ValueError):
            return jsonify({"error": "top_n / n / n_runs_cap must be ints"}), 400

        report = qr.evaluate(
            context.notebook_path,
            top_n=top_n,
            n_runs_cap=n_cap,
            explicit_fingerprints=None,
        )
        report["n_per_fp"] = n_per_fp
        return jsonify(report)

    def api_leaderboard_queue_rerun_apply(nb=None):
        """Apply the auto rerun rule and enqueue followup_tasks.

        Body:
            top_n        int   default 15
            n            int   reruns per fp (default 2, max 5)
            n_runs_cap   int   default 4
            fingerprints list  optional override — apply manual mode to these
                               specific fingerprints instead of the auto rule
        """
        from research.tools import queue_rerun as qr

        body = request.get_json(silent=True) or {}
        try:
            top_n = int(body.get("top_n") or 15)
            n_per_fp = int(body.get("n") or 2)
            n_cap = int(body.get("n_runs_cap") or qr.N_RUNS_CAP_DEFAULT)
        except (TypeError, ValueError):
            return jsonify({"error": "top_n / n / n_runs_cap must be ints"}), 400
        if n_per_fp < 1 or n_per_fp > 5:
            return jsonify({"error": "n must be in [1, 5]"}), 400
        fps_in = body.get("fingerprints") or None
        if fps_in is not None and not isinstance(fps_in, list):
            return jsonify({"error": "fingerprints must be a list"}), 400

        report = qr.evaluate(
            context.notebook_path,
            top_n=top_n,
            n_runs_cap=n_cap,
            explicit_fingerprints=fps_in,
        )
        queued = qr.queue(
            context.notebook_path,
            eligible=report["eligible"],
            n_per_fp=n_per_fp,
            apply=True,
        )
        report["queued"] = queued
        report["n_per_fp"] = n_per_fp
        return jsonify(report)

    def api_discoveries(nb=None):
        """Unified discoveries endpoint merging leaderboard + raw candidates."""
        tier = request.args.get("tier")
        limit = request.args.get("limit", 100, type=int)
        sort_by = request.args.get("sort", "composite_score")
        view = request.args.get("view", "ranked")
        search_query = str(request.args.get("q") or "").strip()
        search_scope = str(request.args.get("scope") or "ranked").strip().lower()
        trusted_only = parse_bool_query(request.args.get("trusted_only"), default=True)
        from ..analytics import ExperimentAnalytics

        analytics = ExperimentAnalytics(nb)
        tier_counts = count_discovery_tiers(nb)
        references = _discovery_references(nb, search_query)
        score_scale = _leaderboard_score_scale(nb)

        if view == "all":
            payload = _all_discoveries_payload(
                nb,
                limit=limit,
                trusted_only=trusted_only,
                references=references,
                tier_counts=tier_counts,
                analytics=analytics,
            )
            payload["score_scale"] = score_scale
            return jsonify(payload)

        if view in ("backlog", "all_graphs", "fingerprint_failed"):
            include_failed = parse_bool_query(
                request.args.get("include_failed"), default=True
            )
            payload = _program_graph_discoveries_payload(
                nb,
                view=view,
                limit=limit,
                include_failed=include_failed,
                search_query=search_query,
                references=references,
                tier_counts=tier_counts,
                analytics=analytics,
            )
            payload["score_scale"] = score_scale
            return jsonify(payload)

        payload = _ranked_discoveries_payload(
            nb,
            tier=tier,
            limit=limit,
            sort_by=sort_by,
            search_query=search_query,
            search_scope=search_scope,
            trusted_only=trusted_only,
            references=references,
            tier_counts=tier_counts,
            analytics=analytics,
        )
        payload["score_scale"] = score_scale
        return jsonify(payload)

    register_notebook_routes(
        app,
        wnb,
        (
            ("/api/leaderboard", "api_leaderboard", api_leaderboard),
            (
                "/api/leaderboard/status",
                "api_leaderboard_update_status",
                api_leaderboard_update_status,
                ("POST",),
            ),
            (
                "/api/leaderboard/pin",
                "api_leaderboard_pin",
                api_leaderboard_pin,
                ("POST",),
            ),
            (
                "/api/leaderboard/rescore",
                "api_leaderboard_rescore",
                api_leaderboard_rescore,
                ("POST",),
            ),
            (
                "/api/leaderboard/queue-rerun-preview",
                "api_leaderboard_queue_rerun_preview",
                api_leaderboard_queue_rerun_preview,
            ),
            (
                "/api/leaderboard/queue-rerun-apply",
                "api_leaderboard_queue_rerun_apply",
                api_leaderboard_queue_rerun_apply,
                ("POST",),
            ),
            ("/api/discoveries", "api_discoveries", api_discoveries),
        ),
    )
