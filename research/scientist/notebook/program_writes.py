from __future__ import annotations

"""Focused helpers for program_results writes."""

import json
from typing import Any, Callable, Dict, Iterable, List, Tuple

from ._shared import sanitize_for_db
from .program_provenance import derive_provenance_fields

BOOL_PROGRAM_RESULT_FIELDS = {
    "stage0_passed",
    "stage05_passed",
    "stage1_passed",
    "rapid_screening_passed",
    "rapid_screening_degraded",
    "extreme_input_passed",
    "random_input_passed",
    "has_nan_output",
    "has_inf_output",
    "has_nan_grad",
    "has_zero_grad",
    "graph_has_gradient_path",
    "graph_uses_math_spaces",
    "graph_uses_frequency_domain",
    "regression_gate_pass",
    "fingerprint_full_ran",
}


_S1_REQUIRED_POST_METRIC_COLUMNS_FOR_GUARDRAIL = (
    "wikitext_perplexity",
    "hellaswag_acc",
    "blimp_overall_accuracy",
    "induction_auc",
    "binding_auc",
    "binding_composite",
    "ar_auc",
)


_LOSS_ONLY_GUARDED_SOURCES = frozenset({"ablation"})


def _enforce_s1_metric_completeness_for_ablation(
    *,
    graph_fingerprint: str,
    kwargs: Dict[str, Any],
    logger,
) -> None:
    """Refuse to write a stage1_passed=True ablation row missing core post-S1 metrics.

    The original ablation runner persisted only loss/basic fields and silently
    shipped a 1500-row dataset that the diagnostics page then tried to draw
    causal conclusions from. Enforce that any ablation S1-passed row coming
    through this path is metric-complete (or explicitly marked partial via
    trust_label='ablation_metric_backfill_replay' which the backfill tool sets
    after the replay actually fills the columns). Fail loud, not silent.
    """
    if not kwargs.get("stage1_passed"):
        return
    model_source = str(kwargs.get("model_source") or "").lower()
    if model_source not in _LOSS_ONLY_GUARDED_SOURCES:
        return
    if str(kwargs.get("trust_label") or "").startswith("ablation_metric_backfill"):
        return
    missing = [
        c
        for c in _S1_REQUIRED_POST_METRIC_COLUMNS_FOR_GUARDRAIL
        if kwargs.get(c) is None
    ]
    if not missing:
        return
    msg = (
        "BLOCKED ablation S1 write missing post-S1 metrics: "
        f"fp={graph_fingerprint[:16]} missing={missing} "
        "(use program_result_kwargs_from_s1 to assemble metrics, or set "
        "trust_label starting with 'ablation_metric_backfill' for backfill paths)."
    )
    logger.error(msg)
    raise ValueError(msg)


def should_record_program_result(
    *,
    graph_fingerprint: str,
    kwargs: Dict[str, Any],
    bypass_quality_gate: bool,
    logger,
) -> bool:
    stage0_passed = kwargs.get("stage0_passed")
    stage1_passed = kwargs.get("stage1_passed")
    loss_ratio = kwargs.get("loss_ratio")
    if bypass_quality_gate:
        logger.info(
            "Quality gate BYPASSED (debug mode): s0=%s s1=%s lr=%s fp=%s",
            stage0_passed,
            stage1_passed,
            loss_ratio,
            graph_fingerprint,
        )
        return True

    if stage0_passed is not None and not stage0_passed and not kwargs.get("error_type"):
        logger.debug(
            "Quality gate: dropping S0 failure with no error_type (fp=%s)",
            graph_fingerprint,
        )
        return False

    novelty_signal = kwargs.get("novelty_score") or kwargs.get("novelty_confidence")
    if (
        stage0_passed
        and not stage1_passed
        and loss_ratio is None
        and not kwargs.get("error_type")
        and not kwargs.get("error_message")
        and not novelty_signal
    ):
        logger.debug(
            "Quality gate: dropping S1 failure with no signal (fp=%s)",
            graph_fingerprint,
        )
        return False

    _enforce_s1_metric_completeness_for_ablation(
        graph_fingerprint=graph_fingerprint,
        kwargs=kwargs,
        logger=logger,
    )
    return True


def normalize_program_result_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(kwargs)
    if (
        normalized.get("novelty_score") is not None
        and "novelty_scoring_policy_version" not in normalized
    ):
        normalized["novelty_scoring_policy_version"] = "gated_lightning_v1"

    for field in BOOL_PROGRAM_RESULT_FIELDS:
        if field in normalized and normalized[field] is not None:
            normalized[field] = int(normalized[field])

    normalized = sanitize_for_db(normalized)
    if "throughput" in normalized:
        normalized.setdefault("throughput_tok_s", normalized.pop("throughput"))
    return normalized


def enrich_program_result_kwargs(
    kwargs: Dict[str, Any],
    *,
    infer_result_cohort: Callable[[Dict[str, Any]], str],
    infer_trust_label: Callable[[Dict[str, Any], str], str],
    infer_comparability_label: Callable[[Dict[str, Any], str, str], str],
    infer_evaluation_protocol_version: Callable[[Dict[str, Any], str, str], str],
    infer_init_regime: Callable[[Dict[str, Any], str], str],
    build_data_provenance: Callable[..., str],
    build_failure_details: Callable[[Dict[str, Any]], Dict[str, Any] | None],
) -> Dict[str, Any]:
    enriched = dict(kwargs)
    for key, value in derive_provenance_fields(enriched).items():
        enriched.setdefault(key, value)
    result_cohort = str(enriched.get("result_cohort") or infer_result_cohort(enriched))
    trust_label = str(
        enriched.get("trust_label") or infer_trust_label(enriched, result_cohort)
    )
    comparability_label = str(
        enriched.get("comparability_label")
        or infer_comparability_label(enriched, result_cohort, trust_label)
    )
    evaluation_protocol_version = str(
        enriched.get("evaluation_protocol_version")
        or infer_evaluation_protocol_version(enriched, result_cohort, trust_label)
    )
    init_regime = str(
        enriched.get("init_regime") or infer_init_regime(enriched, result_cohort)
    )
    enriched.setdefault("result_cohort", result_cohort)
    enriched.setdefault("trust_label", trust_label)
    enriched.setdefault("comparability_label", comparability_label)
    enriched.setdefault("evaluation_protocol_version", evaluation_protocol_version)
    enriched.setdefault("init_regime", init_regime)
    enriched.setdefault(
        "data_provenance_json",
        build_data_provenance(
            enriched,
            result_cohort=result_cohort,
            trust_label=trust_label,
            comparability_label=comparability_label,
            evaluation_protocol_version=evaluation_protocol_version,
            init_regime=init_regime,
        ),
    )

    if "failure_details_json" not in enriched:
        failure_details = build_failure_details(enriched)
        if failure_details:
            enriched["failure_details_json"] = json.dumps(failure_details)
    elif isinstance(enriched.get("failure_details_json"), (dict, list)):
        enriched["failure_details_json"] = json.dumps(enriched["failure_details_json"])

    if isinstance(enriched.get("semantic_warnings_json"), (dict, list)):
        enriched["semantic_warnings_json"] = json.dumps(
            enriched["semantic_warnings_json"]
        )
    return enriched


def filter_known_program_result_columns(
    kwargs: Dict[str, Any],
    valid_columns: Iterable[str],
) -> Tuple[Dict[str, Any], List[str]]:
    valid = set(valid_columns)
    filtered: Dict[str, Any] = {}
    unknown: List[str] = []
    for column, value in kwargs.items():
        if column in valid:
            filtered[column] = value
        else:
            unknown.append(column)
    return filtered, unknown


def build_program_result_insert_payload(
    *,
    result_id: str,
    experiment_id: str,
    timestamp: float,
    graph_fingerprint: str,
    graph_json: str,
    filtered_kwargs: Dict[str, Any],
) -> tuple[List[str], List[Any]]:
    columns = [
        "result_id",
        "experiment_id",
        "timestamp",
        "graph_fingerprint",
        "graph_json",
    ]
    values: List[Any] = [
        result_id,
        experiment_id,
        timestamp,
        graph_fingerprint,
        graph_json,
    ]
    for column, value in filtered_kwargs.items():
        columns.append(column)
        values.append(value)
    return columns, values
