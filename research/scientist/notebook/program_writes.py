from __future__ import annotations

"""Focused helpers for program_results writes."""

import json
from typing import Any, Callable, Dict, Iterable, List, Tuple

from research.eval.nano_contract import NANO
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
    # blimp_overall_accuracy deliberately omitted (2026-05-10): BLiMP is the
    # ~50%-of-screening-time cost reported by codex, while at the ~10M-param
    # screening-tier model size it sits near random (BLiMP needs >100M params
    # for above-baseline signal per standard scaling curves). The column is
    # still produced at capability/validation tier by
    # _helpers_benchmark.py::evaluate_blimp, so the leaderboard's
    # blimp_overall_accuracy is populated where it carries information and
    # left NULL at screening where it doesn't.
    "induction_screening_auc",
    "binding_screening_auc",
    "binding_screening_composite",
    # ar_legacy_auc removed 2026-06-18 (artifact probe, no longer computed —
    # see post_s1_probes._REQUIRED_S1_METRICS; the two lists must stay equal).
)

# Additional post-S1 metrics required for investigation-tier writes (rows that
# claim a deeper probe pass). At least one of these must be populated; if all
# are NULL the investigation didn't actually run its deep probes and the row
# is effectively a duplicate screening result that should not mask the cohort.
# Detected by experiment_type='investigation' OR explicit cohort marker.
_INVESTIGATION_REQUIRED_ANY_OF = (
    "induction_intermediate_auc",
    "binding_intermediate_auc",
    "ar_curriculum_auc_pair_final",
)


_TRUST_LABEL_REPLAY_BYPASS_PREFIXES = (
    "ablation_metric_backfill",
    "metric_backfill",
    "replay_",
    "backfill_observation",
    # test_fixture: synthetic rows in unit tests whose subject is not metric
    # completeness itself (e.g. testing leaderboard dedup, accounting,
    # provenance recovery). Production code will never produce this trust_label
    # because production never sets trust_label='test_fixture'; the runner
    # writes 'candidate_screening', 'candidate_grade', 'runtime_observation',
    # 'exploratory', 'reference', 'backfill_observation', or '' depending on
    # the inferred provenance — none of which start with 'test_fixture'.
    "test_fixture",
)


def _enforce_s1_metric_completeness(
    *,
    graph_fingerprint: str,
    kwargs: Dict[str, Any],
    logger,
) -> None:
    """Refuse to write any stage1_passed=True row missing core post-S1 metrics.

    Any row that claims to have passed Stage 1 must carry the full post-S1
    metric set (wikitext_perplexity, hellaswag_acc, blimp_overall_accuracy,
    induction_screening_auc, binding_screening_auc, binding_screening_composite, ar_legacy_auc). Loss-only S1
    rows are not allowed regardless of model_source — they corrupt
    diagnostics, the leaderboard composite, the construction prior, and
    every downstream rule that conditions on metric presence.

    Originally this guarded only ablation writes (the 2026-04-29 incident
    that shipped 1500 loss-only rows). The user's standing rule is broader:
    "we never enter missing data for any experiments". Extended universally.

    Replay/backfill paths that re-fill metrics in-place are exempted via
    trust_label prefixes in _TRUST_LABEL_REPLAY_BYPASS_PREFIXES. A path
    that legitimately cannot produce all probes must either:
      - set stage1_passed=False explicitly, or
      - tag the write with one of the bypass trust_labels above.

    Fail loud, not silent.
    """
    if not kwargs.get("stage1_passed"):
        return
    trust_label = str(kwargs.get("trust_label") or "")
    if any(trust_label.startswith(p) for p in _TRUST_LABEL_REPLAY_BYPASS_PREFIXES):
        return
    missing = [
        c
        for c in _S1_REQUIRED_POST_METRIC_COLUMNS_FOR_GUARDRAIL
        if kwargs.get(c) is None
    ]
    model_source = str(kwargs.get("model_source") or "unknown")
    if missing:
        msg = (
            f"BLOCKED S1 write missing post-S1 metrics: "
            f"fp={graph_fingerprint[:16]} model_source={model_source} missing={missing} "
            "(assemble via program_result_kwargs_from_s1; if a path legitimately "
            "cannot produce all probes, set stage1_passed=False or tag the write "
            f"with trust_label in {_TRUST_LABEL_REPLAY_BYPASS_PREFIXES})."
        )
        logger.error(msg)
        raise ValueError(msg)
    # Investigation-tier completeness: an investigation write must produce at
    # least one investigation-tier probe metric. Without this, an investigation
    # experiment can complete by re-running the screening probes only, leaving
    # induction_intermediate / binding_intermediate / ar_curriculum NULL — that
    # silently degrades the leaderboard for that fingerprint when the row gets
    # rebound by program_leaderboard_repair (2026-05-09 incident: fp 7fb0412ec57a
    # composite dropped from 330→209 because the new investigation row had no
    # post-S1 deep-probe metrics, then became canonical).
    experiment_type = str(kwargs.get("_experiment_type") or "").lower()
    is_investigation = (
        experiment_type == "investigation"
        or str(kwargs.get("result_cohort") or "") == "investigation"
    )
    if is_investigation and not any(
        kwargs.get(c) is not None for c in _INVESTIGATION_REQUIRED_ANY_OF
    ):
        msg = (
            f"BLOCKED investigation S1 write with no investigation-tier probe metrics: "
            f"fp={graph_fingerprint[:16]} model_source={model_source} "
            f"need any of {list(_INVESTIGATION_REQUIRED_ANY_OF)} populated. "
            "An investigation experiment that only re-runs screening probes is a "
            "duplicate, not an investigation. Either run intermediate/curriculum "
            "probes, set stage1_passed=False, or tag the write as backfill."
        )
        logger.error(msg)
        raise ValueError(msg)


def _enforce_nano_floor(
    *,
    graph_fingerprint: str,
    kwargs: Dict[str, Any],
    logger,
) -> None:
    """Refuse to write a stage1_passed=True row for a sub-floor nano model.

    Positive-control evidence (research/eval/nano_contract): a capable 2-layer
    transformer floors at chance below ~1.2M params, so a model below the nano
    floor cannot carry screening signal — admitting it (as the old gate did at a
    100% pass rate) pollutes the leaderboard with noise. Enforced only when a
    param_count is present; replay/backfill paths are exempt via trust_label.

    Fail loud, not silent.
    """
    if not kwargs.get("stage1_passed"):
        return
    trust_label = str(kwargs.get("trust_label") or "")
    if any(trust_label.startswith(p) for p in _TRUST_LABEL_REPLAY_BYPASS_PREFIXES):
        return
    raw = kwargs.get("param_count")
    if raw is None:
        return
    try:
        param_count = int(raw)
    except (TypeError, ValueError):
        return
    if 0 < param_count < NANO.min_params:
        model_source = str(kwargs.get("model_source") or "unknown")
        msg = (
            f"BLOCKED sub-floor nano S1 write: param_count={param_count:,} < nano "
            f"floor {NANO.min_params:,} (dim {NANO.dim}). A capable 2-layer "
            "transformer floors at chance below this size, so a sub-floor model "
            f"cannot be screening-capable. fp={graph_fingerprint[:16]} "
            f"model_source={model_source}. Use a model at/above the nano floor or "
            "set stage1_passed=False."
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
    _enforce_s1_metric_completeness(
        graph_fingerprint=graph_fingerprint,
        kwargs=kwargs,
        logger=logger,
    )
    _enforce_nano_floor(
        graph_fingerprint=graph_fingerprint,
        kwargs=kwargs,
        logger=logger,
    )
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


# Per-architecture columns that live on `graphs` (mirrors the migration script's
# GRAPH_COLUMNS_FROM_PROGRAM_RESULTS). Anything else is per-run.
GRAPH_TABLE_ARCH_COLUMNS = ("graph_json", "arch_spec_json")


def build_dual_write_statements(
    *,
    result_id: str,
    experiment_id: str,
    timestamp: float,
    graph_fingerprint: str,
    graph_json: str,
    filtered_kwargs: Dict[str, Any],
) -> List[tuple[str, tuple]]:
    """Build the atomic write set: graphs UPSERT + graph_runs INSERT.

    Post-Phase-5b: writes target the new tables only. Reads continue to flow
    through `program_results_compat` (= graph_runs LEFT JOIN graphs), which
    preserves the legacy column shape for callers and tests. The AFTER INSERT
    trigger `_gn_sync_pr_to_runs` still propagates raw `INSERT INTO
    program_results` (e.g. test fixtures, ad-hoc tools) to graphs+graph_runs.
    """
    arch_spec_json = filtered_kwargs.get("arch_spec_json")
    graphs_sql = (
        "INSERT INTO graphs "
        "(graph_fingerprint, graph_json, arch_spec_json, first_seen_ts, last_seen_ts, "
        "graph_json_is_placeholder) "
        "VALUES (?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(graph_fingerprint) DO UPDATE SET "
        "  last_seen_ts = excluded.last_seen_ts, "
        "  arch_spec_json = COALESCE(excluded.arch_spec_json, arch_spec_json), "
        # Upgrade placeholder graph_json the moment a real one arrives.
        "  graph_json = CASE WHEN graph_json_is_placeholder = 1 "
        "                     AND excluded.graph_json_is_placeholder = 0 "
        "                THEN excluded.graph_json ELSE graph_json END, "
        "  graph_json_is_placeholder = MIN(graph_json_is_placeholder, "
        "                                  excluded.graph_json_is_placeholder)"
    )
    has_real_graph = bool(graph_json) and graph_json not in ("", "{}")
    graphs_args = (
        graph_fingerprint,
        graph_json or "{}",
        arch_spec_json,
        timestamp,
        timestamp,
        0 if has_real_graph else 1,
    )

    legacy_cols, legacy_vals = build_program_result_insert_payload(
        result_id=result_id,
        experiment_id=experiment_id,
        timestamp=timestamp,
        graph_fingerprint=graph_fingerprint,
        graph_json=graph_json,
        filtered_kwargs=filtered_kwargs,
    )

    arch_set = set(GRAPH_TABLE_ARCH_COLUMNS)
    runs_cols: List[str] = []
    runs_vals: List[Any] = []
    for c, v in zip(legacy_cols, legacy_vals):
        if c in arch_set:
            continue
        runs_cols.append(c)
        runs_vals.append(v)
    runs_sql = (
        f"INSERT INTO graph_runs ({', '.join(runs_cols)}) "
        f"VALUES ({', '.join(['?'] * len(runs_cols))})"
    )

    return [
        (graphs_sql, graphs_args),
        (runs_sql, tuple(runs_vals)),
    ]
