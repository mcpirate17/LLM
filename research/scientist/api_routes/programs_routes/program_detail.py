"""Program detail / explanation / lineage / refine / list / training-curve handlers."""

from __future__ import annotations

import logging
from flask import jsonify, request

from .._strategy_recommendations import (
    annotate_qkv_usage,
    enrich_program_detail,
    program_lineage_chain,
)
from .._fingerprint_failures import attach_fingerprint_failure_metadata
from ...json_utils import json_safe

from ._shared import (
    attach_candidate_confirmation_status,
    _get_cached_program_explanation,
    _generate_program_explanation,
    _leaderboard_backed_program_detail,
)

logger = logging.getLogger(__name__)


_FINGERPRINT_ROLLUP_FIELDS = (
    "induction_intermediate_auc",
    "induction_intermediate_max_gap_acc",
    "induction_intermediate_gap_accuracies_json",
    "induction_intermediate_steps_trained",
    "induction_intermediate_status",
    "induction_intermediate_elapsed_ms",
    "induction_intermediate_protocol_version",
    "binding_intermediate_auc",
    "binding_intermediate_max_distance_acc",
    "binding_intermediate_distance_accuracies_json",
    "binding_intermediate_train_steps",
    "binding_intermediate_status",
    "binding_intermediate_elapsed_ms",
    "binding_intermediate_protocol_version",
    "ar_gate_metric_version",
    "ar_gate_in_dist_pair_acc",
    "ar_gate_in_dist_class_acc",
    "ar_gate_held_pair_acc",
    "ar_gate_held_class_acc",
    "ar_gate_score",
    "ar_gate_status",
    "ar_gate_elapsed_ms",
    "ar_gate_train_steps_done",
    "champion_floor_protocol_version",
    "champion_steps_to_floor",
    "champion_floor_loss",
    "champion_floor_ppl",
    "champion_floor_loss_std",
    "champion_plateau_detected_step",
    "champion_plateau_window",
    "champion_baseline_result_id",
    "champion_baseline_layers",
    "champion_baseline_protocol_version",
    "champion_steps_to_floor_score",
    "champion_floor_quality_score",
    "champion_floor_stability_score",
    "champion_induction_validation_score",
    "champion_binding_long_context_score",
    "champion_ar_validation_score",
    "champion_tiny_model_score",
    "champion_tiny_model_protocol_version",
    "champion_hard_failure_reason",
    "induction_validation_auc",
    "induction_validation_max_gap_acc",
    "induction_validation_gap_accuracy_cv",
    "induction_validation_gap_accuracies_json",
    "induction_validation_steps_trained",
    "induction_validation_status",
    "induction_validation_elapsed_ms",
    "induction_validation_protocol_version",
    "ar_validation_metric_version",
    "ar_validation_final_acc",
    "ar_validation_held_pair_acc",
    "ar_validation_held_class_acc",
    "ar_validation_learning_curve_json",
    "ar_validation_steps_to_floor",
    "ar_validation_rank_score",
    "ar_validation_status",
    "ar_validation_elapsed_ms",
    "ar_intermediate_metric_version",
    "ar_intermediate_diagnostic_score",
    "ar_intermediate_held_pair_acc",
    "ar_intermediate_held_pair_lift",
    "ar_intermediate_held_class_acc",
    "ar_intermediate_auc_lift",
    "ar_intermediate_best_held_pair_acc",
    "ar_intermediate_improvement",
    "ar_intermediate_status",
    "ar_intermediate_elapsed_ms",
    "binding_multislot_metric_version",
    "binding_multislot_diagnostic_score",
    "binding_multislot_held_entity_slot_acc",
    "binding_multislot_held_slot_lift",
    "binding_multislot_two_plus_slots_acc",
    "binding_multislot_two_plus_slots_lift",
    "binding_multislot_mixed_two_plus_slots_acc",
    "binding_multislot_mixed_two_plus_slots_lift",
    "binding_multislot_all_slots_acc",
    "binding_multislot_auc_lift",
    "binding_multislot_status",
    "binding_multislot_elapsed_ms",
)


def _attach_fingerprint_observation_rollup(nb, program):
    graph_fingerprint = str(program.get("graph_fingerprint") or "").strip()
    if not graph_fingerprint:
        program["same_fingerprint_results"] = []
        return

    rows = nb.conn.execute(
        """
        SELECT pr.result_id, pr.experiment_id, pr.timestamp, pr.model_source,
               pr.intentional_rerun_reason, pr.result_cohort, pr.trust_label,
               pr.comparability_label, e.experiment_type, e.status AS experiment_status,
               e.completed_at
        FROM program_results pr
        LEFT JOIN experiments e ON e.experiment_id = pr.experiment_id
        WHERE pr.graph_fingerprint = ?
        ORDER BY pr.timestamp DESC
        LIMIT 12
        """,
        (graph_fingerprint,),
    ).fetchall()
    observations = [dict(row) for row in rows]
    program["same_fingerprint_results"] = observations
    latest_investigation = next(
        (
            row
            for row in observations
            if str(row.get("experiment_type") or "").lower() == "investigation"
        ),
        None,
    )
    if latest_investigation:
        program["fingerprint_latest_experiment"] = latest_investigation
        program["display_experiment_type"] = "investigation"

    select_cols = ", ".join(_FINGERPRINT_ROLLUP_FIELDS)
    evidence = nb.conn.execute(
        f"""
        SELECT result_id, experiment_id, timestamp, {select_cols}
        FROM program_results
        WHERE graph_fingerprint = ?
          AND (
            induction_intermediate_auc IS NOT NULL
            OR binding_intermediate_auc IS NOT NULL
            OR ar_gate_score IS NOT NULL
            OR champion_tiny_model_score IS NOT NULL
            OR induction_validation_auc IS NOT NULL
            OR ar_validation_rank_score IS NOT NULL
            OR ar_intermediate_diagnostic_score IS NOT NULL
            OR binding_multislot_diagnostic_score IS NOT NULL
          )
        ORDER BY timestamp DESC
        LIMIT 1
        """,
        (graph_fingerprint,),
    ).fetchone()
    if not evidence:
        return

    evidence_dict = dict(evidence)
    program["fingerprint_metric_source_result_id"] = evidence_dict.get("result_id")
    program["fingerprint_metric_source_experiment_id"] = evidence_dict.get(
        "experiment_id"
    )
    for field in _FINGERPRINT_ROLLUP_FIELDS:
        if program.get(field) is None and evidence_dict.get(field) is not None:
            program[field] = evidence_dict[field]


def _load_program_detail_record(nb, requested_result_id: str):
    """Load the requested row, falling back through canonical/leaderboard IDs."""
    result_id = requested_result_id
    program = nb.get_program_detail(result_id)
    if program is None:
        canonical_result_id = nb.resolve_canonical_result_id(requested_result_id)
        result_id = canonical_result_id or requested_result_id
        program = nb.get_program_detail(result_id)
    if program is None:
        program = _leaderboard_backed_program_detail(nb, result_id)
    if program is None:
        # Fallback: resolve fingerprint (architecture_desc) to result_id
        row = nb.conn.execute(
            "SELECT result_id FROM leaderboard WHERE architecture_desc = ? LIMIT 1",
            (result_id,),
        ).fetchone()
        if row:
            resolved_id = row[0] if isinstance(row, (tuple, list)) else row["result_id"]
            program = nb.get_program_detail(resolved_id)
            if program is None:
                program = _leaderboard_backed_program_detail(nb, resolved_id)
    return result_id, program


def _attach_fingerprint_parent_status(nb, result_id: str, program: dict) -> str:
    fingerprint_parent_result_id = result_id
    graph_fingerprint = str(program.get("graph_fingerprint") or "").strip()
    if graph_fingerprint:
        try:
            parent_entry = nb.get_leaderboard_entry_by_fingerprint(graph_fingerprint)
            if parent_entry and parent_entry.get("result_id"):
                fingerprint_parent_result_id = str(parent_entry["result_id"])
                for status_key in (
                    "tier",
                    "composite_score",
                    "investigation_loss_ratio",
                    "investigation_robustness",
                    "investigation_passed",
                    "validation_loss_ratio",
                    "validation_baseline_ratio",
                    "validation_multi_seed_std",
                    "validation_passed",
                    "n_runs",
                    "cv_loss",
                    "cv_understanding",
                    "cv_capability",
                    "score_stability_penalty",
                ):
                    if parent_entry.get(status_key) is not None:
                        program[status_key] = parent_entry.get(status_key)
                for src_key, display_key in (
                    ("result_cohort", "display_result_cohort"),
                    ("trust_label", "display_trust_label"),
                    ("comparability_label", "display_comparability_label"),
                ):
                    if parent_entry.get(src_key):
                        program[display_key] = parent_entry.get(src_key)
        except Exception as exc:
            logger.debug(
                "Failed to load fingerprint parent for result_id=%s: %s",
                result_id,
                exc,
            )
    return fingerprint_parent_result_id


def _attach_program_detail_extras(nb, result_id: str, program: dict) -> None:
    program["display_result_id"] = result_id
    _attach_fingerprint_observation_rollup(nb, program)
    attach_candidate_confirmation_status(nb, program)
    try:
        curve = nb.get_training_curve(result_id)
        program["has_training_curve"] = len(curve) > 0
    except Exception as exc:
        logger.debug(
            "Failed to load training curve for result_id=%s: %s", result_id, exc
        )
        program["has_training_curve"] = False

    cached_explanation = _get_cached_program_explanation(nb, result_id)
    if cached_explanation:
        program["llm_explanation"] = cached_explanation

    try:
        program["lineage_chain"] = program_lineage_chain(nb, result_id)
    except Exception as exc:
        logger.debug(
            "Failed to load lineage chain for result_id=%s: %s", result_id, exc
        )
        program["lineage_chain"] = []


def _attach_causal_rule_evidence(nb, result_id: str, program: dict) -> None:
    try:
        evidence_rows = nb.get_causal_rule_evidence(
            result_id=result_id,
            limit=20,
        )
        for item in evidence_rows:
            evidence_id = item.get("evidence_id")
            if evidence_id:
                children = nb.get_causal_ablation_child_observations(
                    evidence_id=evidence_id,
                    limit=8,
                )
                item["child_observation_count"] = len(children)
                item["child_observations"] = children
        program["causal_rule_evidence"] = evidence_rows
    except Exception as exc:
        logger.debug(
            "Failed to load causal evidence for result_id=%s: %s", result_id, exc
        )
        program["causal_rule_evidence"] = []


def _api_program_detail(result_id, nb=None):
    """Full program detail with parsed graph JSON + fingerprint + all metrics."""
    requested_result_id = str(result_id or "").strip()

    # Program detail must show the requested row when it still exists.  Forced
    # reruns intentionally create same-fingerprint child rows; resolving every
    # request to the newest sibling makes the original parent appear to vanish.
    result_id, program = _load_program_detail_record(nb, requested_result_id)
    if program is None:
        return jsonify({"error": "Not found"}), 404

    program["requested_result_id"] = requested_result_id
    fingerprint_parent_result_id = _attach_fingerprint_parent_status(
        nb, result_id, program
    )
    program["canonical_result_id"] = fingerprint_parent_result_id
    program["fingerprint_parent_result_id"] = fingerprint_parent_result_id
    program["superseded_requested_result"] = requested_result_id != result_id
    _attach_program_detail_extras(nb, result_id, program)
    program = enrich_program_detail(nb, program)
    attach_fingerprint_failure_metadata([program])
    _attach_causal_rule_evidence(nb, result_id, program)
    return jsonify(json_safe(program))


def _api_program_explanation(result_id, nb=None):
    """Generate or fetch cached LLM explanation for a program."""
    requested_result_id = str(result_id or "").strip()
    canonical_result_id = nb.resolve_canonical_result_id(requested_result_id)
    result_id = canonical_result_id or requested_result_id
    force = bool((request.get_json(silent=True) or {}).get("force", False))

    program = nb.get_program_detail(result_id)
    if program is None:
        program = _leaderboard_backed_program_detail(nb, result_id)
    if program is None:
        return jsonify({"error": "Not found"}), 404

    if not force:
        cached_explanation = _get_cached_program_explanation(nb, result_id)
        if cached_explanation:
            return jsonify(
                json_safe(
                    {
                        "result_id": result_id,
                        "requested_result_id": requested_result_id,
                        "canonical_result_id": result_id,
                        "superseded_requested_result": requested_result_id != result_id,
                        "llm_explanation": cached_explanation,
                        "source": "cached",
                    }
                )
            )

    try:
        explanation = _generate_program_explanation(nb, result_id, program)
    except Exception as exc:
        logger.debug(
            "LLM fingerprint explanation failed for result_id=%s: %s",
            result_id,
            exc,
        )
        return (
            jsonify(
                {
                    "result_id": result_id,
                    "requested_result_id": requested_result_id,
                    "canonical_result_id": result_id,
                    "superseded_requested_result": requested_result_id != result_id,
                    "llm_explanation": None,
                    "source": "unavailable",
                    "error": str(exc),
                }
            ),
            503,
        )

    if not explanation:
        return jsonify(
            {
                "result_id": result_id,
                "requested_result_id": requested_result_id,
                "canonical_result_id": result_id,
                "superseded_requested_result": requested_result_id != result_id,
                "llm_explanation": None,
                "source": "unavailable",
            }
        )

    return jsonify(
        json_safe(
            {
                "result_id": result_id,
                "requested_result_id": requested_result_id,
                "canonical_result_id": result_id,
                "superseded_requested_result": requested_result_id != result_id,
                "llm_explanation": explanation,
                "source": "generated",
            }
        )
    )


def _api_program_lineage(result_id: str, nb=None):
    """Program lineage chain for refinement traceability."""
    program = nb.get_program_detail(result_id)
    if program is None:
        return jsonify({"error": "Not found"}), 404
    chain = program_lineage_chain(nb, result_id)
    return jsonify(
        json_safe(
            {
                "result_id": result_id,
                "lineage_chain": chain,
                "depth": len(chain),
            }
        )
    )


def _api_program_refine_analysis(result_id, nb=None):
    from ...analytics import ExperimentAnalytics, RefinementAnalyzer

    program = nb.get_program_detail(result_id)
    if program is None:
        return jsonify({"error": "Not found"}), 404

    analytics = ExperimentAnalytics(nb)
    analyzer = RefinementAnalyzer(analytics)
    analysis = analyzer.analyze_program_for_refinement(result_id, program)
    return jsonify(json_safe(analysis))


def _api_programs(nb=None):
    n = request.args.get("n", 20, type=int)
    sort_by = request.args.get("sort", "novelty_score")
    from ...analytics import ExperimentAnalytics

    analytics = ExperimentAnalytics(nb)
    programs = nb.get_top_programs(n, sort_by)
    annotate_qkv_usage(programs, analytics)
    return jsonify(json_safe(programs))


def _api_training_curve(result_id, nb=None):
    curve = nb.get_training_curve(result_id)
    return jsonify(curve)
