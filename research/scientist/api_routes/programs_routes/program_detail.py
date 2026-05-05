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
    _get_cached_program_explanation,
    _generate_program_explanation,
    _leaderboard_backed_program_detail,
)

logger = logging.getLogger(__name__)


def _api_program_detail(result_id, nb=None):
    """Full program detail with parsed graph JSON + fingerprint + all metrics."""
    requested_result_id = str(result_id or "").strip()
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
    if program is None:
        return jsonify({"error": "Not found"}), 404

    program["requested_result_id"] = requested_result_id
    program["canonical_result_id"] = result_id
    program["superseded_requested_result"] = requested_result_id != result_id

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

    program = enrich_program_detail(nb, program)
    attach_fingerprint_failure_metadata([program])

    try:
        program["lineage_chain"] = program_lineage_chain(nb, result_id)
    except Exception as exc:
        logger.debug(
            "Failed to load lineage chain for result_id=%s: %s", result_id, exc
        )
        program["lineage_chain"] = []

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
