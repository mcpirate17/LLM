"""leaderboard API route registration."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List
from flask import jsonify, request
from ..json_utils import json_safe as _json_safe
from ..leaderboard_rescore import rescore_leaderboard
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


def _dedupe_discovery_rows(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse repeated fingerprints to the strongest representative row."""
    deduped: List[Dict[str, Any]] = []
    index_by_key: Dict[str, int] = {}
    for entry in entries:
        fingerprint = str(entry.get("graph_fingerprint") or "").strip()
        result_id = str(entry.get("result_id") or "").strip()
        key = fingerprint or result_id
        if not key:
            deduped.append(entry)
            continue
        existing_index = index_by_key.get(key)
        if existing_index is None:
            index_by_key[key] = len(deduped)
            deduped.append(entry)
            continue
        existing = deduped[existing_index]
        existing_score = float(existing.get("composite_score") or 0.0)
        new_score = float(entry.get("composite_score") or 0.0)
        if new_score > existing_score:
            deduped[existing_index] = entry
    return deduped


def _current_discovery_tier(entry: Dict[str, Any]) -> str:
    tier = str(entry.get("tier") or "screening").strip().lower()
    if tier == "validation" and not bool(entry.get("validation_passed")):
        return "validation_pending"
    return tier or "screening"


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
            l.discovery_loss_ratio AS leaderboard_discovery_loss_ratio,
            l.is_reference,
            l.reference_name,
            l.model_source AS leaderboard_model_source,
            l.architecture_desc AS leaderboard_architecture_desc,
            l.timestamp AS leaderboard_timestamp
        FROM program_results pr
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
        if tier and _current_discovery_tier(entry) != str(tier).strip().lower():
            continue
        if not include_references and entry.get("is_reference"):
            continue
        if trusted_only and not is_trusted_entry(entry):
            continue
        entries.append(entry)

    deduped = _dedupe_discovery_rows(entries)
    deduped = nb._attach_canonical_program_scores(deduped)
    return deduped[:limit]


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
        "peak_ppl": entry.get("peak_ppl"),
        "robustness_grade": entry.get("robustness_grade"),
        "evaluation_stage": entry.get("evaluation_stage"),
        "steps_to_divergence": entry.get("steps_to_divergence"),
        "loss_improvement_rate": entry.get("loss_improvement_rate"),
        "baseline_loss_ratio": entry.get("baseline_loss_ratio"),
        # HellaSwag commonsense reasoning
        "hellaswag_acc": entry.get("hellaswag_acc"),
        # Binding probes
        "ar_auc": entry.get("ar_auc"),
        "ar_final_acc": entry.get("ar_final_acc"),
        "ar_timed_out": bool(entry.get("ar_timed_out"))
        if entry.get("ar_timed_out") is not None
        else None,
        "ar_above_chance": bool(entry.get("ar_above_chance"))
        if entry.get("ar_above_chance") is not None
        else None,
        "induction_auc": entry.get("induction_auc"),
        "binding_auc": entry.get("binding_auc"),
        "binding_composite": entry.get("binding_composite"),
        "local_only": entry.get("local_only"),
        # v2 investigation-tier probes
        "induction_v2_investigation_auc": entry.get("induction_v2_investigation_auc"),
        "induction_v2_investigation_max_gap_acc": entry.get(
            "induction_v2_investigation_max_gap_acc"
        ),
        "induction_v2_investigation_protocol_version": entry.get(
            "induction_v2_investigation_protocol_version"
        ),
        "binding_v2_investigation_auc": entry.get("binding_v2_investigation_auc"),
        "binding_v2_investigation_max_distance_acc": entry.get(
            "binding_v2_investigation_max_distance_acc"
        ),
        "binding_v2_investigation_protocol_version": entry.get(
            "binding_v2_investigation_protocol_version"
        ),
        # BLiMP linguistic minimal pairs
        "blimp_overall_accuracy": entry.get("blimp_overall_accuracy"),
    }


def _attach_dashboard_entry_metadata(entries: List[Dict[str, Any]]) -> None:
    if not entries:
        return
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
    return stability


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

    def api_discoveries(nb=None):
        """Unified discoveries endpoint merging leaderboard + raw candidates."""
        from ..naming import annotate_display_names

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
        references = nb.get_references()
        annotate_display_names(references)
        if search_query:
            references = [
                entry
                for entry in references
                if _matches_discovery_query(entry, search_query)
            ]

        if view == "all":
            programs = nb.get_top_programs(
                limit,
                sort_by="loss_ratio",
                trusted_only=trusted_only,
            )
            attach_long_context_breakdown(nb, programs)
            annotate_qkv_usage(programs, analytics)
            for p in programs:
                p["architecture_family"] = nb._classify_architecture_family(
                    graph_json=p.get("graph_json"),
                    routing_mode=p.get("routing_mode"),
                )
                p["tier"] = infer_tier_for_program(nb, p)
            annotate_display_names(programs)
            for p in programs:
                p.pop("graph_json", None)
                p.pop("_graph_json", None)
                p.pop("loss_curve", None)
            _annotate_capability_quality(programs)
            _attach_dashboard_entry_metadata(programs)

            return jsonify(
                {
                    "entries": _json_safe(programs),
                    "references": _json_safe(references),
                    "total": len(programs),
                    "counts": tier_counts,
                    "tier_counts": tier_counts,
                    "trusted_only": trusted_only,
                    "view": "all",
                }
            )

        if view in ("backlog", "all_graphs"):
            include_failed = parse_bool_query(
                request.args.get("include_failed"), default=True
            )
            unranked_only = view == "backlog"
            capped_limit = max(min(int(limit), 5000), 1)

            where = ["TRIM(COALESCE(pr.graph_fingerprint, '')) <> ''"]
            params: List[Any] = []
            if unranked_only:
                where.append("l.entry_id IS NULL")
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
                FROM program_results pr
                LEFT JOIN leaderboard l ON l.result_id = pr.result_id
                WHERE {" AND ".join(where)}
                ORDER BY pr.timestamp DESC
                LIMIT ?
            """
            params.append(capped_limit)
            rows = nb.conn.execute(sql, tuple(params)).fetchall()
            programs = nb._attach_canonical_program_scores([dict(row) for row in rows])
            for p in programs:
                # Promote leaderboard_entry_id → entry_id so client filters
                # (`!entry?.entry_id`) and the existing column renderers work
                # uniformly across ranked and unranked rows.
                if p.get("leaderboard_entry_id") and not p.get("entry_id"):
                    p["entry_id"] = p["leaderboard_entry_id"]
            completeness_fields = (
                "rapid_screening_passed",
                "wikitext_perplexity",
                "hellaswag_acc",
                "induction_v2_investigation_auc",
                "binding_v2_investigation_auc",
                "discovery_loss_ratio",
                "validation_loss_ratio",
            )
            for p in programs:
                missing = [f for f in completeness_fields if p.get(f) is None]
                p["missing_metrics"] = missing
                p["missing_metrics_count"] = len(missing)
                p["completeness_ratio"] = 1.0 - len(missing) / len(completeness_fields)
            attach_long_context_breakdown(nb, programs)
            annotate_qkv_usage(programs, analytics)
            for p in programs:
                p["architecture_family"] = nb._classify_architecture_family(
                    graph_json=p.get("graph_json"),
                    routing_mode=p.get("routing_mode"),
                )
                p["tier"] = infer_tier_for_program(nb, p)
            annotate_display_names(programs)
            for p in programs:
                p.pop("graph_json", None)
                p.pop("_graph_json", None)
                p.pop("loss_curve", None)
            _annotate_capability_quality(programs)
            _attach_dashboard_entry_metadata(programs)

            return jsonify(
                {
                    "entries": _json_safe(programs),
                    "references": _json_safe(references),
                    "total": len(programs),
                    "counts": tier_counts,
                    "tier_counts": tier_counts,
                    "trusted_only": False,
                    "view": view,
                    "include_failed": include_failed,
                }
            )

        if search_query and search_scope == "all":
            entries = _search_discoveries(
                nb,
                query=search_query,
                tier=tier,
                limit=limit,
                trusted_only=trusted_only,
                include_references=False,
            )
        else:
            entries = nb.get_leaderboard(
                tier=tier,
                limit=limit,
                sort_by=sort_by,
                include_references=False,
                trusted_only=trusted_only,
                tier_match_mode="current",
            )
        stability = _enrich_ranked_entries(
            nb,
            entries,
            analytics=analytics,
        )
        _attach_dashboard_entry_metadata(entries)
        annotate_display_names(entries)

        return jsonify(
            {
                "entries": _json_safe(entries),
                "references": _json_safe(references),
                "total": len(entries),
                "counts": tier_counts,
                "tier_counts": tier_counts,
                "cross_run_stability_summary": stability.get("summary", {}),
                "cross_run_stability_window": stability.get("window_size", 0),
                "trusted_only": trusted_only,
                "search": {
                    "query": search_query,
                    "scope": search_scope,
                },
                "view": "ranked",
            }
        )

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
            ("/api/discoveries", "api_discoveries", api_discoveries),
        ),
    )
