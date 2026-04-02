"""leaderboard API route registration."""

from __future__ import annotations

import logging
from typing import Any, Dict, List
from flask import jsonify, request
from ..json_utils import json_safe as _json_safe
from .deps import ApiRouteContext
from ._utils import with_notebook_context
from ._strategy_recommendations import (
    annotate_qkv_usage,
    attach_long_context_breakdown,
    compute_cross_run_stability,
    infer_tier_for_program,
    count_discovery_tiers,
)

logger = logging.getLogger(__name__)


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


def _search_discoveries(
    nb,
    *,
    query: str,
    tier: str | None,
    limit: int,
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
            l.screening_loss_ratio,
            l.screening_novelty,
            l.screening_passed,
            l.investigation_loss_ratio,
            l.investigation_robustness,
            l.investigation_passed,
            l.validation_loss_ratio,
            l.validation_baseline_ratio,
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
        if tier and str(entry.get("tier") or "").lower() != str(tier).lower():
            continue
        if not include_references and entry.get("is_reference"):
            continue
        entries.append(entry)

    deduped = _dedupe_discovery_rows(entries)
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
        # Scaling & efficiency (needed by candidateScore)
        "scaling_param_efficiency": entry.get("scaling_param_efficiency"),
        "scaling_gate_passed": entry.get("scaling_gate_passed"),
        # Routing & sparsity (needed by candidateScore)
        "routing_savings_ratio": entry.get("routing_savings_ratio"),
        "routing_utilization_entropy": entry.get("routing_utilization_entropy"),
        "n_routing_ops": entry.get("n_routing_ops"),
        "n_sparse_ops": entry.get("n_sparse_ops"),
        "compression_ratio": entry.get("compression_ratio"),
        "ncd_score": entry.get("ncd_score"),
        "depth_savings_ratio": entry.get("depth_savings_ratio"),
        "recursion_savings_ratio": entry.get("recursion_savings_ratio"),
        "activation_sparsity_score": entry.get("activation_sparsity_score"),
        # Robustness (needed by candidateScore)
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
        # BLiMP linguistic minimal pairs
        "blimp_overall_accuracy": entry.get("blimp_overall_accuracy"),
    }


def register_leaderboard_routes(app, context: ApiRouteContext):
    notebook_path = context.notebook_path
    wnb = with_notebook_context(notebook_path)

    @app.route("/api/leaderboard")
    @wnb
    def api_leaderboard(nb=None):
        """Get leaderboard entries, optionally filtered by tier."""
        tier = request.args.get("tier")
        limit = request.args.get("limit", 50, type=int)
        sort_by = request.args.get("sort", "composite_score")
        quality = str(request.args.get("quality") or "").strip().lower()
        include_references = str(
            request.args.get("include_references", "1")
        ).strip().lower() not in {"0", "false", "no"}
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
        )
        if quality == "promotable":
            entries = [entry for entry in entries if _entry_has_promotion_path(entry)]
            entries = entries[:limit]
        if not compact:
            attach_long_context_breakdown(nb, entries)
            stability = compute_cross_run_stability(
                nb, nb.get_top_programs(20, sort_by="loss_ratio")
            )
            stability_by_result = {
                c.get("result_id"): c
                for c in stability.get("candidates", [])
                if c.get("result_id")
            }
            for entry in entries:
                entry["cross_run_stability"] = stability_by_result.get(
                    entry.get("result_id"),
                    {
                        "trend": "unknown",
                        "seen_runs": 0,
                        "latest_rank": None,
                        "previous_rank": None,
                        "rank_delta": None,
                    },
                )
            annotate_qkv_usage(entries, analytics)
        else:
            entries = [_compact_leaderboard_entry(entry) for entry in entries]
            stability = {"summary": {}, "window_size": 0}
        # Enrich entries with gap_vs_gpt2 and loss_improvement_rate
        # Data is already available from get_leaderboard()'s LEFT JOIN
        import json as _json

        for entry in entries:
            spec_json = entry.get("_arch_spec_json")
            if spec_json:
                try:
                    spec = (
                        _json.loads(spec_json)
                        if isinstance(spec_json, str)
                        else spec_json
                    )
                    if spec.get("gap_nats") is not None:
                        entry["gap_vs_gpt2"] = float(spec["gap_nats"])
                    if (
                        spec.get("improvement_rate") is not None
                        and entry.get("loss_improvement_rate") is None
                    ):
                        entry["loss_improvement_rate"] = float(spec["improvement_rate"])
                except (ValueError, TypeError, _json.JSONDecodeError):
                    pass
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
                "cross_run_stability_summary": stability.get("summary", {}),
                "cross_run_stability_window": stability.get("window_size", 0),
            }
        )

    @app.route("/api/leaderboard/status", methods=["POST"])
    @wnb
    def api_leaderboard_update_status(nb=None):
        body = request.get_json(silent=True) or {}
        tier = str(body.get("tier") or "").strip().lower()
        entry_id = str(body.get("entry_id") or "").strip()
        result_id = str(body.get("result_id") or "").strip()

        valid_tiers = {
            "screening",
            "screened_out",
            "investigation",
            "validation",
            "breakthrough",
        }
        if tier not in valid_tiers:
            return jsonify(
                {
                    "error": "tier must be one of screening, screened_out, investigation, validation, breakthrough"
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

    @app.route("/api/leaderboard/pin", methods=["POST"])
    @wnb
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

    @app.route("/api/discoveries")
    @wnb
    def api_discoveries(nb=None):
        """Unified discoveries endpoint merging leaderboard + raw candidates."""
        from ..naming import annotate_display_names

        tier = request.args.get("tier")
        limit = request.args.get("limit", 100, type=int)
        sort_by = request.args.get("sort", "composite_score")
        view = request.args.get("view", "ranked")
        search_query = str(request.args.get("q") or "").strip()
        search_scope = str(request.args.get("scope") or "ranked").strip().lower()
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
            programs = nb.get_top_programs(limit, sort_by="loss_ratio")
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

            return jsonify(
                {
                    "entries": _json_safe(programs),
                    "references": _json_safe(references),
                    "total": len(programs),
                    "counts": tier_counts,
                    "tier_counts": tier_counts,
                    "view": "all",
                }
            )

        if search_query and search_scope == "all":
            entries = _search_discoveries(
                nb,
                query=search_query,
                tier=tier,
                limit=limit,
                include_references=False,
            )
        else:
            entries = nb.get_leaderboard(
                tier=tier,
                limit=limit,
                sort_by=sort_by,
                include_references=False,
            )
        attach_long_context_breakdown(nb, entries)
        stability = compute_cross_run_stability(
            nb, nb.get_top_programs(20, sort_by="loss_ratio")
        )
        stability_by_result = {
            c.get("result_id"): c
            for c in stability.get("candidates", [])
            if c.get("result_id")
        }
        for entry in entries:
            entry["cross_run_stability"] = stability_by_result.get(
                entry.get("result_id"),
                {
                    "trend": "unknown",
                    "seen_runs": 0,
                    "latest_rank": None,
                    "previous_rank": None,
                    "rank_delta": None,
                },
            )
        annotate_qkv_usage(entries, analytics)
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
                "search": {
                    "query": search_query,
                    "scope": search_scope,
                },
                "view": "ranked",
            }
        )
