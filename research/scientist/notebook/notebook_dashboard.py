"""Mixin for LabNotebook — split from notebook_misc."""

from __future__ import annotations

import json
import math
import statistics
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional

from ._notebook_misc_shared import (
    _cached_extract_op_bigrams,
    _cached_extract_observability_metadata,
    _ObservabilityAccumulator,
    _classify_template_structural,
    _capability_signal_count,
    _reference_metric_baselines,
    _reference_beating_metrics,
    _template_label_from_evidence,
    _summarize_template_stat,
    _empty_template_stat,
    _TEMPLATE_DEF_RE,
    _EMPTY_DATA_ACCOUNTING_SHAPE,
)
from ..json_utils import fast_loads as _json_loads
from ..leaderboard_scoring import (
    compute_efficiency_multiple as _compute_efficiency_multiple,
    compute_pre_investigation_score as _compute_pre_investigation_score,
)

class _DashboardNBMixin:
    """Dashboard summary + data accounting."""

    __slots__ = ()
    _DASHBOARD_SUMMARY_TTL_S = 2.0

    def _empty_data_accounting_summary(self) -> Dict[str, Any]:
        return {
            key: dict(value) if isinstance(value, dict) else value
            for key, value in _EMPTY_DATA_ACCOUNTING_SHAPE.items()
        }

    def get_dashboard_headline_summary(self) -> Dict[str, Any]:
        """Cheap dashboard counters without derived observability payloads."""
        return self.get_dashboard_summary(
            include_data_accounting=False,
            include_template_observability=False,
        )

    def get_dashboard_summary(
        self,
        *,
        include_data_accounting: bool = True,
        include_template_observability: bool = True,
    ) -> Dict:
        """Get aggregate stats for the dashboard.

        Heavy derived sections are opt-in so status-style callers stop paying
        for observability and accounting payloads they do not use.
        """
        now = time.time()
        cache_key = (
            bool(include_data_accounting),
            bool(include_template_observability),
        )
        cached = getattr(self, "_dashboard_summary_cache", {}).get(cache_key)
        expires_at = float(
            getattr(self, "_dashboard_summary_cache_expires_at", 0.0) or 0.0
        )
        if cached is not None and now < expires_at:
            return dict(cached)

        exp_row = self.conn.execute(
            """
            SELECT
                COUNT(*) AS total_experiments,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed_experiments,
                SUM(
                    CASE
                        WHEN status = 'failed'
                         AND n_programs_generated > 0
                         AND aria_summary LIKE 'REPAIRED FROM INTERRUPTED:%'
                        THEN 1
                        ELSE 0
                    END
                ) AS repaired_result_experiments
            FROM experiments
            """
        ).fetchone()
        program_row = self.conn.execute(
            """
            SELECT
                COUNT(*) AS total_programs_evaluated,
                SUM(CASE WHEN stage1_passed = 1 THEN 1 ELSE 0 END) AS stage1_survivors,
                AVG(novelty_score) AS avg_novelty_score,
                MAX(novelty_score) AS top_novelty_score,
                AVG(avg_step_time_ms) AS avg_step_time_ms,
                AVG(throughput_tok_s) AS avg_throughput_tok_s,
                AVG(routing_utilization_entropy) AS avg_routing_entropy,
                AVG(depth_savings_ratio) AS avg_depth_savings,
                AVG(recursion_savings_ratio) AS avg_recursion_savings,
                AVG(CASE WHEN routing_tokens_total > 0
                         THEN CAST(routing_tokens_processed AS REAL) / routing_tokens_total END) AS avg_routing_token_retention,
                AVG(sparsity_ratio) AS avg_sparsity_ratio,
                COUNT(DISTINCT graph_fingerprint) AS unique_fingerprints
            FROM program_results
            """
        ).fetchone()
        insight_row = self.conn.execute(
            "SELECT COUNT(*) AS active_insights FROM insights WHERE status = 'active'"
        ).fetchone()
        learning_row = self.conn.execute(
            """
            SELECT
                COUNT(*) AS learning_events,
                (
                    SELECT description
                    FROM learning_log ll2
                    ORDER BY ll2.timestamp DESC
                    LIMIT 1
                ) AS latest_learning
            FROM learning_log
            """
        ).fetchone()

        latest_perf_report = None
        latest_dedup = None
        latest_perf_row = self.conn.execute(
            """SELECT experiment_id, completed_at, results_json
               FROM experiments
               WHERE status = 'completed'
                  OR (status = 'failed'
                      AND n_programs_generated > 0
                      AND aria_summary LIKE 'REPAIRED FROM INTERRUPTED:%')
                 AND results_json IS NOT NULL
               ORDER BY completed_at DESC
               LIMIT 1"""
        ).fetchone()
        if latest_perf_row and latest_perf_row["results_json"]:
            try:
                rj = latest_perf_row["results_json"]
                latest_results = (
                    self._decompress(rj) if isinstance(rj, bytes) else _json_loads(rj)
                )
                perf_report = (
                    latest_results.get("perf_report")
                    if isinstance(latest_results, dict)
                    else None
                )
                if isinstance(perf_report, dict):
                    queue = perf_report.get("queue_telemetry") or {}
                    kernel_hotspots = perf_report.get("kernel_hotspots") or []
                    top_kernel = kernel_hotspots[0] if kernel_hotspots else None
                    latest_perf_report = {
                        "experiment_id": latest_perf_row["experiment_id"],
                        "completed_at": latest_perf_row["completed_at"],
                        "programs_profiled": int(
                            perf_report.get("programs_profiled", 0) or 0
                        ),
                        "avg_submit_wait_ms": float(
                            queue.get("submit_wait_avg_ms", 0.0) or 0.0
                        ),
                        "avg_scheduling_wait_ms": float(
                            queue.get("scheduling_wait_avg_ms", 0.0) or 0.0
                        ),
                        "gpu_starvation_events": int(
                            (perf_report.get("gpu_starvation") or {}).get(
                                "event_count", 0
                            )
                            or 0
                        ),
                        "top_kernel": top_kernel,
                    }
                # Extract dedup stats from latest experiment
                if isinstance(latest_results, dict) and "dedup_rate" in latest_results:
                    latest_dedup = {
                        "experiment_id": latest_perf_row["experiment_id"],
                        "dedup_rate": latest_results.get("dedup_rate", 0),
                        "skipped_dedup": latest_results.get("skipped_dedup", 0),
                        "novel_count": latest_results.get("dedup_novel_count", 0),
                        "known_fingerprints": latest_results.get(
                            "dedup_known_fingerprints", 0
                        ),
                    }
            except (TypeError, ValueError, json.JSONDecodeError):
                latest_perf_report = None

        total_programs = int(
            (program_row["total_programs_evaluated"] or 0) if program_row else 0
        )
        stage1_survivors = int(
            (program_row["stage1_survivors"] or 0) if program_row else 0
        )
        summary = {
            "total_experiments": int(
                (exp_row["total_experiments"] or 0) if exp_row else 0
            ),
            "completed_experiments": int(
                (exp_row["completed_experiments"] or 0) if exp_row else 0
            ),
            "repaired_result_experiments": int(
                (exp_row["repaired_result_experiments"] or 0) if exp_row else 0
            ),
            "resultful_experiments": int(
                ((exp_row["completed_experiments"] or 0) if exp_row else 0)
                + ((exp_row["repaired_result_experiments"] or 0) if exp_row else 0)
            ),
            "total_programs_evaluated": total_programs,
            "stage1_survivors": stage1_survivors,
            "survival_rate": stage1_survivors / max(total_programs, 1),
            "avg_novelty_score": float(
                (program_row["avg_novelty_score"] or 0.0) if program_row else 0.0
            ),
            "top_novelty_score": float(
                (program_row["top_novelty_score"] or 0.0) if program_row else 0.0
            ),
            "active_insights": int(
                (insight_row["active_insights"] or 0) if insight_row else 0
            ),
            "learning_events": int(
                (learning_row["learning_events"] or 0) if learning_row else 0
            ),
            "latest_learning": (
                learning_row["latest_learning"] if learning_row else None
            ),
            "avg_step_time_ms": float(
                (program_row["avg_step_time_ms"] or 0.0) if program_row else 0.0
            ),
            "avg_throughput_tok_s": float(
                (program_row["avg_throughput_tok_s"] or 0.0) if program_row else 0.0
            ),
            "avg_routing_entropy": (
                float(program_row["avg_routing_entropy"])
                if program_row and program_row["avg_routing_entropy"] is not None
                else None
            ),
            "avg_depth_savings": (
                float(program_row["avg_depth_savings"])
                if program_row and program_row["avg_depth_savings"] is not None
                else None
            ),
            "avg_recursion_savings": (
                float(program_row["avg_recursion_savings"])
                if program_row and program_row["avg_recursion_savings"] is not None
                else None
            ),
            "avg_routing_token_retention": (
                float(program_row["avg_routing_token_retention"])
                if program_row
                and program_row["avg_routing_token_retention"] is not None
                else None
            ),
            "avg_sparsity_ratio": (
                float(program_row["avg_sparsity_ratio"])
                if program_row and program_row["avg_sparsity_ratio"] is not None
                else None
            ),
            "latest_perf_report": latest_perf_report,
            "unique_fingerprints": int(
                (program_row["unique_fingerprints"] or 0) if program_row else 0
            ),
            "latest_dedup": latest_dedup,
            "data_accounting": (
                self.get_data_accounting_summary()
                if include_data_accounting
                else self._empty_data_accounting_summary()
            ),
            "template_observability": (
                self.get_template_slot_observability()
                if include_template_observability
                else {}
            ),
        }
        self._dashboard_summary_cache[cache_key] = dict(summary)
        self._dashboard_summary_cache_expires_at = now + self._DASHBOARD_SUMMARY_TTL_S
        return summary

    def get_data_accounting_summary(self) -> Dict[str, Any]:
        """Separate raw row volume from runs, canonical graphs, and comparable cohorts."""
        entity_row = self.conn.execute(
            """
            WITH curve_rows AS (
                SELECT result_id, COUNT(*) AS curve_rows
                FROM training_curves
                GROUP BY result_id
            ),
            run_rows AS (
                SELECT
                    result_id,
                    graph_fingerprint,
                    evaluation_protocol_version,
                    COALESCE(train_budget_steps, n_train_steps) AS budget_steps,
                    stage0_passed,
                    stage05_passed,
                    stage1_passed,
                    trust_label,
                    comparability_label,
                    data_provenance_json,
                    hellaswag_acc,
                    ar_auc,
                    induction_auc,
                    binding_auc,
                    wikitext_perplexity
                FROM program_results
            )
            SELECT
                (SELECT COUNT(*) FROM run_rows) AS program_result_rows,
                (SELECT COUNT(*) FROM training_curves) AS training_curve_rows,
                (SELECT COUNT(*) FROM leaderboard) AS leaderboard_rows,
                (SELECT COUNT(DISTINCT result_id) FROM run_rows) AS unique_runs,
                (SELECT COUNT(DISTINCT graph_fingerprint) FROM run_rows
                  WHERE TRIM(COALESCE(graph_fingerprint, '')) <> '') AS unique_graphs,
                (SELECT COUNT(DISTINCT graph_fingerprint || '|' || COALESCE(evaluation_protocol_version, ''))
                  FROM run_rows
                  WHERE TRIM(COALESCE(graph_fingerprint, '')) <> '') AS unique_graph_protocols,
                (SELECT COUNT(DISTINCT graph_fingerprint || '|' || COALESCE(evaluation_protocol_version, '') || '|' ||
                                      COALESCE(CAST(budget_steps AS TEXT), ''))
                  FROM run_rows
                  WHERE TRIM(COALESCE(graph_fingerprint, '')) <> '') AS unique_graph_protocol_budgets,
                (SELECT COUNT(*) FROM run_rows WHERE COALESCE(stage0_passed, 0) = 0) AS runs_filtered_pre_s0,
                (SELECT COUNT(*) FROM run_rows
                  WHERE COALESCE(stage0_passed, 0) = 1 AND COALESCE(stage05_passed, 0) = 0) AS runs_filtered_pre_s05,
                (SELECT COUNT(*) FROM run_rows
                  WHERE COALESCE(stage05_passed, 0) = 1 AND COALESCE(stage1_passed, 0) = 0) AS runs_filtered_pre_s1,
                (SELECT COUNT(*) FROM run_rows WHERE COALESCE(stage1_passed, 0) = 1) AS runs_reaching_s1_pass,
                (SELECT COUNT(DISTINCT graph_fingerprint) FROM run_rows
                  WHERE TRIM(COALESCE(graph_fingerprint, '')) <> ''
                    AND COALESCE(stage0_passed, 0) = 0) AS graphs_any_filtered_pre_s0,
                (SELECT COUNT(DISTINCT graph_fingerprint) FROM run_rows
                  WHERE TRIM(COALESCE(graph_fingerprint, '')) <> ''
                    AND COALESCE(stage0_passed, 0) = 1
                    AND COALESCE(stage05_passed, 0) = 0) AS graphs_any_filtered_pre_s05,
                (SELECT COUNT(DISTINCT graph_fingerprint) FROM run_rows
                  WHERE TRIM(COALESCE(graph_fingerprint, '')) <> ''
                    AND COALESCE(stage05_passed, 0) = 1
                    AND COALESCE(stage1_passed, 0) = 0) AS graphs_any_filtered_pre_s1,
                (SELECT COUNT(DISTINCT graph_fingerprint) FROM run_rows
                  WHERE TRIM(COALESCE(graph_fingerprint, '')) <> ''
                    AND COALESCE(stage1_passed, 0) = 1) AS graphs_any_s1_pass,
                (SELECT COUNT(*) FROM (
                    SELECT graph_fingerprint
                    FROM run_rows
                    WHERE TRIM(COALESCE(graph_fingerprint, '')) <> ''
                    GROUP BY graph_fingerprint
                    HAVING MAX(COALESCE(stage0_passed, 0)) = 0
                )) AS graphs_all_filtered_pre_s0,
                (SELECT COUNT(*) FROM (
                    SELECT graph_fingerprint
                    FROM run_rows
                    WHERE TRIM(COALESCE(graph_fingerprint, '')) <> ''
                    GROUP BY graph_fingerprint
                    HAVING MAX(COALESCE(stage0_passed, 0)) = 1
                       AND MAX(COALESCE(stage05_passed, 0)) = 0
                )) AS graphs_all_filtered_pre_s05,
                (SELECT COUNT(*) FROM (
                    SELECT graph_fingerprint
                    FROM run_rows
                    WHERE TRIM(COALESCE(graph_fingerprint, '')) <> ''
                    GROUP BY graph_fingerprint
                    HAVING MAX(COALESCE(stage05_passed, 0)) = 1
                       AND MAX(COALESCE(stage1_passed, 0)) = 0
                )) AS graphs_all_filtered_pre_s1,
                (SELECT COUNT(*) FROM run_rows
                  WHERE hellaswag_acc IS NOT NULL
                     OR ar_auc IS NOT NULL
                     OR induction_auc IS NOT NULL
                     OR binding_auc IS NOT NULL
                     OR wikitext_perplexity IS NOT NULL) AS downstream_eval_runs,
                (SELECT COUNT(DISTINCT graph_fingerprint) FROM run_rows
                  WHERE TRIM(COALESCE(graph_fingerprint, '')) <> ''
                    AND (hellaswag_acc IS NOT NULL
                      OR ar_auc IS NOT NULL
                      OR induction_auc IS NOT NULL
                      OR binding_auc IS NOT NULL
                      OR wikitext_perplexity IS NOT NULL)) AS downstream_eval_graphs,
                (SELECT COUNT(*) FROM run_rows
                  WHERE hellaswag_acc IS NOT NULL
                    AND induction_auc IS NOT NULL
                    AND binding_auc IS NOT NULL
                    AND wikitext_perplexity IS NOT NULL) AS downstream_full_bundle_runs,
                (SELECT COUNT(DISTINCT graph_fingerprint) FROM run_rows
                  WHERE TRIM(COALESCE(graph_fingerprint, '')) <> ''
                    AND hellaswag_acc IS NOT NULL
                    AND induction_auc IS NOT NULL
                    AND binding_auc IS NOT NULL
                    AND wikitext_perplexity IS NOT NULL) AS downstream_full_bundle_graphs,
                (SELECT COUNT(*) FROM run_rows
                  WHERE COALESCE(trust_label, '') IN ('candidate_screening', 'candidate_grade', 'reference')
                    AND COALESCE(comparability_label, '') IN ('screening_only', 'candidate_comparable', 'reference_comparable')) AS trusted_comparable_runs,
                (SELECT COUNT(DISTINCT graph_fingerprint) FROM run_rows
                  WHERE TRIM(COALESCE(graph_fingerprint, '')) <> ''
                    AND COALESCE(trust_label, '') IN ('candidate_screening', 'candidate_grade', 'reference')
                    AND COALESCE(comparability_label, '') IN ('screening_only', 'candidate_comparable', 'reference_comparable')) AS trusted_comparable_graphs,
                (SELECT COUNT(*) FROM run_rows
                  WHERE COALESCE(trust_label, '') IN ('candidate_grade', 'reference')
                    AND COALESCE(comparability_label, '') IN ('candidate_comparable', 'reference_comparable')) AS promotable_runs,
                (SELECT COUNT(DISTINCT graph_fingerprint) FROM run_rows
                  WHERE TRIM(COALESCE(graph_fingerprint, '')) <> ''
                    AND COALESCE(trust_label, '') IN ('candidate_grade', 'reference')
                    AND COALESCE(comparability_label, '') IN ('candidate_comparable', 'reference_comparable')) AS promotable_graphs,
                (SELECT COUNT(*) FROM run_rows
                  WHERE json_extract(COALESCE(data_provenance_json, '{}'), '$.eligible_for_screening_model_training') = 1
                     OR (COALESCE(trust_label, '') IN ('candidate_screening', 'candidate_grade', 'reference')
                         AND COALESCE(comparability_label, '') IN ('screening_only', 'candidate_comparable', 'reference_comparable'))) AS screening_model_eligible_runs,
                (SELECT COUNT(DISTINCT graph_fingerprint) FROM run_rows
                  WHERE TRIM(COALESCE(graph_fingerprint, '')) <> ''
                    AND (json_extract(COALESCE(data_provenance_json, '{}'), '$.eligible_for_screening_model_training') = 1
                      OR (COALESCE(trust_label, '') IN ('candidate_screening', 'candidate_grade', 'reference')
                          AND COALESCE(comparability_label, '') IN ('screening_only', 'candidate_comparable', 'reference_comparable')))) AS screening_model_eligible_graphs,
                (SELECT ROUND(AVG(curve_rows), 2) FROM curve_rows) AS avg_training_curve_rows_per_run,
                (SELECT AVG(curve_rows) FROM (
                    SELECT curve_rows
                    FROM curve_rows
                    ORDER BY curve_rows
                    LIMIT 2 - (SELECT COUNT(*) FROM curve_rows) % 2
                    OFFSET (SELECT (COUNT(*) - 1) / 2 FROM curve_rows)
                )) AS median_training_curve_rows_per_run,
                (SELECT MAX(curve_rows) FROM curve_rows) AS max_training_curve_rows_per_run,
                (SELECT COUNT(*) FROM curve_rows) AS runs_with_training_curves,
                (SELECT (SELECT COUNT(*) FROM run_rows) - COUNT(*) FROM curve_rows) AS runs_without_training_curves
            """
        ).fetchone()

        leaderboard_rows = self.conn.execute(
            """
            SELECT tier, COUNT(*) AS entry_count, COUNT(DISTINCT result_id) AS unique_results
            FROM leaderboard
            GROUP BY tier
            ORDER BY entry_count DESC
            """
        ).fetchall()

        return {
            "row_volume": {
                "program_result_rows": int(entity_row["program_result_rows"] or 0),
                "training_curve_rows": int(entity_row["training_curve_rows"] or 0),
                "leaderboard_rows": int(entity_row["leaderboard_rows"] or 0),
            },
            "run_volume": {
                "unique_runs": int(entity_row["unique_runs"] or 0),
                "trusted_comparable_runs": int(
                    entity_row["trusted_comparable_runs"] or 0
                ),
                "promotable_runs": int(entity_row["promotable_runs"] or 0),
                "screening_model_eligible_runs": int(
                    entity_row["screening_model_eligible_runs"] or 0
                ),
                "downstream_eval_runs": int(entity_row["downstream_eval_runs"] or 0),
                "downstream_full_bundle_runs": int(
                    entity_row["downstream_full_bundle_runs"] or 0
                ),
            },
            "graph_volume": {
                "unique_graphs": int(entity_row["unique_graphs"] or 0),
                "unique_graph_protocols": int(
                    entity_row["unique_graph_protocols"] or 0
                ),
                "unique_graph_protocol_budgets": int(
                    entity_row["unique_graph_protocol_budgets"] or 0
                ),
                "trusted_comparable_graphs": int(
                    entity_row["trusted_comparable_graphs"] or 0
                ),
                "promotable_graphs": int(entity_row["promotable_graphs"] or 0),
                "screening_model_eligible_graphs": int(
                    entity_row["screening_model_eligible_graphs"] or 0
                ),
                "downstream_eval_graphs": int(
                    entity_row["downstream_eval_graphs"] or 0
                ),
                "downstream_full_bundle_graphs": int(
                    entity_row["downstream_full_bundle_graphs"] or 0
                ),
            },
            "filtering": {
                "runs_filtered_pre_s0": int(entity_row["runs_filtered_pre_s0"] or 0),
                "runs_filtered_pre_s05": int(entity_row["runs_filtered_pre_s05"] or 0),
                "runs_filtered_pre_s1": int(entity_row["runs_filtered_pre_s1"] or 0),
                "runs_reaching_s1_pass": int(entity_row["runs_reaching_s1_pass"] or 0),
                "graphs_any_filtered_pre_s0": int(
                    entity_row["graphs_any_filtered_pre_s0"] or 0
                ),
                "graphs_any_filtered_pre_s05": int(
                    entity_row["graphs_any_filtered_pre_s05"] or 0
                ),
                "graphs_any_filtered_pre_s1": int(
                    entity_row["graphs_any_filtered_pre_s1"] or 0
                ),
                "graphs_any_s1_pass": int(entity_row["graphs_any_s1_pass"] or 0),
                "graphs_all_filtered_pre_s0": int(
                    entity_row["graphs_all_filtered_pre_s0"] or 0
                ),
                "graphs_all_filtered_pre_s05": int(
                    entity_row["graphs_all_filtered_pre_s05"] or 0
                ),
                "graphs_all_filtered_pre_s1": int(
                    entity_row["graphs_all_filtered_pre_s1"] or 0
                ),
            },
            "training_curve_density": {
                "runs_with_training_curves": int(
                    entity_row["runs_with_training_curves"] or 0
                ),
                "runs_without_training_curves": int(
                    entity_row["runs_without_training_curves"] or 0
                ),
                "avg_rows_per_run_with_curve": float(
                    entity_row["avg_training_curve_rows_per_run"] or 0.0
                ),
                "median_rows_per_run_with_curve": float(
                    entity_row["median_training_curve_rows_per_run"] or 0.0
                ),
                "max_rows_per_run_with_curve": int(
                    entity_row["max_training_curve_rows_per_run"] or 0
                ),
            },
            "leaderboard_tiers": {
                str(row["tier"] or "unknown"): {
                    "entries": int(row["entry_count"] or 0),
                    "unique_results": int(row["unique_results"] or 0),
                }
                for row in leaderboard_rows
            },
        }

    # ── Leaderboard ──

    # Ops considered "routing" for the structural complexity bonus
    _ROUTING_OPS = frozenset(
        {
            "route_topk",
            "route_lanes",
            "route_recursion",
            "token_merge",
            "mod_topk",
            "early_exit",
            "adaptive_recursion",
            "token_merging",
            "cascade",
            "speculative",
            "moe_topk",
            "adaptive_lane_mixer",
            "mixed_recursion_gate",
            "relu_gate_routing",
            "routing_conditioned_compression",
            "token_type_classifier",
            "entropy_score",
            "progressive_compression_gate",
            "compression_mixture_experts",
            "latent_attention_compressor",
        }
    )

    _SPARSE_OPS = frozenset(
        {
            "nm_sparse_linear",
            "block_sparse_linear",
            "semi_structured_2_4_linear",
            "structured_sparse",
            "block_sparse",
            "semi_structured_2_4",
            "hash_trick",
            "sparse_topk",
            "latent_attention_compressor",
            "routing_conditioned_compression",
            "compression_mixture_experts",
            "progressive_compression_gate",
        }
    )

    _MOE_OPS = frozenset(
        {
            "moe_topk",
            "route_topk",
            "route_lanes",
            "adaptive_lane_mixer",
            "compression_mixture_experts",
            "entropy_score",
        }
    )
    _TIER_ORDER = {
        "screening": 0,
        "investigation": 1,
        "validation": 2,
        "breakthrough": 3,
    }
