"""Mixin for LabNotebook — split from notebook_misc."""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any, Dict, Optional

from ._notebook_misc_shared import (
    _EMPTY_DATA_ACCOUNTING_SHAPE,
)
from ..json_utils import fast_loads as _json_loads


_DATA_ACCOUNTING_PROCESS_CACHE: Dict[
    str, tuple[tuple[Any, ...], float, Dict[str, Any]]
] = {}


def clear_dashboard_process_caches() -> None:
    """Clear cross-notebook dashboard caches after writes."""
    _DATA_ACCOUNTING_PROCESS_CACHE.clear()


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

    def _dashboard_summary_cache_get(
        self,
        cache_key: tuple[bool, bool],
        now: float,
    ) -> Optional[Dict[str, Any]]:
        cached = getattr(self, "_dashboard_summary_cache", {}).get(cache_key)
        expires_at = float(
            getattr(self, "_dashboard_summary_cache_expires_at", 0.0) or 0.0
        )
        if cached is not None and now < expires_at:
            return dict(cached)
        return None

    def _dashboard_summary_cache_set(
        self,
        cache_key: tuple[bool, bool],
        summary: Dict[str, Any],
        now: float,
    ) -> None:
        self._dashboard_summary_cache[cache_key] = dict(summary)
        self._dashboard_summary_cache_expires_at = now + self._DASHBOARD_SUMMARY_TTL_S

    def _dashboard_summary_rows(self) -> tuple[Any, Any, Any, Any]:
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
            FROM program_results_compat
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
        return exp_row, program_row, insight_row, learning_row

    def _latest_performance_dashboard_payloads(
        self,
    ) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
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
        if not latest_perf_row or not latest_perf_row["results_json"]:
            return None, None
        try:
            latest_results = self._latest_results_payload(
                latest_perf_row["results_json"]
            )
            return (
                self._latest_perf_report_payload(latest_perf_row, latest_results),
                self._latest_dedup_payload(latest_perf_row, latest_results),
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            return None, None

    def _latest_results_payload(self, results_json: Any) -> Any:
        return (
            self._decompress(results_json)
            if isinstance(results_json, bytes)
            else _json_loads(results_json)
        )

    @staticmethod
    def _latest_perf_report_payload(
        latest_perf_row: Any,
        latest_results: Any,
    ) -> Optional[Dict[str, Any]]:
        perf_report = (
            latest_results.get("perf_report")
            if isinstance(latest_results, dict)
            else None
        )
        if not isinstance(perf_report, dict):
            return None
        queue = perf_report.get("queue_telemetry") or {}
        kernel_hotspots = perf_report.get("kernel_hotspots") or []
        return {
            "experiment_id": latest_perf_row["experiment_id"],
            "completed_at": latest_perf_row["completed_at"],
            "programs_profiled": int(perf_report.get("programs_profiled", 0) or 0),
            "avg_submit_wait_ms": float(queue.get("submit_wait_avg_ms", 0.0) or 0.0),
            "avg_scheduling_wait_ms": float(
                queue.get("scheduling_wait_avg_ms", 0.0) or 0.0
            ),
            "gpu_starvation_events": int(
                (perf_report.get("gpu_starvation") or {}).get("event_count", 0) or 0
            ),
            "top_kernel": kernel_hotspots[0] if kernel_hotspots else None,
        }

    @staticmethod
    def _latest_dedup_payload(
        latest_perf_row: Any,
        latest_results: Any,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(latest_results, dict) or "dedup_rate" not in latest_results:
            return None
        return {
            "experiment_id": latest_perf_row["experiment_id"],
            "dedup_rate": latest_results.get("dedup_rate", 0),
            "skipped_dedup": latest_results.get("skipped_dedup", 0),
            "novel_count": latest_results.get("dedup_novel_count", 0),
            "known_fingerprints": latest_results.get("dedup_known_fingerprints", 0),
        }

    @staticmethod
    def _row_int(row: Any, field: str, default: int = 0) -> int:
        return int((row[field] or default) if row else default)

    @staticmethod
    def _row_float(row: Any, field: str, default: float = 0.0) -> float:
        return float((row[field] or default) if row else default)

    @staticmethod
    def _row_optional_float(row: Any, field: str) -> Optional[float]:
        if not row or row[field] is None:
            return None
        return float(row[field])

    def _build_dashboard_summary(
        self,
        *,
        exp_row: Any,
        program_row: Any,
        insight_row: Any,
        learning_row: Any,
        latest_perf_report: Optional[Dict[str, Any]],
        latest_dedup: Optional[Dict[str, Any]],
        include_data_accounting: bool,
        include_template_observability: bool,
    ) -> Dict[str, Any]:
        total_programs = self._row_int(program_row, "total_programs_evaluated")
        stage1_survivors = self._row_int(program_row, "stage1_survivors")
        completed = self._row_int(exp_row, "completed_experiments")
        repaired = self._row_int(exp_row, "repaired_result_experiments")
        return {
            "total_experiments": self._row_int(exp_row, "total_experiments"),
            "completed_experiments": completed,
            "repaired_result_experiments": repaired,
            "resultful_experiments": completed + repaired,
            "total_programs_evaluated": total_programs,
            "stage1_survivors": stage1_survivors,
            "survival_rate": stage1_survivors / max(total_programs, 1),
            "avg_novelty_score": self._row_float(program_row, "avg_novelty_score"),
            "top_novelty_score": self._row_float(program_row, "top_novelty_score"),
            "active_insights": self._row_int(insight_row, "active_insights"),
            "learning_events": self._row_int(learning_row, "learning_events"),
            "latest_learning": learning_row["latest_learning"]
            if learning_row
            else None,
            "avg_step_time_ms": self._row_float(program_row, "avg_step_time_ms"),
            "avg_throughput_tok_s": self._row_float(
                program_row, "avg_throughput_tok_s"
            ),
            "avg_routing_entropy": self._row_optional_float(
                program_row, "avg_routing_entropy"
            ),
            "avg_depth_savings": self._row_optional_float(
                program_row, "avg_depth_savings"
            ),
            "avg_recursion_savings": self._row_optional_float(
                program_row,
                "avg_recursion_savings",
            ),
            "avg_routing_token_retention": self._row_optional_float(
                program_row,
                "avg_routing_token_retention",
            ),
            "avg_sparsity_ratio": self._row_optional_float(
                program_row, "avg_sparsity_ratio"
            ),
            "latest_perf_report": latest_perf_report,
            "unique_fingerprints": self._row_int(program_row, "unique_fingerprints"),
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

    def _data_accounting_cache_lookup(
        self,
    ) -> tuple[Optional[str], Optional[tuple[Any, ...]], Optional[Dict[str, Any]]]:
        if getattr(self, "_is_memory", False):
            return None, None, None
        cache_key = str(getattr(self, "db_path", ""))
        signature = self._data_accounting_signature()
        cached = _DATA_ACCOUNTING_PROCESS_CACHE.get(cache_key)
        if (
            cached is not None
            and signature is not None
            and cached[0] == signature
            and time.time() < cached[1]
        ):
            return cache_key, signature, dict(cached[2])
        return cache_key, signature, None

    def _data_accounting_signature(self) -> Optional[tuple[Any, ...]]:
        signature_row = self.conn.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM program_results_compat) AS pr_count,
                (SELECT MAX(timestamp) FROM program_results_compat) AS pr_max_ts,
                (SELECT COUNT(*) FROM training_curves) AS tc_count,
                (SELECT MAX(rowid) FROM training_curves) AS tc_max_rowid,
                (SELECT COUNT(*) FROM leaderboard) AS lb_count,
                (SELECT MAX(timestamp) FROM leaderboard) AS lb_max_ts
            """
        ).fetchone()
        return tuple(signature_row) if signature_row else None

    def _data_accounting_entity_row(self) -> Any:
        return self.conn.execute(
            """
            SELECT
                COUNT(*) AS program_result_rows,
                COUNT(DISTINCT result_id) AS unique_runs,
                COUNT(DISTINCT CASE
                    WHEN TRIM(COALESCE(graph_fingerprint, '')) <> ''
                    THEN graph_fingerprint END) AS unique_graphs,
                COUNT(DISTINCT CASE
                    WHEN TRIM(COALESCE(graph_fingerprint, '')) <> ''
                    THEN graph_fingerprint || '|' || COALESCE(evaluation_protocol_version, '')
                    END) AS unique_graph_protocols,
                COUNT(DISTINCT CASE
                    WHEN TRIM(COALESCE(graph_fingerprint, '')) <> ''
                    THEN graph_fingerprint || '|' || COALESCE(evaluation_protocol_version, '') || '|' ||
                         COALESCE(CAST(COALESCE(train_budget_steps, n_train_steps) AS TEXT), '')
                    END) AS unique_graph_protocol_budgets,
                SUM(CASE WHEN COALESCE(stage0_passed, 0) = 0 THEN 1 ELSE 0 END)
                    AS runs_filtered_pre_s0,
                SUM(CASE
                    WHEN COALESCE(stage0_passed, 0) = 1
                     AND COALESCE(stage05_passed, 0) = 0 THEN 1 ELSE 0 END)
                    AS runs_filtered_pre_s05,
                SUM(CASE
                    WHEN COALESCE(stage05_passed, 0) = 1
                     AND COALESCE(stage1_passed, 0) = 0 THEN 1 ELSE 0 END)
                    AS runs_filtered_pre_s1,
                SUM(CASE WHEN COALESCE(stage1_passed, 0) = 1 THEN 1 ELSE 0 END)
                    AS runs_reaching_s1_pass,
                COUNT(DISTINCT CASE
                    WHEN TRIM(COALESCE(graph_fingerprint, '')) <> ''
                     AND COALESCE(stage0_passed, 0) = 0
                    THEN graph_fingerprint END) AS graphs_any_filtered_pre_s0,
                COUNT(DISTINCT CASE
                    WHEN TRIM(COALESCE(graph_fingerprint, '')) <> ''
                     AND COALESCE(stage0_passed, 0) = 1
                     AND COALESCE(stage05_passed, 0) = 0
                    THEN graph_fingerprint END) AS graphs_any_filtered_pre_s05,
                COUNT(DISTINCT CASE
                    WHEN TRIM(COALESCE(graph_fingerprint, '')) <> ''
                     AND COALESCE(stage05_passed, 0) = 1
                     AND COALESCE(stage1_passed, 0) = 0
                    THEN graph_fingerprint END) AS graphs_any_filtered_pre_s1,
                COUNT(DISTINCT CASE
                    WHEN TRIM(COALESCE(graph_fingerprint, '')) <> ''
                     AND COALESCE(stage1_passed, 0) = 1
                    THEN graph_fingerprint END) AS graphs_any_s1_pass,
                SUM(CASE
                    WHEN hellaswag_acc IS NOT NULL
                      OR ar_legacy_auc IS NOT NULL
                      OR induction_screening_auc IS NOT NULL
                      OR binding_screening_auc IS NOT NULL
                      OR wikitext_perplexity IS NOT NULL
                    THEN 1 ELSE 0 END) AS downstream_eval_runs,
                COUNT(DISTINCT CASE
                    WHEN TRIM(COALESCE(graph_fingerprint, '')) <> ''
                     AND (hellaswag_acc IS NOT NULL
                       OR ar_legacy_auc IS NOT NULL
                       OR induction_screening_auc IS NOT NULL
                       OR binding_screening_auc IS NOT NULL
                       OR wikitext_perplexity IS NOT NULL)
                    THEN graph_fingerprint END) AS downstream_eval_graphs,
                SUM(CASE
                    WHEN hellaswag_acc IS NOT NULL
                     AND induction_screening_auc IS NOT NULL
                     AND binding_screening_auc IS NOT NULL
                     AND wikitext_perplexity IS NOT NULL
                    THEN 1 ELSE 0 END) AS downstream_full_bundle_runs,
                COUNT(DISTINCT CASE
                    WHEN TRIM(COALESCE(graph_fingerprint, '')) <> ''
                     AND hellaswag_acc IS NOT NULL
                     AND induction_screening_auc IS NOT NULL
                     AND binding_screening_auc IS NOT NULL
                     AND wikitext_perplexity IS NOT NULL
                    THEN graph_fingerprint END) AS downstream_full_bundle_graphs,
                SUM(CASE
                    WHEN COALESCE(trust_label, '') IN ('candidate_screening', 'candidate_grade', 'reference')
                     AND COALESCE(comparability_label, '') IN ('screening_only', 'candidate_comparable', 'reference_comparable')
                    THEN 1 ELSE 0 END) AS trusted_comparable_runs,
                COUNT(DISTINCT CASE
                    WHEN TRIM(COALESCE(graph_fingerprint, '')) <> ''
                     AND COALESCE(trust_label, '') IN ('candidate_screening', 'candidate_grade', 'reference')
                     AND COALESCE(comparability_label, '') IN ('screening_only', 'candidate_comparable', 'reference_comparable')
                    THEN graph_fingerprint END) AS trusted_comparable_graphs,
                SUM(CASE
                    WHEN COALESCE(trust_label, '') IN ('candidate_grade', 'reference')
                     AND COALESCE(comparability_label, '') IN ('candidate_comparable', 'reference_comparable')
                    THEN 1 ELSE 0 END) AS promotable_runs,
                COUNT(DISTINCT CASE
                    WHEN TRIM(COALESCE(graph_fingerprint, '')) <> ''
                     AND COALESCE(trust_label, '') IN ('candidate_grade', 'reference')
                     AND COALESCE(comparability_label, '') IN ('candidate_comparable', 'reference_comparable')
                    THEN graph_fingerprint END) AS promotable_graphs,
                SUM(CASE
                    WHEN json_extract(COALESCE(data_provenance_json, '{}'), '$.eligible_for_screening_model_training') = 1
                      OR (COALESCE(trust_label, '') IN ('candidate_screening', 'candidate_grade', 'reference')
                      AND COALESCE(comparability_label, '') IN ('screening_only', 'candidate_comparable', 'reference_comparable'))
                    THEN 1 ELSE 0 END) AS screening_model_eligible_runs,
                COUNT(DISTINCT CASE
                    WHEN TRIM(COALESCE(graph_fingerprint, '')) <> ''
                     AND (
                        json_extract(COALESCE(data_provenance_json, '{}'), '$.eligible_for_screening_model_training') = 1
                        OR (COALESCE(trust_label, '') IN ('candidate_screening', 'candidate_grade', 'reference')
                        AND COALESCE(comparability_label, '') IN ('screening_only', 'candidate_comparable', 'reference_comparable'))
                     )
                    THEN graph_fingerprint END) AS screening_model_eligible_graphs
            FROM program_results_compat
            """
        ).fetchone()

    def _data_accounting_graph_row(self) -> Any:
        return self.conn.execute(
            """
            SELECT
                SUM(CASE WHEN max_s0 = 0 THEN 1 ELSE 0 END)
                    AS graphs_all_filtered_pre_s0,
                SUM(CASE WHEN max_s0 = 1 AND max_s05 = 0 THEN 1 ELSE 0 END)
                    AS graphs_all_filtered_pre_s05,
                SUM(CASE WHEN max_s05 = 1 AND max_s1 = 0 THEN 1 ELSE 0 END)
                    AS graphs_all_filtered_pre_s1
            FROM (
                SELECT
                    graph_fingerprint,
                    MAX(COALESCE(stage0_passed, 0)) AS max_s0,
                    MAX(COALESCE(stage05_passed, 0)) AS max_s05,
                    MAX(COALESCE(stage1_passed, 0)) AS max_s1
                FROM program_results_compat
                WHERE TRIM(COALESCE(graph_fingerprint, '')) <> ''
                GROUP BY graph_fingerprint
            )
            """
        ).fetchone()

    def _training_curve_counts(self) -> list[int]:
        curve_rows = self.conn.execute(
            """
            SELECT result_id, COUNT(*) AS curve_rows
            FROM training_curves
            GROUP BY result_id
            ORDER BY curve_rows
            """
        ).fetchall()
        curve_counts = [int(row["curve_rows"] or 0) for row in curve_rows]
        curve_counts.extend(self._artifact_training_curve_counts())
        return curve_counts

    def _artifact_training_curve_counts(self) -> list[int]:
        try:
            artifact_rows = self.conn.execute(
                """
                SELECT *
                FROM notebook_artifacts
                WHERE table_name = 'training_curves'
                  AND column_name = 'curve_json'
                ORDER BY row_pk, created_at DESC
                """
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        seen_curve_artifacts: set[str] = set()
        counts: list[int] = []
        for artifact in artifact_rows:
            row_pk = str(artifact["row_pk"])
            if row_pk in seen_curve_artifacts:
                continue
            seen_curve_artifacts.add(row_pk)
            try:
                loaded = self._artifact_store.read_json(dict(artifact))
            except Exception:
                loaded = []
            if isinstance(loaded, list):
                counts.append(len(loaded))
        return counts

    @staticmethod
    def _training_curve_density(curve_counts: list[int]) -> Dict[str, Any]:
        training_curve_rows = sum(curve_counts)
        if not curve_counts:
            return {
                "training_curve_rows": training_curve_rows,
                "runs_with_training_curves": 0,
                "avg_rows_per_run_with_curve": 0.0,
                "median_rows_per_run_with_curve": 0.0,
                "max_rows_per_run_with_curve": 0,
            }
        mid = len(curve_counts) // 2
        median_curve_rows = (
            float(curve_counts[mid])
            if len(curve_counts) % 2
            else float(curve_counts[mid - 1] + curve_counts[mid]) / 2
        )
        return {
            "training_curve_rows": training_curve_rows,
            "runs_with_training_curves": len(curve_counts),
            "avg_rows_per_run_with_curve": round(
                training_curve_rows / len(curve_counts), 2
            ),
            "median_rows_per_run_with_curve": median_curve_rows,
            "max_rows_per_run_with_curve": max(curve_counts),
        }

    def _leaderboard_tier_rows(self) -> list[Any]:
        return list(
            self.conn.execute(
                """
                SELECT tier, COUNT(*) AS entry_count, COUNT(DISTINCT result_id) AS unique_results
                FROM leaderboard
                GROUP BY tier
                ORDER BY entry_count DESC
                """
            ).fetchall()
        )

    @staticmethod
    def _leaderboard_tier_payload(leaderboard_rows: list[Any]) -> Dict[str, Any]:
        return {
            str(row["tier"] or "unknown"): {
                "entries": int(row["entry_count"] or 0),
                "unique_results": int(row["unique_results"] or 0),
            }
            for row in leaderboard_rows
        }

    @staticmethod
    def _data_accounting_result(
        *,
        entity_row: Any,
        graph_row: Any,
        curve_density: Dict[str, Any],
        leaderboard_rows: list[Any],
    ) -> Dict[str, Any]:
        leaderboard_count = sum(
            int(row["entry_count"] or 0) for row in leaderboard_rows
        )
        program_result_rows = int(entity_row["program_result_rows"] or 0)
        runs_with_curves = int(curve_density["runs_with_training_curves"])
        return {
            "row_volume": {
                "program_result_rows": program_result_rows,
                "training_curve_rows": int(curve_density["training_curve_rows"]),
                "leaderboard_rows": leaderboard_count,
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
            "graph_volume": _DashboardNBMixin._data_accounting_graph_volume(entity_row),
            "filtering": _DashboardNBMixin._data_accounting_filtering(
                entity_row, graph_row
            ),
            "training_curve_density": {
                "runs_with_training_curves": runs_with_curves,
                "runs_without_training_curves": program_result_rows - runs_with_curves,
                "avg_rows_per_run_with_curve": curve_density[
                    "avg_rows_per_run_with_curve"
                ],
                "median_rows_per_run_with_curve": curve_density[
                    "median_rows_per_run_with_curve"
                ],
                "max_rows_per_run_with_curve": curve_density[
                    "max_rows_per_run_with_curve"
                ],
            },
            "leaderboard_tiers": _DashboardNBMixin._leaderboard_tier_payload(
                leaderboard_rows
            ),
        }

    @staticmethod
    def _data_accounting_graph_volume(entity_row: Any) -> Dict[str, int]:
        return {
            "unique_graphs": int(entity_row["unique_graphs"] or 0),
            "unique_graph_protocols": int(entity_row["unique_graph_protocols"] or 0),
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
            "downstream_eval_graphs": int(entity_row["downstream_eval_graphs"] or 0),
            "downstream_full_bundle_graphs": int(
                entity_row["downstream_full_bundle_graphs"] or 0
            ),
        }

    @staticmethod
    def _data_accounting_filtering(entity_row: Any, graph_row: Any) -> Dict[str, int]:
        return {
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
                graph_row["graphs_all_filtered_pre_s0"] or 0
            ),
            "graphs_all_filtered_pre_s05": int(
                graph_row["graphs_all_filtered_pre_s05"] or 0
            ),
            "graphs_all_filtered_pre_s1": int(
                graph_row["graphs_all_filtered_pre_s1"] or 0
            ),
        }

    def _data_accounting_cache_set(
        self,
        cache_key: Optional[str],
        signature: Optional[tuple[Any, ...]],
        result: Dict[str, Any],
    ) -> None:
        if cache_key is not None and signature is not None:
            _DATA_ACCOUNTING_PROCESS_CACHE[cache_key] = (
                signature,
                time.time() + self._DASHBOARD_SUMMARY_TTL_S,
                dict(result),
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
        cached = self._dashboard_summary_cache_get(cache_key, now)
        if cached is not None:
            return cached
        exp_row, program_row, insight_row, learning_row = self._dashboard_summary_rows()
        latest_perf_report, latest_dedup = self._latest_performance_dashboard_payloads()
        summary = self._build_dashboard_summary(
            exp_row=exp_row,
            program_row=program_row,
            insight_row=insight_row,
            learning_row=learning_row,
            latest_perf_report=latest_perf_report,
            latest_dedup=latest_dedup,
            include_data_accounting=include_data_accounting,
            include_template_observability=include_template_observability,
        )
        self._dashboard_summary_cache_set(cache_key, summary, now)
        return summary

    def get_data_accounting_summary(self) -> Dict[str, Any]:
        """Separate raw row volume from runs, canonical graphs, and comparable cohorts."""
        cache_key, signature, cached = self._data_accounting_cache_lookup()
        if cached is not None:
            return cached
        result = self._data_accounting_result(
            entity_row=self._data_accounting_entity_row(),
            graph_row=self._data_accounting_graph_row(),
            curve_density=self._training_curve_density(self._training_curve_counts()),
            leaderboard_rows=self._leaderboard_tier_rows(),
        )
        self._data_accounting_cache_set(cache_key, signature, result)
        return result

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
