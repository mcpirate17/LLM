from __future__ import annotations

"""Auto-extracted mixin for LabNotebook."""

import json
import sqlite3
import time
import uuid
from typing import Any, Dict, List, Optional

from ._shared import LOGGER, sanitize_for_db
from .graph_artifacts import resolve_graph_json_value
from ..leaderboard_dashboard_fields import (
    PROGRAM_RESULT_DASHBOARD_ALIAS_FIELDS as _PROGRAM_RESULT_DASHBOARD_ALIAS_FIELDS,
)
from ..leaderboard_scoring import (
    build_score_kwargs,
    compute_composite,
    get_scoring_version,
)
from ..thresholds import TIER_RANK
from ..trust_policy import is_promotable_entry, sql_trusted_clause


class DuplicateLeaderboardFingerprintError(Exception):
    """Raised when ``upsert_leaderboard`` would create a second leaderboard
    row for a ``graph_fingerprint`` that already has an entry under a
    different ``result_id``.

    Pass ``allow_fingerprint_duplicate=True`` to bypass, or resolve by calling
    ``promote_to_tier(existing_entry_id, ...)`` on the pre-existing entry so
    metrics merge onto one row instead of creating a duplicate.
    """

    def __init__(
        self,
        graph_fingerprint: str,
        existing_entry_id: str,
        existing_result_id: str,
        attempted_result_id: str,
    ) -> None:
        self.graph_fingerprint = graph_fingerprint
        self.existing_entry_id = existing_entry_id
        self.existing_result_id = existing_result_id
        self.attempted_result_id = attempted_result_id
        super().__init__(
            f"fingerprint {graph_fingerprint} is already on the leaderboard at "
            f"entry_id={existing_entry_id} (result_id={existing_result_id}); "
            f"attempted insert for result_id={attempted_result_id}. "
            f"Call promote_to_tier() on the existing entry, or pass "
            f"allow_fingerprint_duplicate=True if the duplicate is intentional."
        )


_LEADERBOARD_MANAGED_COLUMNS = frozenset(
    {
        "entry_id",
        "result_id",
        "timestamp",
        "model_source",
        "architecture_desc",
        "tier",
        "composite_score",
        "is_reference",
        "reference_name",
        "tags",
        "notes",
    }
)

_PROGRAM_RESULT_UPSERT_SELECT = (
    "SELECT result_id, novelty_confidence, loss_ratio, param_count, flops_forward, "
    "throughput_tok_s, peak_memory_mb, forward_time_ms, graph_json, graph_fingerprint, "
    "result_cohort, trust_label, comparability_label, evaluation_protocol_version, "
    "data_provenance_json "
    "FROM program_results_compat "
)

_LEADERBOARD_BASE_SELECT = (
    "SELECT l.*, pr.graph_json AS _graph_json, "
    "pr.routing_mode AS _routing_mode, "
    "pr.graph_fingerprint AS _graph_fingerprint, "
    "pr.arch_spec_json AS _arch_spec_json, "
    "pr.param_count AS _param_count, "
    "pr.graph_n_params_estimate AS _graph_n_params_estimate, "
    "pr.novelty_confidence AS _novelty_confidence, "
    "pr.novelty_valid_for_promotion AS novelty_valid_for_promotion, "
    "pr.novelty_validity_reason AS novelty_validity_reason, "
    "pr.cka_source AS _cka_source, "
    "pr.stage0_passed AS stage0_passed, "
    "pr.stage1_passed AS stage1_passed, "
    "pr.routing_confidence_mean AS _routing_confidence_mean, "
    "pr.fp_jacobian_spectral_norm AS jacobian_spectral_norm, "
    "pr.fp_jacobian_effective_rank AS fp_jacobian_effective_rank, "
    "pr.fp_sensitivity_uniformity AS fp_sensitivity_uniformity, "
    "pr.fp_jacobian_erf_density AS fp_jacobian_erf_density, "
    "pr.fp_id_collapse_rate AS fp_id_collapse_rate, "
    "pr.fp_id_collapse_rate_normalized AS fp_id_collapse_rate_normalized, "
    "pr.fp_jacobian_erf_decay_slope AS fp_jacobian_erf_decay_slope, "
    "pr.fp_jacobian_erf_first_norm AS fp_jacobian_erf_first_norm, "
    "pr.fp_jacobian_erf_last_norm AS fp_jacobian_erf_last_norm, "
    "pr.fp_logit_margin_velocity AS fp_logit_margin_velocity, "
    "pr.fp_logit_margin_initial AS fp_logit_margin_initial, "
    "pr.fp_logit_margin_final AS fp_logit_margin_final, "
    "pr.fp_logit_margin_delta AS fp_logit_margin_delta, "
    "pr.fp_jacobian_erf_variance AS fp_jacobian_erf_variance, "
    "CASE WHEN pr.fp_jacobian_erf_variance IS NOT NULL "
    "THEN log(abs(pr.fp_jacobian_erf_variance) + 0.000000001) ELSE NULL END AS fp_jacobian_erf_variance_log, "
    "CASE WHEN pr.fp_jacobian_spectral_norm IS NOT NULL "
    "THEN log(abs(pr.fp_jacobian_spectral_norm) + 0.000000001) ELSE NULL END AS fp_jacobian_spectral_norm_log, "
    "pr.fp_icld_velocity AS fp_icld_velocity, "
    "pr.fp_icld_early_loss AS fp_icld_early_loss, "
    "pr.fp_icld_late_loss AS fp_icld_late_loss, "
    "pr.fp_icld_delta_loss AS fp_icld_delta_loss, "
    "pr.loss_ratio AS loss_ratio, "
    "pr.discovery_loss AS discovery_loss, "
    "pr.discovery_loss_ratio AS _pr_discovery_loss_ratio, "
    "pr.validation_loss AS validation_loss, "
    "pr.validation_loss_ratio AS _pr_validation_loss_ratio, "
    "pr.wikitext_perplexity AS _pr_wikitext_perplexity, "
    "pr.wikitext_score AS _pr_wikitext_score, "
    "pr.tinystories_perplexity AS _pr_tinystories_perplexity, "
    "pr.tinystories_score AS _pr_tinystories_score, "
    "pr.hellaswag_acc AS _pr_hellaswag_acc, "
    "pr.hellaswag_metric_version AS _pr_hellaswag_metric_version, "
    "pr.hellaswag_tokenizer_mode AS _pr_hellaswag_tokenizer_mode, "
    "pr.hellaswag_tiktoken_encoding AS _pr_hellaswag_tiktoken_encoding, "
    "pr.blimp_overall_accuracy AS _pr_blimp_overall_accuracy, "
    "pr.blimp_n_subtasks AS _pr_blimp_n_subtasks, "
    "pr.blimp_status AS _pr_blimp_status, "
)

_LEADERBOARD_SUFFIX_SELECT = (
    "pr.language_control_metric_version AS language_control_metric_version, "
    "pr.language_control_s05_sentence_assoc_score AS language_control_s05_sentence_assoc_score, "
    "pr.language_control_s05_binding_order_acc AS language_control_s05_binding_order_acc, "
    "pr.language_control_s05_binding_score AS language_control_s05_binding_score, "
    "pr.language_control_s10_sentence_assoc_score AS language_control_s10_sentence_assoc_score, "
    "pr.language_control_s10_binding_order_acc AS language_control_s10_binding_order_acc, "
    "pr.language_control_s10_binding_score AS language_control_s10_binding_score, "
    "pr.language_control_investigation_sentence_assoc_score AS language_control_investigation_sentence_assoc_score, "
    "pr.language_control_investigation_binding_order_acc AS language_control_investigation_binding_order_acc, "
    "pr.language_control_investigation_binding_score AS language_control_investigation_binding_score, "
    "pr.screening_wikitext_metric_version AS _pr_screening_wikitext_metric_version, "
    "pr.tokenizer_mode AS _pr_tokenizer_mode, "
    "pr.corpus_path AS _pr_corpus_path, "
    "pr.evaluation_protocol_version AS _pr_evaluation_protocol_version, "
    "pr.generalization_gap AS generalization_gap, "
    "pr.novelty_score AS novelty_score, "
    "pr.final_loss AS final_loss, "
    "pr.throughput_tok_s AS throughput_tok_s, "
    "pr.peak_memory_mb AS peak_memory_mb, "
    "pr.loss_improvement_rate AS loss_improvement_rate, "
    "pr.forward_time_ms AS forward_time_ms, "
    "pr.flops_forward AS flops_forward, "
    "pr.flops_per_param AS flops_per_param, "
    "pr.sparsity_ratio AS sparsity_ratio, "
    "pr.baseline_loss_ratio AS baseline_loss_ratio, "
    "pr.routing_utilization_entropy AS routing_utilization_entropy, "
    "pr.routing_drop_rate AS routing_drop_rate, "
    "pr.routing_confidence_std AS routing_confidence_std, "
    "pr.routing_tokens_total AS routing_tokens_total, "
    "pr.routing_tokens_processed AS routing_tokens_processed, "
    "pr.routing_capacity_overflow_count AS routing_capacity_overflow_count, "
    "pr.depth_savings_ratio AS depth_savings_ratio, "
    "pr.effective_depth_ratio AS effective_depth_ratio, "
    "pr.recursion_savings_ratio AS recursion_savings_ratio, "
    "pr.recursion_depth_ratio AS recursion_depth_ratio, "
    "pr.activation_sparsity_score AS activation_sparsity_score, "
    "pr.routing_expert_count AS routing_expert_count, "
    "pr.routing_confidence_mean AS routing_confidence_mean, "
    "pr.max_viable_seq_len AS max_viable_seq_len, "
    "pr.robustness_long_ctx_scaling_score AS robustness_long_ctx_scaling_score, "
    "pr.robustness_long_ctx_assoc_score AS robustness_long_ctx_assoc_score, "
    "pr.robustness_long_ctx_multi_hop_score AS robustness_long_ctx_multi_hop_score, "
    "pr.robustness_long_ctx_passkey_score AS robustness_long_ctx_passkey_score, "
    "pr.external_benchmarks_json AS _external_benchmarks_json, "
    "pr.efficiency_multiple AS _pr_efficiency_multiple "
    "FROM leaderboard l "
    "LEFT JOIN program_results_compat pr ON pr.result_id = l.result_id "
    "WHERE 1=1"
)

_FINGERPRINT_METRIC_TO_SCORE_KWARG = {
    "wikitext_perplexity": (
        "ppl_screening",
        "ppl_investigation",
        "ppl_validation",
    ),
    "blimp_overall_accuracy": ("blimp_accuracy",),
    "hellaswag_acc": (
        "hellaswag_acc_screening",
        "hellaswag_acc_investigation",
        "hellaswag_acc_validation",
    ),
    "tinystories_score": ("tinystories_score",),
    "cross_task_score": ("cross_task_score",),
    "diagnostic_score": ("diagnostic_score",),
    "fp_hierarchy_fitness": ("hierarchy_fitness",),
    "ar_legacy_auc": ("ar_legacy_auc",),
    "ar_gate_score": ("ar_gate_score",),
    "ar_validation_rank_score": ("ar_validation_rank_score",),
    "induction_screening_auc": ("induction_screening_auc",),
    "binding_screening_auc": ("binding_screening_auc",),
    "induction_intermediate_auc": ("induction_intermediate_inv_auc",),
    "binding_intermediate_auc": ("binding_intermediate_inv_auc",),
}


class _LeaderboardMixin:
    """Leaderboard operations for the Lab Notebook."""

    __slots__ = ()

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        try:
            num = float(value)
        except (TypeError, ValueError):
            return None
        return num if num == num else None

    @staticmethod
    def _benchmark_payload(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        raw_payload = entry.pop("_external_benchmarks_json", None)
        if not raw_payload or not isinstance(raw_payload, str):
            return None
        try:
            payload = json.loads(raw_payload)
        except (json.JSONDecodeError, TypeError):
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _screening_wikitext_payload(
        payload: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        screening = payload.get("screening_wikitext") if payload else None
        return screening if isinstance(screening, dict) else None

    @staticmethod
    def _screening_wikitext_metrics(
        screening: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        metrics = screening.get("metrics") if screening else None
        return metrics if isinstance(metrics, dict) else {}

    def _apply_wikitext_ppl_aliases(
        self,
        entry: Dict[str, Any],
        screening_metrics: Dict[str, Any],
    ) -> None:
        wikitext_ppl = self._coerce_float(
            entry.get("wikitext_ppl")
            or entry.get("wikitext_perplexity")
            or screening_metrics.get("wikitext_perplexity")
        )
        if wikitext_ppl is not None:
            entry["wikitext_ppl"] = wikitext_ppl
            entry.setdefault("peak_ppl", wikitext_ppl)

    def _apply_wikitext_improvement_alias(
        self,
        entry: Dict[str, Any],
        screening_metrics: Dict[str, Any],
    ) -> None:
        improvement_ratio = self._coerce_float(
            entry.get("wikitext_ppl_improvement_ratio")
            or entry.get("wikitext_improvement_ratio")
            or entry.get("wikitext_ppl_improvement")
            or screening_metrics.get("wikitext_ppl_improvement")
        )
        if improvement_ratio is not None:
            entry["improvement_ratio"] = improvement_ratio

    def _apply_screening_wikitext_aliases(
        self,
        entry: Dict[str, Any],
        screening: Optional[Dict[str, Any]],
    ) -> None:
        if not screening:
            return
        entry.setdefault("screening_wikitext_status", screening.get("status"))
        entry.setdefault(
            "screening_wikitext_metric_version", screening.get("metric_version")
        )
        entry.setdefault("screening_wikitext_variant", screening.get("variant"))
        elapsed_ms = self._coerce_float(screening.get("elapsed_ms"))
        if elapsed_ms is not None:
            entry.setdefault("screening_wikitext_elapsed_ms", elapsed_ms)

    def _wikitext_trajectory_steps(
        self,
        payload: Optional[Dict[str, Any]],
    ) -> List[tuple[int, float]]:
        trajectory_payload = payload.get("wikitext_trajectory") if payload else None
        checkpoints = (
            trajectory_payload.get("checkpoints")
            if isinstance(trajectory_payload, dict)
            else None
        )
        if not isinstance(checkpoints, dict):
            return []
        ordered_steps: List[tuple[int, float]] = []
        for step, values in checkpoints.items():
            step_pair = self._wikitext_checkpoint_pair(step, values)
            if step_pair is not None:
                ordered_steps.append(step_pair)
        return sorted(ordered_steps, key=lambda item: item[0])

    def _wikitext_checkpoint_pair(
        self,
        step: Any,
        values: Any,
    ) -> Optional[tuple[int, float]]:
        try:
            step_num = int(step)
        except (TypeError, ValueError):
            return None
        if not isinstance(values, dict):
            return None
        ppl = self._coerce_float(values.get("ppl"))
        if ppl is None:
            return None
        return step_num, ppl

    @staticmethod
    def _apply_wikitext_trajectory(
        entry: Dict[str, Any], ordered_steps: List[tuple[int, float]]
    ) -> None:
        if not ordered_steps:
            return
        trajectory = [ppl for _, ppl in ordered_steps]
        entry["wikitext_ppl_trajectory"] = trajectory
        entry["peak_ppl"] = min(trajectory)
        entry["eval_budget_steps"] = ordered_steps[-1][0]
        if len(trajectory) >= 2 and trajectory[1] > 0:
            entry.setdefault("improvement_ratio", trajectory[0] / trajectory[1])

    def _normalize_benchmark_fields(self, entry: Dict[str, Any]) -> None:
        """Backfill stable benchmark aliases from persisted artifact payloads."""
        payload = self._benchmark_payload(entry)
        screening = self._screening_wikitext_payload(payload)
        screening_metrics = self._screening_wikitext_metrics(screening)
        self._apply_wikitext_ppl_aliases(entry, screening_metrics)
        self._apply_wikitext_improvement_alias(entry, screening_metrics)
        self._apply_screening_wikitext_aliases(entry, screening)
        self._apply_wikitext_trajectory(
            entry,
            self._wikitext_trajectory_steps(payload),
        )

    def _highest_tier(self, rows: List[Dict[str, Any]]) -> Optional[str]:
        tiers = [str(r.get("tier") or "").lower() for r in rows if r.get("tier")]
        if not tiers:
            return None
        return max(tiers, key=lambda t: self._TIER_ORDER.get(t, -1))

    def _leaderboard_update_items(
        self, kwargs: Dict[str, Any]
    ) -> List[tuple[str, Any]]:
        allowed = self._get_leaderboard_columns() - _LEADERBOARD_MANAGED_COLUMNS
        update_items: List[tuple[str, Any]] = []
        for col, val in kwargs.items():
            if col not in allowed or val is None:
                continue
            update_items.append((col, int(val) if isinstance(val, bool) else val))
        return update_items

    def _provenance_complete(self, pr_row: Any) -> bool:
        if not pr_row:
            return False
        raw = (
            pr_row["data_provenance_json"]
            if "data_provenance_json" in pr_row.keys()
            else None
        )
        if not raw or not isinstance(raw, str):
            return False
        # data_provenance_json is externalized into a zstd artifact for
        # large payloads, leaving an artifact-pointer stub on the row.
        # _json_loads_maybe_artifact resolves the pointer transparently;
        # without it, payload.get("provenance_complete") always returns
        # False on externalized rows and blocks legitimate promotions.
        try:
            payload = self._json_loads_maybe_artifact(raw)
        except (
            json.JSONDecodeError,
            TypeError,
            ValueError,
            OSError,
            FileNotFoundError,
        ):
            return False
        if not isinstance(payload, dict):
            return False
        return bool(payload.get("provenance_complete"))

    def _resolve_allowed_tier(
        self,
        *,
        requested_tier: str,
        existing_tier: str,
        pr_row: Any,
        is_reference: bool,
    ) -> str:
        requested_rank = TIER_RANK.get(requested_tier, 0)
        existing_rank = TIER_RANK.get(existing_tier, 0)
        if requested_rank <= 0 or requested_rank <= existing_rank:
            return requested_tier if requested_rank >= existing_rank else existing_tier
        if is_reference:
            return requested_tier
        trust_entry = dict(pr_row) if pr_row else {}
        if is_promotable_entry(trust_entry) and self._provenance_complete(pr_row):
            return requested_tier
        LOGGER.warning(
            "Blocked promotion above screening for %s: tier=%s trust=%s comparability=%s provenance_complete=%s",
            trust_entry.get("result_id") or "<missing>",
            requested_tier,
            trust_entry.get("trust_label"),
            trust_entry.get("comparability_label"),
            self._provenance_complete(pr_row),
        )
        return existing_tier or "screening"

    def _lookup_upsert_program_row(
        self,
        result_id: str,
        architecture_desc: str,
        is_reference: bool,
    ) -> Any:
        pr_row = self.conn.execute(
            _PROGRAM_RESULT_UPSERT_SELECT
            + "WHERE result_id = ? OR graph_fingerprint = ? "
            "ORDER BY CASE WHEN result_id = ? THEN 0 ELSE 1 END, timestamp DESC "
            "LIMIT 1",
            (result_id, result_id, result_id),
        ).fetchone()
        if (
            pr_row is not None
            or is_reference
            or not str(architecture_desc or "").strip()
        ):
            return pr_row
        return self.conn.execute(
            _PROGRAM_RESULT_UPSERT_SELECT
            + "WHERE graph_fingerprint = ? ORDER BY timestamp DESC LIMIT 1",
            (str(architecture_desc).strip(),),
        ).fetchone()

    @staticmethod
    def _program_row_fingerprint(pr_row: Any) -> str:
        if not pr_row or not pr_row["graph_fingerprint"]:
            return ""
        return str(pr_row["graph_fingerprint"]).strip()

    def _find_leaderboard_by_fingerprint(self, fp: str) -> Any:
        return self.conn.execute(
            "SELECT * FROM leaderboard WHERE graph_fingerprint = ?",
            (fp,),
        ).fetchone()

    @staticmethod
    def _raise_duplicate_fingerprint(
        *,
        fp: str,
        existing: Any,
        resolved_result_id: str,
    ) -> None:
        raise DuplicateLeaderboardFingerprintError(
            graph_fingerprint=fp,
            existing_entry_id=str(existing["entry_id"]),
            existing_result_id=str(existing["result_id"]),
            attempted_result_id=str(resolved_result_id),
        )

    def _existing_leaderboard_entry_by_id(self, resolved_result_id: str) -> Any:
        return self.conn.execute(
            "SELECT * FROM leaderboard WHERE result_id = ?",
            (resolved_result_id,),
        ).fetchone()

    @staticmethod
    def _matched_reference_fingerprint(
        *,
        existing: Any,
        is_reference: bool,
        resolved_result_id: str,
    ) -> bool:
        return bool(
            existing is not None
            and is_reference
            and str(existing["result_id"] or "") != str(resolved_result_id)
        )

    def _existing_leaderboard_entry_by_fingerprint(
        self,
        *,
        fp: str,
        resolved_result_id: str,
        is_reference: bool,
    ) -> tuple[Any, bool]:
        existing = self._find_leaderboard_by_fingerprint(fp)
        if (
            existing is not None
            and not is_reference
            and str(existing["result_id"]) != str(resolved_result_id)
        ):
            self._raise_duplicate_fingerprint(
                fp=fp,
                existing=existing,
                resolved_result_id=resolved_result_id,
            )
        return existing, self._matched_reference_fingerprint(
            existing=existing,
            is_reference=is_reference,
            resolved_result_id=resolved_result_id,
        )

    def _fallback_existing_reference_entry(
        self,
        *,
        existing: Any,
        pr_row: Any,
        is_reference: bool,
    ) -> tuple[Any, bool]:
        if (
            existing is not None
            or not is_reference
            or not self._program_row_fingerprint(pr_row)
        ):
            return existing, False
        reference_entry = self._find_reference_entry_by_fingerprint(pr_row)
        return reference_entry, reference_entry is not None

    def _find_existing_leaderboard_entry(
        self,
        *,
        pr_row: Any,
        resolved_result_id: str,
        is_reference: bool,
        allow_fingerprint_duplicate: bool,
    ) -> tuple[Any, bool]:
        fp_for_lookup = self._program_row_fingerprint(pr_row)
        existing = None
        matched_reference_fp = False
        if fp_for_lookup and (is_reference or not allow_fingerprint_duplicate):
            existing, matched_reference_fp = (
                self._existing_leaderboard_entry_by_fingerprint(
                    fp=fp_for_lookup,
                    resolved_result_id=resolved_result_id,
                    is_reference=is_reference,
                )
            )
        if existing is None:
            existing = self._existing_leaderboard_entry_by_id(resolved_result_id)
        fallback, fallback_matched = self._fallback_existing_reference_entry(
            existing=existing,
            pr_row=pr_row,
            is_reference=is_reference,
        )
        if fallback_matched:
            return fallback, True
        return existing, matched_reference_fp

    def _find_reference_entry_by_fingerprint(self, pr_row: Any) -> Any:
        fp = str(pr_row["graph_fingerprint"]).strip()
        if not fp:
            return None
        return self.conn.execute(
            "SELECT * FROM leaderboard WHERE graph_fingerprint = ? "
            "ORDER BY COALESCE(is_reference, 0) DESC, timestamp DESC "
            "LIMIT 1",
            (fp,),
        ).fetchone()

    def _block_duplicate_leaderboard_insert(
        self,
        *,
        existing: Any,
        pr_row: Any,
        resolved_result_id: str,
        is_reference: bool,
        allow_fingerprint_duplicate: bool,
    ) -> None:
        if existing or is_reference or allow_fingerprint_duplicate or not pr_row:
            return
        fp = (
            str(pr_row["graph_fingerprint"]).strip()
            if pr_row["graph_fingerprint"]
            else ""
        )
        if not fp:
            return
        fp_dup = self.conn.execute(
            "SELECT l.entry_id, l.result_id FROM leaderboard l "
            "JOIN program_results_compat pr ON l.result_id = pr.result_id "
            "WHERE pr.graph_fingerprint = ? AND l.result_id != ? "
            "LIMIT 1",
            (fp, resolved_result_id),
        ).fetchone()
        if fp_dup is None:
            return
        LOGGER.warning(
            "BLOCKED leaderboard dup insert: fp=%s existing_entry=%s "
            "(result_id=%s) attempted_result_id=%s",
            fp,
            fp_dup["entry_id"],
            fp_dup["result_id"],
            resolved_result_id,
        )
        raise DuplicateLeaderboardFingerprintError(
            graph_fingerprint=fp,
            existing_entry_id=str(fp_dup["entry_id"]),
            existing_result_id=str(fp_dup["result_id"]),
            attempted_result_id=str(resolved_result_id),
        )

    def _merge_upsert_fields(
        self,
        *,
        d: Dict[str, Any],
        kwargs: Dict[str, Any],
        tags: Optional[str],
        notes: Optional[str],
        existing: Any,
        pr_row: Any,
        preserve_reference_parent: bool,
    ) -> Optional[str]:
        for col, val in self._leaderboard_update_items(kwargs):
            d[col] = val
        if tags is not None:
            tags = (
                self._merge_reference_tags(existing, tags)
                if preserve_reference_parent
                else tags
            )
            d["tags"] = tags
        if notes is not None:
            d["notes"] = notes
        if pr_row:
            for key in (
                "result_cohort",
                "trust_label",
                "comparability_label",
                "evaluation_protocol_version",
            ):
                if pr_row[key] is not None and not d.get(key):
                    d[key] = pr_row[key]
                    kwargs.setdefault(key, pr_row[key])
        return tags

    @staticmethod
    def _merge_reference_tags(existing: Any, tags: str) -> str:
        merged_tags = []
        for item in f"{existing['tags'] or ''},{tags}".split(","):
            tag = item.strip()
            if tag and tag not in merged_tags:
                merged_tags.append(tag)
        return ",".join(merged_tags)

    def _apply_upsert_tier_and_identity(
        self,
        *,
        d: Dict[str, Any],
        kwargs: Dict[str, Any],
        requested_tier: str,
        resolved_result_id: str,
        model_source: str,
        architecture_desc: str,
        is_reference: bool,
        reference_name: Optional[str],
        pr_row: Any,
        preserve_reference_parent: bool,
    ) -> str:
        existing_tier = str(d.get("tier") or "screening")
        allowed_tier = self._resolve_allowed_tier(
            requested_tier=requested_tier,
            existing_tier=existing_tier,
            pr_row=pr_row,
            is_reference=bool(is_reference),
        )
        if TIER_RANK.get(allowed_tier, 0) < TIER_RANK.get(existing_tier, 0):
            LOGGER.warning(
                "Blocked tier downgrade for %s: %s -> %s",
                resolved_result_id,
                existing_tier,
                allowed_tier,
            )
            allowed_tier = existing_tier
        d["tier"] = allowed_tier
        if not preserve_reference_parent:
            d["model_source"] = model_source
        scoring_version = get_scoring_version()
        d["scoring_config_hash"] = scoring_version
        kwargs.setdefault("scoring_config_hash", scoring_version)
        if architecture_desc and not preserve_reference_parent:
            d["architecture_desc"] = architecture_desc
        d["is_reference"] = int(is_reference)
        if reference_name:
            d["reference_name"] = reference_name
        return allowed_tier

    @staticmethod
    def _apply_robustness_grade(d: Dict[str, Any], kwargs: Dict[str, Any]) -> None:
        if kwargs.get("robustness_grade"):
            return
        inv_rob = d.get("investigation_robustness")
        if inv_rob is None:
            return
        try:
            inv_rob_f = float(inv_rob)
        except (TypeError, ValueError):
            return
        grade = "A" if inv_rob_f >= 2 / 3 else "B" if inv_rob_f >= 1 / 3 else "C"
        d["robustness_grade"] = grade
        kwargs["robustness_grade"] = grade

    def _apply_fingerprint_aggregates(
        self, fp: Any, kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not fp:
            return {}
        agg = self.get_fingerprint_aggregates(fp)
        if agg.get("n_runs", 0) > 0:
            kwargs.setdefault("replication_n", agg["n_runs"])
            kwargs.setdefault("replication_loss_mean", agg["loss_mean"])
            kwargs.setdefault("replication_loss_std", agg["loss_std"])
            kwargs.setdefault("replication_best_vs_mean_gap", agg["best_vs_mean_gap"])
        metric_agg = self.get_fingerprint_metric_aggregates(fp) or {}
        tier_cv = metric_agg.get("_tier_cv") or {}
        kwargs["n_runs"] = int(metric_agg.get("_n_runs_max") or 0)
        kwargs["cv_loss"] = tier_cv.get("loss")
        kwargs["cv_understanding"] = tier_cv.get("und")
        kwargs["cv_capability"] = tier_cv.get("cap")
        return metric_agg

    def _build_upsert_score_kwargs(
        self,
        *,
        resolved_result_id: str,
        d: Dict[str, Any],
        kwargs: Dict[str, Any],
        is_reference: bool,
        fp: Any,
        metric_agg: Dict[str, Any],
    ) -> Dict[str, Any]:
        score_kwargs = build_score_kwargs(
            self.conn,
            self,
            resolved_result_id,
            d,
            bool(is_reference),
        )
        if fp:
            score_kwargs["replication_n"] = kwargs.get("replication_n")
            score_kwargs["replication_loss_mean"] = kwargs.get("replication_loss_mean")
            score_kwargs["replication_loss_std"] = kwargs.get("replication_loss_std")
            score_kwargs["replication_best_vs_mean_gap"] = kwargs.get(
                "replication_best_vs_mean_gap"
            )
        self._apply_metric_aggregate_score_overrides(score_kwargs, kwargs, metric_agg)
        return score_kwargs

    @staticmethod
    def _apply_metric_aggregate_score_overrides(
        score_kwargs: Dict[str, Any],
        kwargs: Dict[str, Any],
        metric_agg: Dict[str, Any],
    ) -> None:
        if not metric_agg:
            return
        for col, kwarg_names in _FINGERPRINT_METRIC_TO_SCORE_KWARG.items():
            stat = metric_agg.get(col) or {}
            if int(stat.get("n") or 0) < 2 or stat.get("mean") is None:
                continue
            for name in kwarg_names:
                if score_kwargs.get(name) is not None:
                    score_kwargs[name] = stat["mean"]
        score_kwargs["cv_loss"] = kwargs.get("cv_loss")
        score_kwargs["cv_understanding"] = kwargs.get("cv_understanding")
        score_kwargs["cv_capability"] = kwargs.get("cv_capability")
        score_kwargs["n_runs"] = kwargs.get("n_runs")

    @staticmethod
    def _composite_value_and_breakdown(
        composite_dec: Any,
    ) -> tuple[float, Dict[str, Any]]:
        if not isinstance(composite_dec, dict):
            return float(composite_dec), {}
        return (
            float(composite_dec.get("composite_score") or 0.0),
            composite_dec.get("breakdown") or {},
        )

    @staticmethod
    def _set_score_stability_penalty(
        kwargs: Dict[str, Any],
        breakdown: Dict[str, Any],
    ) -> None:
        if not breakdown.get("_cv_penalty_applied"):
            kwargs["score_stability_penalty"] = 1.0
            return
        kwargs["score_stability_penalty"] = (
            float(breakdown.get("_cv_penalty_loss") or 1.0)
            * float(breakdown.get("_cv_penalty_und") or 1.0)
            * float(breakdown.get("_cv_penalty_cap") or 1.0)
        ) ** (1.0 / 3.0)

    @staticmethod
    def _preserve_reference_composite_floor(
        composite: float,
        existing: Any,
        is_reference: bool,
    ) -> float:
        if bool(is_reference) and existing and existing["composite_score"] is not None:
            return max(composite, float(existing["composite_score"] or 0.0))
        return composite

    @staticmethod
    def _compute_upsert_composite(
        score_kwargs: Dict[str, Any],
        kwargs: Dict[str, Any],
        existing: Any,
        is_reference: bool,
    ) -> float:
        composite_dec = compute_composite(decompose=True, **score_kwargs)
        composite, breakdown = _LeaderboardMixin._composite_value_and_breakdown(
            composite_dec
        )
        _LeaderboardMixin._set_score_stability_penalty(kwargs, breakdown)
        return _LeaderboardMixin._preserve_reference_composite_floor(
            composite,
            existing,
            is_reference,
        )

    def _apply_efficiency_multiple(self, pr_row: Any, kwargs: Dict[str, Any]) -> None:
        eff_mult = kwargs.get("efficiency_multiple")
        if eff_mult is None and pr_row:
            eff_result = self.compute_efficiency_multiple(
                loss_ratio=pr_row["loss_ratio"],
                param_count=pr_row["param_count"],
                flops_forward=pr_row["flops_forward"],
                throughput_tok_s=pr_row["throughput_tok_s"],
                peak_memory_mb=pr_row["peak_memory_mb"],
                forward_time_ms=pr_row["forward_time_ms"],
                is_moe=self._upsert_row_is_moe(pr_row),
            )
            if eff_result is not None:
                eff_mult = eff_result["geomean"]
        if eff_mult is not None:
            kwargs["efficiency_multiple"] = eff_mult

    def _upsert_row_is_moe(self, pr_row: Any) -> bool:
        if not pr_row or not pr_row["graph_json"]:
            return False
        from ...synthesis.op_roles import MOE_OPS

        try:
            graph_json = resolve_graph_json_value(
                self.conn,
                self.db_path,
                pr_row["graph_json"],
            )
            if isinstance(graph_json, str):
                graph_json = json.loads(graph_json)
            return any(
                node.get("op_name") in MOE_OPS
                for node in (graph_json.get("nodes") or {}).values()
            )
        except (
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
            AttributeError,
        ):
            return False

    def _write_leaderboard_upsert(
        self,
        *,
        existing: Any,
        resolved_result_id: str,
        model_source: str,
        architecture_desc: str,
        tier: str,
        composite: float,
        is_reference: bool,
        reference_name: Optional[str],
        tags: Optional[str],
        notes: Optional[str],
        update_items: List[tuple[str, Any]],
        pr_row: Any,
        d: Dict[str, Any],
        preserve_reference_parent: bool,
    ) -> str:
        if existing:
            return self._update_leaderboard_entry(
                existing=existing,
                resolved_result_id=resolved_result_id,
                model_source=model_source,
                architecture_desc=architecture_desc,
                tier=tier,
                composite=composite,
                is_reference=is_reference,
                reference_name=reference_name,
                tags=tags,
                notes=notes,
                update_items=update_items,
                d=d,
                preserve_reference_parent=preserve_reference_parent,
            )
        return self._insert_leaderboard_entry(
            resolved_result_id=resolved_result_id,
            model_source=model_source,
            architecture_desc=architecture_desc,
            tier=tier,
            composite=composite,
            is_reference=is_reference,
            reference_name=reference_name,
            tags=tags,
            notes=notes,
            update_items=update_items,
            pr_row=pr_row,
        )

    def _update_leaderboard_entry(
        self,
        *,
        existing: Any,
        resolved_result_id: str,
        model_source: str,
        architecture_desc: str,
        tier: str,
        composite: float,
        is_reference: bool,
        reference_name: Optional[str],
        tags: Optional[str],
        notes: Optional[str],
        update_items: List[tuple[str, Any]],
        d: Dict[str, Any],
        preserve_reference_parent: bool,
    ) -> str:
        entry_id = existing["entry_id"]
        sets = [
            "timestamp = ?",
            "model_source = ?",
            "tier = ?",
            "composite_score = ?",
            "is_reference = ?",
        ]
        params: List[Any] = [
            time.time(),
            d.get("model_source") or model_source,
            tier,
            composite,
            int(is_reference),
        ]
        if not preserve_reference_parent and str(existing["result_id"] or "") != str(
            resolved_result_id
        ):
            sets.append("result_id = ?")
            params.append(resolved_result_id)
        for col, val in self._optional_upsert_update_fields(
            architecture_desc=architecture_desc,
            tags=tags,
            notes=notes,
            reference_name=reference_name,
            preserve_reference_parent=preserve_reference_parent,
        ):
            sets.append(f"{col} = ?")
            params.append(val)
        for col, val in update_items:
            sets.append(f"{col} = ?")
            params.append(val)
        params.append(entry_id)
        self.conn.execute(
            f"UPDATE leaderboard SET {', '.join(sets)} WHERE entry_id = ?",  # nosec B608
            params,
        )
        return entry_id

    @staticmethod
    def _optional_upsert_update_fields(
        *,
        architecture_desc: str,
        tags: Optional[str],
        notes: Optional[str],
        reference_name: Optional[str],
        preserve_reference_parent: bool,
    ) -> List[tuple[str, Any]]:
        fields: List[tuple[str, Any]] = []
        if architecture_desc and not preserve_reference_parent:
            fields.append(("architecture_desc", architecture_desc))
        if tags is not None:
            fields.append(("tags", tags))
        if notes is not None:
            fields.append(("notes", notes))
        if reference_name is not None:
            fields.append(("reference_name", reference_name))
        return fields

    def _insert_leaderboard_entry(
        self,
        *,
        resolved_result_id: str,
        model_source: str,
        architecture_desc: str,
        tier: str,
        composite: float,
        is_reference: bool,
        reference_name: Optional[str],
        tags: Optional[str],
        notes: Optional[str],
        update_items: List[tuple[str, Any]],
        pr_row: Any,
    ) -> str:
        entry_id = str(uuid.uuid4())[:12]
        cols = [
            "entry_id",
            "result_id",
            "timestamp",
            "model_source",
            "architecture_desc",
            "tier",
            "composite_score",
            "is_reference",
            "reference_name",
            "tags",
            "notes",
        ]
        vals: List[Any] = [
            entry_id,
            resolved_result_id,
            time.time(),
            model_source,
            architecture_desc,
            tier,
            composite,
            int(is_reference),
            reference_name,
            tags,
            notes,
        ]
        for col, val in update_items:
            cols.append(col)
            vals.append(val)
        self._append_insert_fingerprint(cols, vals, pr_row)
        placeholders = ", ".join(["?"] * len(cols))
        self.conn.execute(
            f"INSERT INTO leaderboard ({', '.join(cols)}) VALUES ({placeholders})",  # nosec B608
            vals,
        )
        return entry_id

    def _append_insert_fingerprint(
        self, cols: List[str], vals: List[Any], pr_row: Any
    ) -> None:
        fp = None
        if pr_row is not None and pr_row["graph_fingerprint"] is not None:
            fp_val = str(pr_row["graph_fingerprint"]).strip()
            fp = fp_val or None
        if (
            fp is not None
            and "graph_fingerprint" in self._get_leaderboard_columns()
            and "graph_fingerprint" not in cols
        ):
            cols.append("graph_fingerprint")
            vals.append(fp)

    @staticmethod
    def _block_orphan_upsert(
        *,
        result_id: str,
        architecture_desc: str,
        pr_row: Any,
        existing: Any,
        is_reference: bool,
    ) -> bool:
        if pr_row is not None or existing is not None or is_reference:
            return False
        LOGGER.error(
            "Blocked orphan leaderboard insert: result_id=%s architecture_desc=%s",
            str(result_id)[:12],
            str(architecture_desc or "")[:40],
        )
        return True

    @staticmethod
    def _preserve_reference_parent(
        *,
        matched_reference_fp: bool,
        is_reference: bool,
        model_source: str,
    ) -> bool:
        return (
            matched_reference_fp
            and bool(is_reference)
            and str(model_source or "") == "reference_calibration"
        )

    def _apply_upsert_score_fields(
        self,
        *,
        pr_row: Any,
        resolved_result_id: str,
        d: Dict[str, Any],
        kwargs: Dict[str, Any],
        existing: Any,
        is_reference: bool,
    ) -> tuple[float, List[tuple[str, Any]]]:
        self._apply_robustness_grade(d, kwargs)
        fp = pr_row["graph_fingerprint"] if pr_row else None
        metric_agg = self._apply_fingerprint_aggregates(fp, kwargs)
        score_kwargs = self._build_upsert_score_kwargs(
            resolved_result_id=resolved_result_id,
            d=d,
            kwargs=kwargs,
            is_reference=bool(is_reference),
            fp=fp,
            metric_agg=metric_agg,
        )
        composite = self._compute_upsert_composite(
            score_kwargs,
            kwargs,
            existing,
            bool(is_reference),
        )
        update_items = self._leaderboard_update_items(kwargs)
        self._apply_efficiency_multiple(pr_row, kwargs)
        return composite, update_items

    def _build_leaderboard_upsert_write_args(
        self,
        *,
        result_id: str,
        model_source: str,
        architecture_desc: str,
        tier: str,
        tags: Optional[str],
        notes: Optional[str],
        is_reference: bool,
        reference_name: Optional[str],
        allow_fingerprint_duplicate: bool,
        kwargs: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        self.flush_writes()
        pr_row = self._lookup_upsert_program_row(
            result_id, architecture_desc, is_reference
        )
        resolved_result_id = pr_row["result_id"] if pr_row else result_id
        existing, matched_reference_fp = self._find_existing_leaderboard_entry(
            pr_row=pr_row,
            resolved_result_id=resolved_result_id,
            is_reference=bool(is_reference),
            allow_fingerprint_duplicate=allow_fingerprint_duplicate,
        )
        if self._block_orphan_upsert(
            result_id=result_id,
            architecture_desc=architecture_desc,
            pr_row=pr_row,
            existing=existing,
            is_reference=bool(is_reference),
        ):
            return None
        self._block_duplicate_leaderboard_insert(
            existing=existing,
            pr_row=pr_row,
            resolved_result_id=resolved_result_id,
            is_reference=bool(is_reference),
            allow_fingerprint_duplicate=allow_fingerprint_duplicate,
        )
        kwargs = sanitize_for_db(kwargs)
        preserve_reference_parent = self._preserve_reference_parent(
            matched_reference_fp=matched_reference_fp,
            is_reference=bool(is_reference),
            model_source=model_source,
        )
        d = dict(existing) if existing else {}
        tags = self._merge_upsert_fields(
            d=d,
            kwargs=kwargs,
            tags=tags,
            notes=notes,
            existing=existing,
            pr_row=pr_row,
            preserve_reference_parent=preserve_reference_parent,
        )
        tier = self._apply_upsert_tier_and_identity(
            d=d,
            kwargs=kwargs,
            requested_tier=tier,
            resolved_result_id=resolved_result_id,
            model_source=model_source,
            architecture_desc=architecture_desc,
            is_reference=bool(is_reference),
            reference_name=reference_name,
            pr_row=pr_row,
            preserve_reference_parent=preserve_reference_parent,
        )
        composite, update_items = self._apply_upsert_score_fields(
            pr_row=pr_row,
            resolved_result_id=resolved_result_id,
            d=d,
            kwargs=kwargs,
            existing=existing,
            is_reference=bool(is_reference),
        )
        return {
            "existing": existing,
            "resolved_result_id": resolved_result_id,
            "model_source": model_source,
            "architecture_desc": architecture_desc,
            "tier": tier,
            "composite": composite,
            "is_reference": bool(is_reference),
            "reference_name": reference_name,
            "tags": tags,
            "notes": notes,
            "update_items": update_items,
            "pr_row": pr_row,
            "d": d,
            "preserve_reference_parent": preserve_reference_parent,
        }

    def _optional_program_result_select(
        self,
        program_result_columns: set[str],
        column: str,
        alias: str | None = None,
    ) -> str:
        target = alias or column
        if column in program_result_columns:
            return f"pr.{column} AS {target}, "
        return f"NULL AS {target}, "

    def _program_result_dashboard_selects(self) -> str:
        program_result_columns = self._get_program_results_columns()

        def optional_select(column, alias=None):
            return self._optional_program_result_select(
                program_result_columns,
                column,
                alias,
            )

        dashboard_alias_select = "".join(
            optional_select(column, f"_pr_{column}")
            for column in _PROGRAM_RESULT_DASHBOARD_ALIAS_FIELDS
        )
        ar_gate_select = "".join(
            optional_select(column)
            for column in (
                "ar_gate_metric_version",
                "ar_gate_in_dist_pair_acc",
                "ar_gate_in_dist_class_acc",
                "ar_gate_held_pair_acc",
                "ar_gate_held_class_acc",
                "ar_gate_score",
                "ar_gate_status",
                "ar_gate_elapsed_ms",
                "ar_gate_train_steps_done",
            )
        )
        return ar_gate_select + dashboard_alias_select

    @staticmethod
    def _leaderboard_tier_clause(
        *,
        normalized_tier: str,
        include_references: bool,
        tier_match_mode: str,
    ) -> tuple[str, List[Any]]:
        current_status_clause = {
            "screening": "COALESCE(l.tier, 'screening') = 'screening'",
            "failed": (
                "l.tier IN ("
                "'screened_out', "
                "'investigation_failed', "
                "'investigation_fingerprint_incomplete', "
                "'validation_failed', "
                "'failed', "
                "'rejected'"
                ")"
            ),
            "screened_out": "l.tier = 'screened_out'",
            "investigation": "l.tier = 'investigation'",
            "investigation_failed": "l.tier = 'investigation_failed'",
            "investigation_fingerprint_incomplete": (
                "l.tier = 'investigation_fingerprint_incomplete'"
            ),
            "validation": "l.tier = 'validation' AND COALESCE(l.validation_passed, 0) = 1",
            "validation_pending": "l.tier = 'validation' AND COALESCE(l.validation_passed, 0) = 0",
            "validation_failed": "l.tier = 'validation_failed'",
            "breakthrough": "l.tier = 'breakthrough'",
        }
        reached_stage_clause = {
            "investigation": "l.investigation_passed = 1",
            "validation": "l.validation_passed = 1",
        }
        tier_clause = (
            current_status_clause.get(normalized_tier)
            if tier_match_mode == "current"
            else reached_stage_clause.get(normalized_tier)
        )
        if tier_clause and include_references:
            return f" AND ({tier_clause} OR COALESCE(l.is_reference, 0) = 1)", []
        if tier_clause:
            return f" AND {tier_clause} AND COALESCE(l.is_reference, 0) = 0", []
        if include_references:
            return " AND (l.tier = ? OR COALESCE(l.is_reference, 0) = 1)", [
                normalized_tier
            ]
        return " AND l.tier = ? AND COALESCE(l.is_reference, 0) = 0", [normalized_tier]

    def _build_get_leaderboard_query(
        self,
        *,
        tier: Optional[str],
        limit: int,
        sort_by: str,
        include_references: bool,
        trusted_only: bool,
        tier_match_mode: str,
    ) -> tuple[str, List[Any]]:
        dashboard_selects = self._program_result_dashboard_selects()
        query = (
            _LEADERBOARD_BASE_SELECT + dashboard_selects + _LEADERBOARD_SUFFIX_SELECT
        )
        params: List[Any] = []
        if trusted_only:
            query += f" AND {sql_trusted_clause(table_alias='l')}"
        if tier:
            clause, tier_params = self._leaderboard_tier_clause(
                normalized_tier=str(tier).strip().lower(),
                include_references=include_references,
                tier_match_mode=tier_match_mode,
            )
            query += clause
            params.extend(tier_params)
        elif not include_references:
            query += " AND COALESCE(l.is_reference, 0) = 0"
        pr_sort_fields = {"discovery_loss_ratio", "generalization_gap"}
        sort_col = sort_by if sort_by in pr_sort_fields else f"l.{sort_by}"
        query += (
            f" ORDER BY COALESCE(l.is_pinned, 0) DESC, "
            f"COALESCE(l.is_reference, 0) DESC, "
            f"{sort_col} DESC NULLS LAST LIMIT ?"
        )
        params.append(max(limit * 6, 200))
        return query, params

    def _fetch_leaderboard_rows(self, query: str, params: List[Any]) -> List[Any]:
        try:
            return list(self.conn.execute(query, params).fetchall())
        except sqlite3.OperationalError as exc:
            LOGGER.warning(
                "Leaderboard query failed; returning empty results: %s",
                exc,
            )
            return []

    @staticmethod
    def _backfill_discovery_metric(d: Dict[str, Any]) -> None:
        if (
            d.get("discovery_loss_ratio") is None
            and d.get("_pr_discovery_loss_ratio") is not None
        ):
            d["discovery_loss_ratio"] = d.get("_pr_discovery_loss_ratio")

    @staticmethod
    def _should_backfill_validation_metric(d: Dict[str, Any]) -> bool:
        tier = str(d.get("tier") or "").strip().lower()
        return tier in ("validation", "breakthrough") or (
            str(d.get("result_cohort") or "").strip().lower() == "backfill"
        )

    @classmethod
    def _backfill_validation_metric(cls, d: Dict[str, Any]) -> None:
        if (
            cls._should_backfill_validation_metric(d)
            and d.get("validation_loss_ratio") is None
            and d.get("_pr_validation_loss_ratio") is not None
        ):
            d["validation_loss_ratio"] = d.get("_pr_validation_loss_ratio")

    @staticmethod
    def _backfill_phase_metrics(d: Dict[str, Any]) -> None:
        _LeaderboardMixin._backfill_discovery_metric(d)
        _LeaderboardMixin._backfill_validation_metric(d)

    @staticmethod
    def _prefer_pr_value(
        d: Dict[str, Any],
        field: str,
        pr_field: str,
        *,
        force: bool = False,
    ) -> None:
        if (force or d.get(field) is None) and d.get(pr_field) is not None:
            d[field] = d.get(pr_field)

    @classmethod
    def _backfill_benchmark_aliases(cls, d: Dict[str, Any]) -> None:
        pr_eval_is_bpe = d.get("_pr_screening_wikitext_metric_version") == "bpe_eval_v1"
        for field, pr_field in (
            ("wikitext_perplexity", "_pr_wikitext_perplexity"),
            ("wikitext_score", "_pr_wikitext_score"),
            ("tinystories_perplexity", "_pr_tinystories_perplexity"),
            ("tinystories_score", "_pr_tinystories_score"),
            ("hellaswag_acc", "_pr_hellaswag_acc"),
            ("blimp_overall_accuracy", "_pr_blimp_overall_accuracy"),
            ("blimp_n_subtasks", "_pr_blimp_n_subtasks"),
            ("blimp_status", "_pr_blimp_status"),
        ):
            cls._prefer_pr_value(d, field, pr_field, force=pr_eval_is_bpe)
        for field, pr_field in (
            ("hellaswag_metric_version", "_pr_hellaswag_metric_version"),
            ("hellaswag_tokenizer_mode", "_pr_hellaswag_tokenizer_mode"),
            ("hellaswag_tiktoken_encoding", "_pr_hellaswag_tiktoken_encoding"),
            (
                "screening_wikitext_metric_version",
                "_pr_screening_wikitext_metric_version",
            ),
            ("tokenizer_mode", "_pr_tokenizer_mode"),
            ("corpus_path", "_pr_corpus_path"),
            ("evaluation_protocol_version", "_pr_evaluation_protocol_version"),
        ):
            if pr_eval_is_bpe or not d.get(field):
                d[field] = d.get(pr_field) or d.get(field)

    @staticmethod
    def _backfill_dashboard_aliases(d: Dict[str, Any]) -> None:
        for col in _PROGRAM_RESULT_DASHBOARD_ALIAS_FIELDS:
            pr_key = f"_pr_{col}"
            if d.get(col) is None and d.get(pr_key) is not None:
                d[col] = d.get(pr_key)

    @staticmethod
    def _move_program_alias_fields(d: Dict[str, Any]) -> None:
        for field, alias in (
            ("routing_mode", "_routing_mode"),
            ("arch_spec_json", "_arch_spec_json"),
            ("param_count", "_param_count"),
            ("graph_n_params_estimate", "_graph_n_params_estimate"),
            ("novelty_confidence", "_novelty_confidence"),
            ("cka_source", "_cka_source"),
            ("routing_confidence_mean", "_routing_confidence_mean"),
        ):
            d[field] = d.pop(alias, None)
        if (
            d.get("efficiency_multiple") is None
            and d.get("_pr_efficiency_multiple") is not None
        ):
            d["efficiency_multiple"] = d.get("_pr_efficiency_multiple")

    @staticmethod
    def _drop_program_alias_fields(d: Dict[str, Any]) -> None:
        for field in (
            "_pr_discovery_loss_ratio",
            "_pr_validation_loss_ratio",
            "_pr_wikitext_perplexity",
            "_pr_wikitext_score",
            "_pr_tinystories_perplexity",
            "_pr_tinystories_score",
            "_pr_hellaswag_acc",
            "_pr_hellaswag_metric_version",
            "_pr_hellaswag_tokenizer_mode",
            "_pr_hellaswag_tiktoken_encoding",
            "_pr_blimp_overall_accuracy",
            "_pr_blimp_n_subtasks",
            "_pr_blimp_status",
            "_pr_screening_wikitext_metric_version",
            "_pr_tokenizer_mode",
            "_pr_corpus_path",
            "_pr_evaluation_protocol_version",
            "_pr_efficiency_multiple",
        ):
            d.pop(field, None)
        for col in _PROGRAM_RESULT_DASHBOARD_ALIAS_FIELDS:
            d.pop(f"_pr_{col}", None)

    def _parse_leaderboard_entry_details(self, d: Dict[str, Any]) -> None:
        if d.get("investigation_best_training"):
            try:
                d["investigation_best_training_parsed"] = json.loads(
                    d["investigation_best_training"]
                )
            except (json.JSONDecodeError, TypeError):
                pass
        if d.get("is_reference"):
            d["screening_novelty"] = self._reference_novelty_for_display(
                d.get("screening_novelty")
            )
            if d.get("novelty_score") is not None:
                d["novelty_score"] = self._reference_novelty_for_display(
                    d.get("novelty_score")
                )
        d["trusted_candidate"] = bool(is_promotable_entry(d))

    def _leaderboard_row_to_entry(self, row: Any) -> Dict[str, Any]:
        d = dict(row)
        self._backfill_phase_metrics(d)
        self._backfill_benchmark_aliases(d)
        self._backfill_dashboard_aliases(d)
        self._move_program_alias_fields(d)
        self._drop_program_alias_fields(d)
        self._normalize_benchmark_fields(d)
        self._parse_leaderboard_entry_details(d)
        return d

    @staticmethod
    def _replace_if_better_score(
        deduped: List[Dict[str, Any]],
        existing_idx: int,
        entry: Dict[str, Any],
    ) -> None:
        existing_score = deduped[existing_idx].get("composite_score") or 0
        new_score = entry.get("composite_score") or 0
        if new_score > existing_score:
            deduped[existing_idx] = entry

    @classmethod
    def _append_deduped_fingerprint_entry(
        cls,
        *,
        deduped: List[Dict[str, Any]],
        seen: Dict[str, int],
        entry: Dict[str, Any],
    ) -> None:
        fp = entry.get("_graph_fingerprint")
        if fp and fp in seen:
            cls._replace_if_better_score(deduped, seen[fp], entry)
            return
        if fp:
            seen[fp] = len(deduped)
        deduped.append(entry)

    @staticmethod
    def _dedupe_best_by_fingerprint(
        entries: List[Dict[str, Any]],
        *,
        skip_fingerprints: Optional[set[str]] = None,
    ) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
        deduped: List[Dict[str, Any]] = []
        seen: Dict[str, int] = {}
        skip_fingerprints = skip_fingerprints or set()
        for entry in entries:
            fp = entry.get("_graph_fingerprint")
            if fp and fp in skip_fingerprints:
                continue
            _LeaderboardMixin._append_deduped_fingerprint_entry(
                deduped=deduped,
                seen=seen,
                entry=entry,
            )
        return deduped, seen

    @staticmethod
    def _split_reference_entries(
        results: List[Dict[str, Any]],
        include_references: bool,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        references = []
        non_references = []
        for entry in results:
            if include_references and entry.get("is_reference"):
                references.append(entry)
            else:
                non_references.append(entry)
        return references, non_references

    @staticmethod
    def _publish_graph_fingerprint_alias(entries: List[Dict[str, Any]]) -> None:
        for entry in entries:
            entry["graph_fingerprint"] = entry.pop("_graph_fingerprint", None)

    @staticmethod
    def _append_missing_references(
        merged: List[Dict[str, Any]],
        deduped_refs: List[Dict[str, Any]],
        include_references: bool,
    ) -> None:
        if not include_references:
            return
        ref_ids = {entry.get("entry_id") for entry in merged}
        merged.extend(ref for ref in deduped_refs if ref.get("entry_id") not in ref_ids)

    def _dedupe_leaderboard_results(
        self,
        results: List[Dict[str, Any]],
        *,
        include_references: bool,
        limit: int,
    ) -> List[Dict[str, Any]]:
        references, non_references = self._split_reference_entries(
            results,
            include_references,
        )
        deduped_refs, seen_ref_fps = self._dedupe_best_by_fingerprint(references)
        deduped, _ = self._dedupe_best_by_fingerprint(
            non_references,
            skip_fingerprints=set(seen_ref_fps),
        )
        self._publish_graph_fingerprint_alias([*deduped, *deduped_refs])
        merged = deduped[:limit]
        self._append_missing_references(merged, deduped_refs, include_references)
        return merged

    def _attach_leaderboard_families(
        self,
        entries: List[Dict[str, Any]],
        *,
        include_family: bool,
    ) -> None:
        for entry in entries:
            graph_json = entry.pop("_graph_json", None)
            if graph_json:
                graph_json = resolve_graph_json_value(
                    self.conn,
                    self.db_path,
                    graph_json,
                )
            if not include_family:
                continue
            entry["architecture_family"] = self._classify_architecture_family(
                graph_json=graph_json,
                routing_mode=entry.get("routing_mode"),
            )
            if entry.get("architecture_family") == "Unknown" and entry.get(
                "is_reference"
            ):
                entry["architecture_family"] = self._reference_family_fallback(
                    entry.get("reference_name")
                )

    def upsert_leaderboard(
        self,
        result_id: str,
        model_source: str,
        architecture_desc: str = "",
        tier: str = "screening",
        tags: Optional[str] = None,
        notes: Optional[str] = None,
        is_reference: bool = False,
        reference_name: Optional[str] = None,
        allow_fingerprint_duplicate: bool = False,
        **kwargs,
    ) -> str:
        """Insert or update a leaderboard entry."""
        return self._upsert_leaderboard_impl(
            result_id=result_id,
            model_source=model_source,
            architecture_desc=architecture_desc,
            tier=tier,
            tags=tags,
            notes=notes,
            is_reference=is_reference,
            reference_name=reference_name,
            allow_fingerprint_duplicate=allow_fingerprint_duplicate,
            **kwargs,
        )

    def _upsert_leaderboard_impl(
        self,
        result_id: str,
        model_source: str,
        architecture_desc: str = "",
        tier: str = "screening",
        tags: Optional[str] = None,
        notes: Optional[str] = None,
        is_reference: bool = False,
        reference_name: Optional[str] = None,
        allow_fingerprint_duplicate: bool = False,
        **kwargs,
    ) -> str:
        """Insert or update a leaderboard entry.

        Accepts all leaderboard columns as keyword arguments.
        Fields are only updated if provided and not None (prevents accidental NULLing).
        """
        write_args = self._build_leaderboard_upsert_write_args(
            result_id=result_id,
            model_source=model_source,
            architecture_desc=architecture_desc,
            tier=tier,
            tags=tags,
            notes=notes,
            is_reference=bool(is_reference),
            reference_name=reference_name,
            allow_fingerprint_duplicate=allow_fingerprint_duplicate,
            kwargs=kwargs,
        )
        if write_args is None:
            return ""
        entry_id = self._write_leaderboard_upsert(**write_args)
        self._maybe_commit()
        return entry_id

    def get_leaderboard(
        self,
        tier: Optional[str] = None,
        limit: int = 50,
        sort_by: str = "composite_score",
        include_family: bool = True,
        include_references: bool = True,
        trusted_only: bool = False,
        tier_match_mode: str = "reached",
    ) -> List[Dict]:
        """Get leaderboard entries, optionally filtered by tier."""
        return self._get_leaderboard_impl(
            tier=tier,
            limit=limit,
            sort_by=sort_by,
            include_family=include_family,
            include_references=include_references,
            trusted_only=trusted_only,
            tier_match_mode=tier_match_mode,
        )

    def _get_leaderboard_impl(
        self,
        tier: Optional[str] = None,
        limit: int = 50,
        sort_by: str = "composite_score",
        include_family: bool = True,
        include_references: bool = True,
        trusted_only: bool = False,
        tier_match_mode: str = "reached",
    ) -> List[Dict]:
        """Get leaderboard entries, optionally filtered by tier."""
        valid_sorts = {
            "composite_score",
            "screening_loss_ratio",
            "investigation_loss_ratio",
            "validation_loss_ratio",
            "screening_novelty",
            "timestamp",
            "robustness_noise_score",
            "quant_int8_retention",
            "robustness_long_ctx_score",
            "discovery_loss_ratio",
            "generalization_gap",
            "efficiency_multiple",
        }
        sort_by = sort_by if sort_by in valid_sorts else "composite_score"
        query, params = self._build_get_leaderboard_query(
            tier=tier,
            limit=limit,
            sort_by=sort_by,
            include_references=include_references,
            trusted_only=trusted_only,
            tier_match_mode=tier_match_mode,
        )
        rows = self._fetch_leaderboard_rows(query, params)
        results = [self._leaderboard_row_to_entry(row) for row in rows]
        results = self._attach_canonical_program_scores(results)
        merged = self._dedupe_leaderboard_results(
            results,
            include_references=include_references,
            limit=limit,
        )
        self._attach_leaderboard_families(merged, include_family=include_family)
        return merged

    def set_leaderboard_pin(self, entry_id: str, pinned: bool):
        """Pin or unpin a leaderboard entry for dashboard priority."""
        self._submit_write(
            "UPDATE leaderboard SET is_pinned = ? WHERE entry_id = ?",
            (1 if pinned else 0, entry_id),
        )

    def _leaderboard_entry_by_id(self, entry_id: str) -> Any:
        return self.conn.execute(
            "SELECT * FROM leaderboard WHERE entry_id = ?",
            (entry_id,),
        ).fetchone()

    def _program_result_for_promotion(self, result_id: Any) -> Any:
        if not result_id:
            return None
        from ..leaderboard_scoring import (
            _PR_SELECT_COLS,
        )

        return self.conn.execute(
            f"SELECT {_PR_SELECT_COLS}, data_provenance_json, trust_label, comparability_label "  # nosec B608
            "FROM program_results_compat WHERE result_id = ?",
            (result_id,),
        ).fetchone()

    def _promotion_allowed_tier(self, row: Any, pr: Any, tier: str) -> str:
        return self._resolve_allowed_tier(
            requested_tier=tier,
            existing_tier=str(row["tier"] or "screening"),
            pr_row=pr,
            is_reference=bool(row["is_reference"]),
        )

    @staticmethod
    def _promotion_blocked(requested_tier: str, allowed_tier: str) -> bool:
        requested_rank = TIER_RANK.get(str(requested_tier or "").lower(), 0)
        allowed_rank = TIER_RANK.get(str(allowed_tier or "").lower(), 0)
        return requested_rank > allowed_rank

    def _promotion_update_items(
        self,
        *,
        kwargs: Dict[str, Any],
        promotion_blocked: bool,
    ) -> List[tuple[str, Any]]:
        kwargs = sanitize_for_db(kwargs)
        return [] if promotion_blocked else self._leaderboard_update_items(kwargs)

    @staticmethod
    def _promotion_composite(
        row: Any,
        pr: Any,
        update_items: List[tuple[str, Any]],
        allowed_tier: str,
    ) -> Any:
        from ..leaderboard_scoring import (
            _pr_dict_to_score_kwargs,
            compute_composite,
        )

        d = dict(row)
        d.update(dict(update_items))
        d["tier"] = allowed_tier
        pr_d: Dict[str, Any] = dict(pr) if pr else {}
        score_kw = _pr_dict_to_score_kwargs(
            pr_d, d, is_reference=bool(d.get("is_reference"))
        )
        composite = compute_composite(**score_kw)
        return (
            composite["composite_score"] if isinstance(composite, dict) else composite
        )

    @staticmethod
    def _promotion_update_statement(
        *,
        allowed_tier: str,
        update_items: List[tuple[str, Any]],
        composite: Any,
        kwargs: Dict[str, Any],
        entry_id: str,
    ) -> tuple[List[str], List[Any]]:
        sets = ["tier = ?"]
        params: List[Any] = [allowed_tier]
        for col, val in update_items:
            sets.append(f"{col} = ?")
            params.append(val)
        sets.append("composite_score = ?")
        params.append(composite)

        # Handle 'notes' explicitly (it's in _LEADERBOARD_MANAGED_COLUMNS
        # so _leaderboard_update_items filters it out, but promote_to_tier
        # should still allow updating it).
        if "notes" in kwargs and kwargs["notes"] is not None:
            sets.append("notes = ?")
            params.append(kwargs["notes"])

        sets.append("timestamp = ?")
        params.append(time.time())
        params.append(entry_id)
        return sets, params

    def _sync_promoted_entry_fingerprint(self, entry_id: str) -> None:
        try:
            rid_row = self.conn.execute(
                "SELECT result_id FROM leaderboard WHERE entry_id = ?",
                (entry_id,),
            ).fetchone()
            if rid_row and rid_row["result_id"]:
                self._sync_fingerprint_leaderboard(str(rid_row["result_id"]))
        except (KeyError, TypeError, ValueError, sqlite3.OperationalError) as e:
            LOGGER.debug(
                "Fingerprint leaderboard sync skipped for entry %s: %s", entry_id, e
            )

    def promote_to_tier(self, entry_id: str, tier: str, **kwargs) -> None:
        """Update a leaderboard entry's tier and phase-specific results."""
        row = self._leaderboard_entry_by_id(entry_id)
        if not row:
            return
        pr = self._program_result_for_promotion(row["result_id"])
        allowed_tier = self._promotion_allowed_tier(row, pr, tier)
        kwargs = sanitize_for_db(kwargs)
        update_items = self._promotion_update_items(
            kwargs=kwargs,
            promotion_blocked=self._promotion_blocked(tier, allowed_tier),
        )
        composite = self._promotion_composite(row, pr, update_items, allowed_tier)
        sets, params = self._promotion_update_statement(
            allowed_tier=allowed_tier,
            update_items=update_items,
            composite=composite,
            kwargs=kwargs,
            entry_id=entry_id,
        )
        self.conn.execute(
            f"UPDATE leaderboard SET {', '.join(sets)} WHERE entry_id = ?",  # nosec B608
            params,
        )
        self._sync_promoted_entry_fingerprint(entry_id)
        self._maybe_commit()

    # ── Scaling Summary ──

    def _scaling_summary_rows(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            """SELECT l.entry_id, l.scaling_param_efficiency, l.scaling_flop_efficiency,
                      l.scaling_gate_passed, l.scaling_best_family, l.scaling_confidence,
                      l.screening_loss_ratio, l.screening_novelty, l.composite_score,
                      pr.graph_fingerprint
               FROM leaderboard l
               JOIN program_results_compat pr ON l.result_id = pr.result_id
               WHERE l.scaling_param_efficiency IS NOT NULL
               ORDER BY l.scaling_param_efficiency DESC"""
        ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _empty_scaling_summary() -> Dict[str, Any]:
        return {
            "n_evaluated": 0,
            "n_gate_passed": 0,
            "message": "No candidates have been evaluated against external scaling laws yet.",
        }

    @staticmethod
    def _scaling_entry_summary(entry: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "fingerprint": (entry.get("graph_fingerprint") or "")[:12],
            "param_efficiency": entry["scaling_param_efficiency"],
            "family": entry.get("scaling_best_family", "gpt2"),
            "loss_ratio": entry.get("screening_loss_ratio"),
        }

    @staticmethod
    def _scaling_table_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "fingerprint": (entry.get("graph_fingerprint") or "")[:12],
            "param_eff": round(entry["scaling_param_efficiency"], 2),
            "flop_eff": round(entry.get("scaling_flop_efficiency") or 0, 2),
            "gate": bool(entry.get("scaling_gate_passed")),
            "loss_ratio": round(entry.get("screening_loss_ratio") or 0, 4),
        }

    def get_scaling_summary(self) -> Dict:
        """Get a summary of scaling gate results for Aria's context.

        Returns aggregate stats on how candidates compare to external
        baselines (GPT-2/Mamba) in parameter efficiency, plus the best
        and worst performers.
        """
        entries = self._scaling_summary_rows()
        if not entries:
            return self._empty_scaling_summary()
        n_passed = sum(1 for e in entries if e.get("scaling_gate_passed"))
        efficiencies = [e["scaling_param_efficiency"] for e in entries]
        return {
            "n_evaluated": len(entries),
            "n_gate_passed": n_passed,
            "target": 3.0,
            "best_param_efficiency": max(efficiencies),
            "worst_param_efficiency": min(efficiencies),
            "mean_param_efficiency": sum(efficiencies) / len(efficiencies),
            "best_entry": self._scaling_entry_summary(entries[0]),
            "worst_entry": self._scaling_entry_summary(entries[-1]),
            "entries": [self._scaling_table_entry(e) for e in entries[:10]],
        }

    def backfill_replication_aggregates(self) -> int:
        """Backfill replication_n and replication_loss_mean on all leaderboard entries.

        Idempotent — safe to call on every startup. Only touches entries where
        the stored replication_n disagrees with the current count from
        program_results (handles new runs arriving since last backfill).

        Returns the number of entries updated.
        """
        rows = self.conn.execute(
            """SELECT l.entry_id, l.replication_n, pr.graph_fingerprint
               FROM leaderboard l
               JOIN program_results_compat pr ON pr.result_id = l.result_id
               WHERE pr.graph_fingerprint IS NOT NULL"""
        ).fetchall()

        # Batch-fetch all fingerprint aggregates in one query
        fps = list({row["graph_fingerprint"] for row in rows})
        agg_map = self.get_fingerprint_aggregates_batch(fps)

        updated = 0
        for row in rows:
            agg = agg_map.get(row["graph_fingerprint"], {})
            n_runs = agg.get("n_runs", 0)
            if n_runs == 0:
                continue
            if row["replication_n"] == n_runs:
                continue
            self.conn.execute(
                """UPDATE leaderboard
                   SET replication_n = ?,
                       replication_loss_mean = ?,
                       replication_loss_std = ?,
                       replication_best_vs_mean_gap = ?
                   WHERE entry_id = ?""",
                (
                    n_runs,
                    agg.get("loss_mean"),
                    agg.get("loss_std"),
                    agg.get("best_vs_mean_gap"),
                    row["entry_id"],
                ),
            )
            updated += 1

        if updated:
            self._maybe_commit()
            LOGGER.info("backfill_replication_aggregates: updated %d entries", updated)
        return updated
