from __future__ import annotations

"""Program result merge-patch helpers for LabNotebook."""

import json
from typing import Any, Dict, Optional

from .. import shared_utils as _shared_utils


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    return _shared_utils.coerce_finite_float(value, default)


POST_SCREENING_SIGNAL_COLUMNS = (
    "rapid_screening_passed",
    "wikitext_perplexity",
    "wikitext_score",
    "hellaswag_acc",
    "blimp_overall_accuracy",
    "induction_screening_auc",
    "binding_screening_auc",
    "binding_screening_composite",
    "induction_intermediate_auc",
    "binding_intermediate_auc",
    "permutation_composition_score",
    "discovery_loss_ratio",
    "validation_loss_ratio",
)

MERGE_HIGHER_BETTER_COLUMNS = {
    "novelty_score",
    "structural_novelty",
    "behavioral_novelty",
    "novelty_confidence",
    "throughput_tok_s",
    "stability_score",
    "loss_improvement_rate",
    "screening_slope",
    "activation_sparsity_score",
    "routing_confidence_mean",
    "routing_utilization_entropy",
    "routing_savings_ratio",
    "compression_ratio",
    "hellaswag_acc",
    "blimp_overall_accuracy",
    "ar_legacy_auc",
    "ar_legacy_final_acc",
    "induction_screening_auc",
    "binding_screening_auc",
    "binding_screening_composite",
    "validation_robustness_score",
    "wikitext_score",
    "tinystories_score",
    "cross_task_score",
    "diagnostic_score",
    "judgment_score",
    "ncd_score",
    "efficiency_multiple",
    "robustness_long_ctx_scaling_score",
    "robustness_long_ctx_assoc_score",
    "robustness_long_ctx_multi_hop_score",
    "robustness_long_ctx_passkey_score",
    "robustness_long_ctx_retrieval_aggregate",
    "robustness_long_ctx_combined_score",
    "induction_intermediate_auc",
    "induction_intermediate_max_gap_acc",
    "binding_intermediate_auc",
    "binding_intermediate_max_distance_acc",
    "permutation_composition_score",
    "permutation_composition_train_chain_acc",
    "permutation_composition_extrapolation_acc",
}

MERGE_LOWER_BETTER_COLUMNS = {
    "loss_ratio",
    "final_loss",
    "discovery_loss",
    "discovery_loss_ratio",
    "validation_loss",
    "validation_loss_ratio",
    "generalization_gap",
    "baseline_loss_ratio",
    "wikitext_perplexity",
    "wikitext_pre_perplexity",
    "wikitext_ppl_200",
    "wikitext_ppl_500",
    "tinystories_perplexity",
    "ncd_description_length",
    "ncd_description_length_per_param",
    "fp_jacobian_spectral_norm",
    "peak_memory_mb",
    "compile_time_ms",
    "forward_time_ms",
    "backward_time_ms",
    "validation_multi_seed_std",
    "init_sensitivity_std",
    "robustness_noise_score",
}

MERGE_MAX_COLUMNS = {
    "stage0_passed",
    "stage05_passed",
    "stage1_passed",
    "rapid_screening_passed",
    "validation_passed",
    "validation_is_unstable",
    "extreme_input_passed",
    "random_input_passed",
    "has_zero_grad",
    "n_train_steps",
    "train_budget_steps",
    "rapid_screening_steps_completed",
    "rapid_screening_max_steps",
    "routing_tokens_total",
    "routing_tokens_processed",
    "routing_tokens_skipped",
    "routing_capacity_overflow_count",
    "routing_expert_count",
    "graph_n_ops",
    "graph_depth",
    "graph_n_edges",
    "graph_n_unique_ops",
    "max_viable_seq_len",
    "hellaswag_n_examples",
    "blimp_n_subtasks",
    "induction_screening_train_steps",
    "binding_screening_eval_examples",
    "screening_hellaswag_correct",
    "screening_hellaswag_total",
    "induction_intermediate_steps_trained",
    "binding_intermediate_train_steps",
    "permutation_composition_n_items",
    "permutation_composition_train_chain_len",
    "permutation_composition_eval_chain_len",
    "permutation_composition_train_steps",
}

MERGE_REPLACE_COLUMNS = {
    "error_type",
    "error_message",
    "stage0_error",
    "stage_at_death",
    "failure_op",
    "failure_details_json",
    "hellaswag_metric_version",
    "hellaswag_tokenizer_mode",
    "hellaswag_tiktoken_encoding",
    "permutation_composition_metric_version",
    "permutation_composition_status",
}

BACKFILL_RELABEL_COLUMNS = {
    "result_cohort": "backfill",
    "trust_label": "backfill_observation",
    "comparability_label": "reconstructed_init_variant",
    "evaluation_protocol_version": "backfill_replay_v1",
    "init_regime": "reconstructed_fresh_init",
}


def _merge_program_value(column: str, current: Any, candidate: Any) -> Any:
    if candidate is None:
        return current
    if column in MERGE_REPLACE_COLUMNS:
        return candidate
    if current is None:
        return candidate
    if column in MERGE_HIGHER_BETTER_COLUMNS:
        current_f = _safe_float(current)
        candidate_f = _safe_float(candidate)
        if current_f is None:
            return candidate
        if candidate_f is None:
            return current
        return candidate if candidate_f > current_f else current
    if column in MERGE_LOWER_BETTER_COLUMNS:
        current_f = _safe_float(current)
        candidate_f = _safe_float(candidate)
        if current_f is None:
            return candidate
        if candidate_f is None:
            return current
        return candidate if candidate_f < current_f else current
    if column in MERGE_MAX_COLUMNS:
        current_f = _safe_float(current)
        candidate_f = _safe_float(candidate)
        if current_f is None:
            return candidate
        if candidate_f is None:
            return current
        return candidate if candidate_f > current_f else current
    return current


_ARCH_COLS = ("graph_json", "arch_spec_json")


def _build_graphs_set_clause(
    graphs_updates: Dict[str, Any],
) -> tuple[list[str], list[Any]]:
    """Build the SET parts + params for an UPDATE graphs ... statement."""
    set_parts: list[str] = []
    set_vals: list[Any] = []
    if "graph_json" in graphs_updates:
        gj = graphs_updates["graph_json"]
        set_parts.append("graph_json = ?")
        set_parts.append(
            "graph_json_is_placeholder = "
            "CASE WHEN ? IS NULL OR ? IN ('', '{}') THEN 1 ELSE 0 END"
        )
        set_vals.extend([gj, gj, gj])
    if "arch_spec_json" in graphs_updates:
        set_parts.append("arch_spec_json = COALESCE(?, arch_spec_json)")
        set_vals.append(graphs_updates["arch_spec_json"])
    return set_parts, set_vals


class _ProgramResultMergeMixin:
    def _submit_split_program_patch(
        self, result_id: str, updates: Dict[str, Any]
    ) -> None:
        """Route a patch to graph_runs (per-run cols) and graphs (arch cols).

        Shared-fingerprint runs: a graphs-col update affects every run that
        shares the fingerprint — the same cross-run behavior the AFTER-UPDATE
        propagation trigger produced when the legacy write hit program_results.
        """
        runs_updates = {k: v for k, v in updates.items() if k not in _ARCH_COLS}
        graphs_updates = {k: v for k, v in updates.items() if k in _ARCH_COLS}

        if runs_updates:
            set_clause = ", ".join(f"{column} = ?" for column in runs_updates)
            self._submit_write(
                f"UPDATE graph_runs SET {set_clause} WHERE result_id = ?",
                list(runs_updates.values()) + [result_id],
            )

        if not graphs_updates:
            return
        set_parts, set_vals = _build_graphs_set_clause(graphs_updates)
        set_vals.append(result_id)
        self._submit_write(
            "UPDATE graphs SET " + ", ".join(set_parts) + " WHERE graph_fingerprint = ("
            "SELECT graph_fingerprint FROM graph_runs WHERE result_id = ?)",
            set_vals,
        )

    def _merge_patch_base_updates(
        self,
        *,
        current: Dict[str, Any],
        valid_columns: set[str],
        graph_fingerprint: Optional[str],
        graph_json: Optional[str],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        updates: Dict[str, Any] = {}
        if (
            graph_json
            and "graph_json" in valid_columns
            and not current.get("graph_json")
        ):
            updates["graph_json"] = graph_json
            current["graph_json"] = graph_json
        if (
            graph_fingerprint
            and "graph_fingerprint" in valid_columns
            and not str(current.get("graph_fingerprint") or "").strip()
        ):
            updates["graph_fingerprint"] = graph_fingerprint
            current["graph_fingerprint"] = graph_fingerprint

        protected = {
            "result_id",
            "experiment_id",
            "timestamp",
            "graph_fingerprint",
            "graph_json",
        }
        for column, candidate in kwargs.items():
            if column not in valid_columns or column in protected:
                continue
            merged = _merge_program_value(column, current.get(column), candidate)
            if merged != current.get(column):
                updates[column] = merged
                current[column] = merged
        return updates

    @staticmethod
    def _merge_patch_fingerprint_payload(
        current: Dict[str, Any], kwargs: Dict[str, Any]
    ):
        raw_fp_payload = current.get("fingerprint_json")
        if raw_fp_payload:
            try:
                return json.loads(raw_fp_payload)
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        candidate = kwargs.get("fingerprint_json")
        if isinstance(candidate, dict):
            return dict(candidate)
        if isinstance(candidate, str):
            try:
                return json.loads(candidate)
            except (TypeError, ValueError, json.JSONDecodeError):
                return None
        return None

    @staticmethod
    def _clear_stage1_failure_columns(
        *,
        current: Dict[str, Any],
        updates: Dict[str, Any],
        valid_columns: set[str],
    ) -> None:
        if current.get("stage1_passed") not in (1, True):
            return
        for column in (
            "error_type",
            "error_message",
            "stage_at_death",
            "failure_op",
            "failure_details_json",
        ):
            if column in valid_columns and current.get(column) is not None:
                updates[column] = None
                current[column] = None

    @staticmethod
    def _relabel_orphan_backfill(
        *,
        current: Dict[str, Any],
        updates: Dict[str, Any],
        valid_columns: set[str],
        has_leaderboard: bool,
    ) -> None:
        has_post_screening = any(
            current.get(column) is not None for column in POST_SCREENING_SIGNAL_COLUMNS
        )
        if (
            has_leaderboard
            or bool(current.get("stage1_passed"))
            or not has_post_screening
        ):
            return
        for column, value in BACKFILL_RELABEL_COLUMNS.items():
            if column in valid_columns and current.get(column) != value:
                updates[column] = value
                current[column] = value

    def merge_program_result_patch(
        self,
        *,
        result_id: str,
        graph_fingerprint: Optional[str] = None,
        graph_json: Optional[str] = None,
        clear_failure_if_stage1: bool = False,
        relabel_backfill_if_orphan: bool = False,
        **kwargs,
    ) -> bool:
        """Merge new measurements into an existing canonical program row."""
        rid = str(result_id or "").strip()
        if not rid:
            return False
        self.flush_writes()
        existing_row = self.conn.execute(
            "SELECT * FROM program_results_compat WHERE result_id = ?",
            (rid,),
        ).fetchone()
        if existing_row is None:
            return False

        current = dict(existing_row)
        valid_columns = set(self._get_program_results_columns())
        updates = self._merge_patch_base_updates(
            current=current,
            valid_columns=valid_columns,
            graph_fingerprint=graph_fingerprint,
            graph_json=graph_json,
            kwargs=kwargs,
        )

        fp_payload = self._merge_patch_fingerprint_payload(current, kwargs)
        if isinstance(fp_payload, dict):
            canonical_fp_payload = self._canonicalize_fingerprint_payload(
                fp_payload,
                program_values=current,
            )
            canonical_fp_json = json.dumps(canonical_fp_payload)
            if canonical_fp_json != current.get("fingerprint_json"):
                updates["fingerprint_json"] = canonical_fp_json
                current["fingerprint_json"] = canonical_fp_json

        # Failure columns are nonsensical once stage1 has passed. A later
        # patch may carry an exception (e.g. exact_graph_replay re-runs that
        # fail) but stage1_passed is max-merged so it stays 1; without this
        # guard the row ends up reporting `stage_at_death="stage0"` next to
        # `stage1_passed=1`, which renders as "died at stage0" in the UI.
        # Apply unconditionally; `clear_failure_if_stage1` is kept for compat.
        del clear_failure_if_stage1
        self._clear_stage1_failure_columns(
            current=current,
            updates=updates,
            valid_columns=valid_columns,
        )
        if relabel_backfill_if_orphan:
            has_leaderboard = bool(
                self.conn.execute(
                    "SELECT 1 FROM leaderboard WHERE result_id = ? LIMIT 1",
                    (rid,),
                ).fetchone()
            )
            self._relabel_orphan_backfill(
                current=current,
                updates=updates,
                valid_columns=valid_columns,
                has_leaderboard=has_leaderboard,
            )
        if not updates:
            return False

        self._submit_split_program_patch(rid, updates)
        final_fp = str(
            current.get("graph_fingerprint") or graph_fingerprint or ""
        ).strip()
        final_graph_json = str(current.get("graph_json") or graph_json or "").strip()
        if final_fp and final_graph_json:
            self._store_graph_features_async(
                result_id=rid,
                graph_fingerprint=final_fp,
                graph_json=final_graph_json,
            )
        self.upsert_induction_metric_v2(
            graph_fingerprint=final_fp,
            result_id=rid,
            row=current,
            source_cohort="runtime",
        )
        return True
