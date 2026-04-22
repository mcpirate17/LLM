from __future__ import annotations

"""Auto-extracted mixin for LabNotebook."""

import json
import sqlite3
import time
import uuid
from typing import Any, Dict, List, Optional

from ..leaderboard_scoring import build_score_kwargs_from_prefetch, compute_composite
from ..thresholds import TIER_RANK
from .leaderboard_maintenance import (
    leaderboard_consistency_report,
    sync_fingerprint_leaderboard,
)
from .program_provenance import (
    build_data_provenance,
    infer_comparability_label,
    infer_evaluation_protocol_version,
    infer_init_regime,
    infer_result_cohort,
    infer_trust_label,
    merge_experiment_provenance_kwargs,
    normalize_text,
)
from .program_query_views import (
    fetch_report_top_programs_grouped_by_fingerprint,
    fetch_top_programs,
)
from .program_writes import (
    build_program_result_insert_payload,
    enrich_program_result_kwargs,
    filter_known_program_result_columns,
    normalize_program_result_kwargs,
    should_record_program_result,
)
from ..shared_utils import coerce_finite_float as _safe_float
from .notebook_leaderboard import DuplicateLeaderboardFingerprintError
from ._shared import ExperimentEntry, LOGGER, sanitize_for_db


class DuplicateFingerprintError(Exception):
    """Raised when a non-intentional caller tries to record a graph_fingerprint
    that already exists in ``program_results``.

    Callers that legitimately re-run an existing graph (replay, validation
    promotion, reference re-registration, manual override) must pass
    ``intentional_rerun_reason=<short string>`` to bypass the gate.
    """

    def __init__(
        self,
        fingerprint: str,
        existing_result_id: Optional[str],
        existing_experiment_id: Optional[str],
        attempted_experiment_id: Optional[str],
        attempted_model_source: Optional[str],
    ):
        self.fingerprint = fingerprint
        self.existing_result_id = existing_result_id
        self.existing_experiment_id = existing_experiment_id
        self.attempted_experiment_id = attempted_experiment_id
        self.attempted_model_source = attempted_model_source
        super().__init__(
            f"BLOCKED duplicate graph_fingerprint={fingerprint[:16]} "
            f"by experiment={str(attempted_experiment_id)[:12]} "
            f"(model_source={attempted_model_source}); "
            f"already at result_id={existing_result_id} under experiment={str(existing_experiment_id)[:12]}. "
            f"Pass intentional_rerun_reason=<reason> to bypass."
        )


POST_SCREENING_SIGNAL_COLUMNS = (
    "rapid_screening_passed",
    "wikitext_perplexity",
    "wikitext_score",
    "hellaswag_acc",
    "blimp_overall_accuracy",
    "induction_auc",
    "binding_auc",
    "binding_composite",
    "induction_v2_investigation_auc",
    "binding_v2_investigation_auc",
    "discovery_loss_ratio",
    "validation_loss_ratio",
)

SCREENING_EXPERIMENT_TYPES = frozenset(
    {
        "synthesis",
        "novelty",
        "evolution",
        "reference",
        "backfill",
        "forced_exploration",
        "ablation",
    }
)

FINGERPRINT_PROGRAM_TO_PAYLOAD_COLUMNS = {
    "novelty_score": "novelty_score",
    "cka_source": "cka_source",
    "cka_artifact_version": "cka_artifact_version",
    "cka_probe_protocol_hash": "cka_probe_protocol_hash",
    "cka_reference_quality": "cka_reference_quality",
    "novelty_valid_for_promotion": "novelty_valid_for_promotion",
    "novelty_validity_reason": "novelty_validity_reason",
    "novelty_reference_version": "novelty_reference_version",
    "fp_interaction_locality": "interaction_locality",
    "fp_interaction_sparsity": "interaction_sparsity",
    "fp_interaction_symmetry": "interaction_symmetry",
    "fp_interaction_hierarchy": "interaction_hierarchy",
    "fp_intrinsic_dim": "intrinsic_dim",
    "fp_isotropy": "isotropy",
    "fp_rank_ratio": "rank_ratio",
    "fp_jacobian_spectral_norm": "jacobian_spectral_norm",
    "fp_jacobian_effective_rank": "jacobian_effective_rank",
    "fp_sensitivity_uniformity": "sensitivity_uniformity",
    "fp_cka_vs_transformer": "cka_vs_transformer",
    "fp_cka_vs_ssm": "cka_vs_ssm",
    "fp_cka_vs_conv": "cka_vs_conv",
    "fp_hierarchy_fitness": "hierarchy_fitness",
    "fp_gromov_delta": "gromov_delta",
}

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
    "ar_auc",
    "ar_final_acc",
    "induction_auc",
    "binding_auc",
    "binding_composite",
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
    "induction_v2_investigation_auc",
    "induction_v2_investigation_max_gap_acc",
    "binding_v2_investigation_auc",
    "binding_v2_investigation_max_distance_acc",
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
    "induction_probe_train_steps",
    "binding_probe_eval_examples",
    "screening_hellaswag_correct",
    "screening_hellaswag_total",
    "induction_v2_investigation_steps_trained",
    "binding_v2_investigation_train_steps",
}

MERGE_REPLACE_COLUMNS = {
    "error_type",
    "error_message",
    "stage0_error",
    "stage_at_death",
    "failure_op",
    "failure_details_json",
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


class _ProgramsMixin:
    """Programs operations for the Lab Notebook."""

    __slots__ = ()

    def _experiment_type_for_id(self, experiment_id: Any) -> str:
        exp_id = str(experiment_id or "").strip()
        if not exp_id:
            return ""
        row = None
        for attempt in range(2):
            row = self.conn.execute(
                "SELECT experiment_type FROM experiments WHERE experiment_id = ? LIMIT 1",
                (exp_id,),
            ).fetchone()
            if row or attempt:
                break
            self.flush_writes()
        if not row:
            return ""
        return normalize_text(row["experiment_type"])

    def _experiment_config_for_id(self, experiment_id: Any) -> Dict[str, Any]:
        exp_id = str(experiment_id or "").strip()
        if not exp_id:
            return {}
        row = None
        for attempt in range(2):
            row = self.conn.execute(
                "SELECT config_json FROM experiments WHERE experiment_id = ? LIMIT 1",
                (exp_id,),
            ).fetchone()
            if row or attempt:
                break
            self.flush_writes()
        if not row or not row["config_json"]:
            return {}
        try:
            parsed = json.loads(row["config_json"])
        except (json.JSONDecodeError, TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _result_stage_precedence(row: Dict[str, Any]) -> int:
        """Rank rows by authoritative evaluation stage.

        validation > investigation > screening
        """
        experiment_type = normalize_text(row.get("experiment_type"))
        tier = normalize_text(row.get("tier"))
        if experiment_type in {"validation", "breakthrough"} or tier in {
            "validation",
            "validation_failed",
            "breakthrough",
        }:
            return 3
        if experiment_type == "investigation" or tier in {
            "investigation",
            "investigation_failed",
            "investigation_fingerprint_incomplete",
        }:
            return 2
        return 1

    @staticmethod
    def _tier_scope_rank(value: Any) -> int:
        normalized = normalize_text(value)
        if normalized in {"validation", "validation_failed", "breakthrough"}:
            return 3
        if normalized in {
            "investigation",
            "investigation_failed",
            "investigation_fingerprint_incomplete",
        }:
            return 2
        return 1

    def resolve_canonical_result_id(self, result_id: str) -> str:
        """Return the latest authoritative row for the result fingerprint."""
        rid = str(result_id or "").strip()
        if not rid:
            return rid

        row = self.conn.execute(
            "SELECT graph_fingerprint FROM program_results WHERE result_id = ?",
            (rid,),
        ).fetchone()
        if not row or not row["graph_fingerprint"]:
            return rid

        graph_fingerprint = str(row["graph_fingerprint"]).strip()
        if not graph_fingerprint:
            return rid

        self.flush_writes()
        candidates = self.conn.execute(
            """
            SELECT
                pr.result_id,
                pr.timestamp,
                COALESCE(exp.experiment_type, '') AS experiment_type,
                COALESCE(lb.tier, '') AS tier
            FROM program_results pr
            LEFT JOIN experiments exp ON exp.experiment_id = pr.experiment_id
            LEFT JOIN leaderboard lb ON lb.result_id = pr.result_id
            WHERE pr.graph_fingerprint = ?
            """,
            (graph_fingerprint,),
        ).fetchall()
        if not candidates:
            return rid

        best = max(
            (dict(candidate) for candidate in candidates),
            key=lambda candidate: (
                self._result_stage_precedence(candidate),
                float(candidate.get("timestamp") or 0.0),
                str(candidate.get("result_id") or ""),
            ),
        )
        return str(best.get("result_id") or rid)

    def _infer_result_cohort(self, kwargs: Dict[str, Any]) -> str:
        return infer_result_cohort(
            kwargs,
            experiment_type_for_id=self._experiment_type_for_id,
        )

    def _infer_trust_label(self, kwargs: Dict[str, Any], result_cohort: str) -> str:
        return infer_trust_label(kwargs, result_cohort)

    def _infer_comparability_label(
        self,
        kwargs: Dict[str, Any],
        result_cohort: str,
        trust_label: str,
    ) -> str:
        return infer_comparability_label(kwargs, result_cohort, trust_label)

    def _infer_evaluation_protocol_version(
        self,
        kwargs: Dict[str, Any],
        result_cohort: str,
        trust_label: str,
    ) -> str:
        return infer_evaluation_protocol_version(kwargs, result_cohort, trust_label)

    def _infer_init_regime(self, kwargs: Dict[str, Any], result_cohort: str) -> str:
        return infer_init_regime(kwargs, result_cohort)

    def _build_data_provenance(
        self,
        kwargs: Dict[str, Any],
        *,
        result_cohort: str,
        trust_label: str,
        comparability_label: str,
        evaluation_protocol_version: str,
        init_regime: str,
    ) -> str:
        return build_data_provenance(
            kwargs,
            experiment_type_for_id=self._experiment_type_for_id,
            result_cohort=result_cohort,
            trust_label=trust_label,
            comparability_label=comparability_label,
            evaluation_protocol_version=evaluation_protocol_version,
            init_regime=init_regime,
        )

    @staticmethod
    def _build_failure_details(kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Construct a normalized failure payload for persisted program results."""
        stage = kwargs.get("stage_at_death")
        if not stage:
            if kwargs.get("stage0_passed") in (0, False):
                stage = "stage0"
            elif kwargs.get("stage05_passed") in (0, False):
                stage = "stage0.5"
            elif kwargs.get("stage1_passed") in (0, False):
                stage = "stage1"

        error_type = kwargs.get("error_type")
        error_message = kwargs.get("error_message")
        stage0_error = kwargs.get("stage0_error")
        failure_op = kwargs.get("failure_op")
        if not any((stage, error_type, error_message, stage0_error, failure_op)):
            return None

        primary_message = error_message or stage0_error
        traceback_excerpt = None
        if isinstance(primary_message, str) and primary_message:
            lines = [
                line.strip() for line in primary_message.splitlines() if line.strip()
            ]
            if lines:
                traceback_excerpt = "\n".join(lines[-6:])

        return sanitize_for_db(
            {
                "stage": stage,
                "error_type": error_type,
                "error_message": error_message,
                "stage0_error": stage0_error,
                "failure_op": failure_op,
                "root_cause_code": error_type or "unknown",
                "traceback_excerpt": traceback_excerpt,
                "grad_norm": kwargs.get("grad_norm"),
                "max_grad_norm": kwargs.get("max_grad_norm"),
                "stability_score": kwargs.get("stability_score"),
                "param_count": kwargs.get("param_count"),
                "graph_fingerprint": kwargs.get("graph_fingerprint"),
            }
        )

    @staticmethod
    def _fingerprint_confidence_from_payload(
        fp_payload: Dict[str, Any],
    ) -> Optional[float]:
        quality = normalize_text(fp_payload.get("quality"))
        try:
            analyses_succeeded = int(fp_payload.get("analyses_succeeded") or 0)
        except (TypeError, ValueError):
            analyses_succeeded = 0
        if quality == "full":
            return 0.9
        if quality == "partial":
            return min(0.9, 0.4 + (analyses_succeeded * 0.1))
        if fp_payload:
            return 0.3
        return None

    @classmethod
    def _behavioral_fingerprint_program_fields(
        cls,
        fp_payload: Dict[str, Any],
        *,
        novelty_confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        if not isinstance(fp_payload, dict) or not fp_payload:
            return {}

        confidence = novelty_confidence
        if confidence is None:
            confidence = cls._fingerprint_confidence_from_payload(fp_payload)

        patch = {
            "fingerprint_json": json.dumps(sanitize_for_db(fp_payload)),
            "novelty_score": fp_payload.get("novelty_score"),
            "novelty_confidence": confidence,
            "fp_interaction_locality": fp_payload.get("interaction_locality"),
            "fp_interaction_sparsity": fp_payload.get("interaction_sparsity"),
            "fp_interaction_symmetry": fp_payload.get("interaction_symmetry"),
            "fp_interaction_hierarchy": fp_payload.get("interaction_hierarchy"),
            "fp_intrinsic_dim": fp_payload.get("intrinsic_dim"),
            "fp_isotropy": fp_payload.get("isotropy"),
            "fp_rank_ratio": fp_payload.get("rank_ratio"),
            "fp_jacobian_spectral_norm": fp_payload.get("jacobian_spectral_norm"),
            "fp_jacobian_effective_rank": fp_payload.get("jacobian_effective_rank"),
            "fp_sensitivity_uniformity": fp_payload.get("sensitivity_uniformity"),
            "fp_cka_vs_transformer": fp_payload.get("cka_vs_transformer"),
            "fp_cka_vs_ssm": fp_payload.get("cka_vs_ssm"),
            "fp_cka_vs_conv": fp_payload.get("cka_vs_conv"),
            "fp_hierarchy_fitness": fp_payload.get("hierarchy_fitness"),
            "fp_gromov_delta": fp_payload.get("gromov_delta"),
            "cka_source": fp_payload.get("cka_source"),
            "cka_artifact_version": fp_payload.get("cka_artifact_version"),
            "cka_probe_protocol_hash": fp_payload.get("cka_probe_protocol_hash"),
            "cka_reference_quality": fp_payload.get("cka_reference_quality"),
            "novelty_valid_for_promotion": int(
                bool(fp_payload.get("novelty_valid_for_promotion"))
            ),
            "novelty_validity_reason": fp_payload.get("novelty_validity_reason"),
            "novelty_reference_version": fp_payload.get("novelty_reference_version"),
            "fingerprint_full_ran": int(
                normalize_text(fp_payload.get("quality")) == "full"
            ),
        }
        return sanitize_for_db(patch)

    @classmethod
    def _canonicalize_fingerprint_payload(
        cls,
        fp_payload: Dict[str, Any],
        *,
        program_values: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not isinstance(fp_payload, dict) or not fp_payload:
            return {}

        merged = dict(fp_payload)
        for column, payload_key in FINGERPRINT_PROGRAM_TO_PAYLOAD_COLUMNS.items():
            value = program_values.get(column)
            if value is None:
                continue
            if column == "novelty_valid_for_promotion":
                merged[payload_key] = bool(value)
            else:
                merged[payload_key] = value
        return sanitize_for_db(merged)

    def sync_behavioral_fingerprint_result(
        self,
        *,
        result_id: str,
        fp_payload: Dict[str, Any],
        novelty_confidence: Optional[float] = None,
        sync_leaderboard: bool = True,
    ) -> bool:
        rid = str(result_id or "").strip()
        if not rid:
            return False
        patch = self._behavioral_fingerprint_program_fields(
            fp_payload,
            novelty_confidence=novelty_confidence,
        )
        if not patch:
            return False
        valid_columns = set(self._get_program_results_columns())
        patch = {
            column: value for column, value in patch.items() if column in valid_columns
        }
        if not patch:
            return False
        set_clause = ", ".join(f"{column} = ?" for column in patch)
        self._submit_write(
            f"UPDATE program_results SET {set_clause} WHERE result_id = ?",
            [*patch.values(), rid],
        )
        if sync_leaderboard:
            self.flush_writes()
            self._sync_fingerprint_leaderboard(rid)
            self._maybe_commit()
        return True

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
        """Merge new measurements into an existing canonical program row.

        This is used by replay/backfill and promotion flows that should enrich an
        existing fingerprint row instead of creating a sibling ``program_results``
        record.
        """
        rid = str(result_id or "").strip()
        if not rid:
            return False

        self.flush_writes()
        existing_row = self.conn.execute(
            "SELECT * FROM program_results WHERE result_id = ?",
            (rid,),
        ).fetchone()
        if existing_row is None:
            return False

        current = dict(existing_row)
        updates: Dict[str, Any] = {}
        valid_columns = set(self._get_program_results_columns())

        if (
            graph_json
            and "graph_json" in valid_columns
            and not str(current.get("graph_json") or "").strip()
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

        for column, candidate in kwargs.items():
            if column not in valid_columns or column in {
                "result_id",
                "experiment_id",
                "timestamp",
                "graph_fingerprint",
                "graph_json",
            }:
                continue
            merged = _merge_program_value(column, current.get(column), candidate)
            if merged != current.get(column):
                updates[column] = merged
                current[column] = merged

        fp_payload = None
        raw_fp_payload = current.get("fingerprint_json")
        if raw_fp_payload:
            try:
                fp_payload = json.loads(raw_fp_payload)
            except (TypeError, ValueError, json.JSONDecodeError):
                fp_payload = None
        if isinstance(kwargs.get("fingerprint_json"), dict):
            fp_payload = dict(kwargs["fingerprint_json"])
        elif isinstance(kwargs.get("fingerprint_json"), str):
            try:
                fp_payload = json.loads(kwargs["fingerprint_json"])
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        if isinstance(fp_payload, dict):
            canonical_fp_payload = self._canonicalize_fingerprint_payload(
                fp_payload,
                program_values=current,
            )
            canonical_fp_json = json.dumps(canonical_fp_payload)
            if canonical_fp_json != current.get("fingerprint_json"):
                updates["fingerprint_json"] = canonical_fp_json
                current["fingerprint_json"] = canonical_fp_json

        if clear_failure_if_stage1 and current.get("stage1_passed") in (1, True):
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

        if relabel_backfill_if_orphan:
            has_leaderboard = self.conn.execute(
                "SELECT 1 FROM leaderboard WHERE result_id = ? LIMIT 1",
                (rid,),
            ).fetchone()
            has_post_screening = any(
                current.get(column) is not None
                for column in POST_SCREENING_SIGNAL_COLUMNS
            )
            if (
                not has_leaderboard
                and not bool(current.get("stage1_passed"))
                and has_post_screening
            ):
                for column, value in BACKFILL_RELABEL_COLUMNS.items():
                    if column in valid_columns and current.get(column) != value:
                        updates[column] = value
                        current[column] = value

        if not updates:
            return False

        set_clause = ", ".join(f"{column} = ?" for column in updates)
        params = list(updates.values()) + [rid]
        self._submit_write(
            f"UPDATE program_results SET {set_clause} WHERE result_id = ?",
            params,
        )

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

    def _ensure_experiment_row(self, experiment_id: Optional[str]) -> None:
        if not experiment_id:
            return
        now = time.time()
        insert_params = (experiment_id, now, json.dumps({}), now)
        try:
            row = self.conn.execute(
                "SELECT 1 FROM experiments WHERE experiment_id = ? LIMIT 1",
                (experiment_id,),
            ).fetchone()
            if row is not None:
                return
            self.conn.execute(
                """INSERT INTO experiments
                (experiment_id, timestamp, experiment_type, status, config_json, started_at)
                VALUES (?, ?, 'unknown', 'running', ?, ?)""",
                insert_params,
            )
            return
        except sqlite3.OperationalError as exc:
            LOGGER.warning(
                "Primary experiment row check/insert failed for %s; retrying direct: %s",
                experiment_id,
                exc,
            )

        try:
            conn = self._direct_db_conn()
            try:
                row = conn.execute(
                    "SELECT 1 FROM experiments WHERE experiment_id = ? LIMIT 1",
                    (experiment_id,),
                ).fetchone()
                if row is None:
                    conn.execute(
                        """INSERT INTO experiments
                        (experiment_id, timestamp, experiment_type, status, config_json, started_at)
                        VALUES (?, ?, 'unknown', 'running', ?, ?)""",
                        insert_params,
                    )
                    conn.commit()
            finally:
                conn.close()
        except sqlite3.OperationalError as exc:
            LOGGER.warning(
                "Direct experiment row check/insert failed for %s; continuing without placeholder row: %s",
                experiment_id,
                exc,
            )

    def upsert_induction_metric_v2(
        self,
        *,
        graph_fingerprint: str,
        result_id: str,
        row: Dict[str, Any],
        source_cohort: str = "runtime",
    ) -> None:
        """Persist canonical induction metrics keyed by graph fingerprint."""
        auc = row.get("induction_auc")
        if auc is None:
            return
        speed_mode = row.get("induction_probe_speed_mode")
        metric_version = row.get("induction_probe_metric_version")
        if not speed_mode or not metric_version:
            return
        gaps = row.get("induction_probe_gaps") or [4, 8, 16, 32, 64]
        gap_acc = row.get("induction_gap_accuracies") or {}
        try:
            payload = (
                graph_fingerprint,
                result_id,
                source_cohort,
                metric_version,
                speed_mode,
                int(row.get("induction_probe_train_steps") or 0),
                int(row.get("induction_probe_eval_examples") or 0),
                int(row.get("induction_probe_batch_size") or 0),
                int(row.get("induction_probe_pool_size") or 0),
                json.dumps(list(gaps)),
                float(auc),
                float(gap_acc.get(4, 0.0)),
                float(gap_acc.get(8, 0.0)),
                float(gap_acc.get(16, 0.0)),
                float(gap_acc.get(32, 0.0)),
                float(gap_acc.get(64, 0.0)),
                float(row.get("induction_probe_elapsed_ms") or 0.0),
                time.time(),
            )
        except (TypeError, ValueError):
            return
        self._submit_write(
            """
            INSERT INTO induction_metrics_v2 (
                graph_fingerprint, result_id, source_cohort, metric_version, speed_mode,
                train_steps, eval_examples, batch_size, pool_size, gaps_json,
                auc, gap_4, gap_8, gap_16, gap_32, gap_64, wall_ms, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(graph_fingerprint) DO UPDATE SET
                result_id = excluded.result_id,
                source_cohort = excluded.source_cohort,
                metric_version = excluded.metric_version,
                speed_mode = excluded.speed_mode,
                train_steps = excluded.train_steps,
                eval_examples = excluded.eval_examples,
                batch_size = excluded.batch_size,
                pool_size = excluded.pool_size,
                gaps_json = excluded.gaps_json,
                auc = excluded.auc,
                gap_4 = excluded.gap_4,
                gap_8 = excluded.gap_8,
                gap_16 = excluded.gap_16,
                gap_32 = excluded.gap_32,
                gap_64 = excluded.gap_64,
                wall_ms = excluded.wall_ms,
                updated_at = excluded.updated_at
            """,
            payload,
        )

    def purge_junk_programs(self, *, dry_run: bool = False) -> Dict[str, Any]:
        """Delete Stage 0 failure program results that carry no useful data.

        Targets results where stage0_passed = 0 or NULL, excluding any that
        somehow passed stage1 (safety guard).

        Returns dict with 'deleted' or 'would_delete' count and 'dry_run' flag.
        """
        self.flush_writes()
        junk_query = """
            SELECT result_id, experiment_id FROM program_results
            WHERE (stage0_passed = 0 OR stage0_passed IS NULL)
              AND (stage1_passed != 1 OR stage1_passed IS NULL)
        """
        junk_rows = self.conn.execute(junk_query).fetchall()
        count = len(junk_rows)

        if dry_run or count == 0:
            return {"would_delete": count, "dry_run": True}

        junk_ids = [r["result_id"] for r in junk_rows]

        # Never delete protected entries (verified leaders, breakthroughs)
        if junk_ids:
            ph = ",".join("?" * len(junk_ids))
            protected = {
                r[0]
                for r in self.conn.execute(
                    f"SELECT result_id FROM leaderboard "
                    f"WHERE result_id IN ({ph}) AND tags LIKE '%protected%'",
                    junk_ids,
                ).fetchall()
            }
            if protected:
                junk_ids = [rid for rid in junk_ids if rid not in protected]

        affected_experiments = {
            r["experiment_id"] for r in junk_rows if r["experiment_id"]
        }

        # Cascade delete in foreign-key dependency order
        batch_size = 500
        for i in range(0, len(junk_ids), batch_size):
            batch = junk_ids[i : i + batch_size]
            placeholders = ",".join("?" * len(batch))
            self.conn.execute(
                f"DELETE FROM training_curves WHERE result_id IN ({placeholders})",
                batch,
            )
            self.conn.execute(
                f"DELETE FROM leaderboard WHERE result_id IN ({placeholders})", batch
            )
            self.conn.execute(
                f"DELETE FROM program_results WHERE result_id IN ({placeholders})",
                batch,
            )

        self._maybe_commit()

        # Recalculate op success rates for affected experiments
        for exp_id in affected_experiments:
            try:
                self.update_op_success_rates(exp_id)
            except Exception as e:
                LOGGER.debug("op_success_rates update for %s skipped: %s", exp_id, e)

        return {"deleted": count, "dry_run": False}

    # ── Entries ──

    def add_entry(self, entry: ExperimentEntry) -> str:
        """Add a notebook entry."""
        entry_id = str(uuid.uuid4())[:12]
        insert_params = (
            entry_id,
            entry.experiment_id,
            time.time(),
            entry.entry_type,
            entry.title,
            entry.content,
            json.dumps(entry.metadata),
            ",".join(entry.tags),
        )
        try:
            self._ensure_experiment_row(entry.experiment_id)
            self.conn.execute(
                """INSERT INTO entries
                (entry_id, experiment_id, timestamp, entry_type, title, content,
                 metadata_json, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                insert_params,
            )
            self._maybe_commit()
        except sqlite3.OperationalError as exc:
            LOGGER.warning(
                "Entry write failed for %s; continuing without notebook persistence: %s",
                entry.experiment_id or "unscoped",
                exc,
            )
        return entry_id

    # ── Program Results ──

    def has_fingerprint(self, graph_fingerprint: str) -> bool:
        """Check if a computation graph has already been evaluated."""
        if not graph_fingerprint:
            return False
        try:
            row = self.conn.execute(
                "SELECT 1 FROM program_results WHERE graph_fingerprint = ? LIMIT 1",
                (graph_fingerprint,),
            ).fetchone()
            return row is not None
        except sqlite3.OperationalError as exc:
            LOGGER.warning(
                "Fingerprint lookup failed for %s; treating as unseen: %s",
                graph_fingerprint,
                exc,
            )
            return False

    def get_fingerprint_aggregates(self, graph_fingerprint: str) -> dict:
        """Per-fingerprint replication statistics across all persisted runs.

        Counts every persisted run for this fingerprint regardless of
        pass/fail status.  Runs dropped by the record_program_result
        quality gate (S0 failures without error_type, signal-free S1
        failures) are not in the DB and therefore not counted.

        Loss/novelty stats aggregate over all runs that have data,
        not just S1 passes.
        """
        if not graph_fingerprint:
            return {}
        row = self.conn.execute(
            """SELECT
                COUNT(*) AS n_runs_total,
                SUM(CASE WHEN stage1_passed = 1 THEN 1 ELSE 0 END) AS n_s1_passed,
                SUM(CASE WHEN stage0_passed = 1 THEN 1 ELSE 0 END) AS n_s0_passed,
                -- Loss stats over all runs with loss data (not just S1 passes)
                SUM(CASE WHEN loss_ratio IS NOT NULL THEN 1 ELSE 0 END) AS n_with_loss,
                AVG(CASE WHEN loss_ratio IS NOT NULL THEN loss_ratio END) AS loss_mean,
                CASE WHEN SUM(CASE WHEN loss_ratio IS NOT NULL THEN 1 ELSE 0 END) > 1 THEN
                    SQRT(MAX(0,
                        AVG(CASE WHEN loss_ratio IS NOT NULL THEN loss_ratio * loss_ratio END)
                        - AVG(CASE WHEN loss_ratio IS NOT NULL THEN loss_ratio END)
                          * AVG(CASE WHEN loss_ratio IS NOT NULL THEN loss_ratio END)
                    ))
                ELSE NULL END AS loss_std,
                MIN(loss_ratio) AS loss_best,
                AVG(CASE WHEN novelty_score IS NOT NULL THEN novelty_score END) AS novelty_mean,
                CASE WHEN SUM(CASE WHEN novelty_score IS NOT NULL THEN 1 ELSE 0 END) > 1 THEN
                    SQRT(MAX(0,
                        AVG(CASE WHEN novelty_score IS NOT NULL THEN novelty_score * novelty_score END)
                        - AVG(CASE WHEN novelty_score IS NOT NULL THEN novelty_score END)
                          * AVG(CASE WHEN novelty_score IS NOT NULL THEN novelty_score END)
                    ))
                ELSE NULL END AS novelty_std
            FROM program_results
            WHERE graph_fingerprint = ?""",
            (graph_fingerprint,),
        ).fetchone()
        if not row or row["n_runs_total"] == 0:
            return {}
        loss_mean = row["loss_mean"]
        loss_best = row["loss_best"]
        gap = (
            (loss_mean - loss_best)
            if (loss_mean is not None and loss_best is not None)
            else None
        )
        return {
            "n_runs": row["n_runs_total"],
            "n_s1_passed": row["n_s1_passed"],
            "n_s0_passed": row["n_s0_passed"],
            "n_with_loss": row["n_with_loss"],
            "loss_mean": loss_mean,
            "loss_std": row["loss_std"],
            "loss_best": loss_best,
            "best_vs_mean_gap": gap,
            "novelty_mean": row["novelty_mean"],
            "novelty_std": row["novelty_std"],
        }

    def get_fingerprint_aggregates_batch(
        self,
        fingerprints: list[str],
    ) -> dict[str, dict]:
        """Batch version of ``get_fingerprint_aggregates``.

        Returns ``{fingerprint: agg_dict}`` for all fingerprints that have
        at least one run.  Missing fingerprints are absent from the result.
        """
        if not fingerprints:
            return {}
        out: dict[str, dict] = {}
        chunk_size = 900
        for start in range(0, len(fingerprints), chunk_size):
            chunk = fingerprints[start : start + chunk_size]
            placeholders = ",".join("?" * len(chunk))
            rows = self.conn.execute(
                f"""SELECT
                    graph_fingerprint,
                    COUNT(*) AS n_runs_total,
                    SUM(CASE WHEN stage1_passed = 1 THEN 1 ELSE 0 END) AS n_s1_passed,
                    SUM(CASE WHEN stage0_passed = 1 THEN 1 ELSE 0 END) AS n_s0_passed,
                    SUM(CASE WHEN loss_ratio IS NOT NULL THEN 1 ELSE 0 END) AS n_with_loss,
                    AVG(CASE WHEN loss_ratio IS NOT NULL THEN loss_ratio END) AS loss_mean,
                    CASE WHEN SUM(CASE WHEN loss_ratio IS NOT NULL THEN 1 ELSE 0 END) > 1 THEN
                        SQRT(MAX(0,
                            AVG(CASE WHEN loss_ratio IS NOT NULL THEN loss_ratio * loss_ratio END)
                            - AVG(CASE WHEN loss_ratio IS NOT NULL THEN loss_ratio END)
                              * AVG(CASE WHEN loss_ratio IS NOT NULL THEN loss_ratio END)
                        ))
                    ELSE NULL END AS loss_std,
                    MIN(loss_ratio) AS loss_best,
                    AVG(CASE WHEN novelty_score IS NOT NULL THEN novelty_score END) AS novelty_mean,
                    CASE WHEN SUM(CASE WHEN novelty_score IS NOT NULL THEN 1 ELSE 0 END) > 1 THEN
                        SQRT(MAX(0,
                            AVG(CASE WHEN novelty_score IS NOT NULL THEN novelty_score * novelty_score END)
                            - AVG(CASE WHEN novelty_score IS NOT NULL THEN novelty_score END)
                              * AVG(CASE WHEN novelty_score IS NOT NULL THEN novelty_score END)
                        ))
                    ELSE NULL END AS novelty_std
                FROM program_results
                WHERE graph_fingerprint IN ({placeholders})
                GROUP BY graph_fingerprint""",
                chunk,
            ).fetchall()
            for row in rows:
                if row["n_runs_total"] == 0:
                    continue
                loss_mean = row["loss_mean"]
                loss_best = row["loss_best"]
                gap = (
                    (loss_mean - loss_best)
                    if (loss_mean is not None and loss_best is not None)
                    else None
                )
                out[row["graph_fingerprint"]] = {
                    "n_runs": row["n_runs_total"],
                    "n_s1_passed": row["n_s1_passed"],
                    "n_s0_passed": row["n_s0_passed"],
                    "n_with_loss": row["n_with_loss"],
                    "loss_mean": loss_mean,
                    "loss_std": row["loss_std"],
                    "loss_best": loss_best,
                    "best_vs_mean_gap": gap,
                    "novelty_mean": row["novelty_mean"],
                    "novelty_std": row["novelty_std"],
                }
        return out

    def record_program_result(
        self,
        experiment_id: str,
        graph_fingerprint: str,
        graph_json: str,
        result_id: Optional[str] = None,
        bypass_quality_gate: bool = False,
        intentional_rerun_reason: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Record results for a single synthesized program.

        Accepts all program_results columns as keyword arguments.
        Boolean fields (stage0_passed, etc.) are converted to int.

        Quality gate: rejects results that provide no learning signal —
        S0 failures, S1 failures with no loss data, and results with
        errors — to keep the database lean and focused.

        Set bypass_quality_gate=True (via debug mode) to persist all results.

        Cross-experiment dedup gate (slice 4): if ``graph_fingerprint`` is
        already present anywhere in ``program_results``, this method raises
        ``DuplicateFingerprintError`` — unless the caller passes
        ``intentional_rerun_reason``. A SQLite trigger enforces the same
        rule at the schema level for any code path that bypasses this
        method and INSERTs directly.
        """
        if not should_record_program_result(
            graph_fingerprint=graph_fingerprint,
            kwargs=kwargs,
            bypass_quality_gate=bypass_quality_gate,
            logger=LOGGER,
        ):
            return ""

        if not intentional_rerun_reason:
            experiment_type = self._experiment_type_for_id(experiment_id)
            if experiment_type == "validation":
                intentional_rerun_reason = "validation_promotion"
            elif experiment_type == "investigation":
                intentional_rerun_reason = "investigation_followup"

        if (
            graph_fingerprint
            and not intentional_rerun_reason
            and self.has_fingerprint(graph_fingerprint)
        ):
            existing = self.conn.execute(
                "SELECT result_id, experiment_id FROM program_results "
                "WHERE graph_fingerprint = ? LIMIT 1",
                (graph_fingerprint,),
            ).fetchone()
            cls = type(self)
            cls._dup_rejection_count = getattr(cls, "_dup_rejection_count", 0) + 1
            existing_rid = existing["result_id"] if existing else None
            existing_eid = existing["experiment_id"] if existing else None
            attempted_source = kwargs.get("model_source")
            LOGGER.warning(
                "BLOCKED duplicate fp=%s by exp=%s (model_source=%s) — "
                "already at rid=%s under exp=%s. Caller must pass "
                "intentional_rerun_reason=<reason> if this re-run is intentional.",
                graph_fingerprint[:16],
                str(experiment_id)[:12],
                attempted_source,
                existing_rid,
                str(existing_eid)[:12] if existing_eid else None,
            )
            raise DuplicateFingerprintError(
                fingerprint=graph_fingerprint,
                existing_result_id=existing_rid,
                existing_experiment_id=existing_eid,
                attempted_experiment_id=experiment_id,
                attempted_model_source=attempted_source,
            )

        if intentional_rerun_reason:
            kwargs.setdefault("intentional_rerun_reason", intentional_rerun_reason)

        if not result_id:
            result_id = str(uuid.uuid4())[:12]
        now = time.time()
        kwargs = dict(kwargs)
        kwargs.setdefault("experiment_id", experiment_id)
        kwargs = merge_experiment_provenance_kwargs(
            kwargs,
            self._experiment_config_for_id(experiment_id),
        )
        if kwargs.get("fingerprint_json"):
            fp_payload = kwargs.get("fingerprint_json")
            if isinstance(fp_payload, str):
                try:
                    fp_payload = json.loads(fp_payload)
                except (TypeError, ValueError, json.JSONDecodeError):
                    fp_payload = None
            if isinstance(fp_payload, dict):
                fp_payload = self._canonicalize_fingerprint_payload(
                    fp_payload,
                    program_values=kwargs,
                )
                kwargs["fingerprint_json"] = json.dumps(fp_payload)
                for column, value in self._behavioral_fingerprint_program_fields(
                    fp_payload,
                    novelty_confidence=kwargs.get("novelty_confidence"),
                ).items():
                    kwargs.setdefault(column, value)
        kwargs = enrich_program_result_kwargs(
            normalize_program_result_kwargs(kwargs),
            infer_result_cohort=self._infer_result_cohort,
            infer_trust_label=self._infer_trust_label,
            infer_comparability_label=self._infer_comparability_label,
            infer_evaluation_protocol_version=self._infer_evaluation_protocol_version,
            infer_init_regime=self._infer_init_regime,
            build_data_provenance=self._build_data_provenance,
            build_failure_details=self._build_failure_details,
        )
        valid_columns = self._get_program_results_columns()
        filtered_kwargs, unknown_cols = filter_known_program_result_columns(
            kwargs,
            valid_columns,
        )
        if unknown_cols:
            LOGGER.debug(
                "Dropping unknown program_results columns: %s",
                ", ".join(sorted(unknown_cols)),
            )

        all_cols, all_vals = build_program_result_insert_payload(
            result_id=result_id,
            experiment_id=experiment_id,
            timestamp=now,
            graph_fingerprint=graph_fingerprint,
            graph_json=graph_json,
            filtered_kwargs=filtered_kwargs,
        )
        placeholders = ", ".join(["?"] * len(all_cols))
        col_str = ", ".join(all_cols)

        self._submit_write(
            f"INSERT INTO program_results ({col_str}) VALUES ({placeholders})",
            all_vals,
        )
        self._store_graph_features_async(
            result_id=result_id,
            graph_fingerprint=graph_fingerprint,
            graph_json=graph_json,
        )
        self.upsert_induction_metric_v2(
            graph_fingerprint=graph_fingerprint,
            result_id=result_id,
            row=filtered_kwargs,
            source_cohort="runtime",
        )
        experiment_type = str(self._experiment_type_for_id(experiment_id) or "").lower()
        if (
            not str(intentional_rerun_reason or "").strip()
            and experiment_type in SCREENING_EXPERIMENT_TYPES
            and bool(filtered_kwargs.get("stage1_passed"))
        ):
            try:
                self._ensure_screening_leaderboard_entry(
                    result_id=result_id,
                    graph_fingerprint=graph_fingerprint,
                    model_source=str(
                        filtered_kwargs.get("model_source") or "graph_synthesis"
                    ),
                    metrics=filtered_kwargs,
                )
            except DuplicateLeaderboardFingerprintError:
                # Another result for this fingerprint already owns the canonical
                # leaderboard row; keep the fingerprint aggregate in sync instead
                # of silently leaving this survivor without any row coverage.
                self._sync_fingerprint_leaderboard(result_id)
        return result_id

    def save_op_rehabilitation_result(
        self,
        op_name: str,
        compile_passed: bool,
        forward_passed: bool,
        error_message: Optional[str],
        model_dim: int,
    ) -> None:
        """Store a rehabilitation test result."""
        self.conn.execute(
            """INSERT INTO op_rehabilitation_cache
               (op_name, compile_passed, forward_passed, error_message, tested_at, model_dim)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(op_name) DO UPDATE SET
                compile_passed = excluded.compile_passed,
                forward_passed = excluded.forward_passed,
                error_message = excluded.error_message,
                tested_at = excluded.tested_at,
                model_dim = excluded.model_dim""",
            (
                op_name,
                int(compile_passed),
                int(forward_passed),
                error_message,
                time.time(),
                model_dim,
            ),
        )
        self._maybe_commit()

    def get_top_programs(
        self,
        n: int = 20,
        sort_by: str = "novelty_score",
        trusted_only: bool = False,
    ) -> List[Dict]:
        self.flush_writes()
        rows = fetch_top_programs(
            self,
            n=n,
            sort_by=sort_by,
            trusted_only=trusted_only,
        )
        return self._attach_canonical_program_scores(rows)

    def get_report_top_programs_grouped_by_fingerprint(
        self,
        n: int = 20,
        sort_by: str = "loss_ratio",
        trusted_only: bool = False,
    ) -> List[Dict]:
        self.flush_writes()
        rows = fetch_report_top_programs_grouped_by_fingerprint(
            self,
            n=n,
            sort_by=sort_by,
            trusted_only=trusted_only,
        )
        return self._attach_canonical_program_scores(rows)

    def get_program_results(self, experiment_id: str, limit: int = 500) -> List[Dict]:
        """Get ALL program results for an experiment (not just survivors)."""
        rows = self.conn.execute(
            """SELECT * FROM program_results
               WHERE experiment_id = ?
               ORDER BY novelty_score DESC NULLS LAST
               LIMIT ?""",
            (experiment_id, limit),
        ).fetchall()
        return self._attach_canonical_program_scores([dict(r) for r in rows])

    def get_program_detail(self, result_id: str) -> Optional[Dict]:
        """Get full detail for a single program result."""
        row = self.conn.execute(
            "SELECT * FROM program_results WHERE result_id = ?",
            (result_id,),
        ).fetchone()
        if row is None:
            return None
        records = self._attach_canonical_program_scores([dict(row)])
        if not records:
            return None
        return self._parse_program_json_fields(records[0])

    def get_program_details(self, result_ids: List[str]) -> List[Dict]:
        """Batch fetch full details for multiple program results."""
        ids = [rid for rid in result_ids if rid]
        if not ids:
            return []
        placeholders = ",".join(["?"] * len(ids))
        rows = self.conn.execute(
            f"SELECT * FROM program_results WHERE result_id IN ({placeholders})",
            ids,
        ).fetchall()
        by_id = {}
        for d in self._attach_canonical_program_scores([dict(row) for row in rows]):
            d = self._parse_program_json_fields(d)
            by_id[d.get("result_id")] = d
        return [by_id.get(rid) for rid in ids]

    def _attach_canonical_program_scores(
        self, rows: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Annotate raw program rows with backend composite score + breakdown."""
        if not rows:
            return rows

        result_ids = [
            str(row.get("result_id") or "").strip()
            for row in rows
            if row.get("result_id")
        ]
        leaderboard_by_id: Dict[str, Dict[str, Any]] = {}
        if result_ids:
            chunk_size = 900
            for start in range(0, len(result_ids), chunk_size):
                chunk = result_ids[start : start + chunk_size]
                placeholders = ",".join("?" for _ in chunk)
                lb_rows = self.conn.execute(
                    f"SELECT * FROM leaderboard WHERE result_id IN ({placeholders})",
                    chunk,
                ).fetchall()
                for lb_row in lb_rows:
                    leaderboard_by_id[str(lb_row["result_id"])] = dict(lb_row)

        annotated: List[Dict[str, Any]] = []
        for source in rows:
            row = dict(source)
            result_id = str(row.get("result_id") or "").strip()
            lb = leaderboard_by_id.get(result_id, {})
            tier = str(lb.get("tier") or "").strip().lower()
            if not tier:
                tier = "screening" if bool(row.get("stage1_passed")) else "screened_out"
            row["tier"] = tier
            if lb.get("entry_id") and row.get("entry_id") is None:
                row["entry_id"] = lb.get("entry_id")

            score_context = dict(lb)
            score_context.setdefault("tier", tier)
            score_context.setdefault("screening_loss_ratio", row.get("loss_ratio"))
            score_context.setdefault("screening_novelty", row.get("novelty_score"))
            score_context.setdefault(
                "novelty_confidence", row.get("novelty_confidence")
            )
            if (
                row.get("hellaswag_acc") is not None
                and score_context.get("hellaswag_acc") is None
            ):
                score_context["hellaswag_acc"] = row.get("hellaswag_acc")

            stage_rank = self._tier_scope_rank(tier)
            if lb:
                if lb.get("screening_loss_ratio") is not None:
                    row["screening_loss_ratio"] = lb.get("screening_loss_ratio")
                else:
                    row["screening_loss_ratio"] = row.get("loss_ratio")
                row["screening_novelty"] = (
                    lb.get("screening_novelty")
                    if lb.get("screening_novelty") is not None
                    else row.get("novelty_score")
                )
                row["investigation_loss_ratio"] = (
                    lb.get("investigation_loss_ratio") if stage_rank >= 2 else None
                )
                if stage_rank >= 3:
                    row["validation_loss_ratio"] = lb.get("validation_loss_ratio")
                elif (
                    str(row.get("result_cohort") or "").strip().lower() == "backfill"
                    and row.get("validation_loss_ratio") is not None
                ):
                    # Backfill rows may carry validation-style metrics without
                    # representing a promoted validation candidate. Preserve the
                    # observed value so downstream semantic warnings can flag
                    # non-comparable backfill evidence without mislabeling the tier.
                    row["validation_loss_ratio"] = row.get("validation_loss_ratio")
                else:
                    row["validation_loss_ratio"] = None
                row["validation_baseline_ratio"] = (
                    lb.get("validation_baseline_ratio") if stage_rank >= 3 else None
                )
            else:
                row["screening_loss_ratio"] = row.get("loss_ratio")
                row["screening_novelty"] = row.get("novelty_score")
                row["investigation_loss_ratio"] = None
                row["validation_loss_ratio"] = None
                row["validation_baseline_ratio"] = None

            is_reference = bool(
                lb.get("is_reference")
                or str(row.get("trust_label") or "").strip().lower() == "reference"
                or str(row.get("model_source") or "").strip().lower() == "reference"
            )
            try:
                result = compute_composite(
                    decompose=True,
                    **build_score_kwargs_from_prefetch(
                        row, score_context, is_reference
                    ),
                )
            except (TypeError, ValueError, KeyError):
                result = {"composite_score": 0.0, "breakdown": {}}

            computed_score = float(result.get("composite_score") or 0.0)
            row["composite_score"] = (
                float(lb.get("composite_score"))
                if lb.get("composite_score") is not None
                else computed_score
            )
            row["score_breakdown"] = result.get("breakdown") or {}
            annotated.append(row)
        return annotated

    @staticmethod
    def _parse_program_json_fields(d: Dict[str, Any]) -> Dict[str, Any]:
        """Parse known JSON fields for program results in-place."""
        json_fields = (
            "graph_json",
            "fingerprint_json",
            "training_program_json",
            "graph_category_histogram",
            "external_benchmarks_json",
            "perf_report_json",
            "kernel_timings_json",
            "starvation_report_json",
            "diagnostic_tasks_json",
            "sparsity_report_json",
        )
        for json_field in json_fields:
            val = d.get(json_field)
            if val and isinstance(val, str):
                try:
                    d[json_field + "_parsed"] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    pass
        return d

    def _sync_fingerprint_leaderboard(self, result_id: str) -> None:
        sync_fingerprint_leaderboard(self, result_id)

    def backfill_fingerprint_aggregates(self) -> int:
        """[MIGRATION TOOL] Recompute fingerprint-level leaderboard aggregates for all entries."""
        rows = self.conn.execute(
            """
            SELECT DISTINCT l.result_id
            FROM leaderboard l
            JOIN program_results pr ON pr.result_id = l.result_id
            WHERE pr.graph_fingerprint IS NOT NULL
            """
        ).fetchall()
        synced = 0
        seen_fp: set[str] = set()
        for row in rows:
            rid = row["result_id"]
            fp_row = self.conn.execute(
                "SELECT graph_fingerprint FROM program_results WHERE result_id = ?",
                (rid,),
            ).fetchone()
            fp = (
                str(fp_row["graph_fingerprint"])
                if fp_row and fp_row["graph_fingerprint"]
                else ""
            )
            if not fp or fp in seen_fp:
                continue
            seen_fp.add(fp)
            self._sync_fingerprint_leaderboard(rid)
            synced += 1
        self._maybe_commit()
        return synced

    def get_leaderboard_entry(self, result_id: str) -> Optional[Dict]:
        """Fetch a single leaderboard entry by result_id."""
        if not result_id:
            return None
        rows = self.conn.execute(
            "SELECT * FROM leaderboard WHERE result_id = ?",
            (result_id,),
        ).fetchone()
        return dict(rows) if rows else None

    def get_leaderboard_entry_by_fingerprint(
        self, graph_fingerprint: str
    ) -> Optional[Dict]:
        """Fetch the leaderboard entry (if any) for a given graph_fingerprint.

        Leaderboard stores result_id, not fingerprint; this joins through
        program_results. Used by promote callers and the dedup gate to find
        an existing entry before inserting a new one.
        """
        if not graph_fingerprint:
            return None
        row = self.conn.execute(
            "SELECT l.* FROM leaderboard l "
            "JOIN program_results pr ON l.result_id = pr.result_id "
            "WHERE pr.graph_fingerprint = ? "
            "ORDER BY l.timestamp DESC LIMIT 1",
            (str(graph_fingerprint).strip(),),
        ).fetchone()
        return dict(row) if row else None

    def get_leaderboard_consistency_report(self) -> Dict[str, Any]:
        return leaderboard_consistency_report(self)

    def _ensure_screening_leaderboard_entry(
        self,
        *,
        result_id: str,
        graph_fingerprint: str,
        model_source: str,
        metrics: Dict[str, Any],
    ) -> Optional[str]:
        """Create or refresh the canonical screening leaderboard row."""
        return self.upsert_leaderboard(
            result_id=result_id,
            model_source=model_source or "graph_synthesis",
            architecture_desc=str(graph_fingerprint or "")[:40],
            screening_loss_ratio=metrics.get("loss_ratio"),
            screening_novelty=metrics.get("novelty_score"),
            screening_passed=True,
            tier="screening",
            novelty_confidence=metrics.get("novelty_confidence"),
            fp_jacobian_spectral_norm=metrics.get("fp_jacobian_spectral_norm"),
            routing_savings_ratio=metrics.get("routing_savings_ratio"),
            activation_sparsity_score=metrics.get("activation_sparsity_score"),
            depth_savings_ratio=metrics.get("depth_savings_ratio"),
            compression_ratio=metrics.get("compression_ratio"),
            wikitext_perplexity=metrics.get("wikitext_perplexity"),
            wikitext_score=metrics.get("wikitext_score"),
        )

    def backfill_missing_screening_leaderboard_entries(
        self,
        *,
        experiment_types: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """[MIGRATION TOOL] Backfill screening leaderboard entries for uncovered screening survivors."""
        experiment_types = experiment_types or [
            "synthesis",
            "novelty",
            "evolution",
            "reference",
            "backfill",
            "forced_exploration",
            "ablation",
        ]
        placeholders = ",".join("?" for _ in experiment_types)
        params: List[Any] = list(experiment_types)
        query = f"""
            SELECT p.*
            FROM program_results p
            JOIN experiments e ON e.experiment_id = p.experiment_id
            WHERE p.stage1_passed = 1
              AND e.experiment_type IN ({placeholders})
              AND NOT EXISTS (
                    SELECT 1 FROM leaderboard l WHERE l.result_id = p.result_id
              )
              AND NOT EXISTS (
                    SELECT 1
                    FROM leaderboard l
                    JOIN program_results pr2 ON pr2.result_id = l.result_id
                    WHERE pr2.graph_fingerprint = p.graph_fingerprint
              )
            ORDER BY p.timestamp ASC
        """
        if limit is not None:
            query += " LIMIT ?"
            params.append(int(limit))

        rows = self.conn.execute(query, params).fetchall()
        created_entry_ids: List[str] = []
        created_result_ids: List[str] = []
        for row in rows:
            record = dict(row)
            entry_id = self.upsert_leaderboard(
                result_id=str(record["result_id"]),
                model_source=str(record.get("model_source") or "graph_synthesis"),
                architecture_desc=str(record.get("graph_fingerprint") or "")[:40],
                screening_loss_ratio=record.get("loss_ratio"),
                screening_novelty=record.get("novelty_score"),
                screening_passed=True,
                tier="screening",
                novelty_confidence=record.get("novelty_confidence"),
                fp_jacobian_spectral_norm=record.get("fp_jacobian_spectral_norm"),
                routing_savings_ratio=record.get("routing_savings_ratio"),
                activation_sparsity_score=record.get("activation_sparsity_score"),
                depth_savings_ratio=record.get("depth_savings_ratio"),
                compression_ratio=record.get("compression_ratio"),
                wikitext_perplexity=record.get("wikitext_perplexity"),
                wikitext_score=record.get("wikitext_score"),
            )
            created_entry_ids.append(entry_id)
            created_result_ids.append(str(record["result_id"]))
            self._sync_fingerprint_leaderboard(str(record["result_id"]))

        self._maybe_commit()
        return {
            "created_entries": len(created_entry_ids),
            "entry_ids": created_entry_ids,
            "result_ids": created_result_ids,
        }

    def repair_rebindable_orphan_leaderboard_rows(
        self,
        *,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Rebind orphan leaderboard rows to canonical program rows by fingerprint."""
        query = """
            SELECT l.entry_id,
                   l.result_id AS orphan_result_id,
                   l.architecture_desc AS graph_fingerprint,
                   pr.result_id AS canonical_result_id
            FROM leaderboard l
            LEFT JOIN program_results pr0 ON pr0.result_id = l.result_id
            JOIN program_results pr ON pr.graph_fingerprint = l.architecture_desc
            WHERE pr0.result_id IS NULL
              AND l.architecture_desc IS NOT NULL
              AND TRIM(l.architecture_desc) != ''
            ORDER BY l.timestamp DESC
        """
        params: List[Any] = []
        if limit is not None:
            query += " LIMIT ?"
            params.append(int(limit))
        rows = [dict(row) for row in self.conn.execute(query, params).fetchall()]
        if not rows:
            return {
                "rebound_rows": 0,
                "fingerprints_repaired": 0,
                "deleted_duplicate_rows": 0,
                "fingerprints": [],
            }

        grouped: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            fp = str(row.get("graph_fingerprint") or "").strip()
            canonical_rid = str(row.get("canonical_result_id") or "").strip()
            if not fp or not canonical_rid:
                continue
            bucket = grouped.setdefault(
                fp,
                {
                    "canonical_result_id": canonical_rid,
                    "entry_ids": [],
                    "orphan_result_ids": [],
                },
            )
            bucket["entry_ids"].append(str(row["entry_id"]))
            bucket["orphan_result_ids"].append(str(row["orphan_result_id"]))

        rebound_rows = 0
        deleted_duplicate_rows = 0
        repaired_fps: List[str] = []

        for graph_fingerprint, bucket in grouped.items():
            canonical_rid = bucket["canonical_result_id"]
            orphan_entry_ids = bucket["entry_ids"]
            repaired_fps.append(graph_fingerprint)
            rebound_rows += len(orphan_entry_ids)
            placeholders = ",".join("?" for _ in orphan_entry_ids)
            self.conn.execute(
                f"UPDATE leaderboard SET result_id = ? WHERE entry_id IN ({placeholders})",
                [canonical_rid, *orphan_entry_ids],
            )
            self._sync_fingerprint_leaderboard(canonical_rid)

            dup_rows = [
                dict(row)
                for row in self.conn.execute(
                    "SELECT entry_id, tier, timestamp, composite_score "
                    "FROM leaderboard WHERE result_id = ?",
                    (canonical_rid,),
                ).fetchall()
            ]
            if not dup_rows:
                continue
            keep = max(
                dup_rows,
                key=lambda row: (
                    int(TIER_RANK.get(str(row.get("tier") or "").lower(), -1)),
                    float(row.get("timestamp") or 0.0),
                    float(row.get("composite_score") or -1e9),
                ),
            )
            delete_ids = [
                str(row["entry_id"])
                for row in dup_rows
                if str(row["entry_id"]) != str(keep["entry_id"])
            ]
            if delete_ids:
                delete_placeholders = ",".join("?" for _ in delete_ids)
                self.conn.execute(
                    f"DELETE FROM leaderboard WHERE entry_id IN ({delete_placeholders})",
                    delete_ids,
                )
                deleted_duplicate_rows += len(delete_ids)

        self._maybe_commit()
        return {
            "rebound_rows": rebound_rows,
            "fingerprints_repaired": len(repaired_fps),
            "deleted_duplicate_rows": deleted_duplicate_rows,
            "fingerprints": repaired_fps,
        }

    def get_investigated_fingerprints(self) -> set:
        """Return fingerprints that have already been investigated or beyond.

        Checks both leaderboard tiers AND program_results from investigation/
        ablation experiments, so candidates tested in failed/interrupted
        investigations are not re-queued indefinitely.
        """
        fps = set()
        # Tier-based: candidates promoted in leaderboard
        rows = self.conn.execute(
            "SELECT DISTINCT pr.graph_fingerprint "
            "FROM leaderboard l "
            "JOIN program_results pr ON pr.result_id = l.result_id "
            "WHERE l.tier IN ("
            "'investigation', "
            "'investigation_fingerprint_incomplete', "
            "'validation', "
            "'breakthrough'"
            ")"
        ).fetchall()
        fps.update(r[0] for r in rows if r[0])
        # History-based: fingerprints tested in investigation/ablation experiments
        # (catches failed/interrupted investigations that never reached leaderboard)
        rows = self.conn.execute(
            "SELECT DISTINCT pr.graph_fingerprint "
            "FROM program_results pr "
            "JOIN experiments e ON e.experiment_id = pr.experiment_id "
            "WHERE e.experiment_type IN ('investigation', 'ablation')"
        ).fetchall()
        fps.update(r[0] for r in rows if r[0])
        return fps

    def get_tiers_for_result_ids(self, result_ids: List[str]) -> Dict[str, str]:
        """Return {result_id: tier} for given result IDs that have leaderboard entries."""
        if not result_ids:
            return {}
        placeholders = ",".join("?" for _ in result_ids)
        rows = self.conn.execute(
            f"SELECT result_id, tier FROM leaderboard WHERE result_id IN ({placeholders})",
            result_ids,
        ).fetchall()
        return {r["result_id"]: r["tier"] for r in rows}
