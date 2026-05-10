from __future__ import annotations

"""Auto-extracted mixin for LabNotebook."""

import json
import sqlite3
import time
import uuid
from typing import Any, Dict, List, Optional

from ..leaderboard_scoring import (
    build_score_kwargs_from_prefetch,
    compute_composite,
    prefetch_program_results,
)
from .leaderboard_maintenance import (
    leaderboard_consistency_report,
    sync_fingerprint_leaderboard,
)
from .program_leaderboard_repair import _ProgramLeaderboardRepairMixin
from .program_provenance import (
    build_data_provenance,
    infer_comparability_label,
    infer_evaluation_protocol_version,
    infer_init_regime,
    infer_result_cohort,
    infer_trust_label,
    normalize_text,
)
from .program_query_views import (
    fetch_report_top_programs_grouped_by_fingerprint,
    fetch_top_programs,
)
from .program_result_merge import _ProgramResultMergeMixin
from .program_result_recording import (
    DuplicateFingerprintError,
    _ProgramResultRecordingMixin,
)
from ._shared import ExperimentEntry, LOGGER, sanitize_for_db

__all__ = ["DuplicateFingerprintError", "_ProgramsMixin"]


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


class _ProgramsMixin(
    _ProgramResultRecordingMixin,
    _ProgramResultMergeMixin,
    _ProgramLeaderboardRepairMixin,
):
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
            "SELECT graph_fingerprint FROM program_results_compat WHERE result_id = ?",
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
            FROM program_results_compat pr
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
        auc = row.get("induction_screening_auc")
        if auc is None:
            return
        speed_mode = row.get("induction_screening_speed_mode")
        metric_version = row.get("induction_screening_metric_version")
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
                int(row.get("induction_screening_train_steps") or 0),
                int(row.get("induction_screening_eval_examples") or 0),
                int(row.get("induction_screening_batch_size") or 0),
                int(row.get("induction_screening_pool_size") or 0),
                json.dumps(list(gaps)),
                float(auc),
                float(gap_acc.get(4, 0.0)),
                float(gap_acc.get(8, 0.0)),
                float(gap_acc.get(16, 0.0)),
                float(gap_acc.get(32, 0.0)),
                float(gap_acc.get(64, 0.0)),
                float(row.get("induction_screening_elapsed_ms") or 0.0),
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
            SELECT result_id, experiment_id FROM program_results_compat
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
        metadata_json = json.dumps(entry.metadata)
        metadata_json = self._maybe_store_json_artifact(
            table_name="entries",
            row_pk=entry_id,
            column_name="metadata_json",
            payload_json=metadata_json,
        )
        insert_params = (
            entry_id,
            entry.experiment_id,
            time.time(),
            entry.entry_type,
            entry.title,
            entry.content,
            metadata_json,
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
                "SELECT 1 FROM program_results_compat WHERE graph_fingerprint = ? LIMIT 1",
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
            f"""SELECT
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
            FROM program_results_compat
            WHERE graph_fingerprint = ?
              AND {self._FINGERPRINT_AGGREGATE_REASON_FILTER}""",
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

    # Stochastic eval metrics that should be aggregated across runs of the
    # same graph_fingerprint. Each tuple = (column_name, tier_for_cv).
    # Tier "loss"  feeds cv_loss; tier "und"/"cap" feed cv_understanding /
    # cv_capability respectively. Tier None = aggregated mean exposed but
    # not folded into a tier-CV summary.
    _STOCHASTIC_METRICS_FOR_AGG: tuple[tuple[str, Optional[str]], ...] = (
        ("wikitext_perplexity", "loss"),
        ("blimp_overall_accuracy", "und"),
        ("hellaswag_acc", "und"),
        ("tinystories_score", "und"),
        ("cross_task_score", "und"),
        ("diagnostic_score", "und"),
        ("fp_hierarchy_fitness", "und"),
        ("ar_legacy_auc", "cap"),
        ("ar_gate_score", "cap"),
        ("ar_validation_rank_score", "cap"),
        ("induction_screening_auc", "cap"),
        ("binding_screening_auc", "cap"),
        ("induction_intermediate_auc", "cap"),
        ("binding_intermediate_auc", "cap"),
    )

    _FINGERPRINT_AGGREGATE_REASON_FILTER = (
        "COALESCE(intentional_rerun_reason, '') IN "
        "('', 'exact_graph_replay', 'exact_graph_replay_independent_sample')"
    )

    # Metrics that need a per-row provenance filter to be safe to
    # aggregate.  byte-era tokenized rows have ``wikitext_perplexity``
    # in different units from BPE-tokenized rows — they share the column
    # but are NOT comparable.  The filter excludes any row whose
    # screening_wikitext_metric_version is not ``bpe_eval_v1``.
    _METRIC_PROVENANCE_FILTER = {
        # Accept both BPE markers:
        #   - 'bpe_eval_v1' = offline BPE backfill tool (historical
        #     re-eval of pre-BPE rows)
        #   - 'screening_wikitext_v2_bpe' = live screening write after
        #     the 2026-04-27 marker bump (see _SCREENING_METRIC_VERSION
        #     in research/eval/wikitext_eval.py for the rename
        #     rationale).
        # Pre-bump 'screening_wikitext_v1' rows stay excluded — they
        # are a mix of byte-era and BPE that we can't safely
        # disambiguate without re-eval.
        "wikitext_perplexity": (
            "screening_wikitext_metric_version IN "
            "('bpe_eval_v1', 'screening_wikitext_v2_bpe')"
        ),
    }

    def get_fingerprint_metric_aggregates(self, graph_fingerprint: str) -> dict:
        """Per-metric mean / std / n / CV across all runs of a fingerprint.

        Returns ``{metric: {"mean", "std", "n", "cv"}, "_tier_cv": {...},
        "_n_runs_max"}``.  CV is std / |mean| (0 if mean=0).  std is set
        to None for n<2 (single-sample variance is meaningless).

        Rows whose tokenizer/metric provenance does not match the active
        scoring units are excluded per-metric.  Specifically,
        ``wikitext_perplexity`` is only aggregated from rows backfilled
        to BPE (``screening_wikitext_metric_version = 'bpe_eval_v1'``);
        byte-era rows stay in the database but are skipped here.

        ``_tier_cv`` aggregates per-metric CVs into three tier-level
        summaries used by the score-stability penalty:
            - "loss"  = CV(wikitext_perplexity)
            - "und"   = mean of populated understanding-metric CVs
            - "cap"   = mean of populated capability-metric CVs
        Tier-CVs are None if no metric in that tier has n>=2.
        """
        if not graph_fingerprint:
            return {}
        cols = [m for m, _ in self._STOCHASTIC_METRICS_FOR_AGG]
        select_parts: list[str] = ["COUNT(*) AS _n_total"]
        for c in cols:
            extra_filter = self._METRIC_PROVENANCE_FILTER.get(c)
            # Per-metric guard: only count / sum / avg rows whose
            # provenance is compatible with the column's units.
            guard = f"{c} IS NOT NULL"
            if extra_filter:
                guard = f"({guard}) AND ({extra_filter})"
            select_parts.append(f"SUM(CASE WHEN {guard} THEN 1 ELSE 0 END) AS n_{c}")
            select_parts.append(f"AVG(CASE WHEN {guard} THEN {c} END) AS mean_{c}")
            select_parts.append(
                f"CASE WHEN SUM(CASE WHEN {guard} THEN 1 ELSE 0 END) > 1 THEN "
                f"SQRT(MAX(0, "
                f"AVG(CASE WHEN {guard} THEN {c}*{c} END) - "
                f"AVG(CASE WHEN {guard} THEN {c} END) * "
                f"AVG(CASE WHEN {guard} THEN {c} END)"
                f")) ELSE NULL END AS std_{c}"
            )
        sql = (
            f"SELECT {', '.join(select_parts)} FROM program_results_compat "
            f"WHERE graph_fingerprint = ? "
            f"AND {self._FINGERPRINT_AGGREGATE_REASON_FILTER}"
        )
        row = self.conn.execute(sql, (graph_fingerprint,)).fetchone()
        if not row or row["_n_total"] == 0:
            return {}

        per_metric: dict[str, dict] = {}
        tier_cv_lists: dict[str, list[float]] = {"loss": [], "und": [], "cap": []}
        n_runs_max = 0
        for c, tier in self._STOCHASTIC_METRICS_FOR_AGG:
            n = int(row[f"n_{c}"] or 0)
            mean = row[f"mean_{c}"]
            std = row[f"std_{c}"]
            cv: Optional[float] = None
            if n >= 2 and mean is not None and std is not None:
                denom = abs(float(mean))
                cv = (float(std) / denom) if denom > 0 else 0.0
                if tier in tier_cv_lists:
                    tier_cv_lists[tier].append(cv)
            per_metric[c] = {"mean": mean, "std": std, "n": n, "cv": cv}
            if n > n_runs_max:
                n_runs_max = n

        tier_cv: dict[str, Optional[float]] = {}
        for tier, vals in tier_cv_lists.items():
            tier_cv[tier] = (sum(vals) / len(vals)) if vals else None

        per_metric["_tier_cv"] = tier_cv
        per_metric["_n_runs_max"] = n_runs_max
        return per_metric

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
                FROM program_results_compat
                WHERE graph_fingerprint IN ({placeholders})
                  AND {self._FINGERPRINT_AGGREGATE_REASON_FILTER}
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
            """SELECT * FROM program_results_compat
               WHERE experiment_id = ?
               ORDER BY novelty_score DESC NULLS LAST
               LIMIT ?""",
            (experiment_id, limit),
        ).fetchall()
        records = self._attach_canonical_program_scores([dict(r) for r in rows])
        existing_ids = {str(r.get("result_id")) for r in records if r.get("result_id")}
        if len(records) < limit:
            records.extend(self._validation_result_views(experiment_id, existing_ids))
        return records[:limit]

    def get_program_detail(self, result_id: str) -> Optional[Dict]:
        """Get full detail for a single program result."""
        row = self.conn.execute(
            "SELECT * FROM program_results_compat WHERE result_id = ?",
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
            f"SELECT * FROM program_results_compat WHERE result_id IN ({placeholders})",
            ids,
        ).fetchall()
        by_id = {}
        for d in self._attach_canonical_program_scores([dict(row) for row in rows]):
            d = self._parse_program_json_fields(d)
            by_id[d.get("result_id")] = d
        return [by_id.get(rid) for rid in ids]

    def _validation_result_views(
        self, experiment_id: str, existing_result_ids: set[str]
    ) -> List[Dict[str, Any]]:
        """Return source program rows annotated as members of a validation run."""
        row = self.conn.execute(
            """
            SELECT experiment_type, results_json
            FROM experiments
            WHERE experiment_id = ?
            LIMIT 1
            """,
            (experiment_id,),
        ).fetchone()
        if not row or normalize_text(row["experiment_type"]) != "validation":
            return []
        payload = self._decompress(row["results_json"]) if row["results_json"] else {}
        if not isinstance(payload, dict):
            return []
        validation_entries = [
            entry
            for entry in (payload.get("validation_results") or [])
            if isinstance(entry, dict) and entry.get("result_id")
        ]
        missing_ids = [
            str(entry["result_id"])
            for entry in validation_entries
            if str(entry["result_id"]) not in existing_result_ids
        ]
        if not missing_ids:
            return []

        source_rows = {
            row.get("result_id"): row
            for row in self.get_program_details(missing_ids)
            if row is not None
        }
        views: List[Dict[str, Any]] = []
        for entry in validation_entries:
            result_id = str(entry.get("result_id") or "")
            if result_id in existing_result_ids:
                continue
            source = source_rows.get(result_id)
            if not source:
                continue
            view = dict(source)
            view["source_experiment_id"] = source.get("experiment_id")
            view["validation_experiment_id"] = experiment_id
            view["experiment_id"] = experiment_id
            view["is_validation_result_view"] = True
            view["tier"] = entry.get("tier") or (
                "breakthrough" if entry.get("is_breakthrough") else "validation"
            )
            mapping = {
                "val_loss_ratio": "validation_loss_ratio",
                "val_baseline_ratio": "validation_baseline_ratio",
                "val_normalized_ratio": "normalized_baseline_ratio",
                "multi_seed_std": "validation_multi_seed_std",
                "robustness_score": "validation_robustness_score",
                "is_unstable": "validation_is_unstable",
                "param_efficiency": "param_efficiency",
                "novelty_score": "novelty_score",
                "novelty_confidence": "novelty_confidence",
            }
            for src_key, dst_key in mapping.items():
                if entry.get(src_key) is not None:
                    view[dst_key] = entry[src_key]
            view["validation_passed"] = int(entry.get("seeds_passed") or 0) > 0
            view["validation_is_breakthrough"] = bool(entry.get("is_breakthrough"))
            views.append(view)
        return views

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
        program_by_id: Dict[str, Dict[str, Any]] = {}
        if result_ids:
            program_by_id = prefetch_program_results(self.conn, result_ids)
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
                        dict(program_by_id.get(result_id) or row),
                        score_context,
                        is_reference,
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

    def _parse_program_json_fields(self, d: Dict[str, Any]) -> Dict[str, Any]:
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
            "rapid_screening_metrics_json",
            "data_provenance_json",
            "blimp_subtask_accuracies_json",
            "sparsity_report_json",
        )
        for json_field in json_fields:
            val = d.get(json_field)
            if isinstance(val, str) and val.startswith('{"_notebook_artifact"'):
                try:
                    val = self._resolve_artifact_text(val)
                    d[json_field] = val
                except (ValueError, FileNotFoundError, KeyError, TypeError):
                    pass
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
            JOIN program_results_compat pr ON pr.result_id = l.result_id
            WHERE pr.graph_fingerprint IS NOT NULL
            """
        ).fetchall()
        synced = 0
        seen_fp: set[str] = set()
        for row in rows:
            rid = row["result_id"]
            fp_row = self.conn.execute(
                "SELECT graph_fingerprint FROM program_results_compat WHERE result_id = ?",
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
            "JOIN program_results_compat pr ON l.result_id = pr.result_id "
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
            FROM program_results_compat p
            JOIN experiments e ON e.experiment_id = p.experiment_id
            WHERE p.stage1_passed = 1
              AND e.experiment_type IN ({placeholders})
              AND NOT EXISTS (
                    SELECT 1 FROM leaderboard l WHERE l.result_id = p.result_id
              )
              AND NOT EXISTS (
                    SELECT 1
                    FROM leaderboard l
                    JOIN program_results_compat pr2 ON pr2.result_id = l.result_id
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
            "JOIN program_results_compat pr ON pr.result_id = l.result_id "
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
            "FROM program_results_compat pr "
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
