from __future__ import annotations

"""Program result recording mixin for LabNotebook."""

import json
import time
import uuid
from typing import Any, Dict, Optional

from ._shared import LOGGER
from .notebook_leaderboard import DuplicateLeaderboardFingerprintError
from .program_provenance import merge_experiment_provenance_kwargs
from .program_writes import (
    build_dual_write_statements,
    enrich_program_result_kwargs,
    filter_known_program_result_columns,
    normalize_program_result_kwargs,
    should_record_program_result,
)


class DuplicateFingerprintError(Exception):
    """Raised when an unintentional duplicate graph fingerprint is recorded."""

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


class _ProgramResultRecordingMixin:
    def _program_result_rerun_reason(
        self,
        experiment_id: str,
        intentional_rerun_reason: Optional[str],
    ) -> Optional[str]:
        if intentional_rerun_reason:
            return intentional_rerun_reason
        experiment_type = self._experiment_type_for_id(experiment_id)
        if experiment_type == "validation":
            return "validation_promotion"
        if experiment_type == "investigation":
            return "investigation_followup"
        return None

    def _raise_if_unintentional_duplicate_program_result(
        self,
        *,
        experiment_id: str,
        graph_fingerprint: str,
        intentional_rerun_reason: Optional[str],
        kwargs: Dict[str, Any],
    ) -> None:
        if not graph_fingerprint or intentional_rerun_reason:
            return
        if not self.has_fingerprint(graph_fingerprint):
            return

        existing = self.conn.execute(
            "SELECT result_id, experiment_id FROM program_results_compat "
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

    def _canonicalize_program_result_fingerprint_fields(
        self,
        kwargs: Dict[str, Any],
    ) -> None:
        if not kwargs.get("fingerprint_json"):
            return
        fp_payload = kwargs.get("fingerprint_json")
        if isinstance(fp_payload, str):
            try:
                fp_payload = json.loads(fp_payload)
            except (TypeError, ValueError, json.JSONDecodeError):
                fp_payload = None
        if not isinstance(fp_payload, dict):
            return
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

    def _prepare_program_result_kwargs(
        self,
        *,
        experiment_id: str,
        intentional_rerun_reason: Optional[str],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        prepared = dict(kwargs)
        if intentional_rerun_reason:
            prepared.setdefault("intentional_rerun_reason", intentional_rerun_reason)
        prepared.setdefault("experiment_id", experiment_id)
        prepared = merge_experiment_provenance_kwargs(
            prepared,
            self._experiment_config_for_id(experiment_id),
        )
        self._canonicalize_program_result_fingerprint_fields(prepared)
        return enrich_program_result_kwargs(
            normalize_program_result_kwargs(prepared),
            infer_result_cohort=self._infer_result_cohort,
            infer_trust_label=self._infer_trust_label,
            infer_comparability_label=self._infer_comparability_label,
            infer_evaluation_protocol_version=self._infer_evaluation_protocol_version,
            infer_init_regime=self._infer_init_regime,
            build_data_provenance=self._build_data_provenance,
            build_failure_details=self._build_failure_details,
        )

    def _known_program_result_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        filtered_kwargs, unknown_cols = filter_known_program_result_columns(
            kwargs,
            self._get_program_results_columns(),
        )
        if unknown_cols:
            LOGGER.debug(
                "Dropping unknown program_results columns: %s",
                ", ".join(sorted(unknown_cols)),
            )
        return filtered_kwargs

    def _insert_program_result_row(
        self,
        *,
        result_id: str,
        experiment_id: str,
        graph_fingerprint: str,
        graph_json: str,
        filtered_kwargs: Dict[str, Any],
    ) -> None:
        # Atomic dual-write: graphs (UPSERT on fingerprint) + graph_runs +
        # legacy program_results all commit together. The tables stay in
        # lock-step throughout the migration's dual-write window. When Phase 5
        # retires the legacy table, drop the program_results stmt from the
        # group and switch readers off the compat view.
        statements = build_dual_write_statements(
            result_id=result_id,
            experiment_id=experiment_id,
            timestamp=time.time(),
            graph_fingerprint=graph_fingerprint,
            graph_json=graph_json,
            filtered_kwargs=filtered_kwargs,
        )
        self._submit_grouped_write(statements)

    def _sync_program_result_side_effects(
        self,
        *,
        result_id: str,
        experiment_id: str,
        graph_fingerprint: str,
        graph_json: str,
        intentional_rerun_reason: Optional[str],
        filtered_kwargs: Dict[str, Any],
    ) -> None:
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
            str(intentional_rerun_reason or "").strip()
            or experiment_type not in SCREENING_EXPERIMENT_TYPES
            or not bool(filtered_kwargs.get("stage1_passed"))
        ):
            return
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
            self._sync_fingerprint_leaderboard(result_id)

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
        """Record results for a single synthesized program."""
        # Tag kwargs with the parent experiment's type so the post-S1 guardrail
        # can require investigation-tier metrics on investigation writes. The
        # tag itself is consumed by the guard and stripped before the row is
        # filtered for the SQL columns (see _known_program_result_kwargs).
        if "_experiment_type" not in kwargs:
            try:
                exp_type = self._experiment_type_for_id(experiment_id)
            except Exception:  # noqa: BLE001
                exp_type = None
            if exp_type:
                kwargs["_experiment_type"] = exp_type
        if not should_record_program_result(
            graph_fingerprint=graph_fingerprint,
            kwargs=kwargs,
            bypass_quality_gate=bypass_quality_gate,
            logger=LOGGER,
        ):
            return ""

        intentional_rerun_reason = self._program_result_rerun_reason(
            experiment_id,
            intentional_rerun_reason,
        )
        self._raise_if_unintentional_duplicate_program_result(
            experiment_id=experiment_id,
            graph_fingerprint=graph_fingerprint,
            intentional_rerun_reason=intentional_rerun_reason,
            kwargs=kwargs,
        )
        if not result_id:
            result_id = str(uuid.uuid4())[:12]
        kwargs = self._prepare_program_result_kwargs(
            experiment_id=experiment_id,
            intentional_rerun_reason=intentional_rerun_reason,
            kwargs=kwargs,
        )
        filtered_kwargs = self._known_program_result_kwargs(kwargs)
        filtered_kwargs = self._maybe_externalize_program_result_artifacts(
            result_id=result_id,
            filtered_kwargs=filtered_kwargs,
        )
        self._insert_program_result_row(
            result_id=result_id,
            experiment_id=experiment_id,
            graph_fingerprint=graph_fingerprint,
            graph_json=graph_json,
            filtered_kwargs=filtered_kwargs,
        )
        self._sync_program_result_side_effects(
            result_id=result_id,
            experiment_id=experiment_id,
            graph_fingerprint=graph_fingerprint,
            graph_json=graph_json,
            intentional_rerun_reason=intentional_rerun_reason,
            filtered_kwargs=filtered_kwargs,
        )
        return result_id
