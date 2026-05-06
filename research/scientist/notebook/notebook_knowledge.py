from __future__ import annotations

"""Auto-extracted mixin for LabNotebook."""

import json
import math
import sqlite3
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from ._shared import LOGGER, infer_insight_identity, sanitize_for_db


class _KnowledgeMixin:
    """Knowledge operations for the Lab Notebook."""

    __slots__ = ()

    @staticmethod
    def _decode_knowledge_json_field(data: Dict[str, Any], key: str) -> None:
        raw = data.get(key)
        if not raw:
            return
        try:
            data[key] = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            pass

    @classmethod
    def _hydrate_insight_row(cls, row: Any) -> Dict[str, Any]:
        data = dict(row)
        cls._decode_knowledge_json_field(data, "evidence_json")
        return data

    @classmethod
    def _hydrate_selection_insight_trial_row(cls, row: Any) -> Dict[str, Any]:
        item = dict(row)
        for key in ("insight_ids_json", "chosen_result_ids_json", "metadata_json"):
            cls._decode_knowledge_json_field(item, key)
        return item

    @classmethod
    def _hydrate_selection_family_trial_row(cls, row: Any) -> Dict[str, Any]:
        item = dict(row)
        for key in ("chosen_result_ids_json", "metadata_json"):
            cls._decode_knowledge_json_field(item, key)
        return item

    @classmethod
    def _hydrate_followup_task_row(cls, row: Any) -> Dict[str, Any]:
        item = dict(row)
        for key in (
            "result_ids_json",
            "config_json",
            "evidence_pack_json",
            "priority_reasons_json",
            "metadata_json",
        ):
            cls._decode_knowledge_json_field(item, key)
        if "config_json" in item and "config" not in item:
            item["config"] = item.get("config_json")
        if "evidence_pack_json" in item and "evidence_pack" not in item:
            item["evidence_pack"] = item.get("evidence_pack_json")
        if "priority_reasons_json" in item and "priority_reasons" not in item:
            item["priority_reasons"] = item.get("priority_reasons_json")
        if "metadata_json" in item and "metadata" not in item:
            item["metadata"] = item.get("metadata_json")
        return item

    @classmethod
    def _hydrate_threshold_calibration_row(cls, row: Any) -> Dict[str, Any]:
        item = dict(row)
        for key in ("metrics_json", "metadata_json"):
            cls._decode_knowledge_json_field(item, key)
        if "metrics_json" in item and "metrics" not in item:
            item["metrics"] = item.get("metrics_json")
        if "metadata_json" in item and "metadata" not in item:
            item["metadata"] = item.get("metadata_json")
        return item

    @staticmethod
    def _apply_knowledge_confidence_fields(data: Dict[str, Any]) -> Dict[str, Any]:
        base_conf = float(data.get("confidence") or 0.5)
        validated = int(data.get("times_validated") or 0)
        # Validation bonus saturates; repeated confirmations help but are capped.
        val_bonus = min(0.18, 0.05 * math.log1p(max(validated - 1, 0)))
        effective_conf = min(0.95, max(0.0, base_conf) + val_bonus)
        data["effective_confidence"] = round(effective_conf, 4)
        data["validation_bonus"] = round(val_bonus, 4)
        data["confidence_capped"] = effective_conf >= 0.95
        return data

    @classmethod
    def _hydrate_knowledge_row(cls, row: Any) -> Dict[str, Any]:
        data = dict(row)
        cls._apply_knowledge_confidence_fields(data)
        cls._decode_knowledge_json_field(data, "supporting_evidence")
        return data

    # ── Insights ──

    def record_insight(
        self,
        category: str,
        content: str,
        experiment_id: Optional[str] = None,
        confidence: float = 0.5,
        evidence: Optional[str] = None,
        insight_type: Optional[str] = None,
        subject_key: Optional[str] = None,
        semantic_key: Optional[str] = None,
        alpha: float = 1.0,
        beta_: float = 1.0,
        display_only: bool = False,
        insight_level: str = "op",
        evidence_json: Optional[Dict] = None,
    ) -> str:
        """Record an insight/learning, superseding active semantic duplicates.

        If category is 'failure_mode', display_only is forced to True.
        """
        inferred_type, inferred_subject, inferred_semantic = infer_insight_identity(
            category,
            content,
        )
        insight_type = str(insight_type or inferred_type).strip() or inferred_type
        subject_key = str(subject_key or inferred_subject).strip() or inferred_subject
        semantic_key = (
            str(semantic_key or inferred_semantic).strip() or inferred_semantic
        )

        # Hard gate: failure_mode insights are always display-only
        if category == "failure_mode":
            display_only = True

        # Compute confidence from Bayesian posterior.
        # If caller passed explicit confidence but left alpha/beta at defaults,
        # derive alpha/beta from the desired confidence so they stay consistent.
        alpha = max(0.01, float(alpha))
        beta_ = max(0.01, float(beta_))
        if confidence != 0.5 and alpha == 1.0 and beta_ == 1.0:
            # Solve: confidence = alpha / (alpha + beta_) with alpha + beta_ = 2
            alpha = max(0.01, 2.0 * confidence)
            beta_ = max(0.01, 2.0 * (1.0 - confidence))
        confidence = alpha / (alpha + beta_)

        insight_id = str(uuid.uuid4())[:12]
        evidence_json_str = json.dumps(evidence_json) if evidence_json else None
        insert_sql = """INSERT INTO insights
            (insight_id, timestamp, experiment_id, category, insight_type,
             subject_key, semantic_key, content, confidence, supporting_evidence,
             alpha, beta_, display_only, insight_level, n_predictions, n_correct,
             evidence_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?)"""
        insert_params = (
            insight_id,
            time.time(),
            experiment_id,
            category,
            insight_type,
            subject_key,
            semantic_key,
            content,
            confidence,
            evidence,
            alpha,
            beta_,
            1 if display_only else 0,
            insight_level,
            evidence_json_str,
        )
        try:
            if semantic_key:
                existing = self.conn.execute(
                    """SELECT insight_id FROM insights
                       WHERE status = 'active' AND semantic_key = ?
                       ORDER BY confidence DESC, timestamp DESC""",
                    (semantic_key,),
                ).fetchall()
                for row in existing:
                    self.conn.execute(
                        "UPDATE insights SET status = 'superseded' WHERE insight_id = ?",
                        (row["insight_id"],),
                    )

            self.conn.execute(insert_sql, insert_params)
        except sqlite3.IntegrityError:
            if semantic_key:
                self.conn.execute(
                    "UPDATE insights SET status = 'superseded' WHERE status = 'active' AND semantic_key = ?",
                    (semantic_key,),
                )
                self.conn.execute(insert_sql, insert_params)
            else:
                raise
        except sqlite3.OperationalError as exc:
            LOGGER.warning(
                "Insight write failed for experiment %s; continuing without notebook persistence: %s",
                experiment_id or "unscoped",
                exc,
            )
            return insight_id
        self._maybe_commit()
        return insight_id

    def supersede_insight(self, insight_id: str) -> None:
        """Mark an insight as superseded (replaced by a newer version)."""
        self.conn.execute(
            "UPDATE insights SET status = 'superseded' WHERE insight_id = ?",
            (insight_id,),
        )
        self._maybe_commit()

    def get_insights(
        self,
        category: Optional[str] = None,
        status: str = "active",
        limit: int = 50,
        exclude_display_only: bool = False,
        insight_level: Optional[str] = None,
    ) -> List[Dict]:
        query = "SELECT * FROM insights WHERE status = ?"
        params: List[Any] = [status]
        if category:
            query += " AND category = ?"
            params.append(category)
        if exclude_display_only:
            query += " AND display_only = 0"
        if insight_level:
            query += " AND insight_level = ?"
            params.append(insight_level)
        query += " ORDER BY confidence DESC, timestamp DESC LIMIT ?"
        params.append(limit)
        try:
            cursor = self.conn.execute(query, params)
        except sqlite3.OperationalError as exc:
            LOGGER.warning(
                "Insight query failed; returning empty results: %s",
                exc,
            )
            return []
        return [self._hydrate_insight_row(row) for row in cursor]

    def update_insight_bayesian(self, insight_id: str, success: bool) -> None:
        """Increment alpha (success) or beta_ (failure). Recompute confidence.

        This is the single code path for Bayesian updates to insight confidence.
        Confidence = posterior mean of Beta(alpha, beta_).
        """
        col = "alpha" if success else "beta_"
        self.conn.execute(
            f"""UPDATE insights SET
                {col} = {col} + 1,
                confidence = (alpha + CASE WHEN ? THEN 1 ELSE 0 END)
                    / (alpha + beta_ + 1),
                n_predictions = n_predictions + 1,
                n_correct = n_correct + CASE WHEN ? THEN 1 ELSE 0 END
            WHERE insight_id = ?""",
            (int(success), int(success), insight_id),
        )
        self._maybe_commit()

    def record_selection_insight_trial(
        self,
        decision_id: str,
        context: str,
        insight_ids: List[str],
        chosen_result_ids: List[str],
        source_experiment_id: Optional[str] = None,
    ) -> str:
        """Record one insight-bundle trial tied to a selection decision."""
        trial_id = str(uuid.uuid4())[:12]
        now = time.time()
        cleaned_insights = sorted(
            {str(i).strip() for i in (insight_ids or []) if str(i).strip()}
        )
        cleaned_results = sorted(
            {str(r).strip() for r in (chosen_result_ids or []) if str(r).strip()}
        )
        self.conn.execute(
            """INSERT INTO selection_insight_trials
               (trial_id, decision_id, timestamp, context, source_experiment_id,
                insight_ids_json, chosen_result_ids_json, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')""",
            (
                trial_id,
                decision_id,
                now,
                context or "",
                source_experiment_id,
                json.dumps(cleaned_insights),
                json.dumps(cleaned_results),
            ),
        )
        self._maybe_commit()
        return trial_id

    def get_pending_selection_insight_trials(
        self,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return unresolved insight-bundle trials."""
        rows = self.conn.execute(
            """SELECT * FROM selection_insight_trials
               WHERE status = 'pending'
               ORDER BY timestamp ASC
               LIMIT ?""",
            (max(1, int(limit)),),
        )
        return [self._hydrate_selection_insight_trial_row(row) for row in rows]

    def record_selection_family_trial(
        self,
        decision_id: str,
        context: str,
        family: str,
        chosen_result_ids: List[str],
        source_experiment_id: Optional[str] = None,
    ) -> str:
        """Record one family-level selection trial for deferred outcome learning."""
        trial_id = str(uuid.uuid4())[:12]
        now = time.time()
        family_name = str(family or "Unknown").strip() or "Unknown"
        cleaned_results = sorted(
            {str(r).strip() for r in (chosen_result_ids or []) if str(r).strip()}
        )
        self.conn.execute(
            """INSERT INTO selection_family_trials
               (trial_id, decision_id, timestamp, context, source_experiment_id,
                family, chosen_result_ids_json, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')""",
            (
                trial_id,
                decision_id,
                now,
                context or "",
                source_experiment_id,
                family_name,
                json.dumps(cleaned_results),
            ),
        )
        self._maybe_commit()
        return trial_id

    def get_pending_selection_family_trials(
        self,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return unresolved family-level trials."""
        rows = self.conn.execute(
            """SELECT * FROM selection_family_trials
               WHERE status = 'pending'
               ORDER BY timestamp ASC
               LIMIT ?""",
            (max(1, int(limit)),),
        )
        return [self._hydrate_selection_family_trial_row(row) for row in rows]

    def resolve_selection_family_trial(
        self,
        trial_id: str,
        reward: float,
        outcome: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Resolve a family-level trial and update the family reward stats."""
        row = self.conn.execute(
            "SELECT * FROM selection_family_trials WHERE trial_id = ?",
            (trial_id,),
        ).fetchone()
        if row is None:
            return
        trial = dict(row)
        if str(trial.get("status") or "") == "resolved":
            return
        now = time.time()
        reward_value = float(reward or 0.0)
        outcome_text = str(outcome or "inconclusive").strip() or "inconclusive"
        self.conn.execute(
            """UPDATE selection_family_trials
               SET status = 'resolved',
                   reward = ?,
                   outcome = ?,
                   resolved_timestamp = ?,
                   metadata_json = ?
               WHERE trial_id = ?""",
            (
                reward_value,
                outcome_text,
                now,
                json.dumps(metadata or {}),
                trial_id,
            ),
        )
        self.update_selection_family_stats(
            str(trial.get("family") or "Unknown"),
            reward=reward_value,
        )
        self._maybe_commit()

    def resolve_selection_insight_trial(
        self,
        trial_id: str,
        reward: float,
        outcome: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Resolve a trial and update pairwise insight interaction stats."""
        row = self.conn.execute(
            "SELECT * FROM selection_insight_trials WHERE trial_id = ?",
            (trial_id,),
        ).fetchone()
        if row is None:
            return
        trial = dict(row)
        if str(trial.get("status") or "") == "resolved":
            return
        now = time.time()
        reward_value = float(reward or 0.0)
        outcome_text = str(outcome or "inconclusive").strip() or "inconclusive"
        self.conn.execute(
            """UPDATE selection_insight_trials
               SET status = 'resolved',
                   reward = ?,
                   outcome = ?,
                   resolved_timestamp = ?,
                   metadata_json = ?
               WHERE trial_id = ?""",
            (
                reward_value,
                outcome_text,
                now,
                json.dumps(metadata or {}),
                trial_id,
            ),
        )

        try:
            insight_ids = json.loads(trial.get("insight_ids_json") or "[]")
        except (TypeError, json.JSONDecodeError):
            insight_ids = []
        cleaned = sorted(
            {str(i).strip() for i in (insight_ids or []) if str(i).strip()}
        )
        if not cleaned:
            self._maybe_commit()
            return

        # Track singleton and pair interactions. Singleton uses (id, id).
        pairs: List[Tuple[str, str]] = []
        for insight_id in cleaned:
            pairs.append((insight_id, insight_id))
        for i in range(len(cleaned)):
            for j in range(i + 1, len(cleaned)):
                a, b = cleaned[i], cleaned[j]
                if a > b:
                    a, b = b, a
                pairs.append((a, b))

        supported_inc = 1 if outcome_text == "supported" else 0
        not_supported_inc = 1 if outcome_text == "not_supported" else 0
        for insight_a, insight_b in pairs:
            self.conn.execute(
                """INSERT INTO selection_insight_interactions
                   (insight_a, insight_b, n_trials, n_supported, n_not_supported,
                    cumulative_reward, mean_reward, last_reward, last_outcome, last_updated)
                   VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(insight_a, insight_b) DO UPDATE SET
                     n_trials = n_trials + 1,
                     n_supported = n_supported + excluded.n_supported,
                     n_not_supported = n_not_supported + excluded.n_not_supported,
                     cumulative_reward = cumulative_reward + excluded.last_reward,
                     mean_reward = (cumulative_reward + excluded.last_reward) / (n_trials + 1),
                     last_reward = excluded.last_reward,
                     last_outcome = excluded.last_outcome,
                     last_updated = excluded.last_updated""",
                (
                    insight_a,
                    insight_b,
                    supported_inc,
                    not_supported_inc,
                    reward_value,
                    reward_value,
                    reward_value,
                    outcome_text,
                    now,
                ),
            )
        self._maybe_commit()

    # ── Follow-up task queue ──

    def _canonical_followup_result_ids(
        self, result_ids: List[str]
    ) -> tuple[list[str], list[str]]:
        requested = [str(r).strip() for r in (result_ids or []) if str(r).strip()]
        canonical_result_ids = []
        for rid in requested:
            canonical = str(self.resolve_canonical_result_id(rid) or rid).strip()
            if canonical:
                canonical_result_ids.append(canonical)
        return requested, sorted(set(canonical_result_ids))

    @staticmethod
    def _followup_metadata(
        metadata: Optional[Dict[str, Any]],
        requested_result_ids: list[str],
        cleaned_result_ids: list[str],
    ) -> Dict[str, Any]:
        task_metadata = dict(metadata or {})
        if cleaned_result_ids != sorted(set(requested_result_ids)):
            task_metadata.setdefault(
                "requested_result_ids", sorted(set(requested_result_ids))
            )
            task_metadata.setdefault(
                "canonicalized_result_ids", list(cleaned_result_ids)
            )
        return task_metadata

    @staticmethod
    def _followup_json(payload: Optional[Dict[str, Any]]) -> Optional[str]:
        return json.dumps(sanitize_for_db(payload or {})) if payload else None

    def _find_active_followup_task(self, stage: str, result_ids_json: str):
        return self.conn.execute(
            """SELECT task_id FROM followup_tasks
               WHERE stage = ?
                 AND result_ids_json = ?
                 AND status IN ('queued', 'running')
               ORDER BY timestamp DESC
               LIMIT 1""",
            (stage, result_ids_json),
        ).fetchone()

    def _update_followup_task(
        self,
        task_id: str,
        *,
        source_context: Optional[str],
        source_decision_id: Optional[str],
        source_experiment_id: Optional[str],
        hypothesis: str,
        config: Optional[Dict[str, Any]],
        evidence_pack: Optional[Dict[str, Any]],
        priority_score: float,
        priority_reasons: Optional[Dict[str, Any]],
        task_metadata: Dict[str, Any],
    ) -> str:
        self.conn.execute(
            """UPDATE followup_tasks
               SET source_context = COALESCE(?, source_context),
                   source_decision_id = COALESCE(?, source_decision_id),
                   source_experiment_id = COALESCE(?, source_experiment_id),
                   hypothesis = COALESCE(?, hypothesis),
                   config_json = COALESCE(?, config_json),
                   evidence_pack_json = COALESCE(?, evidence_pack_json),
                   priority_score = MAX(priority_score, ?),
                   priority_reasons_json = COALESCE(?, priority_reasons_json),
                   metadata_json = COALESCE(?, metadata_json)
               WHERE task_id = ?""",
            (
                source_context,
                source_decision_id,
                source_experiment_id,
                hypothesis or None,
                self._followup_json(config),
                self._followup_json(evidence_pack),
                float(priority_score or 0.0),
                self._followup_json(priority_reasons),
                self._followup_json(task_metadata),
                task_id,
            ),
        )
        self._maybe_commit()
        return task_id

    def _insert_followup_task(
        self,
        *,
        stage: str,
        result_ids_json: str,
        hypothesis: str,
        config: Optional[Dict[str, Any]],
        evidence_pack: Optional[Dict[str, Any]],
        source_context: Optional[str],
        source_decision_id: Optional[str],
        source_experiment_id: Optional[str],
        priority_score: float,
        priority_reasons: Optional[Dict[str, Any]],
        task_metadata: Dict[str, Any],
    ) -> str:
        task_id = str(uuid.uuid4())[:12]
        self.conn.execute(
            """INSERT INTO followup_tasks
               (task_id, timestamp, stage, status, source_context,
                source_decision_id, source_experiment_id, result_ids_json,
                hypothesis, config_json, evidence_pack_json, priority_score,
                priority_reasons_json, metadata_json)
               VALUES (?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                task_id,
                time.time(),
                stage,
                source_context or "",
                source_decision_id,
                source_experiment_id,
                result_ids_json,
                hypothesis or "",
                self._followup_json(config),
                self._followup_json(evidence_pack),
                float(priority_score or 0.0),
                self._followup_json(priority_reasons),
                self._followup_json(task_metadata),
            ),
        )
        self._maybe_commit()
        return task_id

    def enqueue_followup_task(
        self,
        *,
        stage: str,
        result_ids: List[str],
        hypothesis: str,
        config: Optional[Dict[str, Any]] = None,
        evidence_pack: Optional[Dict[str, Any]] = None,
        source_context: Optional[str] = None,
        source_decision_id: Optional[str] = None,
        source_experiment_id: Optional[str] = None,
        priority_score: float = 0.0,
        priority_reasons: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        bypass_dedup: bool = False,
    ) -> str:
        """Enqueue a follow-up task, deduplicating active duplicates.

        ``bypass_dedup=True`` forces an insert even if an active task
        already exists for this stage+result_ids.  Use this for
        explicitly-batched reruns (e.g. score-stability confirmation
        rounds) where multiple in-flight tasks for the same fingerprint
        is the intended behavior — the runner will claim them
        sequentially.
        """
        cleaned_stage = str(stage or "").strip().lower()
        if not cleaned_stage:
            raise ValueError("stage is required")
        requested_result_ids, cleaned_result_ids = self._canonical_followup_result_ids(
            result_ids
        )
        if not cleaned_result_ids:
            raise ValueError("result_ids is required")
        result_ids_json = json.dumps(cleaned_result_ids)
        task_metadata = self._followup_metadata(
            metadata, requested_result_ids, cleaned_result_ids
        )
        existing = (
            None
            if bypass_dedup
            else self._find_active_followup_task(cleaned_stage, result_ids_json)
        )
        if existing is not None:
            return self._update_followup_task(
                str(existing["task_id"]),
                source_context=source_context,
                source_decision_id=source_decision_id,
                source_experiment_id=source_experiment_id,
                hypothesis=hypothesis,
                config=config,
                evidence_pack=evidence_pack,
                priority_score=priority_score,
                priority_reasons=priority_reasons,
                task_metadata=task_metadata,
            )
        return self._insert_followup_task(
            stage=cleaned_stage,
            result_ids_json=result_ids_json,
            hypothesis=hypothesis,
            config=config,
            evidence_pack=evidence_pack,
            source_context=source_context,
            source_decision_id=source_decision_id,
            source_experiment_id=source_experiment_id,
            priority_score=priority_score,
            priority_reasons=priority_reasons,
            task_metadata=task_metadata,
        )

    def get_followup_tasks(
        self,
        *,
        stage: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM followup_tasks WHERE 1=1"
        params: List[Any] = []
        if stage:
            query += " AND stage = ?"
            params.append(str(stage).strip().lower())
        if status:
            query += " AND status = ?"
            params.append(str(status).strip().lower())
        query += " ORDER BY priority_score DESC, timestamp ASC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self.conn.execute(query, params)
        return [self._hydrate_followup_task_row(row) for row in rows]

    def _claim_followup_task_row(self, row: Any) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        task_id = str(row["task_id"])
        self.conn.execute(
            """UPDATE followup_tasks
               SET status = 'running',
                   started_timestamp = ?
               WHERE task_id = ?
                 AND status = 'queued'""",
            (time.time(), task_id),
        )
        updated = self.conn.execute(
            "SELECT * FROM followup_tasks WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        self._maybe_commit()
        if updated is None:
            return None
        return self._hydrate_followup_task_row(updated)

    def claim_followup_task(self, stage: str) -> Optional[Dict[str, Any]]:
        """Claim the highest-priority queued follow-up task for a stage."""
        cleaned_stage = str(stage or "").strip().lower()
        if not cleaned_stage:
            return None
        row = self.conn.execute(
            """SELECT * FROM followup_tasks
               WHERE stage = ?
                 AND status = 'queued'
               ORDER BY priority_score DESC, timestamp ASC
               LIMIT 1""",
            (cleaned_stage,),
        ).fetchone()
        return self._claim_followup_task_row(row)

    def claim_followup_task_by_id(
        self,
        task_id: str,
        *,
        stage: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Claim a specific queued follow-up task by id."""
        cleaned_task_id = str(task_id or "").strip()
        if not cleaned_task_id:
            return None
        params: List[Any] = [cleaned_task_id]
        stage_clause = ""
        cleaned_stage = str(stage or "").strip().lower()
        if cleaned_stage:
            stage_clause = " AND stage = ?"
            params.append(cleaned_stage)
        row = self.conn.execute(
            f"""SELECT * FROM followup_tasks
                WHERE task_id = ?
                  AND status = 'queued'
                  {stage_clause}
                LIMIT 1""",
            tuple(params),
        ).fetchone()
        return self._claim_followup_task_row(row)

    def complete_followup_task(
        self,
        task_id: str,
        *,
        outcome: str = "launched",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.conn.execute(
            """UPDATE followup_tasks
               SET status = 'completed',
                   outcome = ?,
                   completed_timestamp = ?,
                   metadata_json = COALESCE(?, metadata_json)
               WHERE task_id = ?""",
            (
                str(outcome or "completed"),
                time.time(),
                json.dumps(sanitize_for_db(metadata or {})) if metadata else None,
                task_id,
            ),
        )
        self._maybe_commit()

    def requeue_followup_task(
        self,
        task_id: str,
        *,
        outcome: str = "requeued",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.conn.execute(
            """UPDATE followup_tasks
               SET status = 'queued',
                   outcome = ?,
                   started_timestamp = NULL,
                   metadata_json = COALESCE(?, metadata_json)
               WHERE task_id = ?""",
            (
                str(outcome or "requeued"),
                json.dumps(sanitize_for_db(metadata or {})) if metadata else None,
                task_id,
            ),
        )
        self._maybe_commit()

    # ── Threshold calibration history ──

    def record_threshold_calibration(
        self,
        *,
        context: str,
        tier_clause: str,
        floor: float,
        percentile: float,
        selected_threshold: float,
        fallback_threshold: Optional[float] = None,
        sample_size: int = 0,
        labeled_size: int = 0,
        positive_count: int = 0,
        negative_count: int = 0,
        objective: Optional[float] = None,
        metrics: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        calibration_id = str(uuid.uuid4())[:12]
        now = time.time()
        previous = self.conn.execute(
            """SELECT selected_threshold FROM threshold_calibrations
               WHERE context = ?
               ORDER BY timestamp DESC
               LIMIT 1""",
            (str(context or ""),),
        ).fetchone()
        threshold_delta = None
        if previous is not None and previous["selected_threshold"] is not None:
            threshold_delta = float(selected_threshold) - float(
                previous["selected_threshold"]
            )
        self.conn.execute(
            """INSERT INTO threshold_calibrations
               (calibration_id, timestamp, context, tier_clause, floor, percentile,
                selected_threshold, fallback_threshold, sample_size, labeled_size,
                positive_count, negative_count, objective, threshold_delta,
                metrics_json, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                calibration_id,
                now,
                str(context or ""),
                str(tier_clause or ""),
                float(floor or 0.0),
                float(percentile or 0.0),
                float(selected_threshold),
                float(fallback_threshold) if fallback_threshold is not None else None,
                int(sample_size or 0),
                int(labeled_size or 0),
                int(positive_count or 0),
                int(negative_count or 0),
                float(objective) if objective is not None else None,
                threshold_delta,
                json.dumps(sanitize_for_db(metrics or {})) if metrics else None,
                json.dumps(sanitize_for_db(metadata or {})) if metadata else None,
            ),
        )
        self._maybe_commit()
        return calibration_id

    def get_threshold_calibrations(
        self,
        *,
        context: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM threshold_calibrations WHERE 1=1"
        params: List[Any] = []
        if context:
            query += " AND context = ?"
            params.append(str(context or ""))
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(max(1, int(limit)))
        rows = self.conn.execute(query, params)
        return [self._hydrate_threshold_calibration_row(row) for row in rows]

    def get_selection_insight_interactions(
        self,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return learned insight interaction stats sorted by confidence/reward."""
        rows = self.conn.execute(
            """SELECT * FROM selection_insight_interactions
               ORDER BY n_trials DESC, mean_reward DESC
               LIMIT ?""",
            (max(1, int(limit)),),
        )
        return [dict(row) for row in rows]

    # ── Knowledge Base ──

    def add_knowledge(
        self,
        category: str,
        title: str,
        content: str,
        evidence: Optional[List[str]] = None,
        confidence: float = 0.5,
    ) -> str:
        """Add a knowledge base entry. Returns entry_id."""
        entry_id = str(uuid.uuid4())[:12]
        now = time.time()
        self.conn.execute(
            """INSERT INTO knowledge_base
            (entry_id, timestamp, category, title, content, confidence,
             supporting_evidence, times_validated, last_validated, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, 'active')""",
            (
                entry_id,
                now,
                category,
                title,
                content,
                confidence,
                json.dumps(evidence) if evidence else None,
                now,
            ),
        )
        self._maybe_commit()
        return entry_id

    def get_knowledge(self, category: Optional[str] = None) -> List[Dict]:
        """Get knowledge base entries, optionally filtered by category."""
        query = "SELECT * FROM knowledge_base WHERE status = 'active'"
        params: List[Any] = []
        if category:
            query += " AND category = ?"
            params.append(category)
        query += " ORDER BY timestamp DESC"
        try:
            cursor = self.conn.execute(query, params)
        except sqlite3.OperationalError as exc:
            LOGGER.warning(
                "Knowledge query failed; returning empty results: %s",
                exc,
            )
            return []
        results = [self._hydrate_knowledge_row(row) for row in cursor]
        results.sort(
            key=lambda row: (
                float(row.get("effective_confidence") or 0.0),
                int(row.get("times_validated") or 0),
                float(row.get("timestamp") or 0.0),
            ),
            reverse=True,
        )
        return results

    def validate_knowledge(self, entry_id: str) -> None:
        """Increment times_validated and update last_validated."""
        now = time.time()
        self.conn.execute(
            """UPDATE knowledge_base SET
                times_validated = times_validated + 1,
                last_validated = ?
            WHERE entry_id = ?""",
            (now, entry_id),
        )
        self._maybe_commit()

    def search_knowledge(self, query: str) -> List[Dict]:
        """Simple LIKE search on title + content."""
        pattern = f"%{query}%"
        cursor = self.conn.execute(
            """SELECT * FROM knowledge_base
               WHERE status = 'active'
               AND (title LIKE ? OR content LIKE ?)
               ORDER BY timestamp DESC""",
            (pattern, pattern),
        )
        results = [self._hydrate_knowledge_row(row) for row in cursor]
        results.sort(
            key=lambda row: (
                float(row.get("effective_confidence") or 0.0),
                int(row.get("times_validated") or 0),
                float(row.get("timestamp") or 0.0),
            ),
            reverse=True,
        )
        return results
