from __future__ import annotations
"""Auto-extracted mixin for LabNotebook."""

import json
import math
import sqlite3
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from ._shared import infer_insight_identity


class _KnowledgeMixin:
    """Knowledge operations for the Lab Notebook."""
    __slots__ = ()

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
        semantic_key = str(semantic_key or inferred_semantic).strip() or inferred_semantic

        # Hard gate: failure_mode insights are always display-only
        if category == "failure_mode":
            display_only = True

        # Compute confidence from Bayesian posterior
        alpha = max(0.01, float(alpha))
        beta_ = max(0.01, float(beta_))
        confidence = alpha / (alpha + beta_)

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
        rows = self.conn.execute(query, params).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            # Parse evidence_json if present
            if d.get("evidence_json"):
                try:
                    d["evidence_json"] = json.loads(d["evidence_json"])
                except (json.JSONDecodeError, TypeError):
                    pass
            results.append(d)
        return results


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
        cleaned_insights = sorted({
            str(i).strip() for i in (insight_ids or []) if str(i).strip()
        })
        cleaned_results = sorted({
            str(r).strip() for r in (chosen_result_ids or []) if str(r).strip()
        })
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
        ).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            for key in ("insight_ids_json", "chosen_result_ids_json", "metadata_json"):
                raw = item.get(key)
                if not raw:
                    continue
                try:
                    item[key] = json.loads(raw)
                except (TypeError, json.JSONDecodeError):
                    pass
            out.append(item)
        return out


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
        cleaned = sorted({
            str(i).strip() for i in (insight_ids or []) if str(i).strip()
        })
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
        ).fetchall()
        return [dict(r) for r in rows]


    # ── Knowledge Base ──

    def add_knowledge(self, category: str, title: str, content: str,
                      evidence: Optional[List[str]] = None,
                      confidence: float = 0.5) -> str:
        """Add a knowledge base entry. Returns entry_id."""
        entry_id = str(uuid.uuid4())[:12]
        now = time.time()
        self.conn.execute(
            """INSERT INTO knowledge_base
            (entry_id, timestamp, category, title, content, confidence,
             supporting_evidence, times_validated, last_validated, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, 'active')""",
            (entry_id, now, category, title, content, confidence,
             json.dumps(evidence) if evidence else None, now),
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
        rows = self.conn.execute(query, params).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            base_conf = float(d.get("confidence") or 0.5)
            validated = int(d.get("times_validated") or 0)
            # Validation bonus saturates; repeated confirmations help but are capped.
            val_bonus = min(0.18, 0.05 * math.log1p(max(validated - 1, 0)))
            effective_conf = min(0.95, max(0.0, base_conf) + val_bonus)
            d["effective_confidence"] = round(effective_conf, 4)
            d["validation_bonus"] = round(val_bonus, 4)
            d["confidence_capped"] = effective_conf >= 0.95
            if d.get("supporting_evidence"):
                try:
                    d["supporting_evidence"] = json.loads(d["supporting_evidence"])
                except (json.JSONDecodeError, TypeError):
                    pass
            results.append(d)
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
        rows = self.conn.execute(
            """SELECT * FROM knowledge_base
               WHERE status = 'active'
               AND (title LIKE ? OR content LIKE ?)
               ORDER BY timestamp DESC""",
            (pattern, pattern),
        ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            base_conf = float(d.get("confidence") or 0.5)
            validated = int(d.get("times_validated") or 0)
            val_bonus = min(0.18, 0.05 * math.log1p(max(validated - 1, 0)))
            effective_conf = min(0.95, max(0.0, base_conf) + val_bonus)
            d["effective_confidence"] = round(effective_conf, 4)
            d["validation_bonus"] = round(val_bonus, 4)
            d["confidence_capped"] = effective_conf >= 0.95
            if d.get("supporting_evidence"):
                try:
                    d["supporting_evidence"] = json.loads(d["supporting_evidence"])
                except (json.JSONDecodeError, TypeError):
                    pass
            results.append(d)
        results.sort(
            key=lambda row: (
                float(row.get("effective_confidence") or 0.0),
                int(row.get("times_validated") or 0),
                float(row.get("timestamp") or 0.0),
            ),
            reverse=True,
        )
        return results

