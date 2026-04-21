from __future__ import annotations

"""Auto-extracted mixin for LabNotebook."""

import json
import time
import uuid
from typing import Any, Dict, List, Optional


class _CampaignsMixin:
    """Campaigns operations for the Lab Notebook."""

    __slots__ = ()

    @staticmethod
    def _hydrate_campaign_hypothesis_row(row: Any) -> Dict[str, Any]:
        hypothesis = dict(row)
        raw_meta = hypothesis.get("metadata_json")
        if isinstance(raw_meta, str) and raw_meta.strip():
            try:
                parsed = json.loads(raw_meta)
                hypothesis["metadata"] = parsed if isinstance(parsed, dict) else {}
            except (json.JSONDecodeError, TypeError):
                hypothesis["metadata"] = {}
        else:
            hypothesis["metadata"] = {}
        return hypothesis

    # ── Metrics ──

    def log_metric(
        self,
        metric_name: str,
        value: float,
        experiment_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        """Log a time-series metric."""
        self._submit_write(
            """INSERT INTO metrics_log
            (timestamp, experiment_id, metric_name, metric_value, metadata_json)
            VALUES (?, ?, ?, ?, ?)""",
            (
                time.time(),
                experiment_id,
                metric_name,
                value,
                json.dumps(metadata) if metadata else None,
            ),
        )

    def get_metrics(
        self, metric_name: str, experiment_id: Optional[str] = None, limit: int = 1000
    ) -> List[Dict]:
        query = "SELECT * FROM metrics_log WHERE metric_name = ?"
        params = [metric_name]
        if experiment_id:
            query += " AND experiment_id = ?"
            params.append(experiment_id)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        cursor = self.conn.execute(query, params)
        return [dict(row) for row in cursor]

    # ── Campaigns ──

    def create_campaign(
        self,
        title: str,
        objective: str,
        success_criteria: str,
        parent_id: Optional[str] = None,
    ) -> str:
        """Create a new research campaign. Returns campaign_id."""
        campaign_id = str(uuid.uuid4())[:12]
        now = time.time()
        self.conn.execute(
            """INSERT INTO campaigns
            (campaign_id, timestamp, title, objective, success_criteria,
             status, parent_campaign_id, started_at)
            VALUES (?, ?, ?, ?, ?, 'active', ?, ?)""",
            (campaign_id, now, title, objective, success_criteria, parent_id, now),
        )
        self._maybe_commit()
        return campaign_id

    def get_campaign(self, campaign_id: str) -> Optional[Dict]:
        """Get a campaign by ID."""
        row = self.conn.execute(
            "SELECT * FROM campaigns WHERE campaign_id = ?",
            (campaign_id,),
        ).fetchone()
        return dict(row) if row else None

    def get_active_campaigns(self) -> List[Dict]:
        """Get all active campaigns."""
        cursor = self.conn.execute(
            "SELECT * FROM campaigns WHERE status = 'active' ORDER BY timestamp DESC"
        )
        return [dict(row) for row in cursor]

    def update_campaign(self, campaign_id: str, **kwargs) -> None:
        """Update campaign fields."""
        allowed = {
            "title",
            "objective",
            "success_criteria",
            "status",
            "findings_summary",
            "completed_at",
            "completion_reason",
            "successor_campaign_id",
        }
        sets = []
        params: List[Any] = []
        for k, v in kwargs.items():
            if k in allowed:
                sets.append(f"{k} = ?")
                params.append(v)
        if not sets:
            return
        params.append(campaign_id)
        self.conn.execute(
            f"UPDATE campaigns SET {', '.join(sets)} WHERE campaign_id = ?",
            params,
        )
        self._maybe_commit()

    def get_campaign_hypotheses(self, campaign_id: str) -> List[Dict]:
        """Get all hypotheses for a campaign."""
        cursor = self.conn.execute(
            """SELECT * FROM hypotheses WHERE campaign_id = ?
               ORDER BY timestamp ASC""",
            (campaign_id,),
        )
        return [self._hydrate_campaign_hypothesis_row(row) for row in cursor]

    def get_campaign_decisions(self, campaign_id: str) -> List[Dict]:
        """Get all decisions for a campaign."""
        cursor = self.conn.execute(
            """SELECT * FROM decisions WHERE campaign_id = ?
               ORDER BY timestamp ASC""",
            (campaign_id,),
        )
        return [dict(row) for row in cursor]

    def evaluate_campaign_criteria(self, campaign_id: str) -> Dict:
        """Evaluate campaign success criteria against measured data.

        Returns {
            all_met: bool,        # True if every parseable criterion passes
            n_criteria: int,
            n_passing: int,
            n_at_risk: int,
            n_not_yet: int,
            stale: bool,          # True if 10+ experiments with no progress
            tracker: List[Dict],  # per-criterion status from analytics
        }
        """
        from .analytics import ExperimentAnalytics

        campaign = self.get_campaign(campaign_id)
        if not campaign:
            return {
                "all_met": False,
                "n_criteria": 0,
                "n_passing": 0,
                "n_at_risk": 0,
                "n_not_yet": 0,
                "stale": False,
                "tracker": [],
            }

        experiments = self.get_campaign_experiments(campaign_id)
        hypotheses = self.get_campaign_hypotheses(campaign_id)
        decisions = self.get_campaign_decisions(campaign_id)

        analytics = ExperimentAnalytics(self)
        tracker = analytics.campaign_success_criteria_tracker(
            campaign,
            experiments,
            hypotheses,
            decisions,
        )

        n_passing = sum(1 for t in tracker if t.get("status") == "pass")
        n_at_risk = sum(1 for t in tracker if t.get("status") == "at_risk")
        n_not_yet = sum(1 for t in tracker if t.get("status") == "not_yet")
        n_criteria = len(tracker)

        # All parseable criteria must pass (ignore unknown/not_yet-only)
        all_met = n_criteria > 0 and n_passing == n_criteria

        # Stale: 10+ experiments but zero criteria passing
        stale = len(experiments) >= 10 and n_passing == 0 and n_at_risk > 0

        return {
            "all_met": all_met,
            "n_criteria": n_criteria,
            "n_passing": n_passing,
            "n_at_risk": n_at_risk,
            "n_not_yet": n_not_yet,
            "stale": stale,
            "tracker": tracker,
        }
