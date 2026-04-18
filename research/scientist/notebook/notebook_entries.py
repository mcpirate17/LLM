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
    _load_eval_native_module,
    _TEMPLATE_DEF_RE,
    _EMPTY_DATA_ACCOUNTING_SHAPE,
)
from ..json_utils import fast_loads as _json_loads
from ..leaderboard_scoring import (
    compute_efficiency_multiple as _compute_efficiency_multiple,
    compute_pre_investigation_score as _compute_pre_investigation_score,
)
from ...synthesis.templates import TEMPLATES





class _EntriesMixin:
    """Training curves, entries, and failure analysis."""

    __slots__ = ()

    def store_training_curve(self, result_id: str, curve: List[Dict]) -> None:
        """Store per-step training data for survivors only.

        curve: list of dicts with keys step, loss, grad_norm, step_time_ms
        """
        if not curve or not result_id:
            return
        self.flush_writes()
        # Only store curves for results that passed S1 (survivors).
        # S1 failure learning signal is captured in loss_ratio, not per-step curves.
        row = self.conn.execute(
            "SELECT stage1_passed FROM program_results WHERE result_id = ?",
            (result_id,),
        ).fetchone()
        if row is None or row[0] != 1:
            return
        self.conn.executemany(
            """INSERT OR REPLACE INTO training_curves
               (result_id, step, loss, grad_norm, step_time_ms)
               VALUES (?, ?, ?, ?, ?)""",
            [
                (
                    result_id,
                    d.get("step", i),
                    d.get("loss"),
                    d.get("grad_norm"),
                    d.get("step_time_ms"),
                )
                for i, d in enumerate(curve)
            ],
        )
        self._maybe_commit()

    def get_training_curve(self, result_id: str) -> List[Dict]:
        """Get per-step training data for a program."""
        rows = self.conn.execute(
            """SELECT step, loss, grad_norm, step_time_ms
               FROM training_curves WHERE result_id = ?
               ORDER BY step""",
            (result_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def strip_graph_json_for_failures(self, experiment_id: str) -> int:
        """Clear graph_json for S1 failures with no loss data.

        Called after update_op_success_rates() has already consumed the graphs.
        Sets to empty string (NOT NULL constraint on column).
        Returns the number of rows stripped.
        """
        cur = self.conn.execute(
            """UPDATE program_results SET graph_json = ''
               WHERE experiment_id = ?
                 AND stage0_passed = 1 AND stage1_passed = 0
                 AND loss_ratio IS NULL AND length(graph_json) > 0""",
            (experiment_id,),
        )
        n = cur.rowcount
        if n:
            self._maybe_commit()
        return n

    def merge_op_failure_counts(self, op_counts: Dict[str, Dict[str, int]]) -> None:
        """Merge S0 failure op counts into op_success_rates.

        Called after update_op_success_rates() to incorporate ops from programs
        that failed S0/S0.5 and were not stored in program_results.

        Args:
            op_counts: {op_name: {"n_used": int, "n_s0": int, "n_s05": int}}
        """
        if not op_counts:
            return
        now = time.time()
        for op_name, counts in op_counts.items():
            self.conn.execute(
                """INSERT INTO op_success_rates
                   (op_name, n_used, n_stage0_passed, n_stage05_passed,
                    n_stage1_passed, last_updated)
                   VALUES (?, ?, ?, ?, 0, ?)
                   ON CONFLICT(op_name) DO UPDATE SET
                    n_used = n_used + excluded.n_used,
                    n_stage0_passed = n_stage0_passed + excluded.n_stage0_passed,
                    n_stage05_passed = n_stage05_passed + excluded.n_stage05_passed,
                    last_updated = excluded.last_updated""",
                (
                    op_name,
                    counts.get("n_used", 0),
                    counts.get("n_s0", 0),
                    counts.get("n_s05", 0),
                    now,
                ),
            )
        self._maybe_commit()

    # ── Failure Signatures ──

    @staticmethod
    def _extract_op_bigrams(graph_json: str) -> List[str]:
        """Extract sorted op-pair bigrams from a graph JSON.

        A bigram is "opA->opB" for each edge in the graph.  Returns a
        sorted deduplicated list, giving a compact structural fingerprint
        of what-connects-to-what.
        """
        if not isinstance(graph_json, str) or not graph_json:
            return []
        return list(_cached_extract_op_bigrams(graph_json))

    def get_entries(
        self,
        experiment_id: Optional[str] = None,
        entry_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict]:
        query = "SELECT * FROM entries WHERE 1=1"
        params = []
        if experiment_id:
            query += " AND experiment_id = ?"
            params.append(experiment_id)
        if entry_type:
            query += " AND entry_type = ?"
            params.append(entry_type)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def set_external_benchmarks(self, result_id: str, payload: Any) -> bool:
        """Store external benchmark payload for a program result."""
        if not result_id:
            return False
        serialized = None
        try:
            if payload is None:
                serialized = None
            elif isinstance(payload, dict):
                # Merge partial benchmark updates (for example, scaling-only writes)
                # with any previously stored benchmark families (for example, long_context).
                existing = self.conn.execute(
                    "SELECT external_benchmarks_json FROM program_results WHERE result_id = ?",
                    (result_id,),
                ).fetchone()
                merged: Dict[str, Any] = {}
                if existing and existing["external_benchmarks_json"]:
                    try:
                        parsed = _json_loads(existing["external_benchmarks_json"])
                        if isinstance(parsed, dict):
                            merged.update(parsed)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        pass
                merged.update(payload)
                serialized = json.dumps(merged)
            else:
                serialized = json.dumps(payload)
        except (TypeError, ValueError):
            return False
        cur = self.conn.execute(
            "UPDATE program_results SET external_benchmarks_json = ? WHERE result_id = ?",
            (serialized, result_id),
        )
        self._maybe_commit()
        return cur.rowcount > 0

    def get_failure_analysis(self, experiment_id: str) -> Dict:
        """Get failure analysis data for an experiment."""
        programs = self.get_program_results(experiment_id)
        total = len(programs)
        if total == 0:
            return {
                "total": 0,
                "funnel": {},
                "errors": {},
                "stage_deaths": {},
                "root_causes": {},
                "exemplars": [],
            }

        s0_pass = sum(1 for p in programs if p.get("stage0_passed"))
        s05_pass = sum(1 for p in programs if p.get("stage05_passed"))
        s1_pass = sum(1 for p in programs if p.get("stage1_passed"))

        # Error type distribution (use classified error_type if available)
        errors: Dict[str, int] = {}
        root_causes: Dict[str, int] = {}
        exemplars: List[Dict[str, Any]] = []
        for p in programs:
            err_type = p.get("error_type") or ""
            err_msg = p.get("error_message") or p.get("stage0_error") or ""
            key = err_type if err_type else err_msg[:80].strip()
            if key:
                errors[key] = errors.get(key, 0) + 1
            failure_details = {}
            raw_failure = p.get("failure_details_json")
            if raw_failure:
                try:
                    failure_details = (
                        _json_loads(raw_failure)
                        if isinstance(raw_failure, str)
                        else raw_failure
                    )
                except (json.JSONDecodeError, TypeError, ValueError):
                    failure_details = {}
            root_cause = (
                failure_details.get("root_cause_code")
                or err_type
                or p.get("stage_at_death")
                or "unknown"
            )
            root_causes[root_cause] = root_causes.get(root_cause, 0) + 1
            if failure_details and len(exemplars) < 10:
                exemplars.append(
                    {
                        "result_id": p.get("result_id"),
                        "graph_fingerprint": p.get("graph_fingerprint"),
                        "stage": failure_details.get("stage")
                        or p.get("stage_at_death"),
                        "root_cause_code": root_cause,
                        "error_type": failure_details.get("error_type") or err_type,
                        "error_message": failure_details.get("error_message")
                        or err_msg,
                        "failure_op": failure_details.get("failure_op"),
                        "traceback_excerpt": failure_details.get("traceback_excerpt"),
                    }
                )

        # Stage-at-death histogram
        stage_deaths = {"validation": 0, "stage0": 0, "stage0.5": 0, "stage1": 0}
        for p in programs:
            sad = p.get("stage_at_death")
            if sad and sad in stage_deaths:
                stage_deaths[sad] += 1
            elif not p.get("stage0_passed"):
                stage_deaths["stage0"] += 1
            elif not p.get("stage05_passed"):
                stage_deaths["stage0.5"] += 1
            elif not p.get("stage1_passed"):
                stage_deaths["stage1"] += 1

        return {
            "total": total,
            "funnel": {
                "generated": total,
                "stage0_passed": s0_pass,
                "stage05_passed": s05_pass,
                "stage1_passed": s1_pass,
            },
            "errors": dict(sorted(errors.items(), key=lambda x: -x[1])[:10]),
            "root_causes": dict(sorted(root_causes.items(), key=lambda x: -x[1])[:10]),
            "stage_deaths": stage_deaths,
            "exemplars": exemplars,
        }
