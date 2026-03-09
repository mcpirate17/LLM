from __future__ import annotations
"""Auto-extracted mixin for LabNotebook."""

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ._shared import LOGGER


class _AnalyticsMixin:
    """Analytics operations for the Lab Notebook."""
    __slots__ = ()

    # ── Op Success Rates ──

    def update_op_success_rates(self, experiment_id: str) -> None:
        """Recompute op success rates from program results in this experiment.

        Uses a targeted query (only needed columns) and avoids dict(r)
        conversion overhead from get_program_results.
        """
        rows = self.conn.execute(
            """SELECT graph_json, stage0_passed, stage05_passed, stage1_passed,
                      loss_ratio, novelty_score, novelty_confidence
               FROM program_results
               WHERE experiment_id = ? AND graph_json IS NOT NULL""",
            (experiment_id,),
        ).fetchall()

        op_stats: Dict[str, Dict] = {}
        # Reusable reference to avoid repeated dict key hashing
        _OP_NAME = "op_name"

        for r in rows:
            graph_json = r[0]  # access by index — faster than by name
            if not graph_json:
                continue
            try:
                graph_data = json.loads(graph_json)
                nodes = graph_data.get("nodes", {})
            except (json.JSONDecodeError, TypeError):
                continue

            ops_in_graph = set()
            for node_data in nodes.values():
                op_name = node_data.get(_OP_NAME, "")
                if op_name and op_name != "input":
                    ops_in_graph.add(op_name)

            s0 = r[1]   # stage0_passed
            s05 = r[2]  # stage05_passed
            s1 = r[3]   # stage1_passed
            lr = r[4]   # loss_ratio
            nov = r[5]  # novelty_score
            nov_conf = r[6]  # novelty_confidence

            for op_name in ops_in_graph:
                if op_name not in op_stats:
                    op_stats[op_name] = {
                        "n_used": 0, "n_s0": 0, "n_s05": 0, "n_s1": 0,
                        "lr_sum": 0.0, "lr_n": 0,
                        "nov_sum": 0.0, "nov_n": 0,
                        "nov_conf_sum": 0.0, "nov_conf_n": 0,
                    }
                stats = op_stats[op_name]
                stats["n_used"] += 1
                if s0:
                    stats["n_s0"] += 1
                if s05:
                    stats["n_s05"] += 1
                if s1:
                    stats["n_s1"] += 1
                if lr is not None:
                    stats["lr_sum"] += lr
                    stats["lr_n"] += 1
                if nov is not None:
                    stats["nov_sum"] += nov
                    stats["nov_n"] += 1
                if nov_conf is not None:
                    stats["nov_conf_sum"] += nov_conf
                    stats["nov_conf_n"] += 1

        now = time.time()
        for op_name, stats in op_stats.items():
            avg_lr = stats["lr_sum"] / stats["lr_n"] if stats["lr_n"] else None
            avg_nov = stats["nov_sum"] / stats["nov_n"] if stats["nov_n"] else None
            avg_nov_conf = (stats["nov_conf_sum"] / stats["nov_conf_n"]
                           if stats["nov_conf_n"] else None)
            self.conn.execute(
                """INSERT INTO op_success_rates
                   (op_name, n_used, n_stage0_passed, n_stage05_passed,
                    n_stage1_passed, avg_loss_ratio, avg_novelty,
                    avg_novelty_confidence, last_updated)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(op_name) DO UPDATE SET
                    n_used = n_used + excluded.n_used,
                    n_stage0_passed = n_stage0_passed + excluded.n_stage0_passed,
                    n_stage05_passed = n_stage05_passed + excluded.n_stage05_passed,
                    n_stage1_passed = n_stage1_passed + excluded.n_stage1_passed,
                    avg_loss_ratio = excluded.avg_loss_ratio,
                    avg_novelty = excluded.avg_novelty,
                    avg_novelty_confidence = excluded.avg_novelty_confidence,
                    last_updated = excluded.last_updated""",
                (op_name, stats["n_used"], stats["n_s0"], stats["n_s05"],
                 stats["n_s1"], avg_lr, avg_nov, avg_nov_conf, now),
            )
        self._maybe_commit()


    def get_op_success_rates(self) -> List[Dict]:
        """Get all op success rates."""
        rows = self.conn.execute(
            """SELECT * FROM op_success_rates
               ORDER BY n_stage1_passed DESC, n_used DESC"""
        ).fetchall()
        return [dict(r) for r in rows]


    def update_failure_signatures(self, experiment_id: str) -> None:
        """Update failure_signatures table from program results in this experiment.

        Extracts op-pair bigrams from each graph and tracks how often
        each bigram appears in failed vs successful programs.  This gives
        Aria a compact memory of which structural patterns to avoid.
        """
        rows = self.conn.execute(
            """SELECT graph_json, stage1_passed, error_type
               FROM program_results
               WHERE experiment_id = ? AND graph_json IS NOT NULL
                 AND stage0_passed = 1 AND stage05_passed = 1""",
            (experiment_id,),
        ).fetchall()

        sig_stats: Dict[str, Dict] = {}
        for r in rows:
            bigrams = self._extract_op_bigrams(r[0])
            s1 = r[1]
            err = r[2] or ""
            for bg in bigrams:
                if bg not in sig_stats:
                    sig_stats[bg] = {"n_f": 0, "n_s": 0, "errs": set()}
                if s1:
                    sig_stats[bg]["n_s"] += 1
                else:
                    sig_stats[bg]["n_f"] += 1
                    if err:
                        sig_stats[bg]["errs"].add(err)

        now = time.time()
        for sig, st in sig_stats.items():
            # Keep error_types compact: top 3, comma-separated
            errs_str = ",".join(sorted(st["errs"])[:3]) if st["errs"] else None
            self.conn.execute(
                """INSERT INTO failure_signatures
                   (signature, n_failures, n_successes, error_types, last_updated)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(signature) DO UPDATE SET
                    n_failures = n_failures + excluded.n_failures,
                    n_successes = n_successes + excluded.n_successes,
                    error_types = COALESCE(excluded.error_types, error_types),
                    last_updated = excluded.last_updated""",
                (sig, st["n_f"], st["n_s"], errs_str, now),
            )
        self._maybe_commit()


    def backfill_failure_signatures(self) -> int:
        """One-time backfill of failure_signatures from all existing results.

        Skips if the table already has data.  Returns count of signatures created.
        """
        existing = self.conn.execute(
            "SELECT COUNT(*) FROM failure_signatures"
        ).fetchone()[0]
        if existing > 0:
            return 0
        rows = self.conn.execute(
            """SELECT graph_json, stage1_passed, error_type
               FROM program_results
               WHERE graph_json IS NOT NULL
                 AND stage0_passed = 1 AND stage05_passed = 1"""
        ).fetchall()
        sig_stats: Dict[str, Dict] = {}
        for r in rows:
            bigrams = self._extract_op_bigrams(r[0])
            s1 = r[1]
            err = r[2] or ""
            for bg in bigrams:
                if bg not in sig_stats:
                    sig_stats[bg] = {"n_f": 0, "n_s": 0, "errs": set()}
                if s1:
                    sig_stats[bg]["n_s"] += 1
                else:
                    sig_stats[bg]["n_f"] += 1
                    if err:
                        sig_stats[bg]["errs"].add(err)
        now = time.time()
        for sig, st in sig_stats.items():
            errs_str = ",".join(sorted(st["errs"])[:3]) if st["errs"] else None
            self.conn.execute(
                """INSERT INTO failure_signatures
                   (signature, n_failures, n_successes, error_types, last_updated)
                   VALUES (?, ?, ?, ?, ?)""",
                (sig, st["n_f"], st["n_s"], errs_str, now),
            )
        self._maybe_commit()
        LOGGER.info("Backfilled %d failure signatures from existing results", len(sig_stats))
        return len(sig_stats)


    def recompute_failure_signatures(self) -> int:
        """Delete and rebuild failure_signatures from scratch using S1-only failures.

        Unlike backfill_failure_signatures(), this always runs (even if data exists)
        and only counts programs that passed S0+S0.5 but failed at S1 as failures.
        This cleans up historically contaminated data from S0.5 causality failures.
        """
        self.conn.execute("DELETE FROM failure_signatures")
        rows = self.conn.execute(
            """SELECT graph_json, stage1_passed, error_type
               FROM program_results
               WHERE graph_json IS NOT NULL
                 AND stage0_passed = 1 AND stage05_passed = 1"""
        ).fetchall()
        sig_stats: Dict[str, Dict] = {}
        for r in rows:
            bigrams = self._extract_op_bigrams(r[0])
            s1 = r[1]
            err = r[2] or ""
            for bg in bigrams:
                if bg not in sig_stats:
                    sig_stats[bg] = {"n_f": 0, "n_s": 0, "errs": set()}
                if s1:
                    sig_stats[bg]["n_s"] += 1
                else:
                    sig_stats[bg]["n_f"] += 1
                    if err:
                        sig_stats[bg]["errs"].add(err)
        now = time.time()
        for sig, st in sig_stats.items():
            errs_str = ",".join(sorted(st["errs"])[:3]) if st["errs"] else None
            self.conn.execute(
                """INSERT INTO failure_signatures
                   (signature, n_failures, n_successes, error_types, last_updated)
                   VALUES (?, ?, ?, ?, ?)""",
                (sig, st["n_f"], st["n_s"], errs_str, now),
            )
        self._maybe_commit()
        LOGGER.info("Recomputed %d failure signatures (S1-only failures)", len(sig_stats))
        return len(sig_stats)


    def get_failure_signature_blocklist(self, min_seen: int = 20,
                                        max_fail_rate: float = 0.95) -> Dict[str, float]:
        """Return op-pair bigrams that consistently fail.

        Returns {signature: penalty} where penalty is 0.0 (hard block) for
        100% failure bigrams and scales up to 1.0.  Only includes bigrams
        seen at least ``min_seen`` times with failure rate >= ``max_fail_rate``.
        """
        rows = self.conn.execute(
            """SELECT signature, n_failures, n_successes
               FROM failure_signatures
               WHERE (n_failures + n_successes) >= ?""",
            (min_seen,),
        ).fetchall()
        blocklist: Dict[str, float] = {}
        for r in rows:
            total = r[1] + r[2]
            fail_rate = r[1] / total if total else 0
            if fail_rate >= max_fail_rate:
                # Scale: 100% fail → 0.0, max_fail_rate → 0.3
                penalty = max(0.0, 0.3 * (1.0 - fail_rate) / (1.0 - max_fail_rate))
                blocklist[r[0]] = round(penalty, 2)
        return blocklist


    # ── Op Rehabilitation Cache ──

    def get_op_rehabilitation_cache(self, max_age_hours: float = 24.0) -> Dict[str, Dict]:
        """Return cached op rehabilitation results, filtered by recency.

        Returns {op_name: {compile_passed, forward_passed, error_message, tested_at, model_dim}}.
        """
        cutoff = time.time() - max_age_hours * 3600
        rows = self.conn.execute(
            """SELECT op_name, compile_passed, forward_passed, error_message, tested_at, model_dim
               FROM op_rehabilitation_cache
               WHERE tested_at >= ?""",
            (cutoff,),
        ).fetchall()
        cache: Dict[str, Dict] = {}
        for r in rows:
            cache[r[0]] = {
                "compile_passed": bool(r[1]),
                "forward_passed": bool(r[2]),
                "error_message": r[3],
                "tested_at": r[4],
                "model_dim": r[5],
            }
        return cache


    # ── Learning Log ──

    def log_learning_event(self, event_type: str, description: str,
                           old_weights: Optional[Dict] = None,
                           new_weights: Optional[Dict] = None,
                           evidence: Optional[str] = None,
                           **event_data: Any) -> None:
        """Log a grammar weight change or learning decision.

        Backward-compatible with callers that pass extra structured keyword
        fields (e.g. ``changes=...``, ``excluded_ops=...``).
        """
        if old_weights is None and "old_weights" in event_data:
            old_weights = event_data.pop("old_weights")
        if new_weights is None and "new_weights" in event_data:
            new_weights = event_data.pop("new_weights")

        if event_data:
            serialized_extra = json.dumps(event_data, sort_keys=True, default=str)
            if evidence:
                evidence = f"{evidence}\n\nmeta={serialized_extra}"
            else:
                evidence = serialized_extra

        self.conn.execute(
            """INSERT INTO learning_log
               (timestamp, event_type, description, old_weights,
                new_weights, evidence)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (time.time(), event_type, description,
             json.dumps(old_weights) if old_weights else None,
             json.dumps(new_weights) if new_weights else None,
             evidence),
        )
        self._maybe_commit()


    def get_learning_log(self, limit: int = 100) -> List[Dict]:
        """Get recent learning log entries."""
        rows = self.conn.execute(
            "SELECT * FROM learning_log ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            for f in ("old_weights", "new_weights"):
                if d.get(f):
                    try:
                        d[f] = json.loads(d[f])
                    except (json.JSONDecodeError, TypeError):
                        pass
            results.append(d)
        return results


    def save_effective_weights(self, weights: Dict[str, float],
                               s1_rate: float,
                               experiment_id: Optional[str] = None) -> None:
        """Save the final applied grammar weights and S1 outcome for EMA continuity."""
        self.log_learning_event(
            "effective_weights_snapshot",
            f"Effective weights after {experiment_id or 'unknown'} (S1={s1_rate:.3f})",
            new_weights=weights,
            evidence=json.dumps({"s1_rate": s1_rate, "experiment_id": experiment_id}),
        )


    def load_last_effective_weights(self) -> Optional[tuple]:
        """Load the most recent effective weights snapshot.

        Returns (weights_dict, s1_rate) or None if no snapshot exists.
        """
        row = self.conn.execute(
            "SELECT new_weights, evidence FROM learning_log "
            "WHERE event_type='effective_weights_snapshot' "
            "ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        if not row or not row[0]:
            return None
        try:
            weights = json.loads(row[0])
            meta = json.loads(row[1]) if row[1] else {}
            return (weights, meta.get("s1_rate", 0.0))
        except (json.JSONDecodeError, TypeError):
            return None


    # ── Workflow Definitions ──

    def save_workflow_definition(
        self,
        workflow_id: str,
        name: str,
        graph_json: str,
        metadata: Optional[Dict] = None,
        author: str = "user",
    ) -> None:
        """Save a visual designer workflow definition."""
        now = time.time()
        self.conn.execute(
            """INSERT INTO workflow_definitions
               (workflow_id, name, timestamp, graph_json, metadata_json, author)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(workflow_id) DO UPDATE SET
                 name = excluded.name,
                 timestamp = excluded.timestamp,
                 graph_json = excluded.graph_json,
                 metadata_json = excluded.metadata_json,
                 author = excluded.author""",
            (workflow_id, name, now, graph_json, json.dumps(metadata or {}), author),
        )
        self._maybe_commit()


    def get_workflow_definition(self, workflow_id: str) -> Optional[Dict]:
        """Get a specific workflow definition."""
        row = self.conn.execute(
            "SELECT * FROM workflow_definitions WHERE workflow_id = ?",
            (workflow_id,),
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        if d.get("metadata_json"):
            try:
                d["metadata"] = json.loads(d["metadata_json"])
            except (json.JSONDecodeError, TypeError):
                d["metadata"] = {}
        return d


    def list_workflow_definitions(self, limit: int = 50) -> List[Dict]:
        """List recent workflow definitions."""
        rows = self.conn.execute(
            """SELECT workflow_id, name, timestamp, author
               FROM workflow_definitions
               ORDER BY timestamp DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


    # ── Designer Run Lineage ──

    def save_designer_run_lineage(
        self,
        run_id: str,
        workflow_id: str,
        *,
        workflow_version: Optional[int] = None,
        graph_fingerprint: Optional[str] = None,
        status: str = "unknown",
        source: str = "aria_designer",
        total_time_ms: Optional[float] = None,
        metrics: Optional[Dict[str, Any]] = None,
        payload: Optional[Dict[str, Any]] = None,
        created_at: Optional[float] = None,
    ) -> None:
        """Upsert lineage metadata for runs produced by Aria Designer."""
        now = time.time()
        created_ts = float(created_at) if created_at is not None else now
        self.conn.execute(
            """INSERT INTO designer_run_lineage
               (run_id, workflow_id, workflow_version, graph_fingerprint, status, source,
                total_time_ms, metrics_json, payload_json, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(run_id) DO UPDATE SET
                 workflow_id = excluded.workflow_id,
                 workflow_version = excluded.workflow_version,
                 graph_fingerprint = excluded.graph_fingerprint,
                 status = excluded.status,
                 source = excluded.source,
                 total_time_ms = excluded.total_time_ms,
                 metrics_json = excluded.metrics_json,
                 payload_json = excluded.payload_json,
                 updated_at = excluded.updated_at""",
            (
                run_id,
                workflow_id,
                workflow_version,
                graph_fingerprint,
                status,
                source,
                total_time_ms,
                json.dumps(metrics or {}),
                json.dumps(payload or {}),
                created_ts,
                now,
            ),
        )
        self._maybe_commit()


    def get_designer_run_lineage(self, run_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM designer_run_lineage WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        try:
            d["metrics"] = json.loads(d.get("metrics_json") or "{}")
        except (TypeError, json.JSONDecodeError):
            d["metrics"] = {}
        try:
            d["payload"] = json.loads(d.get("payload_json") or "{}")
        except (TypeError, json.JSONDecodeError):
            d["payload"] = {}
        return d


    def list_designer_run_lineage(
        self, *, workflow_id: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM designer_run_lineage"
        params: List[Any] = []
        if workflow_id:
            query += " WHERE workflow_id = ?"
            params.append(workflow_id)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(int(max(1, limit)))
        rows = self.conn.execute(query, params).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            d = dict(row)
            try:
                d["metrics"] = json.loads(d.get("metrics_json") or "{}")
            except (TypeError, json.JSONDecodeError):
                d["metrics"] = {}
            out.append(d)
        return out


    def get_report_snapshot(
        self,
        snapshot_key: str,
        scope: str,
        min_latest_completed_ts: float,
    ) -> Optional[Dict[str, Any]]:
        if not snapshot_key:
            return None
        row = self.conn.execute(
            """SELECT payload_json, latest_completed_ts
               FROM report_snapshots
               WHERE snapshot_key = ? AND scope = ?""",
            (snapshot_key, scope),
        ).fetchone()
        if not row:
            return None
        cached_latest = float(row["latest_completed_ts"] or 0.0)
        if cached_latest < float(min_latest_completed_ts or 0.0):
            return None
        payload = row["payload_json"]
        if not payload:
            return None
        try:
            parsed = json.loads(payload)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None


    def save_report_snapshot(
        self,
        snapshot_key: str,
        scope: str,
        query: Dict[str, Any],
        payload: Dict[str, Any],
        latest_completed_ts: float,
    ) -> None:
        if not snapshot_key or not scope:
            return
        now = time.time()
        self.conn.execute(
            """INSERT INTO report_snapshots (
                   snapshot_key, scope, query_json, payload_json,
                   latest_completed_ts, created_at, updated_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(snapshot_key) DO UPDATE SET
                   scope = excluded.scope,
                   query_json = excluded.query_json,
                   payload_json = excluded.payload_json,
                   latest_completed_ts = excluded.latest_completed_ts,
                   updated_at = excluded.updated_at""",
            (
                snapshot_key,
                scope,
                json.dumps(query or {}, sort_keys=True, separators=(",", ":")),
                json.dumps(payload or {}, separators=(",", ":")),
                float(latest_completed_ts or 0.0),
                now,
                now,
            ),
        )
        self._maybe_commit()

        cleanup_interval_seconds = 300.0
        last_cleanup = float(self.__class__._last_report_snapshot_cleanup_at or 0.0)
        if (now - last_cleanup) >= cleanup_interval_seconds:
            try:
                ttl_seconds = int(os.environ.get("ARIA_REPORT_SNAPSHOT_TTL_SECONDS", str(7 * 24 * 3600)))
            except Exception:
                ttl_seconds = 7 * 24 * 3600
            try:
                max_rows_per_scope = int(os.environ.get("ARIA_REPORT_SNAPSHOT_MAX_ROWS_PER_SCOPE", "400"))
            except Exception:
                max_rows_per_scope = 400
            self.cleanup_report_snapshots(
                ttl_seconds=max(60, ttl_seconds),
                max_rows_per_scope=max(20, max_rows_per_scope),
            )
            self.__class__._last_report_snapshot_cleanup_at = now


    def cleanup_report_snapshots(
        self,
        ttl_seconds: int = 7 * 24 * 3600,
        max_rows_per_scope: int = 400,
    ) -> Dict[str, int]:
        ttl = max(60, int(ttl_seconds or 0))
        cap = max(1, int(max_rows_per_scope or 0))
        cutoff = time.time() - float(ttl)

        stats = {
            "deleted_expired": 0,
            "deleted_capped": 0,
            "remaining": 0,
        }

        cur = self.conn.execute(
            "DELETE FROM report_snapshots WHERE updated_at < ?",
            (cutoff,),
        )
        stats["deleted_expired"] = int(cur.rowcount or 0)

        scopes = self.conn.execute(
            "SELECT DISTINCT scope FROM report_snapshots"
        ).fetchall()
        for row in scopes:
            scope = row[0]
            if not scope:
                continue
            cur = self.conn.execute(
                """DELETE FROM report_snapshots
                   WHERE snapshot_key IN (
                       SELECT snapshot_key
                       FROM report_snapshots
                       WHERE scope = ?
                       ORDER BY updated_at DESC
                       LIMIT -1 OFFSET ?
                   )""",
                (scope, cap),
            )
            stats["deleted_capped"] += int(cur.rowcount or 0)

        remaining_row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM report_snapshots"
        ).fetchone()
        stats["remaining"] = int(remaining_row["n"] or 0) if remaining_row else 0
        self._maybe_commit()
        return stats


    def get_report_snapshot_stats(self) -> Dict[str, Any]:
        now = time.time()
        rows = self.conn.execute(
            """SELECT scope,
                      COUNT(*) AS count,
                      MIN(updated_at) AS oldest_updated_at,
                      MAX(updated_at) AS newest_updated_at
               FROM report_snapshots
               GROUP BY scope
               ORDER BY count DESC, scope ASC"""
        ).fetchall()

        scopes: List[Dict[str, Any]] = []
        total = 0
        oldest_seen: Optional[float] = None
        newest_seen: Optional[float] = None
        for row in rows:
            count = int(row["count"] or 0)
            oldest = float(row["oldest_updated_at"] or 0.0)
            newest = float(row["newest_updated_at"] or 0.0)
            total += count
            if oldest > 0 and (oldest_seen is None or oldest < oldest_seen):
                oldest_seen = oldest
            if newest > 0 and (newest_seen is None or newest > newest_seen):
                newest_seen = newest

            scopes.append({
                "scope": row["scope"],
                "count": count,
                "oldest_age_seconds": round(max(0.0, now - oldest), 2) if oldest > 0 else None,
                "newest_age_seconds": round(max(0.0, now - newest), 2) if newest > 0 else None,
            })

        return {
            "total_snapshots": total,
            "n_scopes": len(scopes),
            "oldest_age_seconds": round(max(0.0, now - oldest_seen), 2) if oldest_seen else None,
            "newest_age_seconds": round(max(0.0, now - newest_seen), 2) if newest_seen else None,
            "scopes": scopes,
        }


    # ── Attribution Reports ──

    def record_attribution_report(
        self,
        hypothesis_id: Optional[str],
        supporting_experiments: Optional[List[str]],
        ablation_experiments: Optional[List[str]],
        outcome: str,
        report: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Persist an attribution report row linking evidence and ablations."""
        report_id = str(uuid.uuid4())[:12]
        now = time.time()
        self.conn.execute(
            """INSERT INTO attribution_reports
            (report_id, timestamp, hypothesis_id, supporting_experiments,
             ablation_experiments, outcome, report_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                report_id,
                now,
                hypothesis_id,
                json.dumps(supporting_experiments or []),
                json.dumps(ablation_experiments or []),
                outcome,
                json.dumps(report or {}),
            ),
        )
        self._maybe_commit()
        return report_id


    def get_attribution_reports(self, hypothesis_id: Optional[str] = None,
                                limit: int = 100) -> List[Dict]:
        """Return attribution reports, newest first."""
        query = "SELECT * FROM attribution_reports WHERE 1=1"
        params: List[Any] = []
        if hypothesis_id:
            query += " AND hypothesis_id = ?"
            params.append(hypothesis_id)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        out: List[Dict] = []
        for row in rows:
            item = dict(row)
            for key in ("supporting_experiments", "ablation_experiments", "report_json"):
                raw = item.get(key)
                if raw:
                    try:
                        item[key] = json.loads(raw)
                    except (TypeError, json.JSONDecodeError):
                        pass
            out.append(item)
        return out


    # ── Report Markdown Export ──

    def save_report_markdown(self, content: str, reason: str,
                             summary: Optional[Dict] = None) -> Optional[Path]:
        """Save a report as a markdown file alongside the database.

        Creates a reports/ directory next to lab_notebook.db and writes
        the report content as a .md file with a frontmatter-style header.

        Returns the path to the created file, or None on failure.
        """
        logger = logging.getLogger(__name__)
        try:
            reports_dir = self.db_path.parent / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)

            now = datetime.now()
            timestamp_str = now.strftime("%Y-%m-%d_%H-%M")
            safe_reason = reason.replace(" ", "_").replace("/", "-")[:40]
            filename = f"report_{timestamp_str}_{safe_reason}.md"
            filepath = reports_dir / filename

            # Build frontmatter header
            header_lines = [
                "---",
                f"generated: {now.isoformat()}",
                f"reason: {reason}",
            ]
            if summary:
                header_lines.append(
                    f"experiments: {summary.get('total_experiments', '?')}")
                total_prog = summary.get("total_programs_evaluated", 0)
                s1 = summary.get("stage1_survivors", 0)
                rate = s1 / max(total_prog, 1) * 100
                header_lines.append(f"s1_pass_rate: {rate:.1f}%")
                header_lines.append(f"stage1_survivors: {s1}")
            header_lines.append("---")
            header_lines.append("")

            full_content = "\n".join(header_lines) + content

            filepath.write_text(full_content, encoding="utf-8")
            logger.info(f"Report saved to {filepath}")
            return filepath
        except Exception as e:
            logger.warning(f"Failed to save report markdown: {e}")
            return None

