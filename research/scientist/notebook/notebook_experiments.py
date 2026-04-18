from __future__ import annotations

"""Auto-extracted mixin for LabNotebook."""

import json
import math
import sqlite3
import time
import uuid
import zlib
from typing import Any, Dict, List, Optional

from ..runtime_events import get_runtime_event_services, publish_lifecycle_event
from ._shared import ExperimentEntry, LOGGER, sanitize_for_db

try:
    from ..preregistration import PreregistrationError, validate_preregistration
except (ImportError, ModuleNotFoundError):
    from pathlib import Path as _Path
    import importlib.util as _importlib_util
    import sys as _sys

    _prereg_path = _Path(__file__).parent.parent / "preregistration.py"
    _prereg_spec = _importlib_util.spec_from_file_location(
        "_notebook_preregistration_fallback", str(_prereg_path)
    )
    _prereg_mod = _importlib_util.module_from_spec(_prereg_spec)
    assert _prereg_spec is not None and _prereg_spec.loader is not None
    _sys.modules[_prereg_spec.name] = _prereg_mod
    _prereg_spec.loader.exec_module(_prereg_mod)
    PreregistrationError = _prereg_mod.PreregistrationError
    validate_preregistration = _prereg_mod.validate_preregistration


class _ExperimentsMixin:
    """Experiments operations for the Lab Notebook."""

    __slots__ = ()

    def _direct_db_conn(self):
        """Return the shared native connection for critical lifecycle writes.

        Previously opened a one-off sqlite3.Connection which caused SHM
        teardown on close.  Now returns the shared NativeConnectionWrapper
        which delegates to the Rust singleton — no close, no SHM teardown.
        """
        if getattr(self, "_use_native", False):
            return self.conn
        # Fallback for in-memory test DBs.
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        for pragma in (
            "PRAGMA foreign_keys=ON",
            "PRAGMA wal_autocheckpoint=0",
            "PRAGMA busy_timeout=15000",
        ):
            try:
                conn.execute(pragma)
            except sqlite3.OperationalError as exc:
                LOGGER.warning("Direct DB pragma failed (%s): %s", pragma, exc)
        return conn

    def _experiment_exists_direct(self, experiment_id: str) -> bool:
        try:
            conn = self._direct_db_conn()
            try:
                row = conn.execute(
                    "SELECT 1 FROM experiments WHERE experiment_id = ?",
                    (experiment_id,),
                ).fetchone()
                return row is not None
            finally:
                conn.close()
        except sqlite3.OperationalError as exc:
            LOGGER.warning(
                "Direct experiment existence check failed for %s: %s",
                experiment_id,
                exc,
            )
            return False

    def _experiment_exists_primary(self, experiment_id: str) -> bool:
        try:
            row = self.conn.execute(
                "SELECT 1 FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
            return row is not None
        except sqlite3.OperationalError as exc:
            LOGGER.warning(
                "Primary experiment existence check failed for %s: %s",
                experiment_id,
                exc,
            )
            return False

    def _experiment_status_direct(self, experiment_id: str) -> Optional[str]:
        try:
            conn = self._direct_db_conn()
            try:
                row = conn.execute(
                    "SELECT status FROM experiments WHERE experiment_id = ?",
                    (experiment_id,),
                ).fetchone()
                return str(row[0]) if row and row[0] is not None else None
            finally:
                conn.close()
        except sqlite3.OperationalError as exc:
            LOGGER.warning(
                "Direct experiment status check failed for %s: %s",
                experiment_id,
                exc,
            )
            return None

    def _execute_direct_write(self, sql: str, params: tuple[Any, ...]) -> None:
        conn = self._direct_db_conn()
        try:
            conn.execute(sql, params)
            conn.commit()
        finally:
            conn.close()

    def _preregistration_exists_direct(self, preregistration_id: str) -> bool:
        try:
            conn = self._direct_db_conn()
            try:
                row = conn.execute(
                    """SELECT 1 FROM hypothesis_preregistrations
                       WHERE preregistration_id = ?""",
                    (preregistration_id,),
                ).fetchone()
                return row is not None
            finally:
                conn.close()
        except sqlite3.OperationalError as exc:
            LOGGER.warning(
                "Direct preregistration existence check failed for %s: %s",
                preregistration_id,
                exc,
            )
            return False

    def _preregistration_exists_primary(self, preregistration_id: str) -> bool:
        try:
            row = self.conn.execute(
                """SELECT 1 FROM hypothesis_preregistrations
                   WHERE preregistration_id = ?""",
                (preregistration_id,),
            ).fetchone()
            return row is not None
        except sqlite3.OperationalError as exc:
            LOGGER.warning(
                "Primary preregistration existence check failed for %s: %s",
                preregistration_id,
                exc,
            )
            return False

    def _publish_lifecycle_event_safe(
        self,
        *,
        event_type: str,
        run_id: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            current = get_runtime_event_services(self.db_path).registry.get(run_id)
            if current is not None and current.last_event.event_type == event_type:
                return
            publish_lifecycle_event(
                notebook_path=self.db_path,
                event_type=event_type,
                producer="notebook.experiments",
                run_id=run_id,
                payload=payload,
            )
        except Exception as exc:
            LOGGER.warning(
                "Runtime lifecycle publish failed for %s (%s): %s",
                event_type,
                run_id,
                exc,
            )

    def cleanup_stale_experiments(
        self,
        timeout_minutes: int = 60,
        startup_failure_minutes: int = 15,
    ) -> int:
        """Mark stale or startup-failed running experiments as failed.

        - Long-running stale experiments are cleaned after ``timeout_minutes``.
        - Runs with no progress signals are cleaned after
          ``startup_failure_minutes`` to handle interrupted startup paths.

        Returns the number of experiments cleaned up.
        """
        self.flush_writes()
        now = time.time()
        cutoff = now - (timeout_minutes * 60)
        startup_cutoff = now - (startup_failure_minutes * 60)

        stale_rows = self.conn.execute(
            "SELECT experiment_id FROM experiments "
            "WHERE status = 'running' AND started_at < ?",
            (cutoff,),
        ).fetchall()
        stale_ids = {r["experiment_id"] for r in stale_rows}

        startup_failed_rows = self.conn.execute(
            """
            SELECT e.experiment_id
            FROM experiments e
            WHERE e.status = 'running'
              AND e.started_at < ?
              AND NOT EXISTS (
                SELECT 1 FROM program_results pr WHERE pr.experiment_id = e.experiment_id
              )
              AND NOT EXISTS (
                SELECT 1 FROM metrics_log ml WHERE ml.experiment_id = e.experiment_id
              )
              AND NOT EXISTS (
                SELECT 1
                FROM entries en
                WHERE en.experiment_id = e.experiment_id
                  AND en.entry_type != 'hypothesis'
              )
            """,
            (startup_cutoff,),
        ).fetchall()
        startup_failed_ids = {r["experiment_id"] for r in startup_failed_rows}

        if not stale_ids and not startup_failed_ids:
            return 0

        updates = []
        all_ids = stale_ids | startup_failed_ids
        for experiment_id in all_ids:
            if experiment_id in startup_failed_ids and experiment_id not in stale_ids:
                reason = "Startup failed before any progress was recorded"
            else:
                reason = "Process terminated while running"
            updates.append((reason, experiment_id))

        self.conn.executemany(
            "UPDATE experiments SET status = 'failed', "
            "results_json = json_set(COALESCE(results_json, '{}'), '$.failure_reason', ?) "
            "WHERE experiment_id = ?",
            updates,
        )
        self._maybe_commit()
        for reason, experiment_id in updates:
            self._publish_lifecycle_event_safe(
                event_type="experiment_failed",
                run_id=experiment_id,
                payload={
                    "completed_at": now,
                    "error": reason,
                    "results": None,
                    "reason": "stale_recovery_cleanup",
                },
            )
        return len(all_ids)

    def get_resumable_experiment(self, experiment_id: str) -> Optional[Dict]:
        """Get experiment data for resume if status is 'running', 'failed', or 'interrupted'.

        Returns dict with config_json, experiment_type, hypothesis, started_at,
        or None if the experiment doesn't exist or isn't resumable.
        """
        row = self.conn.execute(
            "SELECT experiment_id, experiment_type, status, config_json, "
            "hypothesis, started_at FROM experiments "
            "WHERE experiment_id = ? AND status IN ('running', 'failed', 'interrupted')",
            (experiment_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "experiment_id": row["experiment_id"],
            "experiment_type": row["experiment_type"],
            "status": row["status"],
            "config_json": row["config_json"],
            "hypothesis": row["hypothesis"],
            "started_at": row["started_at"],
        }

    def resume_experiment(self, experiment_id: str) -> None:
        """Mark an experiment as running again for resume compatibility."""
        resume_sql = "UPDATE experiments SET status = 'running' WHERE experiment_id = ?"
        resume_params = (experiment_id,)
        try:
            self.conn.execute(resume_sql, resume_params)
            self._maybe_commit()
        except sqlite3.OperationalError as exc:
            LOGGER.warning(
                "Primary resume_experiment write failed for %s; retrying direct: %s",
                experiment_id,
                exc,
            )
            self._execute_direct_write(resume_sql, resume_params)
        status = self._experiment_status_direct(experiment_id)
        if status != "running":
            LOGGER.warning(
                "Experiment %s status is %r after resume_experiment; retrying direct write",
                experiment_id,
                status,
            )
            self._execute_direct_write(resume_sql, resume_params)

    def interrupt_experiment(self, experiment_id: str, aria_summary: str) -> None:
        """Mark an experiment as interrupted for shutdown compatibility."""
        interrupt_sql = """UPDATE experiments SET
               status = 'interrupted',
               completed_at = ?,
               aria_summary = ?
               WHERE experiment_id = ?"""
        interrupt_params = (
            time.time(),
            aria_summary,
            experiment_id,
        )
        try:
            self.conn.execute(interrupt_sql, interrupt_params)
            self._maybe_commit()
        except sqlite3.OperationalError as exc:
            LOGGER.warning(
                "Primary interrupt_experiment write failed for %s; retrying direct: %s",
                experiment_id,
                exc,
            )
            self._execute_direct_write(interrupt_sql, interrupt_params)
        status = self._experiment_status_direct(experiment_id)
        if status != "interrupted":
            LOGGER.warning(
                "Experiment %s status is %r after interrupt_experiment; retrying direct write",
                experiment_id,
                status,
            )
            self._execute_direct_write(interrupt_sql, interrupt_params)

    # ── Hypothesis Preregistration ──

    def create_preregistration(
        self,
        experiment_type: str,
        preregistration: Dict[str, Any],
        created_by: str = "runner",
        notes: Optional[str] = None,
    ) -> str:
        """Create a structured preregistration entry."""
        validate_preregistration(preregistration)
        prereg_id = str(uuid.uuid4())[:12]
        now = time.time()
        insert_sql = """INSERT INTO hypothesis_preregistrations
            (preregistration_id, timestamp, experiment_type, status,
             hypothesis_json, analysis_plan_json, falsification_json,
             confounders_json, exploratory, created_by, notes)
            VALUES (?, ?, ?, 'registered', ?, ?, ?, ?, ?, ?, ?)"""
        insert_params = (
            prereg_id,
            now,
            experiment_type,
            json.dumps(preregistration.get("hypothesis") or {}, default=str),
            json.dumps(preregistration.get("analysis_plan") or {}, default=str),
            json.dumps(
                preregistration.get("falsification_conditions") or [], default=str
            ),
            json.dumps(preregistration.get("confounders_checklist") or [], default=str),
            int(bool(preregistration.get("exploratory"))),
            created_by,
            notes,
        )
        try:
            self.conn.execute(insert_sql, insert_params)
            self._maybe_commit()
        except sqlite3.OperationalError as exc:
            LOGGER.warning(
                "Primary create_preregistration write failed for %s; retrying direct: %s",
                prereg_id,
                exc,
            )
            self._execute_direct_write(insert_sql, insert_params)

        direct_visible = self._preregistration_exists_direct(prereg_id)
        primary_visible = self._preregistration_exists_primary(prereg_id)
        if not direct_visible:
            LOGGER.warning(
                "Preregistration %s missing after primary commit; retrying via direct connection",
                prereg_id,
            )
            if not primary_visible:
                self._execute_direct_write(insert_sql, insert_params)
                direct_visible = self._preregistration_exists_direct(prereg_id)
                primary_visible = self._preregistration_exists_primary(prereg_id)
                if not direct_visible and not primary_visible:
                    raise sqlite3.OperationalError(
                        f"Preregistration {prereg_id} was not durably persisted"
                    )
            else:
                LOGGER.warning(
                    "Preregistration %s is visible on primary connection but direct verification failed; continuing",
                    prereg_id,
                )
        return prereg_id

    def get_preregistration(self, preregistration_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM hypothesis_preregistrations WHERE preregistration_id = ?",
            (preregistration_id,),
        ).fetchone()
        if row is None:
            return None
        out = dict(row)
        for field in (
            "hypothesis_json",
            "analysis_plan_json",
            "falsification_json",
            "confounders_json",
        ):
            raw = out.get(field)
            if raw:
                try:
                    out[field] = json.loads(raw)
                except (TypeError, json.JSONDecodeError):
                    pass
        return out

    def get_preregistration_for_experiment(
        self, experiment_id: str
    ) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT preregistration_id FROM experiments WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
        if not row or not row["preregistration_id"]:
            return None
        return self.get_preregistration(row["preregistration_id"])

    def get_preregistration_deviations(
        self, experiment_id: str
    ) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            """SELECT * FROM preregistration_deviations
               WHERE experiment_id = ?
               ORDER BY timestamp DESC""",
            (experiment_id,),
        ).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            d = dict(row)
            if d.get("details_json"):
                try:
                    d["details_json"] = json.loads(d["details_json"])
                except (TypeError, json.JSONDecodeError):
                    pass
            out.append(d)
        return out

    def log_preregistration_deviation(
        self,
        experiment_id: str,
        rationale: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Record explicit exploratory deviation from preregistered plan."""
        exp = self.conn.execute(
            "SELECT preregistration_id FROM experiments WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
        prereg_id = exp["preregistration_id"] if exp else None
        dev_id = str(uuid.uuid4())[:12]
        self.conn.execute(
            """INSERT INTO preregistration_deviations
            (deviation_id, preregistration_id, experiment_id, timestamp,
             deviation_type, rationale, details_json)
            VALUES (?, ?, ?, ?, 'exploratory', ?, ?)""",
            (
                dev_id,
                prereg_id,
                experiment_id,
                time.time(),
                rationale,
                json.dumps(details or {}),
            ),
        )
        self._maybe_commit()
        return dev_id

    # ── Experiments ──

    def start_experiment(
        self,
        experiment_type: str,
        config: Dict,
        hypothesis: Optional[str] = None,
        research_question: Optional[str] = None,
        hypothesis_metadata: Optional[Dict] = None,
        preregistration_id: Optional[str] = None,
        require_preregistration: bool = False,
    ) -> str:
        """Start a new experiment. Returns experiment ID."""
        if require_preregistration and not preregistration_id:
            raise PreregistrationError(
                "Experiment start blocked: missing preregistration_id."
            )
        exp_id = str(uuid.uuid4())[:12]
        now = time.time()
        config_payload = dict(config)
        config_payload.setdefault("code_version", self._detect_code_version())
        insert_params = (
            exp_id,
            now,
            experiment_type,
            hypothesis,
            research_question,
            preregistration_id,
            json.dumps(config_payload, default=str),
            now,
        )

        self.conn.execute(
            """INSERT INTO experiments
            (experiment_id, timestamp, experiment_type, status, hypothesis,
             research_question, preregistration_id, config_json, started_at)
            VALUES (?, ?, ?, 'running', ?, ?, ?, ?, ?)""",
            insert_params,
        )
        if preregistration_id:
            self.conn.execute(
                """UPDATE hypothesis_preregistrations
                   SET experiment_id = ?, status = 'linked'
                   WHERE preregistration_id = ?""",
                (exp_id, preregistration_id),
            )
        self._maybe_commit()

        direct_visible = self._experiment_exists_direct(exp_id)
        primary_visible = self._experiment_exists_primary(exp_id)
        if not direct_visible:
            LOGGER.warning(
                "Experiment %s missing after primary commit; retrying via direct connection",
                exp_id,
            )
            if not primary_visible:
                conn = self._direct_db_conn()
                try:
                    row = conn.execute(
                        "SELECT 1 FROM experiments WHERE experiment_id = ?",
                        (exp_id,),
                    ).fetchone()
                    if row is None:
                        conn.execute(
                            """INSERT INTO experiments
                            (experiment_id, timestamp, experiment_type, status, hypothesis,
                             research_question, preregistration_id, config_json, started_at)
                            VALUES (?, ?, ?, 'running', ?, ?, ?, ?, ?)""",
                            insert_params,
                        )
                    if preregistration_id:
                        conn.execute(
                            """UPDATE hypothesis_preregistrations
                               SET experiment_id = ?, status = 'linked'
                               WHERE preregistration_id = ?""",
                            (exp_id, preregistration_id),
                        )
                    conn.commit()
                finally:
                    conn.close()
                direct_visible = self._experiment_exists_direct(exp_id)
                primary_visible = self._experiment_exists_primary(exp_id)
                if not direct_visible and not primary_visible:
                    raise sqlite3.OperationalError(
                        f"Experiment {exp_id} was not durably persisted"
                    )
            else:
                LOGGER.warning(
                    "Experiment %s is visible on primary connection but direct verification failed; continuing",
                    exp_id,
                )

        # Log entry
        source = (hypothesis_metadata or {}).get("source", "unknown")
        confidence = (hypothesis_metadata or {}).get("confidence")
        critique_confidence = (hypothesis_metadata or {}).get("critique_confidence")
        critique = (hypothesis_metadata or {}).get("critique")
        effective_confidence = (
            confidence if confidence is not None else critique_confidence
        )
        confidence_text = (
            effective_confidence if effective_confidence is not None else "not provided"
        )
        if isinstance(critique, dict):
            verdict = critique.get("verdict") or "unknown"
            gate = critique.get("gate") or "n/a"
            concerns = critique.get("concerns") or []
            concern_hint = concerns[0] if concerns else "no concerns recorded"
            critique_text = f"{verdict} (gate={gate}) — {concern_hint}"
        else:
            critique_text = critique if critique else "not provided"
        self.add_entry(
            ExperimentEntry(
                entry_type="hypothesis",
                title=f"Experiment {exp_id} started",
                content=(
                    f"Type: {experiment_type}\n"
                    f"Hypothesis: {hypothesis or 'exploratory'}\n"
                    f"Provenance: {source}\n"
                    f"Confidence: {confidence_text}\n"
                    f"Critique: {critique_text}"
                ),
                experiment_id=exp_id,
                tags=["experiment_start"],
                metadata=hypothesis_metadata or {},
            )
        )

        self._publish_lifecycle_event_safe(
            event_type="experiment_started",
            run_id=exp_id,
            payload={
                "timestamp": now,
                "started_at": now,
                "experiment_type": experiment_type,
                "hypothesis": hypothesis,
                "research_question": research_question,
                "preregistration_id": preregistration_id,
                "config": config_payload,
            },
        )

        return exp_id

    def complete_experiment(
        self,
        experiment_id: str,
        results: Dict,
        aria_summary: str = "",
        aria_mood: str = "contemplative",
        insights: Optional[List[str]] = None,
        llm_analysis: Optional[str] = None,
        exploratory_deviation_reason: Optional[str] = None,
    ):
        """Mark an experiment as completed with results."""
        n_total = results.get("total", 0)
        if n_total == 0:
            return self.fail_experiment(
                experiment_id,
                error="Experiment completed with 0 programs generated (possible synthesis failure).",
                results=results,
            )

        now = time.time()
        try:
            started = self.conn.execute(
                "SELECT started_at FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
        except sqlite3.OperationalError as exc:
            LOGGER.warning(
                "Primary complete_experiment read failed for %s; using zero duration: %s",
                experiment_id,
                exc,
            )
            started = None
        duration = now - started["started_at"] if started else 0

        update_sql = """UPDATE experiments SET
                status = 'completed',
                results_json = ?,
                n_programs_generated = ?,
                n_stage0_passed = ?,
                n_stage05_passed = ?,
                n_stage1_passed = ?,
                best_loss_ratio = ?,
                best_novelty_score = ?,
                aria_summary = ?,
                aria_mood = ?,
                insights_json = ?,
                llm_analysis = ?,
                completed_at = ?,
                duration_seconds = ?
            WHERE experiment_id = ?"""
        update_params = (
            self._compress(results),
            results.get("total", 0),
            results.get("stage0_passed", 0),
            results.get("stage05_passed", 0),
            results.get("stage1_passed", 0),
            float(results["best_loss_ratio"])
            if results.get("best_loss_ratio") is not None
            else None,
            float(results["best_novelty_score"])
            if results.get("best_novelty_score") is not None
            else None,
            aria_summary,
            aria_mood,
            self._compress(insights or []),
            llm_analysis,
            now,
            duration,
            experiment_id,
        )
        complete_persisted = False
        try:
            self.conn.execute(update_sql, update_params)
            self._maybe_commit()
            complete_persisted = True
        except sqlite3.OperationalError as exc:
            LOGGER.warning(
                "Primary complete_experiment write failed for %s; retrying direct: %s",
                experiment_id,
                exc,
            )
            try:
                self._execute_direct_write(update_sql, update_params)
                complete_persisted = True
            except sqlite3.OperationalError as direct_exc:
                LOGGER.warning(
                    "Direct complete_experiment write failed for %s; continuing without notebook persistence: %s",
                    experiment_id,
                    direct_exc,
                )
        if complete_persisted:
            status = self._experiment_status_direct(experiment_id)
            if status != "completed":
                LOGGER.warning(
                    "Experiment %s status is %r after complete_experiment; retrying direct write",
                    experiment_id,
                    status,
                )
                try:
                    self._execute_direct_write(update_sql, update_params)
                except sqlite3.OperationalError as exc:
                    LOGGER.warning(
                        "Final complete_experiment retry failed for %s; continuing without notebook persistence: %s",
                        experiment_id,
                        exc,
                    )

        prereg = None
        is_exploratory = bool(exploratory_deviation_reason)
        try:
            prereg = self.get_preregistration_for_experiment(experiment_id)
            if is_exploratory:
                self.log_preregistration_deviation(
                    experiment_id,
                    rationale=exploratory_deviation_reason
                    or "Post-hoc exploratory deviation.",
                    details={"source": "complete_experiment"},
                )
            self.add_entry(
                ExperimentEntry(
                    entry_type="analysis",
                    title="Post-hoc Analysis Link",
                    content=(
                        "Analysis linked to preregistration."
                        if prereg
                        else "Analysis has no preregistration link and is exploratory."
                    ),
                    experiment_id=experiment_id,
                    tags=["analysis_traceability"],
                    metadata={
                        "preregistration_id": prereg.get("preregistration_id")
                        if prereg
                        else None,
                        "analysis_mode": "exploratory"
                        if is_exploratory or not prereg
                        else "confirmatory",
                        "deviation_reason": exploratory_deviation_reason,
                    },
                )
            )
        except sqlite3.OperationalError as exc:
            LOGGER.warning(
                "Post-completion notebook follow-up failed for %s; continuing without notebook persistence: %s",
                experiment_id,
                exc,
            )
        self._publish_lifecycle_event_safe(
            event_type="experiment_completed",
            run_id=experiment_id,
            payload={
                "completed_at": now,
                "results": results,
                "aria_summary": aria_summary,
                "aria_mood": aria_mood,
                "insights": insights or [],
                "llm_analysis": llm_analysis,
            },
        )

    def fail_experiment(
        self, experiment_id: str, error: str, results: Optional[Dict] = None
    ):
        """Mark an experiment as failed. Deletes record if it contains no useful information."""
        self.flush_writes()
        results_blob = self._compress(results) if results else None
        n_prog = results.get("total", 0) if results else 0

        # First update so we have the state
        fail_sql = """UPDATE experiments SET 
               status = 'failed', 
               completed_at = ?,
               aria_summary = ?,
               results_json = ?,
               n_programs_generated = ?
               WHERE experiment_id = ?"""
        fail_params = (
            time.time(),
            f"FAILED: {error}",
            results_blob,
            n_prog,
            experiment_id,
        )
        fail_persisted = False
        try:
            self.conn.execute(fail_sql, fail_params)
            self._maybe_commit()
            fail_persisted = True
        except sqlite3.OperationalError as exc:
            LOGGER.warning(
                "Primary fail_experiment write failed for %s; retrying direct: %s",
                experiment_id,
                exc,
            )
            try:
                self._execute_direct_write(fail_sql, fail_params)
                fail_persisted = True
            except sqlite3.OperationalError as direct_exc:
                LOGGER.warning(
                    "Direct fail_experiment write failed for %s; continuing without notebook persistence: %s",
                    experiment_id,
                    direct_exc,
                )
        if fail_persisted:
            status = self._experiment_status_direct(experiment_id)
            if status != "failed":
                LOGGER.warning(
                    "Experiment %s status is %r after fail_experiment; retrying direct write",
                    experiment_id,
                    status,
                )
                try:
                    self._execute_direct_write(fail_sql, fail_params)
                except sqlite3.OperationalError as exc:
                    LOGGER.warning(
                        "Final fail_experiment retry failed for %s; continuing without notebook persistence: %s",
                        experiment_id,
                        exc,
                    )

        # Delete if it's total junk (no programs AND no LLM insights AND no results)
        try:
            row = self.conn.execute(
                "SELECT llm_analysis, experiment_type, hypothesis, research_question "
                "FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
            has_results = self.conn.execute(
                "SELECT 1 FROM program_results WHERE experiment_id = ? LIMIT 1",
                (experiment_id,),
            ).fetchone()
        except sqlite3.OperationalError as exc:
            LOGGER.warning(
                "Post-failure notebook follow-up failed for %s; continuing without notebook persistence: %s",
                experiment_id,
                exc,
            )
            row = None
            has_results = True

        # Preserve expensive or user-driven failed runs. Investigation/validation
        # jobs may do substantial work without creating new program_results rows,
        # and "unknown" experiment types are often externally-triggered live runs.
        preserve_failed = bool(
            row
            and (
                row["experiment_type"] == "unknown"
                or row["hypothesis"]
                or row["research_question"]
            )
        )

        if (
            not preserve_failed
            and n_prog == 0
            and (not row or not row["llm_analysis"])
            and not has_results
        ):
            self._delete_experiment_cascade(experiment_id)
            LOGGER.info("Deleted zero-value failed experiment %s", experiment_id)

        self._publish_lifecycle_event_safe(
            event_type="experiment_failed",
            run_id=experiment_id,
            payload={
                "completed_at": fail_params[0],
                "error": error,
                "results": results,
            },
        )

    def _delete_experiment_cascade(self, experiment_id: str) -> None:
        """Delete an experiment and all FK-dependent child rows.

        Deletion order respects FK chains:
        attribution_reports → hypotheses → experiments,
        healer_task_events → healer_tasks → experiments, etc.
        """
        # attribution_reports → hypotheses (grandchild)
        hypothesis_rows = self.conn.execute(
            "SELECT hypothesis_id FROM hypotheses WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchall()
        if hypothesis_rows:
            hypothesis_ids = [r[0] for r in hypothesis_rows]
            ph = ",".join("?" for _ in hypothesis_ids)
            self.conn.execute(
                f"DELETE FROM attribution_reports WHERE hypothesis_id IN ({ph})",
                hypothesis_ids,
            )

        # healer_task_events → healer_tasks (grandchild)
        task_rows = self.conn.execute(
            "SELECT task_id FROM healer_tasks WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchall()
        if task_rows:
            task_ids = [r[0] for r in task_rows]
            ph = ",".join("?" for _ in task_ids)
            self.conn.execute(
                f"DELETE FROM healer_task_events WHERE task_id IN ({ph})",
                task_ids,
            )

        # Direct children of experiments
        for table in (
            "entries",
            "insights",
            "hypotheses",
            "preregistration_deviations",
            "hypothesis_preregistrations",
            "healer_tasks",
        ):
            try:
                self.conn.execute(
                    f"DELETE FROM {table} WHERE experiment_id = ?",
                    (experiment_id,),
                )
            except Exception as e:
                LOGGER.debug("Cascade delete from %s skipped: %s", table, e)

        self.conn.execute(
            "DELETE FROM experiments WHERE experiment_id = ?",
            (experiment_id,),
        )
        self._maybe_commit()

    def purge_empty_experiments(self) -> int:
        """Delete failed experiments that produced no program_results.

        Call periodically (e.g. between experiment cycles) to prevent
        empty experiments from accumulating.  Returns count deleted.
        """
        self.flush_writes()
        rows = self.conn.execute("""
            SELECT experiment_id FROM experiments
            WHERE status = 'failed'
            AND NOT EXISTS (
                SELECT 1 FROM program_results p
                WHERE p.experiment_id = experiments.experiment_id
            )
        """).fetchall()
        if not rows:
            return 0
        for r in rows:
            self._delete_experiment_cascade(r[0])
        LOGGER.debug("Purged %d empty failed experiments", len(rows))
        return len(rows)

    def cancel_experiment(self, experiment_id: str) -> bool:
        """Cancel a running experiment by marking it as failed.

        Returns True if the experiment was cancelled, False if not found or
        not in a cancellable state.
        """
        row = self.conn.execute(
            "SELECT status FROM experiments WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
        if not row or row["status"] != "running":
            return False
        self.conn.execute(
            """UPDATE experiments SET status = 'failed', completed_at = ?,
               aria_summary = 'Cancelled by user'
               WHERE experiment_id = ?""",
            (time.time(), experiment_id),
        )
        self._maybe_commit()
        if self._experiment_exists_direct(experiment_id):
            try:
                conn = self._direct_db_conn()
                try:
                    conn.execute(
                        """UPDATE experiments SET status = 'failed', completed_at = ?,
                           aria_summary = 'Cancelled by user'
                           WHERE experiment_id = ?""",
                        (time.time(), experiment_id),
                    )
                    conn.commit()
                finally:
                    conn.close()
            except sqlite3.OperationalError as exc:
                LOGGER.warning(
                    "Direct cancel persistence failed for %s: %s",
                    experiment_id,
                    exc,
                )
        self._publish_lifecycle_event_safe(
            event_type="experiment_failed",
            run_id=experiment_id,
            payload={
                "completed_at": time.time(),
                "error": "Cancelled by user",
                "results": None,
                "cancelled": True,
            },
        )
        return True

    # ── Queries ──

    def get_experiment(self, experiment_id: str) -> Optional[Dict]:
        row = self.conn.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        if d.get("results_json"):
            d["results"] = self._decompress(d["results_json"])
        if d.get("insights_json"):
            d["insights"] = self._decompress(d["insights_json"])
        return d

    def backfill_experiment_metrics(self, experiment_id: str) -> Dict[str, Any]:
        """Backfill missing summary metrics on an existing experiment row.

        Uses already-recorded program_results/results_json only (no rerun).
        """
        exp = self.conn.execute(
            "SELECT experiment_id, best_loss_ratio, best_novelty_score, results_json "
            "FROM experiments WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
        if exp is None:
            return {"found": False, "updated_fields": [], "updated": False}

        agg = self.conn.execute(
            """SELECT
                    MIN(loss_ratio) AS min_loss_ratio,
                    MAX(novelty_score) AS max_novelty_score,
                    AVG(throughput_tok_s) AS avg_throughput_tok_s,
                    COUNT(*) AS n_results
               FROM program_results
               WHERE experiment_id = ?""",
            (experiment_id,),
        ).fetchone()

        min_loss = agg["min_loss_ratio"] if agg else None
        max_novelty = agg["max_novelty_score"] if agg else None
        avg_tp = agg["avg_throughput_tok_s"] if agg else None
        n_results = int(agg["n_results"] or 0) if agg else 0

        perf_tp = None
        raw_results = exp["results_json"]
        if isinstance(raw_results, str) and raw_results:
            try:
                parsed = self._decompress(raw_results)
                perf = parsed.get("perf_report") if isinstance(parsed, dict) else None
                if isinstance(perf, dict):
                    perf_tp = perf.get("avg_throughput_tok_s")
            except (json.JSONDecodeError, KeyError, TypeError, ValueError, zlib.error):
                perf_tp = None

        updates: List[str] = []
        params: List[Any] = []
        updated_fields: List[str] = []

        if exp["best_loss_ratio"] is None and min_loss is not None:
            updates.append("best_loss_ratio = ?")
            params.append(float(min_loss))
            updated_fields.append("best_loss_ratio")

        if exp["best_novelty_score"] is None and max_novelty is not None:
            updates.append("best_novelty_score = ?")
            params.append(float(max_novelty))
            updated_fields.append("best_novelty_score")

        if updates:
            params.append(experiment_id)
            self.conn.execute(
                f"UPDATE experiments SET {', '.join(updates)} WHERE experiment_id = ?",
                tuple(params),
            )
            self._maybe_commit()

        throughput_available = (avg_tp is not None and float(avg_tp) > 0) or (
            perf_tp is not None and float(perf_tp) > 0
        )

        return {
            "found": True,
            "updated": bool(updated_fields),
            "updated_fields": updated_fields,
            "n_program_results": n_results,
            "throughput_available": bool(throughput_available),
        }

    def get_recent_experiments(self, n: int = 20, offset: int = 0) -> List[Dict]:
        n = max(1, int(n))
        offset = max(0, int(offset))
        try:
            rows = self.conn.execute(
                """SELECT experiment_id, timestamp, experiment_type, status,
                          hypothesis, research_question,
                          n_programs_generated, n_stage0_passed, n_stage05_passed,
                          n_stage1_passed,
                          best_loss_ratio, best_novelty_score, aria_mood,
                          aria_summary, duration_seconds
                   FROM experiments ORDER BY timestamp DESC LIMIT ? OFFSET ?""",
                (n, offset),
            ).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.OperationalError as exc:
            LOGGER.warning(
                "Recent experiment query failed; returning empty history: %s",
                exc,
            )
            return []

    def get_latest_completed_experiment_timestamp(self) -> float:
        row = self.conn.execute(
            "SELECT MAX(timestamp) AS latest_ts FROM experiments "
            "WHERE status = 'completed' "
            "   OR (status = 'failed' "
            "       AND n_programs_generated > 0 "
            "       AND aria_summary LIKE 'REPAIRED FROM INTERRUPTED:%')"
        ).fetchone()
        if not row:
            return 0.0
        try:
            return float(row["latest_ts"] or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def get_experiment_trends(self, limit: int = 50) -> List[Dict]:
        """Get cross-experiment trend data for charts."""

        def _mode_factor(mode: Optional[str]) -> float:
            normalized = str(mode or "").strip().lower()
            if normalized in {"investigation", "validation", "single"}:
                return 0.55
            if normalized in {
                "continuous",
                "evolution",
                "synthesis",
                "morphological",
                "training",
            }:
                return 1.0
            return 0.8

        def _resolve_mode(row: Dict) -> str:
            config_mode = None
            raw_config = row.get("config_json")
            if isinstance(raw_config, str) and raw_config.strip():
                try:
                    parsed = json.loads(raw_config)
                    if isinstance(parsed, dict):
                        config_mode = (
                            parsed.get("mode")
                            or parsed.get("run_mode")
                            or parsed.get("experiment_mode")
                        )
                except (json.JSONDecodeError, TypeError):
                    config_mode = None
            return str(config_mode or row.get("experiment_type") or "unknown")

        rows = self.conn.execute(
            """SELECT experiment_id, timestamp, experiment_type, config_json, results_json,
                      n_programs_generated, n_stage0_passed, n_stage05_passed, n_stage1_passed,
                      best_loss_ratio, best_novelty_score, duration_seconds
               FROM experiments
               WHERE status = 'completed'
                  OR (status = 'failed'
                      AND n_programs_generated > 0
                      AND aria_summary LIKE 'REPAIRED FROM INTERRUPTED:%')
               ORDER BY timestamp ASC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        exp_ids = [row["experiment_id"] for row in rows if row["experiment_id"]]
        program_metrics_by_exp: Dict[str, Dict[str, Any]] = {}
        if exp_ids:
            available_cols = self._get_program_results_columns()
            select_parts = ["experiment_id"]

            def _avg(col: str, alias: Optional[str] = None) -> None:
                if col in available_cols:
                    select_parts.append(f"AVG({col}) as {alias or 'avg_' + col}")

            _avg("throughput_tok_s", "avg_throughput_tok_s_programs")
            _avg("routing_tokens_total", "avg_routing_tokens_total")
            _avg("routing_tokens_processed", "avg_routing_tokens_processed")
            _avg("routing_drop_rate", "avg_routing_drop_rate")
            _avg("routing_utilization_entropy", "avg_routing_utilization_entropy")
            _avg("routing_confidence_mean", "avg_routing_confidence_mean")
            _avg(
                "routing_capacity_overflow_count", "avg_routing_capacity_overflow_count"
            )
            _avg("discovery_loss_ratio", "avg_discovery_loss_ratio")
            _avg("validation_loss_ratio", "avg_validation_loss_ratio")
            _avg("generalization_gap", "avg_generalization_gap")
            if (
                "routing_tokens_total" in available_cols
                and "routing_tokens_processed" in available_cols
            ):
                select_parts.append(
                    "AVG(CASE WHEN routing_tokens_total > 0 "
                    "THEN CAST(routing_tokens_processed AS REAL) / routing_tokens_total END) "
                    "as avg_routing_token_retention"
                )
            _avg("depth_savings_ratio", "avg_depth_savings_ratio")
            _avg("effective_depth_ratio", "avg_effective_depth_ratio")
            _avg("recursion_savings_ratio", "avg_recursion_savings_ratio")
            _avg("recursion_depth_ratio", "avg_recursion_depth_ratio")

            if len(select_parts) > 1:
                placeholders = ",".join("?" for _ in exp_ids)
                query = (
                    f"SELECT {', '.join(select_parts)} "
                    f"FROM program_results WHERE experiment_id IN ({placeholders}) "
                    f"GROUP BY experiment_id"
                )
                agg_rows = self.conn.execute(query, exp_ids).fetchall()
                program_metrics_by_exp = {
                    row["experiment_id"]: dict(row) for row in agg_rows
                }
        trends = []
        total_programs = 0
        total_stage1 = 0
        for r in rows:
            d = dict(r)
            exp_id = d.get("experiment_id")
            if exp_id and exp_id in program_metrics_by_exp:
                d.update(program_metrics_by_exp[exp_id])

            # Extract perf report if available
            results_json = d.get("results_json")
            if results_json:
                try:
                    res = self._decompress(results_json)
                    perf = res.get("perf_report")
                    if isinstance(perf, dict):
                        d["avg_step_time_ms"] = perf.get("trace_avg_ms", {}).get(
                            "forward_pass", 0
                        ) + perf.get("trace_avg_ms", {}).get("backward_pass", 0)
                        d["avg_throughput_tok_s"] = perf.get("avg_throughput_tok_s", 0)
                        d["gpu_starvation_ms"] = perf.get("gpu_starvation", {}).get(
                            "total_stall_ms", 0
                        )
                except (
                    json.JSONDecodeError,
                    KeyError,
                    TypeError,
                    ValueError,
                    zlib.error,
                ):
                    pass
            if (
                d.get("avg_throughput_tok_s") in (None, 0)
                and d.get("avg_throughput_tok_s_programs") is not None
            ):
                d["avg_throughput_tok_s"] = d.get("avg_throughput_tok_s_programs")

            n_programs = max(int(d.get("n_programs_generated") or 0), 0)
            n_stage1 = max(int(d.get("n_stage1_passed") or 0), 0)
            total = max(n_programs, 1)
            raw_s1_rate = n_stage1 / total

            trend_mode = _resolve_mode(d)
            mode_factor = _mode_factor(trend_mode)
            effective_n = max(1.0, n_programs * mode_factor)
            trend_weight = min(1.0, effective_n / 20.0)

            d["s1_pass_rate"] = raw_s1_rate
            d["trend_mode"] = trend_mode
            d["_effective_n"] = effective_n
            d["_trend_weight"] = trend_weight

            total_programs += n_programs
            total_stage1 += n_stage1
            trends.append(d)

        if not trends:
            return trends

        overall_rate = total_stage1 / max(total_programs, 1)
        prior_strength = 12.0

        for d in trends:
            raw_rate = d.get("s1_pass_rate") or 0.0
            effective_n = d.get("_effective_n") or 1.0
            trend_weight = d.get("_trend_weight") or 0.0

            shrinkage = effective_n / (effective_n + prior_strength)
            adjusted_rate = overall_rate + shrinkage * (raw_rate - overall_rate)

            variance = max(adjusted_rate * (1.0 - adjusted_rate), 0.0)
            stderr = math.sqrt(variance / max(effective_n, 1.0))
            halfwidth = 1.96 * stderr
            lower = max(0.0, adjusted_rate - halfwidth)
            upper = min(1.0, adjusted_rate + halfwidth)

            if effective_n >= 20:
                confidence = "high"
            elif effective_n >= 8:
                confidence = "medium"
            else:
                confidence = "low"

            d["adjusted_s1_pass_rate"] = round(adjusted_rate, 6)
            d["s1_confidence_lower"] = round(lower, 6)
            d["s1_confidence_upper"] = round(upper, 6)
            d["s1_confidence_halfwidth"] = round(halfwidth, 6)
            d["trend_weight"] = round(trend_weight, 4)
            d["trend_confidence"] = confidence

            d.pop("_effective_n", None)
            d.pop("_trend_weight", None)
            # Remove raw blob columns that are not JSON-serializable
            d.pop("results_json", None)
            d.pop("config_json", None)

        return sanitize_for_db(trends)

    def get_campaign_experiments(self, campaign_id: str) -> List[Dict]:
        """Get all experiments for a campaign."""
        rows = self.conn.execute(
            """SELECT experiment_id, timestamp, experiment_type, status,
                      hypothesis, n_programs_generated, n_stage1_passed,
                      best_loss_ratio, best_novelty_score, aria_mood,
                      duration_seconds
               FROM experiments WHERE campaign_id = ?
               ORDER BY timestamp ASC""",
            (campaign_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Hypotheses ──

    def record_hypothesis(
        self,
        campaign_id: Optional[str],
        prediction: str,
        reasoning: str,
        test_method: str,
        success_metric: str,
        parent_id: Optional[str] = None,
        confidence: float = 0.5,
        experiment_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Record a structured hypothesis. Returns hypothesis_id."""
        hypothesis_id = str(uuid.uuid4())[:12]
        now = time.time()
        self.conn.execute(
            """INSERT INTO hypotheses
            (hypothesis_id, campaign_id, experiment_id, timestamp,
             prediction, reasoning, test_method, success_metric,
             parent_hypothesis_id, status, confidence_before, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)""",
            (
                hypothesis_id,
                campaign_id,
                experiment_id,
                now,
                prediction,
                reasoning,
                test_method,
                success_metric,
                parent_id,
                confidence,
                json.dumps(metadata) if metadata else None,
            ),
        )
        # Update parent's child list
        if parent_id:
            parent = self.conn.execute(
                "SELECT child_hypotheses FROM hypotheses WHERE hypothesis_id = ?",
                (parent_id,),
            ).fetchone()
            if parent:
                children = json.loads(parent["child_hypotheses"] or "[]")
                children.append(hypothesis_id)
                self.conn.execute(
                    "UPDATE hypotheses SET child_hypotheses = ? WHERE hypothesis_id = ?",
                    (json.dumps(children), parent_id),
                )
        self._maybe_commit()
        return hypothesis_id

    def resolve_hypothesis(
        self,
        hypothesis_id: str,
        status: str,
        evidence: str,
        summary: str,
        confidence_after: float,
    ) -> None:
        """Resolve a hypothesis with outcome."""
        self.conn.execute(
            """UPDATE hypotheses SET
                status = ?, outcome_evidence = ?, outcome_summary = ?,
                confidence_after = ?
            WHERE hypothesis_id = ?""",
            (status, evidence, summary, confidence_after, hypothesis_id),
        )
        self._maybe_commit()

    def get_hypothesis_chain(
        self, hypothesis_id: str, max_depth: int = 500
    ) -> List[Dict]:
        """Trace lineage from root to all descendants."""
        # Find root (with cycle detection)
        current = hypothesis_id
        visited = {current}
        for _ in range(max_depth):
            row = self.conn.execute(
                "SELECT * FROM hypotheses WHERE hypothesis_id = ?",
                (current,),
            ).fetchone()
            if row is None:
                break
            parent = row["parent_hypothesis_id"]
            if parent is None or parent in visited:
                break
            visited.add(parent)
            current = parent

        # BFS from root (with max nodes limit)
        chain: List[Dict] = []
        queue_ids = [current]
        seen: set = set()
        while queue_ids and len(chain) < max_depth:
            hid = queue_ids.pop(0)
            if hid in seen:
                continue
            seen.add(hid)
            row = self.conn.execute(
                "SELECT * FROM hypotheses WHERE hypothesis_id = ?",
                (hid,),
            ).fetchone()
            if row is None:
                continue
            d = dict(row)
            chain.append(d)
            children = json.loads(d.get("child_hypotheses") or "[]")
            queue_ids.extend(children)
        return chain
