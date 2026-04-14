from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from typing import Optional

from ..schema import LIFECYCLE_EVENT_TYPES, RuntimeEvent
from ..spool import NdjsonEventSpool, SpoolOffset
from ..state_machine import LifecycleStateMachine

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProjectorStatus:
    last_offset: Optional[SpoolOffset]
    last_event_id: Optional[str]
    applied_count: int
    degraded: bool


class LifecycleProjector:
    """Projects lifecycle events from the spool into SQLite."""

    PROJECTOR_NAME = "lifecycle"

    def __init__(self, conn: sqlite3.Connection, *, spool: NdjsonEventSpool) -> None:
        self.conn = conn
        self.spool = spool
        self.machine = LifecycleStateMachine()
        self._ensure_tables()

    def replay_once(self) -> ProjectorStatus:
        checkpoint = self._load_checkpoint()
        last_offset = checkpoint
        last_event_id: Optional[str] = None
        applied_count = 0
        degraded = False
        try:
            for record in self.spool.replay(after=checkpoint):
                if (
                    record.event.event_type not in LIFECYCLE_EVENT_TYPES
                    or not record.event.run_id
                ):
                    continue
                if self._is_applied(record.event.event_id):
                    last_offset = record.offset
                    last_event_id = record.event.event_id
                    continue
                self._apply_event(record.event)
                self._mark_applied(record.event)
                self._store_checkpoint(record.offset)
                self.conn.commit()
                logger.info(
                    "Projected lifecycle event: type=%s run_id=%s event_id=%s offset=%s:%d",
                    record.event.event_type,
                    record.event.run_id,
                    record.event.event_id[:12],
                    record.offset.segment,
                    record.offset.line_number,
                )
                last_offset = record.offset
                last_event_id = record.event.event_id
                applied_count += 1
        except sqlite3.DatabaseError as exc:
            degraded = True
            logger.error(
                "Projector degraded — SQLite error during replay: %s: %s",
                type(exc).__name__,
                exc,
            )
            self.conn.rollback()
        return ProjectorStatus(
            last_offset=last_offset,
            last_event_id=last_event_id,
            applied_count=applied_count,
            degraded=degraded,
        )

    def _apply_event(self, event: RuntimeEvent) -> None:
        current_type = self._load_current_event_type(event.run_id)
        if event.event_type == current_type:
            return  # duplicate, already applied
        if not self.machine.is_valid_transition(current_type, event.event_type):
            return  # invalid transition, skip silently in projector
        handler = getattr(self, f"_apply_{event.event_type}")
        handler(event)

    def _apply_experiment_start_requested(self, event: RuntimeEvent) -> None:
        # Spool-only in the first cut; checkpoint for ordering but no row write yet.
        return

    def _apply_experiment_started(self, event: RuntimeEvent) -> None:
        payload = event.payload
        self.conn.execute(
            """INSERT INTO experiments
               (experiment_id, timestamp, experiment_type, status, hypothesis,
                research_question, preregistration_id, config_json, started_at)
               VALUES (?, ?, ?, 'running', ?, ?, ?, ?, ?)
               ON CONFLICT(experiment_id) DO UPDATE SET
                 status = excluded.status,
                 hypothesis = excluded.hypothesis,
                 research_question = excluded.research_question,
                 preregistration_id = excluded.preregistration_id,
                 config_json = excluded.config_json,
                 started_at = excluded.started_at""",
            (
                event.run_id,
                float(payload.get("timestamp", event.created_at)),
                str(payload.get("experiment_type", "unknown")),
                payload.get("hypothesis"),
                payload.get("research_question"),
                payload.get("preregistration_id"),
                json.dumps(payload.get("config") or {}),
                float(payload.get("started_at", event.created_at)),
            ),
        )

    def _apply_experiment_start_failed(self, event: RuntimeEvent) -> None:
        self.conn.execute(
            """INSERT OR IGNORE INTO entries
               (entry_id, experiment_id, timestamp, entry_type, title, content, metadata_json, tags)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event.event_id,
                event.run_id,
                event.created_at,
                "error",
                f"Experiment {event.run_id} launch failed",
                str(event.payload.get("error") or "unknown launch failure"),
                json.dumps(event.payload),
                "event_bus,lifecycle,start_failed",
            ),
        )

    def _apply_experiment_completed(self, event: RuntimeEvent) -> None:
        payload = event.payload
        results = payload.get("results") or {}
        started_at = self._fetch_started_at(event.run_id)
        completed_at = float(payload.get("completed_at", event.created_at))
        duration_seconds = (
            max(0.0, completed_at - started_at) if started_at is not None else None
        )
        self.conn.execute(
            """UPDATE experiments SET
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
               WHERE experiment_id = ?""",
            (
                json.dumps(results),
                int(results.get("total", 0)),
                int(results.get("stage0_passed", 0)),
                int(results.get("stage05_passed", 0)),
                int(results.get("stage1_passed", 0)),
                results.get("best_loss_ratio"),
                results.get("best_novelty_score"),
                payload.get("aria_summary", ""),
                payload.get("aria_mood", "contemplative"),
                json.dumps(payload.get("insights") or []),
                payload.get("llm_analysis"),
                completed_at,
                duration_seconds,
                event.run_id,
            ),
        )

    def _apply_experiment_failed(self, event: RuntimeEvent) -> None:
        payload = event.payload
        self.conn.execute(
            """UPDATE experiments SET
                 status = 'failed',
                 completed_at = ?,
                 aria_summary = ?,
                 results_json = ?,
                 n_programs_generated = ?
               WHERE experiment_id = ?""",
            (
                float(payload.get("completed_at", event.created_at)),
                f"FAILED: {payload.get('error', 'unknown')}",
                json.dumps(payload.get("results")) if payload.get("results") else None,
                int((payload.get("results") or {}).get("total", 0)),
                event.run_id,
            ),
        )

    def _ensure_tables(self) -> None:
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS applied_runtime_events (
                   event_id TEXT PRIMARY KEY,
                   event_type TEXT NOT NULL,
                   run_id TEXT,
                   applied_at REAL NOT NULL
               )"""
        )
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS runtime_projector_checkpoints (
                   projector_name TEXT PRIMARY KEY,
                   segment TEXT NOT NULL,
                   line_number INTEGER NOT NULL,
                   updated_at REAL NOT NULL
               )"""
        )
        self.conn.commit()

    def _mark_applied(self, event: RuntimeEvent) -> None:
        self.conn.execute(
            """INSERT OR IGNORE INTO applied_runtime_events
               (event_id, event_type, run_id, applied_at)
               VALUES (?, ?, ?, ?)""",
            (event.event_id, event.event_type, event.run_id, time.time()),
        )

    def _is_applied(self, event_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM applied_runtime_events WHERE event_id = ?",
            (event_id,),
        ).fetchone()
        return row is not None

    def _load_checkpoint(self) -> Optional[SpoolOffset]:
        row = self.conn.execute(
            """SELECT segment, line_number
               FROM runtime_projector_checkpoints
               WHERE projector_name = ?""",
            (self.PROJECTOR_NAME,),
        ).fetchone()
        if row is None:
            return None
        return SpoolOffset(segment=row[0], line_number=int(row[1]))

    def _store_checkpoint(self, offset: SpoolOffset) -> None:
        self.conn.execute(
            """INSERT INTO runtime_projector_checkpoints
               (projector_name, segment, line_number, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(projector_name) DO UPDATE SET
                 segment = excluded.segment,
                 line_number = excluded.line_number,
                 updated_at = excluded.updated_at""",
            (self.PROJECTOR_NAME, offset.segment, offset.line_number, time.time()),
        )

    def _load_current_event_type(self, run_id: str) -> Optional[str]:
        row = self.conn.execute(
            """SELECT event_type FROM applied_runtime_events
               WHERE run_id = ? ORDER BY applied_at DESC LIMIT 1""",
            (run_id,),
        ).fetchone()
        return str(row[0]) if row else None

    def _fetch_started_at(self, run_id: str) -> Optional[float]:
        row = self.conn.execute(
            "SELECT started_at FROM experiments WHERE experiment_id = ?",
            (run_id,),
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return float(row[0])
