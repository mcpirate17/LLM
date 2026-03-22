"""
Action Queue — Persistent storage for autonomous actions.

Stores Aria's autonomous decisions in SQLite alongside the lab notebook.
Supports approve/dismiss/undo operations with time-windowed undo snapshots.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS autonomous_actions (
    action_id TEXT PRIMARY KEY,
    decision_type TEXT NOT NULL,
    behavior TEXT NOT NULL,         -- auto / notify / ask
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    detail_json TEXT,
    status TEXT NOT NULL DEFAULT 'pending',  -- pending / executed / undone / dismissed / failed / expired
    created_at REAL NOT NULL,
    executed_at REAL,
    undo_snapshot_json TEXT,
    experiment_id TEXT,
    result_id TEXT,
    source TEXT DEFAULT 'autonomy'
);

CREATE INDEX IF NOT EXISTS idx_actions_status ON autonomous_actions(status);
CREATE INDEX IF NOT EXISTS idx_actions_created ON autonomous_actions(created_at);
"""

UNDO_WINDOW_SECONDS = 300  # 5 minutes


class ActionStore:
    """SQLite-backed persistent action queue."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        try:
            self.conn.executescript(_SCHEMA)
        except sqlite3.OperationalError:
            # Table already exists
            pass

    def insert(
        self,
        action_id: str,
        decision_type: str,
        behavior: str,
        title: str,
        summary: str,
        detail: Optional[Dict] = None,
        status: str = "pending",
        executed_at: Optional[float] = None,
        undo_snapshot: Optional[Dict] = None,
        experiment_id: Optional[str] = None,
        result_id: Optional[str] = None,
        source: str = "autonomy",
    ) -> str:
        """Insert a new action. Returns action_id."""
        self.conn.execute(
            """INSERT OR REPLACE INTO autonomous_actions
               (action_id, decision_type, behavior, title, summary, detail_json,
                status, created_at, executed_at, undo_snapshot_json,
                experiment_id, result_id, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                action_id,
                decision_type,
                behavior,
                title,
                summary,
                json.dumps(detail) if detail else None,
                status,
                time.time(),
                executed_at,
                json.dumps(undo_snapshot) if undo_snapshot else None,
                experiment_id,
                result_id,
                source,
            ),
        )
        self.conn.commit()
        return action_id

    def update_status(
        self,
        action_id: str,
        status: str,
        executed_at: Optional[float] = None,
        undo_snapshot: Optional[Dict] = None,
    ) -> bool:
        """Update an action's status. Returns True if a row was updated."""
        fields = ["status = ?"]
        params: list = [status]
        if executed_at is not None:
            fields.append("executed_at = ?")
            params.append(executed_at)
        if undo_snapshot is not None:
            fields.append("undo_snapshot_json = ?")
            params.append(json.dumps(undo_snapshot))
        params.append(action_id)
        cursor = self.conn.execute(
            f"UPDATE autonomous_actions SET {', '.join(fields)} WHERE action_id = ?",
            params,
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def get(self, action_id: str) -> Optional[Dict[str, Any]]:
        """Get a single action by ID."""
        row = self.conn.execute(
            "SELECT * FROM autonomous_actions WHERE action_id = ?",
            (action_id,),
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def get_pending(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get pending + recently executed (undoable) actions."""
        now = time.time()
        cutoff = now - UNDO_WINDOW_SECONDS
        rows = self.conn.execute(
            """SELECT * FROM autonomous_actions
               WHERE status = 'pending'
                  OR (status = 'executed' AND behavior = 'notify')
                  OR (status = 'executed' AND executed_at >= ? AND undo_snapshot_json IS NOT NULL)
               ORDER BY created_at DESC
               LIMIT ?""",
            (cutoff, limit),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent actions for the activity feed."""
        rows = self.conn.execute(
            """SELECT * FROM autonomous_actions
               ORDER BY created_at DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def expire_old_pending(self, max_age_seconds: int = 3600) -> int:
        """Expire pending actions older than max_age_seconds."""
        cutoff = time.time() - max_age_seconds
        cursor = self.conn.execute(
            "UPDATE autonomous_actions SET status = 'expired' WHERE status = 'pending' AND created_at < ?",
            (cutoff,),
        )
        self.conn.commit()
        return cursor.rowcount

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        # Parse JSON fields
        for json_field in ("detail_json", "undo_snapshot_json"):
            raw = d.pop(json_field, None)
            key = json_field.replace("_json", "")
            if raw:
                try:
                    d[key] = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    d[key] = None
            else:
                d[key] = None

        # Compute undoable
        now = time.time()
        d["undoable"] = (
            d.get("status") == "executed"
            and d.get("executed_at") is not None
            and d.get("undo_snapshot") is not None
            and (now - (d.get("executed_at") or 0)) < UNDO_WINDOW_SECONDS
        )
        if d["undoable"] and d.get("executed_at"):
            d["undo_remaining_seconds"] = max(
                0, int(UNDO_WINDOW_SECONDS - (now - d["executed_at"]))
            )
        else:
            d["undo_remaining_seconds"] = 0

        return d
