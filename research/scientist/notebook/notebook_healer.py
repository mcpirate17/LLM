from __future__ import annotations

"""Auto-extracted mixin for LabNotebook."""

import json
import sqlite3
import time
import uuid
from typing import Any, Dict, List, Optional

from ._shared import LOGGER


class _HealerMixin:
    """Healer operations for the Lab Notebook."""

    __slots__ = ()

    # ── Code Healer ──

    def create_healer_task(
        self,
        experiment_id: Optional[str],
        trigger_type: str,
        scope: str,
        reproduction_steps: List[str],
        acceptance_tests: List[str],
        model_endpoint: Optional[str],
        sandbox_policy: Dict[str, Any],
        trigger_payload: Optional[Dict[str, Any]] = None,
    ) -> str:
        task_id = f"heal-{uuid.uuid4().hex[:10]}"
        now = time.time()
        trigger_payload_json = json.dumps(trigger_payload or {})
        trigger_payload_json = self._maybe_store_json_artifact(
            table_name="healer_tasks",
            row_pk=task_id,
            column_name="trigger_payload_json",
            payload_json=trigger_payload_json,
        )
        try:
            self.conn.execute(
                """INSERT INTO healer_tasks
                (task_id, timestamp, experiment_id, trigger_type, trigger_payload_json,
                 scope, reproduction_steps_json, acceptance_tests_json, model_endpoint,
                 sandbox_policy_json, state)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')""",
                (
                    task_id,
                    now,
                    experiment_id,
                    trigger_type,
                    trigger_payload_json,
                    scope,
                    json.dumps(reproduction_steps or []),
                    json.dumps(acceptance_tests or []),
                    model_endpoint,
                    json.dumps(sandbox_policy or {}),
                ),
            )
            self._maybe_commit()
        except sqlite3.OperationalError as exc:
            LOGGER.warning(
                "Healer task write failed for experiment %s; continuing without notebook persistence: %s",
                experiment_id or "unscoped",
                exc,
            )
        return task_id

    def update_healer_task(
        self,
        task_id: str,
        state: Optional[str] = None,
        patch_summary: Optional[str] = None,
        risk_assessment: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        completed: bool = False,
    ) -> None:
        sets: List[str] = []
        params: List[Any] = []
        if state is not None:
            sets.append("state = ?")
            params.append(state)
        if patch_summary is not None:
            sets.append("patch_summary = ?")
            params.append(patch_summary)
        if risk_assessment is not None:
            sets.append("risk_assessment = ?")
            params.append(risk_assessment)
        if result is not None:
            sets.append("result_json = ?")
            result_json = json.dumps(result)
            params.append(
                self._maybe_store_json_artifact(
                    table_name="healer_tasks",
                    row_pk=task_id,
                    column_name="result_json",
                    payload_json=result_json,
                )
            )
        if completed:
            sets.append("completed_at = ?")
            params.append(time.time())
        if not sets:
            return
        params.append(task_id)
        self.conn.execute(
            f"UPDATE healer_tasks SET {', '.join(sets)} WHERE task_id = ?",
            params,
        )
        self._maybe_commit()

    def add_healer_event(
        self,
        task_id: str,
        message: str,
        state: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> str:
        event_id = str(uuid.uuid4())[:12]
        try:
            self.conn.execute(
                """INSERT INTO healer_task_events
                (event_id, task_id, timestamp, state, message, payload_json)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    event_id,
                    task_id,
                    time.time(),
                    state,
                    message,
                    json.dumps(payload or {}),
                ),
            )
            self._maybe_commit()
        except sqlite3.OperationalError as exc:
            LOGGER.warning(
                "Healer event write failed for task %s; continuing without notebook persistence: %s",
                task_id,
                exc,
            )
        return event_id

    def get_healer_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        try:
            row = self.conn.execute(
                "SELECT * FROM healer_tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        except sqlite3.OperationalError as exc:
            LOGGER.warning(
                "Healer task query failed for %s; returning no task: %s",
                task_id,
                exc,
            )
            return None
        if row is None:
            return None
        out = dict(row)
        for field in (
            "trigger_payload_json",
            "reproduction_steps_json",
            "acceptance_tests_json",
            "sandbox_policy_json",
            "result_json",
        ):
            raw = out.get(field)
            if raw:
                try:
                    out[field] = self._json_loads_maybe_artifact(raw)
                except (
                    TypeError,
                    json.JSONDecodeError,
                    ValueError,
                    FileNotFoundError,
                    KeyError,
                ):
                    pass
        return out

    def get_recent_healer_tasks(self, limit: int = 20) -> List[Dict[str, Any]]:
        try:
            rows = self.conn.execute(
                """SELECT * FROM healer_tasks
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            LOGGER.warning(
                "Recent healer task query failed; returning empty results: %s",
                exc,
            )
            return []
        out: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            for key in (
                "trigger_payload_json",
                "reproduction_steps_json",
                "acceptance_tests_json",
                "sandbox_policy_json",
                "result_json",
            ):
                raw = item.get(key)
                if raw:
                    try:
                        item[key] = self._json_loads_maybe_artifact(raw)
                    except (
                        TypeError,
                        json.JSONDecodeError,
                        ValueError,
                        FileNotFoundError,
                        KeyError,
                    ):
                        pass
            out.append(item)
        return out

    def get_healer_events(self, task_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            """SELECT * FROM healer_task_events
               WHERE task_id = ?
               ORDER BY timestamp DESC
               LIMIT ?""",
            (task_id, limit),
        ).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            raw = item.get("payload_json")
            if raw:
                try:
                    item["payload_json"] = json.loads(raw)
                except (TypeError, json.JSONDecodeError):
                    pass
            out.append(item)
        return out
