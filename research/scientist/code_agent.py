"""Code Agent Interface — breaks circular runner <-> api dependency.

Runner submodules (cycle.py, dashboard.py) import from this module instead
of from api.py, avoiding the circular import chain:
  api.py -> runner/__init__.py -> runner/cycle.py -> api.py

Contains the actual implementations of code agent spawn/snapshot functions.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _should_autospawn_self_repair(error_message: str) -> bool:
    """Heuristic: should we automatically spawn a code agent to fix this error?

    Pure function -- no api.py dependency needed.
    """
    lowered = str(error_message or "").lower()
    triggers = (
        "unexpected keyword argument",
        "attributeerror",
        "nameerror",
        "importerror",
        "modulenotfounderror",
        "syntaxerror",
        "typeerror",
        "keyerror",
        "valueerror",
        "traceback",
    )
    return any(token in lowered for token in triggers)


def _spawn_code_agent_task(
    goal: str,
    notebook_path: str,
    allow_write: bool = True,
    session_id: str = "",
) -> Dict[str, Any]:
    """Spawn a background code agent task for autonomous repair/refactor.

    Records the task in the shared _CODE_AGENT_TASKS dict and returns
    a task descriptor dict.
    """
    from .api_routes._helpers import _CODE_AGENT_TASKS, _CODE_AGENT_TASKS_LOCK

    task_id = f"agent-{uuid.uuid4().hex[:12]}"
    task = {
        "task_id": task_id,
        "goal": str(goal or "").strip(),
        "notebook_path": notebook_path,
        "allow_write": bool(allow_write),
        "session_id": str(session_id or "").strip(),
        "status": "queued",
        "created_at": time.time(),
        "started_at": None,
        "completed_at": None,
        "result": None,
        "error": None,
    }

    with _CODE_AGENT_TASKS_LOCK:
        _CODE_AGENT_TASKS[task_id] = task

    # No actual code agent implementation exists — mark unavailable synchronously.
    now = time.time()
    task["status"] = "unavailable"
    task["started_at"] = now
    task["completed_at"] = now
    task["result"] = {
        "status": "unavailable",
        "message": "Code agent not implemented",
    }
    with _CODE_AGENT_TASKS_LOCK:
        _CODE_AGENT_TASKS[task_id] = task
    logger.warning(f"Code agent {task_id} unavailable (not implemented): {goal[:120]}")

    return task


def _code_agent_task_snapshot(task_id: str) -> Optional[Dict[str, Any]]:
    """Get a point-in-time copy of a task's state."""
    from .api_routes._helpers import _CODE_AGENT_TASKS, _CODE_AGENT_TASKS_LOCK

    with _CODE_AGENT_TASKS_LOCK:
        task = _CODE_AGENT_TASKS.get(task_id)
    if task is None:
        return None
    return dict(task)
