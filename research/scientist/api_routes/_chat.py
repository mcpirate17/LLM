"""Chat-related helper functions for Aria chat endpoints.

Contains message classification, guardrail tracking, local agent execution,
action parsing, and text utility helpers.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Message classification ─────────────────────────────────────────────


def chat_requests_detailed_response(question: str) -> bool:
    """Check if the user's question requests a detailed/verbose response."""
    q = question.lower().strip()
    return any(
        kw in q
        for kw in (
            "detail",
            "explain in detail",
            "verbose",
            "thorough",
            "comprehensive",
            "in depth",
        )
    )


def chat_requests_summary_response(question: str) -> bool:
    """Check if the user's question requests a summary."""
    q = question.lower().strip()
    return any(
        kw in q
        for kw in (
            "summarize",
            "summary",
            "overview",
            "what happened",
            "status update",
            "progress report",
        )
    )


def chat_requests_brief_response(question: str) -> bool:
    """Check if the user wants a brief/concise response."""
    q = question.lower().strip()
    return any(
        kw in q
        for kw in ("brief", "short", "concise", "quick", "tl;dr", "tldr", "one line")
    )


def chat_requests_self_fix_now(question: str) -> bool:
    """Check if the user is asking Aria to fix itself."""
    q = question.lower().strip()
    return any(
        kw in q
        for kw in (
            "fix yourself",
            "fix what's wrong",
            "self-repair",
            "self repair",
            "heal yourself",
            "fix it yourself",
        )
    )


def chat_requests_codebase_fix(question: str) -> bool:
    """Check if the user is requesting a codebase/code fix."""
    q = question.lower().strip()
    return any(
        kw in q
        for kw in (
            "fix the",
            "fix this",
            "fix bug",
            "patch",
            "repair",
            "debug this",
            "fix code",
            "fix error",
        )
    )


# ── Guardrail tracking ─────────────────────────────────────────────────
_GUARDRAIL_LOCK = threading.Lock()
_GUARDRAIL_EVENTS: List[Dict[str, Any]] = []


def record_chat_guardrail_event(
    *,
    actionable: bool,
    advice_only: bool,
    summary_text: str,
) -> None:
    """Record a chat guardrail event for rate limiting and auditing."""
    event = {
        "timestamp": time.time(),
        "actionable": actionable,
        "advice_only": advice_only,
        "summary_length": len(summary_text),
    }
    with _GUARDRAIL_LOCK:
        _GUARDRAIL_EVENTS.append(event)
        # Keep only last 500 events
        if len(_GUARDRAIL_EVENTS) > 500:
            del _GUARDRAIL_EVENTS[: len(_GUARDRAIL_EVENTS) - 500]


def chat_guardrail_snapshot(*, window: int = 200) -> Dict[str, Any]:
    """Return a snapshot of recent chat guardrail events."""
    with _GUARDRAIL_LOCK:
        recent = list(_GUARDRAIL_EVENTS[-window:])
    total = len(recent)
    actionable = sum(1 for e in recent if e.get("actionable"))
    advice_only = sum(1 for e in recent if e.get("advice_only"))
    return {
        "total_events": total,
        "actionable": actionable,
        "advice_only": advice_only,
        "actionable_rate": round(actionable / max(total, 1), 4),
        "window": window,
    }


# ── Code agent task helpers ─────────────────────────────────────────────


def code_agent_task_snapshot(task_id: str) -> Optional[Dict[str, Any]]:
    """Get a snapshot of a code agent task by ID.

    Looks up the task in the shared _CODE_AGENT_TASKS dict from _helpers.
    """
    from ._helpers import _CODE_AGENT_TASKS, _CODE_AGENT_TASKS_LOCK

    with _CODE_AGENT_TASKS_LOCK:
        task = _CODE_AGENT_TASKS.get(task_id)
    if task is None:
        return None
    return dict(task)


def summarize_agent_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """Produce a concise milestone summary for an agent task."""
    status = str(task.get("status") or "unknown").strip()
    goal = str(task.get("goal") or "").strip()
    task_id = str(task.get("task_id") or "").strip()

    milestone = "queued"
    if status == "running":
        milestone = "executing"
    elif status == "completed":
        milestone = "done"
    elif status == "failed":
        milestone = "failed"

    return {
        "task_id": task_id,
        "status": status,
        "milestone_summary": milestone,
        "goal_preview": goal[:120] + ("..." if len(goal) > 120 else ""),
    }


# ── Local chat agent ────────────────────────────────────────────────────


def run_local_chat_agent(
    *,
    question: str,
    runner,
    nb,
    notebook_path: str,
    enable_code_tools: bool = True,
) -> Dict[str, Any]:
    """Run local lightweight chat agent to gather context for the question.

    Returns dict with 'tools_used', 'summary', 'code_hits'.
    """
    result: Dict[str, Any] = {"tools_used": [], "summary": "", "code_hits": []}

    if not enable_code_tools:
        return result

    q_lower = question.lower()

    # Search workspace for relevant code if the question mentions code/files
    code_keywords = (
        "code",
        "file",
        "function",
        "class",
        "error",
        "bug",
        "fix",
        "module",
    )
    if any(kw in q_lower for kw in code_keywords):
        result["tools_used"].append("workspace_search")
        try:
            ws_root = chat_workspace_root(notebook_path)
            hits = query_file_index(question, ws_root, max_results=5)
            result["code_hits"] = hits
            if hits:
                result["summary"] = f"Found {len(hits)} relevant files."
        except Exception:
            pass

    return result


def chat_workspace_root(notebook_path: str) -> Path:
    """Determine the workspace root directory for code search."""
    return Path(notebook_path).parent


def query_file_index(
    query: str,
    workspace_root: Path,
    max_results: int = 5,
) -> List[Dict[str, Any]]:
    """Query the file index for files matching the query terms.

    Simple keyword-based file matching against the workspace.
    """
    from ._helpers import (
        _WORKSPACE_FILE_INDEX,
        _WORKSPACE_FILE_INDEX_LOCK,
        _WORKSPACE_FILE_INDEX_BUILT_AT,
    )

    # Build index if stale (older than 5 minutes)
    now = time.time()
    with _WORKSPACE_FILE_INDEX_LOCK:
        if now - _WORKSPACE_FILE_INDEX_BUILT_AT > 300:
            _rebuild_file_index(workspace_root)

    terms = set(query.lower().split())
    scored: List[tuple] = []

    with _WORKSPACE_FILE_INDEX_LOCK:
        items = list(_WORKSPACE_FILE_INDEX.items())

    for rel_path, info in items:
        path_lower = rel_path.lower()
        score = sum(1 for t in terms if t in path_lower) * 10
        try:
            full_path = workspace_root / rel_path
            if full_path.is_file() and full_path.stat().st_size < 350_000:
                content = full_path.read_text(errors="ignore").lower()
                score += sum(content.count(t) for t in terms)
        except Exception:
            pass
        if score > 0:
            scored.append((score, rel_path, info))

    scored.sort(key=lambda x: -x[0])
    return [
        {
            "rel_path": item[1],
            "path": item[1],
            "abs_path": str(workspace_root / item[1]),
            "score": item[0],
            "line": 0,
        }
        for item in scored[:max_results]
    ]


def _rebuild_file_index(workspace_root: Path) -> None:
    """Rebuild the workspace file index.

    Must be called while *not* holding _WORKSPACE_FILE_INDEX_LOCK, or by a
    caller that already holds it (the current call-site in query_file_index
    holds the lock).
    """
    import research.scientist.api_routes._helpers as _h

    include_ext = {".py", ".js", ".ts", ".tsx", ".md", ".json"}
    skip_dirs = {
        ".git",
        "node_modules",
        "__pycache__",
        "build",
        "dist",
        ".venv",
        "venv",
        ".mypy_cache",
        ".pytest_cache",
    }

    index: Dict[str, Dict[str, Any]] = {}
    try:
        for path in workspace_root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in include_ext:
                continue
            if any(part in skip_dirs for part in path.parts):
                continue
            try:
                rel = str(path.relative_to(workspace_root))
                index[rel] = {"size": path.stat().st_size}
            except Exception:
                continue
    except Exception:
        pass

    # Update the shared index (caller already holds the lock)
    _h._WORKSPACE_FILE_INDEX.clear()
    _h._WORKSPACE_FILE_INDEX.update(index)
    _h._WORKSPACE_FILE_INDEX_BUILT_AT = time.time()


# ── Action response parsing ────────────────────────────────────────────


def parse_action_contract_response(text: str) -> Dict[str, Any]:
    """Parse an LLM response following the action contract format.

    Looks for ```action JSON blocks and extracts executable actions.
    Returns dict with 'actions' list, 'summary' text, and 'advice_only' flag.
    """
    actions: List[Dict[str, Any]] = []
    summary_parts: List[str] = []

    # Extract ```action blocks
    pattern = r"```action\s*\n(.*?)\n```"
    matches = re.findall(pattern, text, re.DOTALL)

    for match in matches:
        try:
            action = json.loads(match.strip())
            if isinstance(action, dict) and action.get("type"):
                actions.append(action)
        except (json.JSONDecodeError, TypeError):
            continue

    # Everything outside action blocks is summary
    cleaned = re.sub(pattern, "", text, flags=re.DOTALL).strip()
    if cleaned:
        summary_parts.append(cleaned)

    return {
        "actions": actions,
        "summary": " ".join(summary_parts),
        "advice_only": len(actions) == 0,
    }


# ── Text utilities ─────────────────────────────────────────────────────


def truncate_summary(text: str, max_len: int = 200) -> str:
    """Truncate text to max_len, appending '...' if truncated."""
    text = str(text or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def estimate_tokens(text: str) -> int:
    """Rough token count estimate (~4 chars per token)."""
    return max(1, len(str(text or "")) // 4)


# ── Ollama helpers ─────────────────────────────────────────────────────


def get_local_ollama_settings() -> Dict[str, Any]:
    """Get local Ollama helper settings from environment."""
    return {
        "host": os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        "enabled": os.environ.get("ARIA_OLLAMA_ENABLED", "0").lower()
        in ("1", "true", "yes"),
        "max_small_workers": int(os.environ.get("ARIA_OLLAMA_MAX_SMALL_WORKERS", "3")),
        "model_3b": os.environ.get("ARIA_OLLAMA_MODEL_3B", "phi3:3b"),
        "model_7b": os.environ.get("ARIA_OLLAMA_MODEL_7B", "codellama:7b"),
    }


def local_ollama_helper_status(llm) -> Dict[str, Any]:
    """Report Ollama local helper availability."""
    settings = get_local_ollama_settings()
    if not settings.get("enabled"):
        return {"available": False, "reason": "disabled"}

    host = settings.get("host", "http://localhost:11434")
    try:
        import requests as _requests

        resp = _requests.get(f"{host}/api/tags", timeout=2)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            return {
                "available": True,
                "host": host,
                "models": [m.get("name") for m in models[:10]],
                "model_count": len(models),
            }
        return {"available": False, "reason": f"ollama_status_{resp.status_code}"}
    except Exception as exc:
        return {"available": False, "reason": f"connection_error: {str(exc)[:60]}"}
