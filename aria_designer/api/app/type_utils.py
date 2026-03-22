"""Safe type-conversion helpers for API dict data.

Replaces scattered inline patterns like:
    str(x.get("k") or "")
    float(x.get("k") or 0.0)
    (x.get("a") or {}).get("b")
"""

from __future__ import annotations

from typing import Any


def safe_str(value: Any, *, lower: bool = False, strip: bool = False) -> str:
    """Convert *value* to str, treating None/missing as ''."""
    s = str(value) if value is not None else ""
    if strip:
        s = s.strip()
    if lower:
        s = s.lower()
    return s


def safe_float(value: Any, default: float = 0.0) -> float:
    """Best-effort float conversion; returns *default* on failure."""
    if value is None:
        return default
    try:
        f = float(value)
        return f if f == f else default  # NaN check
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Best-effort int conversion; returns *default* on failure."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def dig(d: Any, *keys: str, default: Any = None) -> Any:
    """Safely traverse nested dicts: ``dig(d, "a", "b", "c")`` == ``d["a"]["b"]["c"]``.

    Returns *default* whenever a key is missing or an intermediate value
    is not a dict.
    """
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur
