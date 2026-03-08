"""Shared utility functions for the scientist package.

Consolidates duplicate helper functions previously scattered across
api.py, evidence.py, analyzer.py, runner.py, and other modules.
"""

from __future__ import annotations

import ast
import json
import math
import struct
from typing import Any, Dict, Optional


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Convert a value to float, handling bytes/blobs/NaN/Inf/None gracefully.

    Superset of the various ``_safe_float()`` implementations across the
    codebase.  Handles:
    - None → default
    - bytes/bytearray (4-byte float32, 8-byte float64, or UTF-8 string)
    - Stringified bytes like ``"b'3.14'"``
    - NaN/Inf → default
    - Any other type via ``float()``
    """
    if value is None:
        return default

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        f = float(value)
        return f if math.isfinite(f) else default

    # Handle stringified bytes: "b'...'"
    if isinstance(value, str) and value.startswith("b'") and value.endswith("'"):
        try:
            value = ast.literal_eval(value)
        except Exception:
            pass

    if isinstance(value, (bytes, bytearray)):
        try:
            if len(value) == 4:
                value = struct.unpack("<f", value)[0]
            elif len(value) == 8:
                value = struct.unpack("<d", value)[0]
            else:
                value = value.decode("utf-8", errors="ignore")
        except Exception:
            return default

    try:
        f = float(value)
    except (TypeError, ValueError):
        return default

    return f if math.isfinite(f) else default


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to the range [lo, hi]."""
    return max(lo, min(hi, value))


def safe_json_loads(text: Any, default: Any = None) -> Any:
    """Parse JSON text, returning *default* on any failure."""
    if not isinstance(text, str) or not text.strip():
        return default
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return default


def ensure_metadata_dict(entry: Dict[str, Any],
                         json_key: str = "metadata_json",
                         dict_key: str = "metadata") -> Dict[str, Any]:
    """Ensure *entry* has a parsed metadata dict.

    If ``entry[dict_key]`` is already a dict, return as-is.
    Otherwise parse ``entry[json_key]`` and store the result in
    ``entry[dict_key]``.
    """
    meta = entry.get(dict_key)
    if isinstance(meta, dict):
        return meta
    raw = entry.get(json_key)
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                entry[dict_key] = parsed
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    entry[dict_key] = {}
    return {}


def canonicalize_text(text: str) -> str:
    """Collapse whitespace, lowercase, strip numbers for fuzzy comparison."""
    import re
    normalized = " ".join(text.split()).lower()
    normalized = re.sub(r"\d+", "N", normalized)
    return re.sub(r"[^a-z0-9_ ]", "", normalized).strip()
