"""Unified JSON serialization utilities.

Single source of truth for safely converting Python objects (including
numpy/torch types, NaN/Inf floats, Path objects) to JSON-serializable
primitives.  All callers should use ``json_safe()`` instead of rolling
their own conversion logic.

Hot-path functions (``fast_dumps`` / ``fast_loads``) use ``orjson`` when
available (3-10x faster, native numpy support).  Falls back to stdlib
``json`` transparently.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Union

try:
    import orjson as _orjson
except ImportError:  # pragma: no cover
    _orjson = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def json_safe(value: Any) -> Any:
    """Recursively convert *value* to a JSON-serializable primitive.

    Handles:
    - ``float('nan')``, ``float('inf')`` → ``None``
    - ``pathlib.Path`` → ``str``
    - numpy scalars (``int64``, ``float32``, ``bool_``, …) → Python native
    - numpy arrays → nested lists
    - torch tensors → detached CPU lists/scalars
    - ``dict`` / ``list`` / ``tuple`` / ``set`` → recursive clean
    - ``bytes`` / ``memoryview`` → ``str`` or ``None``
    - Everything else → ``str(value)``
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return None

    if isinstance(value, (memoryview, bytearray)):
        return None

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]

    # Torch-like tensors (duck-typed to avoid importing torch)
    if hasattr(value, "detach") and callable(getattr(value, "detach")):
        try:
            t = value.detach()
            if hasattr(t, "cpu"):
                t = t.cpu()
            if hasattr(t, "tolist"):
                return json_safe(t.tolist())
            if hasattr(t, "item"):
                return json_safe(t.item())
            return str(t)
        except Exception as exc:
            logger.debug("Returning default due to error: %s", exc)
            return str(value)

    # Numpy-like arrays/scalars (duck-typed to avoid importing numpy)
    if hasattr(value, "tolist") and callable(getattr(value, "tolist")):
        try:
            return json_safe(value.tolist())
        except Exception as exc:
            logger.debug("Suppressed error: %s", exc)

    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return json_safe(value.item())
        except Exception as exc:
            logger.debug("Suppressed error: %s", exc)

    return str(value)


class SafeJSONEncoder(json.JSONEncoder):
    """JSONEncoder subclass that handles numpy/torch types and bytes."""

    def default(self, o: Any) -> Any:
        # numpy scalar types
        type_name = type(o).__name__
        if type_name in ("bool_", "int64", "int32", "float64", "float32", "float16"):
            return o.item()
        if isinstance(o, bytes):
            try:
                return o.decode("utf-8")
            except UnicodeDecodeError:
                return None
        if isinstance(o, (memoryview, bytearray)):
            return None
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


# ── Fast JSON (orjson when available) ──────────────────────────────────


def fast_dumps(obj: Any, *, safe: bool = False) -> str:
    """Serialize *obj* to a JSON string using orjson if available.

    Parameters
    ----------
    obj : Any
        Object to serialize.
    safe : bool
        If True, run ``json_safe()`` on *obj* first to sanitize
        numpy/torch/NaN values.  Adds overhead but guarantees
        serializability.

    Returns
    -------
    str
        JSON string (always str, never bytes).
    """
    if safe:
        obj = json_safe(obj)
    if _orjson is not None:
        # orjson.dumps returns bytes; decode to str for API compat
        return _orjson.dumps(
            obj,
            option=_orjson.OPT_NON_STR_KEYS | _orjson.OPT_SERIALIZE_NUMPY,
        ).decode("utf-8")
    return json.dumps(obj, cls=SafeJSONEncoder)


def fast_loads(data: Union[str, bytes]) -> Any:
    """Deserialize JSON string or bytes using orjson if available."""
    if _orjson is not None:
        return _orjson.loads(data)
    return json.loads(data)
