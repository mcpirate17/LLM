"""Shared utility functions for the scientist package.

Consolidates duplicate helper functions previously scattered across
api.py, evidence.py, analyzer.py, runner.py, and other modules.
"""

from __future__ import annotations

import ast
import logging
import math
import struct
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


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
        except Exception as exc:
            logger.debug("Suppressed error: %s", exc)

    if isinstance(value, (bytes, bytearray)):
        try:
            if len(value) == 4:
                value = struct.unpack("<f", value)[0]
            elif len(value) == 8:
                value = struct.unpack("<d", value)[0]
            else:
                value = value.decode("utf-8", errors="ignore")
        except Exception as exc:
            logger.debug("Returning default due to error: %s", exc)
            return default

    try:
        f = float(value)
    except (TypeError, ValueError):
        return default

    return f if math.isfinite(f) else default


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to the range [lo, hi]."""
    return max(lo, min(hi, value))


def coerce_dict_payload(payload: Any) -> Optional[Dict[str, Any]]:
    """Return a dict payload from either a mapping or a ``to_dict`` object."""
    if isinstance(payload, dict):
        return payload
    to_dict = getattr(payload, "to_dict", None)
    if callable(to_dict):
        try:
            coerced = to_dict()
        except Exception as exc:
            logger.debug("Returning default due to error: %s", exc)
            return None
        if isinstance(coerced, dict):
            return coerced
    return None


def resolve_device(_config_device: Optional[str]) -> "torch.device":
    """Resolve the runtime device.  Always prefer CUDA when available.

    Training on CPU is orders of magnitude slower and wastes experiment
    budget producing no useful results in any reasonable time.
    """
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
