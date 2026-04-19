"""Lightweight sensitivity-skip counters.

Kept separate from ``fingerprint_sensitivity`` so diagnostics endpoints can
read skip stats without importing the full torch/synthesis stack.
"""

from __future__ import annotations

import logging
import time
from typing import Dict

logger = logging.getLogger(__name__)

_SENSITIVITY_SKIP_COUNTS: Dict[str, int] = {}
_SENSITIVITY_SKIP_LAST_LOG_TS: float = 0.0
_SENSITIVITY_SKIP_LOG_INTERVAL_S: float = 60.0


def record_sensitivity_skip(reason: str) -> None:
    global _SENSITIVITY_SKIP_LAST_LOG_TS
    key = str(reason or "unknown")
    _SENSITIVITY_SKIP_COUNTS[key] = _SENSITIVITY_SKIP_COUNTS.get(key, 0) + 1
    now = time.monotonic()
    if (now - _SENSITIVITY_SKIP_LAST_LOG_TS) < _SENSITIVITY_SKIP_LOG_INTERVAL_S:
        return
    _SENSITIVITY_SKIP_LAST_LOG_TS = now
    total = sum(_SENSITIVITY_SKIP_COUNTS.values())
    breakdown = ", ".join(
        f"{name}={count}" for name, count in sorted(_SENSITIVITY_SKIP_COUNTS.items())
    )
    logger.debug("Sensitivity probes skipped (%d total): %s", total, breakdown)


def get_sensitivity_skip_stats(reset: bool = False) -> Dict[str, object]:
    global _SENSITIVITY_SKIP_LAST_LOG_TS
    by_reason = dict(_SENSITIVITY_SKIP_COUNTS)
    payload = {
        "total": int(sum(by_reason.values())),
        "by_reason": by_reason,
        "log_interval_seconds": _SENSITIVITY_SKIP_LOG_INTERVAL_S,
        "last_log_monotonic": _SENSITIVITY_SKIP_LAST_LOG_TS,
    }
    if reset:
        _SENSITIVITY_SKIP_COUNTS.clear()
        _SENSITIVITY_SKIP_LAST_LOG_TS = 0.0
    return payload
