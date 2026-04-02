from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Optional

from .core import (
    _FALLBACK_METRICS,
    _SELECTIVE_GUARDRAIL,
    _SELECTIVE_GUARDRAIL_HISTORY,
    _SELECTIVE_GUARDRAIL_HISTORY_MAX,
    _env_flag,
)
from .telemetry import _legacy_compile_count

logger = logging.getLogger(__name__)
_legacy_only_deprecation_warned = False


def _maybe_fail_on_fallback_rate() -> None:
    max_rate_env = os.environ.get("NATIVE_RUNNER_MAX_FALLBACK_RATE")
    if max_rate_env is None:
        if not _env_flag("NATIVE_RUNNER_FAIL_ON_FALLBACK_RATE", False):
            return
        max_rate_env = os.environ.get("NATIVE_RUNNER_FALLBACK_RATE_MAX", "1.0")
    try:
        max_rate_raw = float(str(max_rate_env))
    except (TypeError, ValueError):
        max_rate_raw = 1.0
    max_rate_raw = max(0.0, min(1.0, max_rate_raw))
    try:
        min_samples = int(
            str(os.environ.get("NATIVE_RUNNER_FALLBACK_MIN_SAMPLES", "1"))
        )
    except (TypeError, ValueError):
        min_samples = 1

    total = int(_FALLBACK_METRICS.get("native_enabled_compiles") or 0)
    fallback = int(_FALLBACK_METRICS.get("fallback_compiles") or 0)
    if total < max(1, min_samples):
        return
    rate = float(fallback) / float(total)
    if rate > max_rate_raw:
        raise RuntimeError(
            "Native runner fallback rate exceeded threshold: "
            f"rate={rate:.3f} threshold={max_rate_raw:.3f} total={total}"
        )


def _maybe_fail_on_legacy_compile_usage() -> None:
    max_legacy_env = os.environ.get("NATIVE_RUNNER_MAX_LEGACY_COMPILE_INVOCATIONS")
    if max_legacy_env is None:
        return
    try:
        max_legacy = int(str(max_legacy_env))
    except (TypeError, ValueError):
        max_legacy = -1
    max_legacy = max(0, max_legacy)
    used = _legacy_compile_count()
    if used > max_legacy:
        raise RuntimeError(
            "Native runner legacy compile usage exceeded threshold: "
            f"used={used} threshold={max_legacy}"
        )


def _record_guardrail_event(
    event: str,
    *,
    reason: Optional[str],
    threshold: int,
    source: Optional[str] = None,
) -> None:
    timestamp = (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
    entry = {
        "event": str(event),
        "timestamp": timestamp,
        "source": source,
        "reason": reason,
        "consecutive_requested_not_candidate": int(
            _SELECTIVE_GUARDRAIL.get("consecutive_requested_not_candidate") or 0
        ),
        "threshold": int(threshold),
        "trigger_count": int(_SELECTIVE_GUARDRAIL.get("trigger_count") or 0),
    }
    _SELECTIVE_GUARDRAIL_HISTORY.append(entry)
    if len(_SELECTIVE_GUARDRAIL_HISTORY) > _SELECTIVE_GUARDRAIL_HISTORY_MAX:
        del _SELECTIVE_GUARDRAIL_HISTORY[
            0 : len(_SELECTIVE_GUARDRAIL_HISTORY) - _SELECTIVE_GUARDRAIL_HISTORY_MAX
        ]


def _maybe_warn_deprecated_legacy_only_flag() -> None:
    global _legacy_only_deprecation_warned
    if _legacy_only_deprecation_warned:
        return
    logger.warning(
        "NATIVE_RUNNER_LEGACY_ONLY is deprecated and scheduled for Phase-D removal; "
        "prefer NATIVE_RUNNER_ABI_MODEL_ONLY=0 and NATIVE_RUNNER_ALLOW_LEGACY_FALLBACK=1 "
        "for controlled rollback behavior."
    )
    _legacy_only_deprecation_warned = True


def _maybe_enforce_fallback_guardrails() -> None:
    _maybe_fail_on_fallback_rate()
    _maybe_fail_on_legacy_compile_usage()
