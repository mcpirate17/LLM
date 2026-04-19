from __future__ import annotations

import os
import time
import logging
from typing import Any, Dict, List

from .core import (
    _FALLBACK_METRICS,
    _NATIVE_FALLBACK_LOG_STATE,
    _NATIVE_FALLBACK_LOG_WINDOW_S,
    _SELECTIVE_GUARDRAIL,
    _SELECTIVE_GUARDRAIL_HISTORY,
    _env_flag,
    detect_native_state,
)
from ..native_runner_adapter import capability_handshake

logger = logging.getLogger(__name__)


def reset_native_runner_telemetry() -> None:
    for key in list(_FALLBACK_METRICS.keys()):
        _FALLBACK_METRICS[key] = 0
    _SELECTIVE_GUARDRAIL["consecutive_requested_not_candidate"] = 0
    _SELECTIVE_GUARDRAIL["triggered"] = False
    _SELECTIVE_GUARDRAIL["trigger_count"] = 0
    _SELECTIVE_GUARDRAIL["last_reason"] = None
    _SELECTIVE_GUARDRAIL_HISTORY.clear()


def native_runner_capability_report(*, deep: bool = True) -> Dict[str, Any]:
    report = capability_handshake(deep=deep)
    state = detect_native_state()
    # Phase D: ABI model-only is always active when native is enabled.
    disable_legacy_compile = _env_flag("NATIVE_RUNNER_DISABLE_LEGACY_COMPILE", False)
    disable_legacy_compile_native_enabled = _env_flag(
        "NATIVE_RUNNER_DISABLE_LEGACY_COMPILE_NATIVE_ENABLED",
        False,
    )
    legacy_disabled = bool(
        disable_legacy_compile
        or (state.enabled and disable_legacy_compile_native_enabled)
    )
    if legacy_disabled and state.enabled and disable_legacy_compile_native_enabled:
        legacy_disabled_reason = "native_enabled_gate"
    elif legacy_disabled:
        legacy_disabled_reason = "env_flag"
    else:
        legacy_disabled_reason = None

    if legacy_disabled:
        execution_mode = "legacy_disabled"
    elif state.enabled:
        execution_mode = "native_abi_model_only"
    else:
        execution_mode = "legacy_only"

    report["execution_mode_classification"] = execution_mode
    report["legacy_compile_disabled"] = legacy_disabled
    report["legacy_compile_disabled_reason"] = legacy_disabled_reason
    total = int(_FALLBACK_METRICS.get("total_compiles") or 0)
    native_total = int(_FALLBACK_METRICS.get("native_enabled_compiles") or 0)
    fallback = int(_FALLBACK_METRICS.get("fallback_compiles") or 0)
    legacy_count = _legacy_compile_count()
    hybrid = int(_FALLBACK_METRICS.get("hybrid_compiles") or 0)
    report["fallback_metrics"] = {
        **_FALLBACK_METRICS,
        "legacy_compile_count": legacy_count,
        "hybrid_compiles": hybrid,
        "fallback_rate": (float(fallback) / float(native_total))
        if native_total > 0
        else 0.0,
        "hybrid_rate": (float(hybrid) / float(native_total))
        if native_total > 0
        else 0.0,
        "max_allowed_fallback_rate": os.environ.get("NATIVE_RUNNER_MAX_FALLBACK_RATE"),
        "max_allowed_legacy_compile_count": os.environ.get(
            "NATIVE_RUNNER_MAX_LEGACY_COMPILE_INVOCATIONS"
        ),
        "samples_considered": native_total,
        "all_compile_calls": total,
    }
    checks: List[Dict[str, Any]] = []
    fallback_limit_raw = os.environ.get("NATIVE_RUNNER_MAX_FALLBACK_RATE")
    if fallback_limit_raw is not None:
        try:
            fallback_limit = max(0.0, min(1.0, float(str(fallback_limit_raw))))
        except (TypeError, ValueError):
            fallback_limit = 1.0
        fallback_rate = report["fallback_metrics"]["fallback_rate"]
        checks.append(
            {
                "name": "fallback_rate",
                "active": True,
                "pass": bool(fallback_rate <= fallback_limit),
                "actual": float(fallback_rate),
                "limit": float(fallback_limit),
            }
        )

    legacy_limit_raw = os.environ.get("NATIVE_RUNNER_MAX_LEGACY_COMPILE_INVOCATIONS")
    if legacy_limit_raw is not None:
        try:
            legacy_limit = max(0, int(str(legacy_limit_raw)))
        except (TypeError, ValueError):
            legacy_limit = 0
        legacy_used = legacy_count
        checks.append(
            {
                "name": "legacy_compile_count",
                "active": True,
                "pass": bool(legacy_used <= legacy_limit),
                "actual": int(legacy_used),
                "limit": int(legacy_limit),
            }
        )

    require_parity = _env_flag("NATIVE_RUNNER_REQUIRE_PARITY_PASS", False)
    parity_samples = int(_FALLBACK_METRICS.get("parity_samples") or 0)
    parity_failures = int(_FALLBACK_METRICS.get("parity_failures") or 0)
    if require_parity:
        if parity_samples <= 0:
            checks.append(
                {
                    "name": "parity",
                    "active": True,
                    "pass": None,
                    "actual": "no_samples",
                    "limit": "no_failures",
                }
            )
        else:
            checks.append(
                {
                    "name": "parity",
                    "active": True,
                    "pass": bool(parity_failures == 0),
                    "actual": int(parity_failures),
                    "limit": 0,
                }
            )

    active_checks = [c for c in checks if c.get("active")]
    if not active_checks:
        cutover_ready = None
        cutover_status = "waiting"
    elif any(c.get("pass") is None for c in active_checks):
        cutover_ready = None
        cutover_status = "waiting"
    else:
        cutover_ready = all(bool(c.get("pass")) for c in active_checks)
        cutover_status = "ready" if cutover_ready else "blocked"
    report["cutover_gate"] = {
        "ready": cutover_ready,
        "status": cutover_status,
        "checks": active_checks,
    }
    try:
        threshold = int(
            str(os.environ.get("NATIVE_RUNNER_SELECTIVE_GUARDRAIL_WINDOW", "5"))
        )
    except (TypeError, ValueError):
        threshold = 5
    threshold = max(1, threshold)
    report["selective_guardrail"] = {
        "consecutive_requested_not_candidate": int(
            _SELECTIVE_GUARDRAIL.get("consecutive_requested_not_candidate") or 0
        ),
        "threshold": threshold,
        "triggered": bool(_SELECTIVE_GUARDRAIL.get("triggered")),
        "trigger_count": int(_SELECTIVE_GUARDRAIL.get("trigger_count") or 0),
        "last_reason": _SELECTIVE_GUARDRAIL.get("last_reason"),
        "history": [dict(item) for item in _SELECTIVE_GUARDRAIL_HISTORY],
    }
    return report


def _log_native_fallback_coverage(op_support: Dict[str, Any]) -> None:
    """Log native coverage fallback with burst deduplication.

    During investigation cycles, many compile attempts can produce identical
    coverage diagnostics in rapid succession. Emit the first message and
    suppress repeats within a short window to reduce log noise.
    """
    coverage = float(op_support.get("native_coverage") or 0.0)
    supported_count = len(op_support.get("supported") or [])
    all_count = len(op_support.get("all_ops") or [])
    unsupported = list(op_support.get("unsupported") or [])

    signature = (supported_count, all_count, tuple(unsupported))
    now = time.time()
    state = _NATIVE_FALLBACK_LOG_STATE

    same_signature = signature == state["signature"]
    within_window = (
        now - float(state["last_ts"] or 0.0)
    ) <= _NATIVE_FALLBACK_LOG_WINDOW_S

    if same_signature and within_window:
        state["suppressed"] = int(state["suppressed"] or 0) + 1
        return

    suppressed = int(state.get("suppressed") or 0)
    if suppressed > 0 and state.get("signature") is not None:
        logger.debug(
            "Suppressed %d repeated native fallback coverage log(s) in the last %.0fs.",
            suppressed,
            _NATIVE_FALLBACK_LOG_WINDOW_S,
        )

    logger.debug(
        "Native kernel coverage %.1f%% (%d/%d ops). Unsupported: %s. "
        "Falling back to legacy compile.",
        coverage * 100,
        supported_count,
        all_count,
        unsupported,
    )

    state["signature"] = signature
    state["last_ts"] = now
    state["suppressed"] = 0


def _record_legacy_compile_invocation() -> None:
    _FALLBACK_METRICS["legacy_compile_count"] += 1


def _legacy_compile_count() -> int:
    return int(_FALLBACK_METRICS.get("legacy_compile_count") or 0)
