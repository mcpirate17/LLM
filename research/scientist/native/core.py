from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..native_runner_adapter import detect_adapter_state

logger = logging.getLogger(__name__)

_cython_bridge_cache: Any = False
_rust_scheduler_cache: Any = False
_native_lib_cache: Any = False

PARTIAL_NATIVE_COVERAGE_THRESHOLD = 0.7
_SELECTIVE_GUARDRAIL_HISTORY_MAX = 25
_SELECTIVE_GUARDRAIL_HISTORY = []
_SELECTIVE_GUARDRAIL = {
    "consecutive_requested_not_candidate": 0,
    "triggered": False,
    "trigger_count": 0,
    "last_reason": None,
}
_NATIVE_FALLBACK_LOG_WINDOW_S = 30.0
_NATIVE_FALLBACK_LOG_STATE = {
    "signature": None,
    "last_ts": 0.0,
    "suppressed": 0,
}
_FALLBACK_METRICS = {
    "total_compiles": 0,
    "native_enabled_compiles": 0,
    "native_dispatch_compiles": 0,
    "fallback_compiles": 0,
    "hybrid_compiles": 0,
    "probe_successes": 0,
    "probe_failures": 0,
    "parity_samples": 0,
    "parity_passes": 0,
    "parity_failures": 0,
    "legacy_compile_count": 0,
    "selective_mode_candidates": 0,
    "selective_mode_activations": 0,
    "selective_mode_activation_failures": 0,
}


def _env_flag(name: str, default: bool) -> bool:
    # Backward-compat shim for callers importing this helper directly.
    from ..native_runner_adapter import _env_flag as _adapter_env_flag

    return _adapter_env_flag(name, default)


@dataclass
class NativeRunnerState:
    enabled: bool
    strict: bool
    designer_runtime_available: bool
    reason: str


def detect_native_state() -> NativeRunnerState:
    adapter_state = detect_adapter_state()
    return NativeRunnerState(
        enabled=adapter_state.enabled,
        strict=adapter_state.strict,
        designer_runtime_available=adapter_state.designer_runtime_available,
        reason=adapter_state.reason,
    )


def _try_import_cython_bridge() -> Any:
    """Try to import the Cython bridge module (aria_bridge).

    Adds the cython build directory to sys.path if needed. The result is
    cached module-level so subsequent calls are free.

    Returns the aria_bridge module, or None if unavailable.
    """
    global _cython_bridge_cache
    if _cython_bridge_cache not in {False, None}:
        return _cython_bridge_cache

    try:
        from .abi import _try_load_native_lib

        _try_load_native_lib()
    except Exception as exc:
        logger.debug(
            "Native runtime preload unavailable before aria_bridge import: %s", exc
        )

    # Try direct import first (may already be on sys.path).
    try:
        import aria_bridge  # type: ignore[import-untyped]

        _cython_bridge_cache = aria_bridge
        logger.info("Loaded Cython bridge (aria_bridge) via direct import")
        return _cython_bridge_cache
    except ImportError:
        pass

    # Add the cython directory to sys.path and retry.
    cython_dir = str(
        Path(__file__).resolve().parents[2] / "runtime" / "native" / "cython"
    )
    if cython_dir not in sys.path:
        sys.path.insert(0, cython_dir)
    try:
        import aria_bridge  # type: ignore[import-untyped]

        _cython_bridge_cache = aria_bridge
        logger.info("Loaded Cython bridge (aria_bridge) from %s", cython_dir)
        return _cython_bridge_cache
    except ImportError as exc:
        logger.debug("Cython bridge not available: %s", exc)
        _cython_bridge_cache = False
        return None


def _reset_cython_bridge_cache() -> None:
    """Reset the Cython bridge cache (used in tests)."""
    global _cython_bridge_cache
    _cython_bridge_cache = False


def _try_import_rust_scheduler() -> Any:
    """Try to import the Rust scheduler module (aria_scheduler)."""
    global _rust_scheduler_cache
    if _rust_scheduler_cache not in {False, None}:
        return _rust_scheduler_cache

    try:
        from . import aria_scheduler

        _rust_scheduler_cache = aria_scheduler
        logger.info("Loaded Rust scheduler (aria_scheduler)")
        return _rust_scheduler_cache
    except ImportError as exc:
        logger.debug("Package-local Rust scheduler not available: %s", exc)

    try:
        import aria_scheduler  # type: ignore[import-untyped]

        _rust_scheduler_cache = aria_scheduler
        logger.info("Loaded Rust scheduler (aria_scheduler) via top-level import")
        return _rust_scheduler_cache
    except ImportError as exc:
        logger.debug("Rust scheduler not available: %s", exc)
        _rust_scheduler_cache = False
        return None


__all__ = [
    "NativeRunnerState",
    "_env_flag",
    "detect_native_state",
    "_try_import_cython_bridge",
    "_reset_cython_bridge_cache",
    "_try_import_rust_scheduler",
    "_native_lib_cache",
    "PARTIAL_NATIVE_COVERAGE_THRESHOLD",
    "_SELECTIVE_GUARDRAIL_HISTORY_MAX",
    "_SELECTIVE_GUARDRAIL_HISTORY",
    "_SELECTIVE_GUARDRAIL",
    "_NATIVE_FALLBACK_LOG_WINDOW_S",
    "_NATIVE_FALLBACK_LOG_STATE",
    "_FALLBACK_METRICS",
]
