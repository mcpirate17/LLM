from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
from unittest.mock import patch

from .native_runner import (
    compile_model_native_first,
    native_runner_capability_report,
    reset_native_runner_telemetry,
)


@dataclass
class NativeRunnerSoakResult:
    iterations: int
    native_enabled_compiles: int
    fallback_compiles: int
    fallback_rate: float
    probe_successes: int
    probe_failures: int


def run_native_runner_fallback_soak(
    iterations: int = 1000,
    probe_succeeds: bool = True,
    max_fallback_rate: float = 1.0,
) -> NativeRunnerSoakResult:
    """Run deterministic compile-loop soak for fallback telemetry stability.

    This is intentionally lightweight and avoids real model compilation by patching
    the legacy compiler call. It validates that native-runner counters stay coherent
    under long-running repeated compile calls.
    """
    iterations = max(1, int(iterations))

    # Phase D: NATIVE_RUNNER_ABI_MODEL_ONLY removed. Use NATIVE_RUNNER_ENABLED=0
    # since this soak test exercises legacy compile paths for telemetry validation.
    env = {
        "NATIVE_RUNNER_ENABLED": "0",
        "NATIVE_RUNNER_STRICT": "0",
        "NATIVE_RUNNER_MAX_FALLBACK_RATE": str(max_fallback_rate),
        "NATIVE_RUNNER_FALLBACK_MIN_SAMPLES": "1",
    }

    class _DummyModel:
        pass

    probe_payload: Dict[str, Any] = {
        "attempted": True,
        "succeeded": bool(probe_succeeds),
        "parity_ok": bool(probe_succeeds),
        "reason": "ok" if probe_succeeds else "probe_error:simulated",
    }

    reset_native_runner_telemetry()

    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner.os.environ", env
    ), patch(
        "research.scientist.native_runner_adapter.Path.exists", return_value=True
    ), patch(
        "research.scientist.native_runner.try_designer_runtime_probe",
        return_value=probe_payload,
    ), patch(
        "research.scientist.native_runner._legacy_compile_model", return_value=_DummyModel()
    ):
        for _ in range(iterations):
            compile_model_native_first([object()], vocab_size=128, max_seq_len=32)

        report = native_runner_capability_report()

    fallback = (report.get("fallback_metrics") or {})
    return NativeRunnerSoakResult(
        iterations=iterations,
        native_enabled_compiles=int(fallback.get("native_enabled_compiles") or 0),
        fallback_compiles=int(fallback.get("fallback_compiles") or 0),
        fallback_rate=float(fallback.get("fallback_rate") or 0.0),
        probe_successes=int(fallback.get("probe_successes") or 0),
        probe_failures=int(fallback.get("probe_failures") or 0),
    )
