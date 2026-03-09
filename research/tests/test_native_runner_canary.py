from __future__ import annotations

import pytest

from research.scientist.native_runner_canary import run_selective_canary_latency_benchmark

pytestmark = pytest.mark.native


def test_selective_canary_benchmark_reports_probe_vs_selective_latency_shapes():
    result = run_selective_canary_latency_benchmark(iterations=8, seed=123)

    assert result.iterations == 8
    assert result.seed == 123

    assert result.probe_avg_latency_ms >= 0.0
    assert result.selective_avg_latency_ms >= 0.0
    assert isinstance(result.latency_delta_ms, float)
    assert result.latency_ratio >= 0.0

    # Phase D: canary runs with NATIVE_RUNNER_ENABLED=0, so both modes use
    # the legacy_disabled execution path (native ABI is not available).
    assert sum(result.probe_execution_paths.values()) == 8
    assert sum(result.selective_execution_paths.values()) == 8
    assert result.probe_execution_paths.get("legacy_disabled", 0) == 8
    assert result.selective_execution_paths.get("legacy_disabled", 0) == 8
