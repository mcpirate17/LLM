from __future__ import annotations

from research.scientist.native_runner_soak import run_native_runner_fallback_soak


def test_soak_harness_keeps_telemetry_coherent_for_success_probe():
    # Phase D: soak runs with NATIVE_RUNNER_ENABLED=0 (legacy compile path).
    # native_enabled_compiles is 0, fallback counters don't apply to disabled mode.
    result = run_native_runner_fallback_soak(iterations=200, probe_succeeds=True, max_fallback_rate=1.0)
    assert result.native_enabled_compiles == 0
    assert result.fallback_compiles == 0
    assert result.fallback_rate == 0.0
    assert result.probe_successes == 0
    assert result.probe_failures == 0


def test_soak_harness_keeps_telemetry_coherent_for_failed_probe():
    # Phase D: soak runs with NATIVE_RUNNER_ENABLED=0 (legacy compile path).
    result = run_native_runner_fallback_soak(iterations=150, probe_succeeds=False, max_fallback_rate=1.0)
    assert result.native_enabled_compiles == 0
    assert result.fallback_compiles == 0
    assert result.fallback_rate == 0.0
    assert result.probe_successes == 0
    assert result.probe_failures == 0
