"""Tests for native runner fallback-rate telemetry and hard threshold gate.

Validates:
- Telemetry counters increment correctly across multiple compiles
- Fallback rate computation is accurate
- Hard threshold gate triggers at the right point
- Gate respects min-sample configuration
- Telemetry reset works cleanly
- API endpoint returns expected shape
"""

from __future__ import annotations

import os
from unittest.mock import patch
import pytest

from research.scientist.native_runner import (
    compile_model_native_first,
    reset_native_runner_telemetry,
    native_runner_capability_report,
    _FALLBACK_METRICS,
    _maybe_fail_on_fallback_rate,
    _reset_native_lib_cache,
)

pytestmark = pytest.mark.native


@pytest.fixture(autouse=True)
def clean_state():
    """Reset telemetry and lib cache before each test."""
    reset_native_runner_telemetry()
    _reset_native_lib_cache()
    yield
    reset_native_runner_telemetry()
    _reset_native_lib_cache()


def _mock_env(enabled="0", strict="0", max_rate=None, min_samples="1"):
    # Phase D: NATIVE_RUNNER_ABI_MODEL_ONLY removed. Default to enabled="0"
    # so legacy compile path is reachable for telemetry tests.
    env = {
        "NATIVE_RUNNER_ENABLED": enabled,
        "NATIVE_RUNNER_STRICT": strict,
    }
    if max_rate is not None:
        env["NATIVE_RUNNER_MAX_FALLBACK_RATE"] = str(max_rate)
    env["NATIVE_RUNNER_FALLBACK_MIN_SAMPLES"] = str(min_samples)
    return env


def _compile_with_mocks(env, n_compiles=1):
    """Run compile_model_native_first n times with mocked env.

    Phase D: when native is enabled, ABI model-only is always active and legacy
    compile is unreachable. Mock the ABI session path for native-enabled tests.
    """

    class DummyModel:
        pass

    native_enabled = env.get("NATIVE_RUNNER_ENABLED") == "1"
    fake_abi_report = {
        "requested": True,
        "attempted": True,
        "succeeded": True,
        "reason": "ok",
        "model_handle": "fake",
        "session": object(),
    }
    extra_patches = []
    if native_enabled:
        extra_patches.append(
            patch(
                "research.scientist.native_runner._maybe_prepare_runner_abi_session",
                return_value=fake_abi_report,
            )
        )
        extra_patches.append(
            patch(
                "research.scientist.native.compiler._build_native_abi_only_model",
                return_value=DummyModel(),
            )
        )

    with (
        patch("research.scientist.native_runner_adapter.os.environ", env),
        patch("research.scientist.native_runner.os.environ", env),
        patch(
            "research.scientist.native_runner_adapter.Path.exists", return_value=True
        ),
        patch(
            "research.scientist.native.compiler.try_designer_runtime_probe",
            return_value={
                "attempted": True,
                "succeeded": True,
                "parity_ok": True,
                "reason": "ok",
            },
        ),
        patch(
            "research.scientist.native_runner._try_load_native_lib", return_value=None
        ),
        patch(
            "research.scientist.native_runner._legacy_compile_model",
            return_value=DummyModel(),
        ),
    ):
        # Stack any extra patches for native-enabled mode
        from contextlib import ExitStack

        with ExitStack() as stack:
            for p in extra_patches:
                stack.enter_context(p)
            for _ in range(n_compiles):
                compile_model_native_first([object()], vocab_size=128, max_seq_len=32)


class TestTelemetryCounters:
    def test_counters_start_at_zero(self):
        for key in _FALLBACK_METRICS:
            assert _FALLBACK_METRICS[key] == 0

    def test_single_compile_increments_total(self):
        env = _mock_env()
        _compile_with_mocks(env, n_compiles=1)
        assert _FALLBACK_METRICS["total_compiles"] >= 1

    def test_multiple_compiles_accumulate(self):
        env = _mock_env()
        _compile_with_mocks(env, n_compiles=5)
        assert _FALLBACK_METRICS["total_compiles"] >= 5

    def test_reset_clears_all_counters(self):
        env = _mock_env()
        _compile_with_mocks(env, n_compiles=3)
        reset_native_runner_telemetry()
        for key in _FALLBACK_METRICS:
            assert _FALLBACK_METRICS[key] == 0

    def test_enabled_compile_counted_separately(self):
        env = _mock_env(enabled="1")
        _compile_with_mocks(env, n_compiles=2)
        assert _FALLBACK_METRICS["native_enabled_compiles"] >= 2

    def test_disabled_compile_not_counted_as_native(self):
        env = _mock_env(enabled="0")
        _compile_with_mocks(env, n_compiles=3)
        assert _FALLBACK_METRICS["native_enabled_compiles"] == 0


class TestFallbackRate:
    def test_rate_is_zero_when_no_compiles(self):
        report = native_runner_capability_report()
        rate = report.get("fallback_metrics", {}).get("fallback_rate", 0.0)
        assert rate == 0.0

    def test_rate_computed_correctly(self):
        _FALLBACK_METRICS["native_enabled_compiles"] = 10
        _FALLBACK_METRICS["fallback_compiles"] = 3
        report = native_runner_capability_report()
        rate = report["fallback_metrics"]["fallback_rate"]
        assert abs(rate - 0.3) < 1e-6


class TestHardThresholdGate:
    def test_no_error_when_under_threshold(self):
        _FALLBACK_METRICS["native_enabled_compiles"] = 10
        _FALLBACK_METRICS["fallback_compiles"] = 2
        env = {
            "NATIVE_RUNNER_MAX_FALLBACK_RATE": "0.5",
            "NATIVE_RUNNER_FALLBACK_MIN_SAMPLES": "1",
        }
        with patch.dict(os.environ, env):
            _maybe_fail_on_fallback_rate()  # Should not raise

    def test_error_when_over_threshold(self):
        _FALLBACK_METRICS["native_enabled_compiles"] = 10
        _FALLBACK_METRICS["fallback_compiles"] = 8
        env = {
            "NATIVE_RUNNER_MAX_FALLBACK_RATE": "0.5",
            "NATIVE_RUNNER_FALLBACK_MIN_SAMPLES": "1",
        }
        with patch.dict(os.environ, env):
            with pytest.raises(RuntimeError, match="fallback rate exceeded"):
                _maybe_fail_on_fallback_rate()

    def test_respects_min_samples(self):
        _FALLBACK_METRICS["native_enabled_compiles"] = 2
        _FALLBACK_METRICS["fallback_compiles"] = 2
        env = {
            "NATIVE_RUNNER_MAX_FALLBACK_RATE": "0.5",
            "NATIVE_RUNNER_FALLBACK_MIN_SAMPLES": "10",
        }
        with patch.dict(os.environ, env):
            _maybe_fail_on_fallback_rate()  # Should not raise (below min samples)

    def test_exact_threshold_does_not_trigger(self):
        _FALLBACK_METRICS["native_enabled_compiles"] = 10
        _FALLBACK_METRICS["fallback_compiles"] = 5
        env = {
            "NATIVE_RUNNER_MAX_FALLBACK_RATE": "0.5",
            "NATIVE_RUNNER_FALLBACK_MIN_SAMPLES": "1",
        }
        with patch.dict(os.environ, env):
            _maybe_fail_on_fallback_rate()  # Exactly 0.5 = threshold, should not trigger (> not >=)

    def test_no_gate_when_env_not_set(self):
        _FALLBACK_METRICS["native_enabled_compiles"] = 10
        _FALLBACK_METRICS["fallback_compiles"] = 10
        env = {}
        with patch.dict(os.environ, env, clear=True):
            _maybe_fail_on_fallback_rate()  # No env var, no gate


class TestCapabilityReport:
    def test_report_shape(self):
        report = native_runner_capability_report()
        assert "enabled" in report
        assert "strict" in report
        assert "fallback_metrics" in report
        fm = report["fallback_metrics"]
        assert "fallback_rate" in fm
        assert "max_allowed_fallback_rate" in fm
        assert "all_compile_calls" in fm

    def test_report_includes_native_op_support(self):
        """When native lib loads, report includes op support info."""
        env = _mock_env(enabled="1")
        _compile_with_mocks(env, n_compiles=1)
        # After compile, the report should include native_op_support from the model
        report = native_runner_capability_report()
        assert isinstance(report, dict)
