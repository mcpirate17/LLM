"""Tests for native profiling integration.

Verifies that:
1. Profiling can be enabled/disabled via the Rust scheduler
2. execute_graph_with_stats returns node_profiles when profiling is on
3. Profiling data contains expected fields (node_id, op_name, duration_us)
4. No profiling data when profiling is disabled
5. Python-level enable_native_profiling / get_native_profile helpers work
"""

import json
import pytest

pytestmark = pytest.mark.native

# Skip entire module if the Rust scheduler is not available.
try:
    from research.scientist.native import core as _native_core
    from research.scientist.native import profiling as _native_profiling

    rust = _native_core._try_import_rust_scheduler()
    if rust is None:
        pytest.skip(
            "Rust scheduler (aria_scheduler) not available", allow_module_level=True
        )
except Exception:
    pytest.skip("native_runner import failed", allow_module_level=True)


SIMPLE_GRAPH_JSON = json.dumps(
    {
        "schema_version": "0.1",
        "model_dim": 4,
        "output_node_id": 1,
        "nodes": [
            {
                "id": 0,
                "op_name": "input",
                "input_ids": [],
                "config": {},
                "is_input": True,
                "is_output": False,
            },
            {
                "id": 1,
                "op_name": "relu",
                "input_ids": [0],
                "config": {},
                "is_input": False,
                "is_output": True,
            },
        ],
        "edges": [
            {"source": 0, "target": 1},
        ],
        "metadata": None,
    }
)


class TestProfilerControl:
    """Test profiler enable/disable via Rust bridge."""

    def test_enable_disable(self):
        rust.profiler_enable(True)
        assert rust.profiler_enabled() is True
        rust.profiler_enable(False)
        assert rust.profiler_enabled() is False


class TestProfiledExecution:
    """Test that profiling data appears in execution results."""

    def test_no_profiles_when_disabled(self):
        rust.profiler_enable(False)
        result = rust.execute_graph_with_stats(
            SIMPLE_GRAPH_JSON, [1.0, -1.0, 2.0, -2.0]
        )
        assert "output" in result
        # node_profiles should be absent or empty when profiling is off
        profiles = result.get("node_profiles")
        assert profiles is None or len(profiles) == 0

    def test_profiles_present_when_enabled(self):
        rust.profiler_enable(True)
        try:
            result = rust.execute_graph_with_stats(
                SIMPLE_GRAPH_JSON, [1.0, -1.0, 2.0, -2.0]
            )
        finally:
            rust.profiler_enable(False)

        assert "output" in result
        profiles = result.get("node_profiles")
        assert profiles is not None, "node_profiles missing from result"
        assert len(profiles) > 0, "no profiling events recorded"

        # Check structure of profile entries
        for p in profiles:
            assert "node_id" in p
            assert "op_name" in p
            assert "duration_us" in p
            assert "start_ns" in p
            assert "end_ns" in p
            assert p["end_ns"] >= p["start_ns"]
            assert p["duration_us"] >= 0.0

    def test_profiles_contain_relu_node(self):
        """The graph has a relu node (id=1); it should appear in profiles."""
        rust.profiler_enable(True)
        try:
            result = rust.execute_graph_with_stats(
                SIMPLE_GRAPH_JSON, [1.0, -1.0, 2.0, -2.0]
            )
        finally:
            rust.profiler_enable(False)

        profiles = result.get("node_profiles", [])
        op_names = [p["op_name"] for p in profiles]
        assert "relu" in op_names, f"expected relu in profiled ops, got {op_names}"

    def test_output_correctness_unaffected(self):
        """Profiling should not change output values."""
        input_data = [1.0, -1.0, 2.0, -2.0]

        # Without profiling
        rust.profiler_enable(False)
        result_off = rust.execute_graph_with_stats(SIMPLE_GRAPH_JSON, input_data)

        # With profiling
        rust.profiler_enable(True)
        try:
            result_on = rust.execute_graph_with_stats(SIMPLE_GRAPH_JSON, input_data)
        finally:
            rust.profiler_enable(False)

        assert result_off["output"] == result_on["output"]

    def test_peak_memory_reported(self):
        """When profiling is on, peak_memory_bytes should be in result."""
        rust.profiler_enable(True)
        try:
            result = rust.execute_graph_with_stats(
                SIMPLE_GRAPH_JSON, [1.0, -1.0, 2.0, -2.0]
            )
        finally:
            rust.profiler_enable(False)

        # peak_memory_bytes should be present (may be 0 if no memory events emitted)
        assert "peak_memory_bytes" in result


class TestPythonHelpers:
    """Test the Python-level profiling API."""

    def test_enable_native_profiling(self):
        result = _native_profiling.enable_native_profiling(True)
        assert result is True
        result = _native_profiling.enable_native_profiling(False)
        assert result is False

    def test_get_native_profile_returns_none_when_disabled(self):
        _native_profiling.enable_native_profiling(False)
        # Reset cached data
        _native_profiling._last_profile_data = None
        profile = _native_profiling.get_native_profile()
        assert profile is None
