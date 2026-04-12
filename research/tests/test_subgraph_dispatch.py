"""Tests for SubgraphDispatcher — batch native subgraph execution.

Verifies that SubgraphDispatcher correctly identifies all-native graphs,
dispatches them through the Rust scheduler, and falls back gracefully when
ops are unsupported or the scheduler is unavailable.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure the research package is importable.
_root = str(Path(__file__).resolve().parents[1].parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from research.synthesis.graph import ComputationGraph
from research.scientist.native import dispatch as native_dispatch_module
from research.scientist.native.autograd import SubgraphDispatcher

pytestmark = pytest.mark.native


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_rust_dispatch(monkeypatch, fake_rust, *, compile_handle=None):
    """Common monkeypatch setup for Rust dispatch tests."""
    monkeypatch.setattr(
        native_dispatch_module,
        "_prepare_graph_input",
        lambda graph, input_data: (np.zeros((1, 2, 4), dtype=np.float32), "fake-ir"),
    )
    monkeypatch.setattr(
        native_dispatch_module,
        "_try_import_rust_scheduler",
        lambda: fake_rust,
    )
    if compile_handle is not None:
        monkeypatch.setattr(
            native_dispatch_module,
            "_compile_rust_graph_handle",
            lambda ir: compile_handle,
        )


def _make_simple_graph(
    model_dim: int = 4, ops: list[str] | None = None
) -> ComputationGraph:
    """Build a simple chain graph: input -> op1 -> op2 -> ... -> output.

    Default ops: ["relu", "add"] (add uses the input as both operands).
    """
    if ops is None:
        ops = ["relu"]
    g = ComputationGraph(model_dim)
    inp = g.add_input()
    prev = inp
    for op_name in ops:
        prim = None
        try:
            from research.synthesis.primitives import get_primitive

            prim = get_primitive(op_name)
        except Exception:
            pass

        n_inputs = 1
        if prim is not None and prim.n_inputs == 2:
            n_inputs = 2

        if n_inputs == 2:
            nid = g.add_op(op_name, [prev, inp])
        else:
            nid = g.add_op(op_name, [prev])
        prev = nid
    g.set_output(prev)
    return g


def _make_diamond_graph(model_dim: int = 4) -> ComputationGraph:
    """Build a diamond graph: input -> relu, input -> gelu, relu+gelu -> add -> output."""
    g = ComputationGraph(model_dim)
    inp = g.add_input()
    relu_id = g.add_op("relu", [inp])
    gelu_id = g.add_op("gelu", [inp])
    add_id = g.add_op("add", [relu_id, gelu_id])
    g.set_output(add_id)
    return g


# ---------------------------------------------------------------------------
# Test 1: SubgraphDispatcher detects all-native graph
# ---------------------------------------------------------------------------


def test_subgraph_dispatcher_all_native():
    """SubgraphDispatcher.all_native should be True when all ops are supported."""
    g = _make_simple_graph(ops=["relu"])
    supported = {"relu", "add", "gelu", "silu"}
    dispatcher = SubgraphDispatcher(g, supported)
    assert dispatcher.all_native is True


def test_subgraph_dispatcher_not_all_native():
    """SubgraphDispatcher.all_native should be False when an op is unsupported."""
    g = _make_simple_graph(ops=["relu"])
    supported = {"add"}  # relu not included
    dispatcher = SubgraphDispatcher(g, supported)
    assert dispatcher.all_native is False


def test_subgraph_dispatcher_diamond_all_native():
    """Diamond graph with all supported ops should be all-native."""
    g = _make_diamond_graph()
    supported = {"relu", "gelu", "add"}
    dispatcher = SubgraphDispatcher(g, supported)
    assert dispatcher.all_native is True


def test_subgraph_dispatcher_diamond_partial():
    """Diamond graph with one unsupported op should not be all-native."""
    g = _make_diamond_graph()
    supported = {"relu", "add"}  # gelu missing
    dispatcher = SubgraphDispatcher(g, supported)
    assert dispatcher.all_native is False


def test_subgraph_dispatcher_treats_structural_ops_as_native():
    """Structural ops should not block whole-graph native dispatch."""
    g = _make_simple_graph(ops=["identity", "relu"])
    dispatcher = SubgraphDispatcher(g, {"relu"})
    assert dispatcher.all_native is True


def test_subgraph_dispatcher_excludes_scheduler_unsupported_ops():
    """Per-op native support must not imply Rust scheduler support."""
    g = _make_simple_graph(ops=["layernorm"])
    dispatcher = SubgraphDispatcher(g, {"layernorm"})
    assert dispatcher.all_native is False


# ---------------------------------------------------------------------------
# Test 2: try_dispatch returns None when not all-native
# ---------------------------------------------------------------------------


def test_try_dispatch_returns_none_when_not_all_native():
    """When some ops are unsupported, try_dispatch returns None."""
    g = _make_simple_graph(ops=["relu"])
    supported = {"add"}
    dispatcher = SubgraphDispatcher(g, supported)
    x = np.zeros((1, 2, 4), dtype=np.float32)
    result = dispatcher.try_dispatch(x)
    assert result is None
    assert dispatcher.stats["subgraph_dispatches"] == 0


# ---------------------------------------------------------------------------
# Test 3: try_dispatch calls dispatch_graph_native_cached when all-native
# ---------------------------------------------------------------------------


@patch("research.scientist.native.autograd.dispatch_graph_native_cached")
def test_try_dispatch_calls_graph_dispatch(mock_cached_dispatch):
    """When all ops are native, try_dispatch should call dispatch_graph_native_cached."""
    g = _make_simple_graph(ops=["relu"])
    supported = {"relu"}
    dispatcher = SubgraphDispatcher(g, supported)

    # Verify the IR was cached at init time
    assert dispatcher._ir_json is not None

    fake_output = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    mock_cached_dispatch.return_value = fake_output

    x = np.zeros((1, 2, 4), dtype=np.float32)
    result = dispatcher.try_dispatch(x)

    assert result is not None
    np.testing.assert_array_equal(result, fake_output)
    mock_cached_dispatch.assert_called_once_with(dispatcher._ir_json, g, x)
    assert dispatcher.stats["subgraph_dispatches"] == 1


# ---------------------------------------------------------------------------
# Test 5: try_dispatch converts torch tensors
# ---------------------------------------------------------------------------


@patch("research.scientist.native.autograd.dispatch_graph_native_cached")
def test_try_dispatch_torch_conversion(mock_cached_dispatch):
    """When input is a torch tensor, result should also be a torch tensor."""
    torch = pytest.importorskip("torch")

    g = _make_simple_graph(ops=["relu"])
    supported = {"relu"}
    dispatcher = SubgraphDispatcher(g, supported)

    fake_output = np.array([0.0, 0.0, 2.0, 3.5], dtype=np.float32)
    mock_cached_dispatch.return_value = fake_output

    x = torch.tensor([[-1.0, 0.0, 2.0, 3.5]], dtype=torch.float32)
    result = dispatcher.try_dispatch(x)

    assert result is not None
    assert isinstance(result, torch.Tensor)
    expected = torch.tensor([0.0, 0.0, 2.0, 3.5], dtype=torch.float32)
    torch.testing.assert_close(result, expected)


@patch("research.scientist.native.autograd.dispatch_graph_native_cached")
def test_try_dispatch_skips_host_bridge_for_non_cpu_tensors(
    mock_cached_dispatch, monkeypatch
):
    g = _make_simple_graph(ops=["relu"])
    dispatcher = SubgraphDispatcher(g, {"relu"})

    monkeypatch.setattr(
        "research.scientist.native.autograd.supports_host_array_bridge",
        lambda *values: False,
    )

    result = dispatcher.try_dispatch(np.zeros((1, 2, 4), dtype=np.float32))

    assert result is None
    mock_cached_dispatch.assert_not_called()
    assert dispatcher.stats["subgraph_dispatches"] == 0
    assert (
        dispatcher.stats["last_refusal_reason"]
        == "host_array_bridge_unsupported_device"
    )


# ---------------------------------------------------------------------------
# Test 6: empty graph returns all_native=False
# ---------------------------------------------------------------------------


def test_subgraph_dispatcher_empty_graph():
    """A graph with no non-input nodes should not be all-native."""
    g = ComputationGraph(4)
    # Just add input, no ops
    g.add_input()
    # Don't set output — incomplete graph
    dispatcher = SubgraphDispatcher(g, {"relu"})
    # Only has input node, which is skipped, so vacuously True
    # But try_dispatch will fail because dispatch_graph_native needs a valid graph
    assert dispatcher.all_native is True


# ---------------------------------------------------------------------------
# Test 7: stats tracking
# ---------------------------------------------------------------------------


@patch("research.scientist.native.autograd.dispatch_graph_native_cached")
def test_stats_tracking(mock_cached_dispatch):
    """Verify dispatch/fallback counts are tracked correctly."""
    g = _make_simple_graph(ops=["relu"])
    supported = {"relu"}
    dispatcher = SubgraphDispatcher(g, supported)

    mock_cached_dispatch.return_value = np.zeros(4, dtype=np.float32)
    x = np.zeros((1, 2, 4), dtype=np.float32)

    # Two successful dispatches
    dispatcher.try_dispatch(x)
    dispatcher.try_dispatch(x)

    # One failure
    mock_cached_dispatch.side_effect = RuntimeError("fail")
    dispatcher.try_dispatch(x)

    stats = dispatcher.stats
    assert stats["subgraph_dispatches"] == 2
    assert stats["subgraph_fallbacks"] == 1
    assert stats["all_native"] is True


@patch("research.scientist.native.autograd.dispatch_graph_native_cached")
def test_try_dispatch_disables_scheduler_after_unsupported_runtime_failure(
    mock_cached_dispatch,
):
    """Unsupported-op runtime failures should disable future scheduler attempts."""
    g = _make_simple_graph(ops=["relu"])
    dispatcher = SubgraphDispatcher(g, {"relu"})
    mock_cached_dispatch.side_effect = RuntimeError(
        "op layernorm not registered in native runtime"
    )

    x = np.zeros((1, 2, 4), dtype=np.float32)
    first = dispatcher.try_dispatch(x)
    second = dispatcher.try_dispatch(x)

    assert first is None
    assert second is None
    assert dispatcher.all_native is False
    assert mock_cached_dispatch.call_count == 1


def test_dispatch_graph_forward_saved_prefers_rust(monkeypatch):
    g = _make_simple_graph(model_dim=4, ops=["relu"])

    class FakeRust:
        @staticmethod
        def execute_graph_forward_saved(graph_json, input_flat):
            assert graph_json == "fake-ir"
            assert input_flat == [0.0] * 8
            return {
                "output": [1.0] * 8,
                "saved_activations": {0: [0.0] * 8, 1: [1.0] * 8},
                "arena_bytes_used": 64,
                "arena_capacity": 128,
            }

    _setup_rust_dispatch(monkeypatch, FakeRust())

    result = native_dispatch_module.dispatch_graph_forward_native_saved(
        g, np.zeros((1, 2, 4), dtype=np.float32)
    )

    assert result["output"].shape == (1, 2, 4)
    np.testing.assert_array_equal(
        result["output"], np.ones((1, 2, 4), dtype=np.float32)
    )
    assert set(result["saved_activations"]) == {0, 1}
    assert result["arena_bytes_used"] == 64
    assert result["arena_capacity"] == 128


def test_dispatch_graph_forward_saved_prefers_rust_array_buffers(monkeypatch):
    g = _make_simple_graph(model_dim=4, ops=["relu"])

    class FakeRust:
        @staticmethod
        def execute_graph_forward_saved_arrays(graph_json, input_array):
            assert graph_json == "fake-ir"
            assert input_array.shape == (1, 2, 4)
            return {
                "output": np.ones((1, 2, 4), dtype=np.float32),
                "saved_activations": {
                    0: np.zeros((1, 2, 4), dtype=np.float32),
                    1: np.ones((1, 2, 4), dtype=np.float32),
                },
                "arena_bytes_used": 64,
                "arena_capacity": 128,
            }

        @staticmethod
        def execute_graph_forward_saved(graph_json, input_flat):
            raise AssertionError("list fallback should not be used")

    _setup_rust_dispatch(monkeypatch, FakeRust())

    result = native_dispatch_module.dispatch_graph_forward_native_saved(
        g, np.zeros((1, 2, 4), dtype=np.float32)
    )

    assert result["output"].shape == (1, 2, 4)
    np.testing.assert_array_equal(
        result["output"], np.ones((1, 2, 4), dtype=np.float32)
    )
    assert set(result["saved_activations"]) == {0, 1}


def test_dispatch_graph_forward_saved_prefers_rust_saved_state_handle(monkeypatch):
    g = _make_simple_graph(model_dim=4, ops=["relu"])
    saved_state = object()

    class FakeRust:
        @staticmethod
        def execute_graph_forward_saved_arrays_handle(graph_json, input_array):
            assert graph_json == "fake-ir"
            assert input_array.shape == (1, 2, 4)
            return {
                "output": np.ones((1, 2, 4), dtype=np.float32),
                "saved_state": saved_state,
                "arena_bytes_used": 64,
                "arena_capacity": 128,
            }

        @staticmethod
        def execute_graph_forward_saved_arrays(graph_json, input_array):
            raise AssertionError("dict fallback should not be used")

    _setup_rust_dispatch(monkeypatch, FakeRust())

    result = native_dispatch_module.dispatch_graph_forward_native_saved(
        g, np.zeros((1, 2, 4), dtype=np.float32)
    )

    assert result["saved_activations"] is saved_state
    assert result["output"].shape == (1, 2, 4)


def test_dispatch_graph_forward_saved_prefers_compiled_graph_handle(monkeypatch):
    g = _make_simple_graph(model_dim=4, ops=["relu"])
    saved_state = object()
    compiled_graph = object()

    class FakeRust:
        @staticmethod
        def execute_graph_forward_saved_compiled_arrays_handle(handle, input_array):
            assert handle is compiled_graph
            assert input_array.shape == (1, 2, 4)
            return {
                "output": np.ones((1, 2, 4), dtype=np.float32),
                "saved_state": saved_state,
                "arena_bytes_used": 64,
                "arena_capacity": 128,
            }

        @staticmethod
        def execute_graph_forward_saved_arrays_handle(graph_json, input_array):
            raise AssertionError("json path should not be used")

    _setup_rust_dispatch(monkeypatch, FakeRust(), compile_handle=compiled_graph)

    result = native_dispatch_module.dispatch_graph_forward_native_saved(
        g, np.zeros((1, 2, 4), dtype=np.float32)
    )

    assert result["saved_activations"] is saved_state
    assert result["output"].shape == (1, 2, 4)


def test_dispatch_graph_backward_prefers_rust(monkeypatch):
    g = _make_simple_graph(model_dim=4, ops=["relu"])

    class FakeRust:
        @staticmethod
        def execute_graph_backward(graph_json, grad_output, saved_activations):
            assert graph_json == "fake-ir"
            assert grad_output == [1.0] * 8
            assert saved_activations[0] == [0.0] * 8
            return {
                "grads": {0: [2.0] * 8, 1: [1.0] * 8},
                "arena_bytes_used": 32,
            }

    monkeypatch.setattr(
        native_dispatch_module,
        "_try_import_rust_scheduler",
        lambda: FakeRust(),
    )

    result = native_dispatch_module.dispatch_graph_backward_native(
        g,
        np.ones((1, 2, 4), dtype=np.float32),
        {0: np.zeros((1, 2, 4), dtype=np.float32)},
        ir_json="fake-ir",
    )

    assert set(result) == {0, 1}
    np.testing.assert_array_equal(result[0], np.full(8, 2.0, dtype=np.float32))
    np.testing.assert_array_equal(result[1], np.full(8, 1.0, dtype=np.float32))


def test_dispatch_graph_backward_prefers_rust_array_buffers(monkeypatch):
    g = _make_simple_graph(model_dim=4, ops=["relu"])

    class FakeRust:
        @staticmethod
        def execute_graph_backward_arrays(graph_json, grad_output, saved_activations):
            assert graph_json == "fake-ir"
            assert grad_output.shape == (1, 2, 4)
            assert saved_activations[0].shape == (1, 2, 4)
            return {
                "grads": {
                    0: np.full((8,), 2.0, dtype=np.float32),
                    1: np.full((8,), 1.0, dtype=np.float32),
                },
                "arena_bytes_used": 32,
            }

        @staticmethod
        def execute_graph_backward(graph_json, grad_output, saved_activations):
            raise AssertionError("list fallback should not be used")

    monkeypatch.setattr(
        native_dispatch_module,
        "_try_import_rust_scheduler",
        lambda: FakeRust(),
    )

    result = native_dispatch_module.dispatch_graph_backward_native(
        g,
        np.ones((1, 2, 4), dtype=np.float32),
        {0: np.zeros((1, 2, 4), dtype=np.float32)},
        ir_json="fake-ir",
    )

    assert set(result) == {0, 1}
    np.testing.assert_array_equal(result[0], np.full(8, 2.0, dtype=np.float32))
    np.testing.assert_array_equal(result[1], np.full(8, 1.0, dtype=np.float32))


def test_dispatch_graph_backward_prefers_rust_saved_state_handle(monkeypatch):
    g = _make_simple_graph(model_dim=4, ops=["relu"])
    saved_state = object()

    class FakeRust:
        @staticmethod
        def execute_graph_backward_arrays_handle(graph_json, grad_output, saved_handle):
            assert graph_json == "fake-ir"
            assert grad_output.shape == (1, 2, 4)
            assert saved_handle is saved_state
            return {
                "grads": {
                    0: np.full((8,), 2.0, dtype=np.float32),
                    1: np.full((8,), 1.0, dtype=np.float32),
                },
                "arena_bytes_used": 32,
            }

        @staticmethod
        def execute_graph_backward_arrays(graph_json, grad_output, saved_activations):
            raise AssertionError("dict fallback should not be used")

    monkeypatch.setattr(
        native_dispatch_module,
        "_try_import_rust_scheduler",
        lambda: FakeRust(),
    )

    result = native_dispatch_module.dispatch_graph_backward_native(
        g,
        np.ones((1, 2, 4), dtype=np.float32),
        saved_state,
        ir_json="fake-ir",
    )

    assert set(result) == {0, 1}
    np.testing.assert_array_equal(result[0], np.full(8, 2.0, dtype=np.float32))


def test_dispatch_graph_backward_prefers_compiled_graph_handle(monkeypatch):
    g = _make_simple_graph(model_dim=4, ops=["relu"])
    saved_state = object()
    compiled_graph = object()

    class FakeRust:
        @staticmethod
        def execute_graph_backward_compiled_arrays_handle(
            handle, grad_output, saved_handle
        ):
            assert handle is compiled_graph
            assert grad_output.shape == (1, 2, 4)
            assert saved_handle is saved_state
            return {
                "grads": {
                    0: np.full((8,), 2.0, dtype=np.float32),
                    1: np.full((8,), 1.0, dtype=np.float32),
                },
                "arena_bytes_used": 32,
            }

        @staticmethod
        def execute_graph_backward_arrays_handle(graph_json, grad_output, saved_handle):
            raise AssertionError("json path should not be used")

    _setup_rust_dispatch(monkeypatch, FakeRust(), compile_handle=compiled_graph)

    result = native_dispatch_module.dispatch_graph_backward_native(
        g,
        np.ones((1, 2, 4), dtype=np.float32),
        saved_state,
        ir_json="fake-ir",
    )

    assert set(result) == {0, 1}
    np.testing.assert_array_equal(result[0], np.full(8, 2.0, dtype=np.float32))


def test_dispatch_graph_native_cached_prefers_compiled_graph_handle(monkeypatch):
    g = _make_simple_graph(model_dim=4, ops=["relu"])
    compiled_graph = object()

    class FakeRust:
        @staticmethod
        def execute_graph_with_stats_compiled_arrays(handle, input_array):
            assert handle is compiled_graph
            assert input_array.shape == (1, 2, 4)
            return {
                "output": np.ones((8,), dtype=np.float32),
                "arena_bytes_used": 64,
                "arena_capacity": 128,
                "arena_alloc_count": 1,
                "heap_fallback_count": 0,
            }

        @staticmethod
        def execute_graph_with_stats_arrays(ir_json, input_array):
            raise AssertionError("json path should not be used")

    _setup_rust_dispatch(monkeypatch, FakeRust(), compile_handle=compiled_graph)

    result = native_dispatch_module.dispatch_graph_native_cached(
        "fake-ir",
        g,
        np.zeros((1, 2, 4), dtype=np.float32),
    )

    assert result.shape == (1, 2, 4)


# ---------------------------------------------------------------------------
# Test 8: CompiledLayer uses subgraph dispatcher when attached
# ---------------------------------------------------------------------------


def test_compiled_layer_subgraph_dispatch_integration():
    """CompiledLayer.forward() should use _subgraph_dispatcher when attached."""
    torch = pytest.importorskip("torch")
    from research.synthesis.compiler import CompiledLayer

    g = _make_simple_graph(model_dim=8, ops=["relu"])
    layer = CompiledLayer(g)

    mock_dispatcher = MagicMock()
    fake_out = torch.ones(1, 4, 8)
    mock_dispatcher.try_dispatch.return_value = fake_out
    layer._subgraph_dispatcher = mock_dispatcher

    x = torch.randn(1, 4, 8)
    result = layer(x)

    mock_dispatcher.try_dispatch.assert_called_once_with(x)
    assert torch.equal(result, fake_out)


def test_compiled_layer_falls_through_when_dispatcher_returns_none():
    """CompiledLayer should fall through to per-op when dispatcher returns None."""
    torch = pytest.importorskip("torch")
    from research.synthesis.compiler import CompiledLayer

    g = _make_simple_graph(model_dim=8, ops=["relu"])
    layer = CompiledLayer(g)

    mock_dispatcher = MagicMock()
    mock_dispatcher.try_dispatch.return_value = None
    layer._subgraph_dispatcher = mock_dispatcher

    x = torch.randn(1, 4, 8)
    result = layer(x)

    mock_dispatcher.try_dispatch.assert_called_once_with(x)
    # Result should be relu(x) via per-op path
    expected = torch.relu(x)
    assert torch.allclose(result, expected)


def test_compiled_layer_no_dispatcher_unchanged():
    """CompiledLayer without a dispatcher should behave exactly as before."""
    torch = pytest.importorskip("torch")
    from research.synthesis.compiler import CompiledLayer

    g = _make_simple_graph(model_dim=8, ops=["relu"])
    layer = CompiledLayer(g)
    assert not hasattr(layer, "_subgraph_dispatcher")

    x = torch.randn(1, 4, 8)
    result = layer(x)

    expected = torch.relu(x)
    assert torch.allclose(result, expected)


# ---------------------------------------------------------------------------
# Test 9: Multi-op chain
# ---------------------------------------------------------------------------


def test_subgraph_dispatcher_multi_op_chain():
    """SubgraphDispatcher handles a chain of multiple ops."""
    g = _make_simple_graph(ops=["relu", "add", "gelu"])
    supported = {"relu", "add", "gelu"}
    dispatcher = SubgraphDispatcher(g, supported)
    assert dispatcher.all_native is True


def test_subgraph_dispatcher_multi_op_chain_partial():
    """A chain with one unsupported op should not be all-native."""
    # relu and add are supported, but softmax_last is not
    g = ComputationGraph(4)
    inp = g.add_input()
    r = g.add_op("relu", [inp])
    s = g.add_op("softmax_last", [r])
    g.set_output(s)

    supported = {"relu", "add"}
    dispatcher = SubgraphDispatcher(g, supported)
    assert dispatcher.all_native is False


# ---------------------------------------------------------------------------
# Test 10: graph_to_native_ir_json interop
# ---------------------------------------------------------------------------


def test_graph_converts_to_native_ir():
    """Ensure the test graphs produce valid native_ir JSON."""
    import json
    from research.synthesis.native_ir_converter import graph_to_native_ir_json

    g = _make_diamond_graph()
    ir_json = graph_to_native_ir_json(g)
    ir = json.loads(ir_json)

    assert ir["schema_version"] == "native_ir.v1"
    assert ir["model_dim"] == 4
    assert len(ir["nodes"]) == 4  # input, relu, gelu, add
    assert len(ir["edges"]) == 4  # relu<-input, gelu<-input, add<-relu, add<-gelu


# ---------------------------------------------------------------------------
# Test 11: IR JSON caching in SubgraphDispatcher
# ---------------------------------------------------------------------------


def test_subgraph_dispatcher_caches_ir_json():
    """SubgraphDispatcher should pre-convert and cache the IR JSON at init."""
    import json
    from research.synthesis.native_ir_converter import graph_to_native_ir_json

    g = _make_simple_graph(ops=["relu"])
    supported = {"relu"}
    dispatcher = SubgraphDispatcher(g, supported)

    # The cached IR should be a valid JSON string matching direct conversion
    assert dispatcher._ir_json is not None
    expected_ir = graph_to_native_ir_json(g)
    assert dispatcher._ir_json == expected_ir

    # Verify it parses as valid native_ir.v1
    ir = json.loads(dispatcher._ir_json)
    assert ir["schema_version"] == "native_ir.v1"


def test_subgraph_dispatcher_no_cache_when_not_all_native():
    """IR JSON should not be cached when the graph is not all-native."""
    g = _make_simple_graph(ops=["relu"])
    supported = {"add"}  # relu not included
    dispatcher = SubgraphDispatcher(g, supported)

    assert dispatcher.all_native is False
    assert dispatcher._ir_json is None


@patch("research.scientist.native.autograd.dispatch_graph_native_cached")
def test_subgraph_dispatcher_reuses_cached_ir_across_calls(mock_cached_dispatch):
    """The same cached IR JSON should be passed on every try_dispatch call."""
    g = _make_simple_graph(ops=["relu"])
    supported = {"relu"}
    dispatcher = SubgraphDispatcher(g, supported)

    cached_ir = dispatcher._ir_json
    assert cached_ir is not None

    mock_cached_dispatch.return_value = np.zeros(4, dtype=np.float32)
    x = np.zeros((1, 2, 4), dtype=np.float32)

    dispatcher.try_dispatch(x)
    dispatcher.try_dispatch(x)
    dispatcher.try_dispatch(x)

    assert mock_cached_dispatch.call_count == 3
    for call in mock_cached_dispatch.call_args_list:
        # First positional arg should be the same cached IR string
        assert call[0][0] is cached_ir


@patch("research.scientist.native.autograd.dispatch_graph_native")
def test_subgraph_dispatcher_fallback_when_ir_cache_none(mock_dispatch):
    """If IR caching fails, try_dispatch falls back to dispatch_graph_native."""
    g = _make_simple_graph(ops=["relu"])
    supported = {"relu"}
    dispatcher = SubgraphDispatcher(g, supported)

    # Force the cache to None to simulate conversion failure
    dispatcher._ir_json = None

    fake_output = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    mock_dispatch.return_value = fake_output

    x = np.zeros((1, 2, 4), dtype=np.float32)
    result = dispatcher.try_dispatch(x)

    assert result is not None
    np.testing.assert_array_equal(result, fake_output)
    mock_dispatch.assert_called_once_with(g, x)
