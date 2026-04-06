from __future__ import annotations

import pytest
import torch

from research.scientist.native.dispatch import (
    dispatch_graph_forward_saved_multi_input_cached,
    dispatch_graph_native_multi_input_cached,
)


def test_dispatch_graph_native_multi_input_cached_returns_torch(monkeypatch):
    import research.scientist.native.dispatch as native_dispatch

    class FakeRust:
        @staticmethod
        def execute_graph_multi_input(ir_json, inputs):
            assert ir_json == "fake-ir"
            assert len(inputs) == 2
            return [sum(inputs[0]), sum(inputs[1])]

    monkeypatch.setattr(
        native_dispatch, "_try_import_rust_scheduler", lambda: FakeRust()
    )

    x = torch.tensor([1.0, 2.0], dtype=torch.float32)
    y = torch.tensor([3.0, 4.0], dtype=torch.float32)
    result = dispatch_graph_native_multi_input_cached(
        "fake-ir",
        [x, y],
        output_shape=(2,),
    )

    assert isinstance(result, torch.Tensor)
    torch.testing.assert_close(result, torch.tensor([3.0, 7.0]))


def test_dispatch_graph_native_multi_input_cached_prefers_array_buffers(monkeypatch):
    import research.scientist.native.dispatch as native_dispatch

    calls = {"arrays": 0, "lists": 0}

    class FakeRust:
        @staticmethod
        def execute_graph_multi_input_arrays(ir_json, inputs):
            assert ir_json == "fake-ir"
            assert len(inputs) == 2
            calls["arrays"] += 1
            return [float(inputs[0].sum()), float(inputs[1].sum())]

        @staticmethod
        def execute_graph_multi_input(ir_json, inputs):
            calls["lists"] += 1
            raise AssertionError("list fallback should not be used")

    monkeypatch.setattr(
        native_dispatch, "_try_import_rust_scheduler", lambda: FakeRust()
    )

    x = torch.tensor([1.0, 2.0], dtype=torch.float32)
    y = torch.tensor([3.0, 4.0], dtype=torch.float32)
    result = dispatch_graph_native_multi_input_cached(
        "fake-ir",
        [x, y],
        output_shape=(2,),
    )

    assert isinstance(result, torch.Tensor)
    torch.testing.assert_close(result, torch.tensor([3.0, 7.0]))
    assert calls == {"arrays": 1, "lists": 0}


def test_dispatch_graph_forward_saved_multi_input_cached_returns_saved(monkeypatch):
    import research.scientist.native.dispatch as native_dispatch

    class FakeRust:
        @staticmethod
        def execute_graph_forward_saved_multi_input(ir_json, inputs):
            assert ir_json == "fake-ir"
            assert len(inputs) == 2
            return {
                "output": [3.0, 7.0],
                "saved_activations": {0: inputs[0], 1: inputs[1], 2: [3.0, 7.0]},
                "arena_bytes_used": 32,
                "arena_capacity": 64,
            }

    monkeypatch.setattr(
        native_dispatch, "_try_import_rust_scheduler", lambda: FakeRust()
    )

    x = torch.tensor([1.0, 2.0], dtype=torch.float32)
    y = torch.tensor([3.0, 4.0], dtype=torch.float32)
    result = dispatch_graph_forward_saved_multi_input_cached(
        "fake-ir",
        [x, y],
        output_shape=(2,),
    )

    assert isinstance(result["output"], torch.Tensor)
    torch.testing.assert_close(result["output"], torch.tensor([3.0, 7.0]))
    assert set(result["saved_activations"]) == {0, 1, 2}


def test_dispatch_graph_forward_saved_multi_input_cached_prefers_array_buffers(
    monkeypatch,
):
    import research.scientist.native.dispatch as native_dispatch

    calls = {"arrays": 0, "lists": 0}

    class FakeRust:
        @staticmethod
        def execute_graph_forward_saved_multi_input_arrays(ir_json, inputs):
            assert ir_json == "fake-ir"
            assert len(inputs) == 2
            calls["arrays"] += 1
            return {
                "output": [3.0, 7.0],
                "saved_activations": {
                    0: inputs[0].tolist(),
                    1: inputs[1].tolist(),
                    2: [3.0, 7.0],
                },
                "arena_bytes_used": 32,
                "arena_capacity": 64,
            }

        @staticmethod
        def execute_graph_forward_saved_multi_input(ir_json, inputs):
            calls["lists"] += 1
            raise AssertionError("list fallback should not be used")

    monkeypatch.setattr(
        native_dispatch, "_try_import_rust_scheduler", lambda: FakeRust()
    )

    x = torch.tensor([1.0, 2.0], dtype=torch.float32)
    y = torch.tensor([3.0, 4.0], dtype=torch.float32)
    result = dispatch_graph_forward_saved_multi_input_cached(
        "fake-ir",
        [x, y],
        output_shape=(2,),
    )

    assert isinstance(result["output"], torch.Tensor)
    torch.testing.assert_close(result["output"], torch.tensor([3.0, 7.0]))
    assert set(result["saved_activations"]) == {0, 1, 2}
    assert calls == {"arrays": 1, "lists": 0}


def test_dispatch_graph_native_multi_input_cached_rejects_non_cpu_host_bridge(
    monkeypatch,
):
    import research.scientist.native.dispatch as native_dispatch

    monkeypatch.setattr(
        native_dispatch,
        "supports_host_array_bridge",
        lambda *values: False,
    )
    monkeypatch.setattr(
        native_dispatch,
        "_try_import_rust_scheduler",
        lambda: (_ for _ in ()).throw(AssertionError("rust scheduler should not load")),
    )

    x = torch.tensor([1.0, 2.0], dtype=torch.float32)
    y = torch.tensor([3.0, 4.0], dtype=torch.float32)
    with pytest.raises(RuntimeError, match="Host array bridge does not support"):
        dispatch_graph_native_multi_input_cached("fake-ir", [x, y], output_shape=(2,))
