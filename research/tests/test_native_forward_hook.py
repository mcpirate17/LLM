"""Tests for native forward wrapper hook integration at the CompiledOp level."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

# Ensure the research package is importable.
_root = str(Path(__file__).resolve().parents[1].parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from research.synthesis.compiler import CompiledOp, _execute_op
from research.synthesis.graph import ShapeInfo

pytestmark = pytest.mark.native


def _make_compiled_op(op_name: str = "relu", dim: int = 8) -> CompiledOp:
    """Helper to build a minimal CompiledOp for testing."""
    shape = ShapeInfo(dim=dim)
    return CompiledOp(op_name=op_name, config={}, input_shape=shape,
                      output_shape=shape, model_dim=dim)


# ── Test 1: wrapper intercepts when dispatch returns a tensor ─────

def test_compiled_op_uses_native_wrapper_when_set():
    op = _make_compiled_op("relu")
    mock_wrapper = MagicMock()
    fake_result = torch.tensor([1.0, 2.0, 3.0])
    mock_wrapper.dispatch.return_value = fake_result

    op._native_wrapper = mock_wrapper

    x = torch.tensor([-1.0, 0.0, 3.0])
    result = op(x)

    mock_wrapper.dispatch.assert_called_once_with("relu", x)
    assert torch.equal(result, fake_result)


# ── Test 2: wrapper returns None → falls through to _execute_op ──

def test_compiled_op_falls_through_when_wrapper_returns_none():
    op = _make_compiled_op("relu")
    mock_wrapper = MagicMock()
    mock_wrapper.dispatch.return_value = None

    op._native_wrapper = mock_wrapper

    x = torch.randn(1, 4, 8)
    result = op(x)

    mock_wrapper.dispatch.assert_called_once()
    # relu fallback: negative values zeroed
    assert (result >= 0).all()


# ── Test 3: no wrapper set → original behavior unchanged ─────────

def test_compiled_op_no_wrapper_uses_default_path():
    op = _make_compiled_op("relu")
    # No _native_wrapper attribute set at all
    assert not hasattr(op, '_native_wrapper')

    x = torch.tensor([[[-1.0, 0.0, 2.0, 3.0, -5.0, 1.0, 0.5, -0.5]]])
    result = op(x)
    expected = torch.relu(x)
    assert torch.allclose(result, expected)


# ── Test 4: wrapper propagation to ops via model.layers ──────────

def test_wrapper_propagation_to_ops():
    """Simulate the propagation loop from compile_model_native_first."""
    # Build a fake model structure: model.layers[i].ops dict of CompiledOps
    op1 = _make_compiled_op("relu")
    op2 = _make_compiled_op("gelu")

    class FakeLayer:
        def __init__(self, ops_dict):
            self.ops = ops_dict

    layer = FakeLayer({"0": op1, "1": op2})

    class FakeModel:
        def __init__(self):
            self.layers = [layer]

    model = FakeModel()
    wrapper = MagicMock()

    # Propagation logic (mirrors native_runner.py)
    for mdl_layer in getattr(model, 'layers', []):
        ops = getattr(mdl_layer, 'ops', None)
        if ops is not None:
            for op in ops.values():
                if hasattr(op, 'forward'):
                    op._native_wrapper = wrapper

    assert op1._native_wrapper is wrapper
    assert op2._native_wrapper is wrapper


# ── Test 5: end-to-end native dispatch for relu ──────────────────

def test_native_dispatch_produces_correct_relu_output():
    """CompiledOp with a real-ish wrapper dispatching relu."""
    op = _make_compiled_op("relu")

    class SimpleReluWrapper:
        """Minimal wrapper that handles relu natively via torch."""
        def dispatch(self, op_name, *tensors):
            if op_name == "relu":
                return torch.relu(tensors[0])
            return None

    op._native_wrapper = SimpleReluWrapper()

    x = torch.tensor([[[- 2.0, 0.0, 1.5, -0.3, 4.0, -1.0, 0.0, 3.0]]])
    result = op(x)
    expected = torch.relu(x)
    assert torch.allclose(result, expected), f"Expected {expected}, got {result}"
