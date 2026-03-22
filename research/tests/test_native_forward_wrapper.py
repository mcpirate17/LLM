"""Tests for NativeForwardWrapper in native_runner.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from research.scientist.native_runner import NativeForwardWrapper

pytestmark = pytest.mark.native


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wrapper(supported_ops=None):
    """Create a NativeForwardWrapper with a mock model."""
    model = MagicMock()
    ops = supported_ops if supported_ops is not None else {"relu", "add", "mul"}
    return NativeForwardWrapper(model, ops)


def _fake_dispatch_op_native(op_name, *tensors):
    """Simple fake dispatch that mimics relu and add for testing."""
    if op_name == "relu":
        return np.maximum(tensors[0], 0.0)
    if op_name == "add":
        return tensors[0] + tensors[1]
    if op_name == "mul":
        return tensors[0] * tensors[1]
    raise ValueError(f"Unsupported op: {op_name}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@patch(
    "research.scientist.native_runner.dispatch_op_native",
    side_effect=_fake_dispatch_op_native,
)
def test_wrapper_dispatches_relu_through_native(mock_dispatch):
    """Wrapper should route relu through the native dispatch path."""
    wrapper = _make_wrapper({"relu"})
    x = np.array([-1.0, 0.0, 2.0, 3.5], dtype=np.float32)
    result = wrapper.dispatch("relu", x)
    assert result is not None
    np.testing.assert_array_equal(
        result, np.array([0.0, 0.0, 2.0, 3.5], dtype=np.float32)
    )
    mock_dispatch.assert_called_once()


def test_wrapper_returns_none_for_unsupported_op():
    """Wrapper should return None for ops not in supported_ops set."""
    wrapper = _make_wrapper({"relu"})
    x = np.array([1.0, 2.0], dtype=np.float32)
    result = wrapper.dispatch("softmax_last", x)
    assert result is None
    assert wrapper.stats["native_dispatches"] == 0
    assert wrapper.stats["fallbacks"] == 0


@patch(
    "research.scientist.native_runner.dispatch_op_native",
    side_effect=_fake_dispatch_op_native,
)
def test_wrapper_handles_torch_tensor_conversion(mock_dispatch):
    """Wrapper should convert torch tensors to numpy, dispatch, and convert back."""
    torch = pytest.importorskip("torch")
    wrapper = _make_wrapper({"relu"})
    x = torch.tensor([-2.0, 0.0, 1.0, 4.0], dtype=torch.float32)
    result = wrapper.dispatch("relu", x)
    assert result is not None
    assert isinstance(result, torch.Tensor)
    expected = torch.tensor([0.0, 0.0, 1.0, 4.0], dtype=torch.float32)
    torch.testing.assert_close(result, expected)


@patch(
    "research.scientist.native_runner.dispatch_op_native",
    side_effect=_fake_dispatch_op_native,
)
def test_wrapper_handles_numpy_input(mock_dispatch):
    """Wrapper should handle raw numpy arrays without conversion errors."""
    wrapper = _make_wrapper({"add"})
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    result = wrapper.dispatch("add", a, b)
    assert result is not None
    np.testing.assert_array_equal(
        result, np.array([11.0, 22.0, 33.0], dtype=np.float32)
    )


@patch(
    "research.scientist.native_runner.dispatch_op_native",
    side_effect=_fake_dispatch_op_native,
)
def test_wrapper_stats_tracking(mock_dispatch):
    """Wrapper should accurately track dispatch and fallback counts."""
    wrapper = _make_wrapper({"relu", "add"})
    x = np.array([1.0, -1.0], dtype=np.float32)

    # Two successful dispatches
    wrapper.dispatch("relu", x)
    wrapper.dispatch("add", x, x)

    # One unsupported op (returns None, no stat change)
    wrapper.dispatch("unknown_op", x)

    stats = wrapper.stats
    assert stats["native_dispatches"] == 2
    assert stats["fallbacks"] == 0


@patch(
    "research.scientist.native_runner.dispatch_op_native",
    side_effect=RuntimeError("kernel crashed"),
)
def test_wrapper_fallback_on_dispatch_error(mock_dispatch):
    """Wrapper should increment fallback count and return None on dispatch error."""
    wrapper = _make_wrapper({"relu"})
    x = np.array([1.0, 2.0], dtype=np.float32)
    result = wrapper.dispatch("relu", x)
    assert result is None
    assert wrapper.stats["native_dispatches"] == 0
    assert wrapper.stats["fallbacks"] == 1
