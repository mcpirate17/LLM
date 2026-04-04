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


def _fake_dispatch_op_native(op_name, *tensors, **kwargs):
    """Simple fake dispatch that mimics relu and add for testing."""
    if op_name == "relu":
        return np.maximum(tensors[0], 0.0)
    if op_name == "add":
        return tensors[0] + tensors[1]
    if op_name == "mul":
        return tensors[0] * tensors[1]
    if op_name == "softmax_attention":
        return tensors[0]
    if op_name == "selective_scan":
        return tensors[0]
    if op_name == "state_space":
        return tensors[0]
    if op_name == "gated_delta":
        return tensors[0]
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
def test_wrapper_dispatches_per_op_bridge_only_attention(mock_dispatch):
    wrapper = _make_wrapper({"softmax_attention"})
    x = np.zeros((1, 2, 4), dtype=np.float32)
    w = np.zeros((4, 4), dtype=np.float32)
    module = type(
        "_AttentionModule",
        (),
        {
            "q_proj": type("_Proj", (), {"weight": w})(),
            "k_proj": type("_Proj", (), {"weight": w})(),
            "v_proj": type("_Proj", (), {"weight": w})(),
            "o_proj": type("_Proj", (), {"weight": w})(),
            "n_heads": 2,
        },
    )()
    result = wrapper.dispatch(
        "softmax_attention",
        x,
        module=module,
    )
    assert result is not None
    np.testing.assert_array_equal(result, x)
    mock_dispatch.assert_called_once()


@patch(
    "research.scientist.native_runner.dispatch_op_native",
    side_effect=_fake_dispatch_op_native,
)
def test_wrapper_dispatches_per_op_bridge_only_selective_scan(mock_dispatch):
    wrapper = _make_wrapper({"selective_scan"})
    x = np.zeros((1, 2, 4), dtype=np.float32)
    module = type(
        "_SelectiveScanModule",
        (),
        {
            "A_log": np.zeros((4,), dtype=np.float32),
            "dt_proj": np.zeros((4,), dtype=np.float32),
            "B_proj": type(
                "_Proj", (), {"weight": np.zeros((4, 4), dtype=np.float32)}
            )(),
            "C_proj": type(
                "_Proj", (), {"weight": np.zeros((4, 4), dtype=np.float32)}
            )(),
        },
    )()
    result = wrapper.dispatch("selective_scan", x, module=module)
    assert result is not None
    np.testing.assert_array_equal(result, x)
    mock_dispatch.assert_called_once()


@patch(
    "research.scientist.native_runner.dispatch_op_native",
    side_effect=_fake_dispatch_op_native,
)
def test_wrapper_dispatches_per_op_bridge_only_state_space(mock_dispatch):
    wrapper = _make_wrapper({"state_space"})
    x = np.zeros((1, 2, 4), dtype=np.float32)
    module = type(
        "_StateSpaceModule",
        (),
        {
            "ssm_A": np.zeros((4, 16), dtype=np.float32),
            "ssm_B": type(
                "_Proj", (), {"weight": np.zeros((64, 4), dtype=np.float32)}
            )(),
            "ssm_C": type(
                "_Proj", (), {"weight": np.zeros((4, 64), dtype=np.float32)}
            )(),
            "ssm_D": np.zeros((4,), dtype=np.float32),
            "ssm_dt": type(
                "_Proj",
                (),
                {
                    "weight": np.zeros((4, 4), dtype=np.float32),
                    "bias": np.zeros((4,), dtype=np.float32),
                },
            )(),
        },
    )()
    result = wrapper.dispatch("state_space", x, module=module)
    assert result is not None
    np.testing.assert_array_equal(result, x)
    mock_dispatch.assert_called_once()


@patch(
    "research.scientist.native_runner.dispatch_op_native",
    side_effect=_fake_dispatch_op_native,
)
def test_wrapper_dispatches_per_op_bridge_only_gated_delta(mock_dispatch):
    wrapper = _make_wrapper({"gated_delta"})
    x = np.zeros((1, 2, 4), dtype=np.float32)
    w = np.zeros((4, 4), dtype=np.float32)
    module = type(
        "_GatedDeltaModule",
        (),
        {
            "model_dim": 4,
            "q_proj": type("_Proj", (), {"weight": w})(),
            "k_proj": type("_Proj", (), {"weight": w})(),
            "v_proj": type("_Proj", (), {"weight": w})(),
            "alpha_proj": type("_Proj", (), {"weight": w})(),
            "beta_proj": type("_Proj", (), {"weight": w})(),
            "o_proj": type("_Proj", (), {"weight": w})(),
            "_gated_delta_heads": 2,
        },
    )()
    result = wrapper.dispatch("gated_delta", x, module=module)
    assert result is not None
    np.testing.assert_array_equal(result, x)
    mock_dispatch.assert_called_once()


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


def test_wrapper_skips_host_bridge_for_non_cpu_tensors(monkeypatch):
    import research.scientist.native.autograd as native_autograd

    wrapper = native_autograd.NativeForwardWrapper(MagicMock(), {"relu"})
    dispatch_calls = []

    monkeypatch.setattr(
        native_autograd,
        "supports_host_array_bridge",
        lambda *values: False,
    )
    monkeypatch.setattr(
        native_autograd,
        "dispatch_op_native",
        lambda *args, **kwargs: dispatch_calls.append((args, kwargs)),
    )

    result = wrapper.dispatch("relu", np.array([1.0], dtype=np.float32))

    assert result is None
    assert dispatch_calls == []
    assert wrapper.stats["native_dispatches"] == 0
    assert wrapper.stats["fallbacks"] == 0
    assert (
        wrapper.stats["last_fallback_reason"] == "host_array_bridge_unsupported_device"
    )


def test_wrapper_uses_native_autograd_for_linear_proj(monkeypatch):
    torch = pytest.importorskip("torch")
    import research.scientist.native.autograd as native_autograd

    wrapper = native_autograd.NativeForwardWrapper(MagicMock(), {"linear_proj"})
    calls = []

    def _fake_native_autograd_dispatch(op_name, *inputs):
        calls.append((op_name, len(inputs)))
        return inputs[0]

    monkeypatch.setattr(
        "research.scientist.native_autograd.native_autograd_dispatch",
        _fake_native_autograd_dispatch,
    )

    module = type(
        "_LinearModule", (), {"weight": torch.randn(8, 8, requires_grad=True)}
    )()
    x = torch.randn(2, 3, 8, requires_grad=True)
    result = wrapper.dispatch("linear_proj", x, module=module)

    assert result is x
    assert calls == [("linear", 2)]


def test_wrapper_uses_native_autograd_for_rmsnorm(monkeypatch):
    torch = pytest.importorskip("torch")
    import research.scientist.native.autograd as native_autograd

    wrapper = native_autograd.NativeForwardWrapper(MagicMock(), {"rmsnorm"})
    calls = []

    def _fake_native_autograd_dispatch(op_name, *inputs):
        calls.append((op_name, len(inputs)))
        return inputs[0]

    monkeypatch.setattr(
        "research.scientist.native_autograd.native_autograd_dispatch",
        _fake_native_autograd_dispatch,
    )

    module = type("_NormModule", (), {"weight": torch.ones(8, requires_grad=True)})()
    x = torch.randn(2, 3, 8, requires_grad=True)
    result = wrapper.dispatch("rmsnorm", x, module=module)

    assert result is x
    assert calls == [("rmsnorm", 2)]


@pytest.mark.parametrize(
    "op_name",
    [
        "conv1d_seq",
        "rwkv_channel",
        "swiglu_mlp",
        "gated_linear",
        "rwkv_time_mixing",
        "softmax_attention",
        "selective_scan",
        "state_space",
        "gated_delta",
    ],
)
def test_wrapper_uses_bound_single_op_native_dispatch_for_composite_ops(
    monkeypatch, op_name
):
    torch = pytest.importorskip("torch")
    import research.scientist.native.autograd as native_autograd

    wrapper = native_autograd.NativeForwardWrapper(MagicMock(), {op_name})
    calls = []

    def _fake_bound_dispatch(op_name_arg, module, x, *, supported_ops):
        calls.append((op_name_arg, module, tuple(sorted(supported_ops))))
        return x

    monkeypatch.setattr(
        "research.scientist.native.autograd.dispatch_single_op_bound_native",
        _fake_bound_dispatch,
    )

    module = MagicMock()
    x = torch.randn(2, 3, 8, requires_grad=True)
    result = wrapper.dispatch(op_name, x, module=module)

    assert result is x
    assert calls == [(op_name, module, (op_name,))]
