from __future__ import annotations

import torch

from research.scientist.native.autograd import NativeForwardWrapper
from research.synthesis.compiler import CompiledOp
from research.synthesis.graph import ShapeInfo


def _run_grad_case(
    op_name: str,
    config: dict,
    dim: int = 8,
    seq: int = 5,
    *,
    forward_rtol: float = 1e-3,
    forward_atol: float = 1e-4,
    grad_rtol: float = 1e-2,
    grad_atol: float = 1e-3,
    param_rtol: float = 2e-2,
    param_atol: float = 2e-3,
):
    shape = ShapeInfo(dim=dim)
    native_op = CompiledOp(op_name, config, shape, shape, dim)
    python_op = CompiledOp(op_name, config, shape, shape, dim)
    python_op.load_state_dict(native_op.state_dict(), strict=False)
    wrapper = NativeForwardWrapper(native_op, {op_name})
    native_op._native_wrapper = wrapper

    x_native = torch.randn(2, seq, dim, dtype=torch.float32, requires_grad=True)
    x_python = x_native.detach().clone().requires_grad_(True)

    y_native = native_op(x_native)
    y_python = python_op(x_python)
    torch.testing.assert_close(
        y_native,
        y_python,
        rtol=forward_rtol,
        atol=forward_atol,
    )

    loss_native = y_native.square().mean()
    loss_python = y_python.square().mean()
    loss_native.backward()
    loss_python.backward()

    torch.testing.assert_close(
        x_native.grad,
        x_python.grad,
        rtol=grad_rtol,
        atol=grad_atol,
    )
    for (name_native, param_native), (_, param_python) in zip(
        native_op.named_parameters(), python_op.named_parameters()
    ):
        assert param_native.grad is not None, name_native
        assert param_python.grad is not None, name_native
        torch.testing.assert_close(
            param_native.grad,
            param_python.grad,
            rtol=param_rtol,
            atol=param_atol,
        )


def test_native_gated_linear_backward_matches_python():
    _run_grad_case("gated_linear", {"out_dim": 8})


def test_native_softmax_attention_backward_matches_python():
    _run_grad_case("softmax_attention", {})


def test_native_selective_scan_backward_matches_python():
    _run_grad_case("selective_scan", {})


def test_native_state_space_backward_matches_python():
    _run_grad_case("state_space", {})


def test_native_gated_delta_backward_matches_python():
    _run_grad_case("gated_delta", {})


def test_native_rwkv_time_mixing_backward_matches_python():
    _run_grad_case(
        "rwkv_time_mixing",
        {},
        forward_rtol=5e-2,
        forward_atol=4e-3,
    )


def test_native_conv1d_seq_backward_matches_python():
    _run_grad_case("conv1d_seq", {})


def test_native_rwkv_channel_backward_matches_python():
    _run_grad_case(
        "rwkv_channel",
        {},
        forward_rtol=1e-1,
        forward_atol=5e-4,
    )


def test_native_swiglu_backward_matches_python():
    _run_grad_case("swiglu_mlp", {})
