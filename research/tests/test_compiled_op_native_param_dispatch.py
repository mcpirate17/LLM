from __future__ import annotations

import torch

from research.synthesis.compiler import CompiledOp
from research.synthesis.graph import ShapeInfo


class _TrackingWrapper:
    def __init__(self):
        self.calls = []

    def dispatch(self, op_name, *tensors, module=None):
        self.calls.append(
            {
                "op_name": op_name,
                "n_tensors": len(tensors),
                "module": module,
            }
        )
        return tensors[0]


def test_compiled_op_passes_module_to_native_wrapper_for_gated_linear():
    shape = ShapeInfo(dim=8)
    op = CompiledOp(
        op_name="gated_linear",
        config={"out_dim": 8},
        input_shape=shape,
        output_shape=shape,
        model_dim=8,
    )
    wrapper = _TrackingWrapper()
    op._native_wrapper = wrapper

    x = torch.randn(2, 3, 8)
    result = op(x)

    assert torch.equal(result, x)
    assert wrapper.calls == [
        {
            "op_name": "gated_linear",
            "n_tensors": 1,
            "module": op,
        }
    ]


def test_compiled_op_passes_module_to_native_wrapper_for_linear_proj():
    shape = ShapeInfo(dim=8)
    op = CompiledOp(
        op_name="linear_proj",
        config={"out_dim": 8},
        input_shape=shape,
        output_shape=shape,
        model_dim=8,
    )
    wrapper = _TrackingWrapper()
    op._native_wrapper = wrapper

    x = torch.randn(2, 3, 8)
    result = op(x)

    assert torch.equal(result, x)
    assert wrapper.calls == [
        {
            "op_name": "linear_proj",
            "n_tensors": 1,
            "module": op,
        }
    ]


def test_compiled_op_passes_module_to_native_wrapper_for_rmsnorm():
    shape = ShapeInfo(dim=8)
    op = CompiledOp(
        op_name="rmsnorm",
        config={},
        input_shape=shape,
        output_shape=shape,
        model_dim=8,
    )
    wrapper = _TrackingWrapper()
    op._native_wrapper = wrapper

    x = torch.randn(2, 3, 8)
    result = op(x)

    assert torch.equal(result, x)
    assert wrapper.calls == [
        {
            "op_name": "rmsnorm",
            "n_tensors": 1,
            "module": op,
        }
    ]


def test_compiled_op_passes_module_to_native_wrapper_for_rwkv_time_mixing():
    shape = ShapeInfo(dim=8)
    op = CompiledOp(
        op_name="rwkv_time_mixing",
        config={},
        input_shape=shape,
        output_shape=shape,
        model_dim=8,
    )
    wrapper = _TrackingWrapper()
    op._native_wrapper = wrapper

    x = torch.randn(2, 3, 8)
    result = op(x)

    assert torch.equal(result, x)
    assert wrapper.calls == [
        {
            "op_name": "rwkv_time_mixing",
            "n_tensors": 1,
            "module": op,
        }
    ]


def test_compiled_op_passes_module_to_native_wrapper_for_softmax_attention():
    shape = ShapeInfo(dim=8)
    op = CompiledOp(
        op_name="softmax_attention",
        config={},
        input_shape=shape,
        output_shape=shape,
        model_dim=8,
    )
    wrapper = _TrackingWrapper()
    op._native_wrapper = wrapper

    x = torch.randn(2, 3, 8)
    result = op(x)

    assert torch.equal(result, x)
    assert wrapper.calls == [
        {
            "op_name": "softmax_attention",
            "n_tensors": 1,
            "module": op,
        }
    ]


def test_compiled_op_passes_module_to_native_wrapper_for_selective_scan():
    shape = ShapeInfo(dim=8)
    op = CompiledOp(
        op_name="selective_scan",
        config={},
        input_shape=shape,
        output_shape=shape,
        model_dim=8,
    )
    wrapper = _TrackingWrapper()
    op._native_wrapper = wrapper

    x = torch.randn(2, 3, 8)
    result = op(x)

    assert torch.equal(result, x)
    assert wrapper.calls == [
        {
            "op_name": "selective_scan",
            "n_tensors": 1,
            "module": op,
        }
    ]


def test_compiled_op_passes_module_to_native_wrapper_for_state_space():
    shape = ShapeInfo(dim=8)
    op = CompiledOp(
        op_name="state_space",
        config={},
        input_shape=shape,
        output_shape=shape,
        model_dim=8,
    )
    wrapper = _TrackingWrapper()
    op._native_wrapper = wrapper

    x = torch.randn(2, 3, 8)
    result = op(x)

    assert torch.equal(result, x)
    assert wrapper.calls == [
        {
            "op_name": "state_space",
            "n_tensors": 1,
            "module": op,
        }
    ]


def test_compiled_op_passes_module_to_native_wrapper_for_gated_delta():
    shape = ShapeInfo(dim=8)
    op = CompiledOp(
        op_name="gated_delta",
        config={},
        input_shape=shape,
        output_shape=shape,
        model_dim=8,
    )
    wrapper = _TrackingWrapper()
    op._native_wrapper = wrapper

    x = torch.randn(2, 3, 8)
    result = op(x)

    assert torch.equal(result, x)
    assert wrapper.calls == [
        {
            "op_name": "gated_delta",
            "n_tensors": 1,
            "module": op,
        }
    ]
