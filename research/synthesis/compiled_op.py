from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .compiled_op_params import CompiledOpParamInitMixin
from .compiled_op_runtime import CompiledOpRuntimeMixin
from .compiler_constants import MATHSPACE_OPS
from .compiler_registry import OP_DISPATCH
from .graph import ShapeInfo
from .primitives import PRIMITIVE_REGISTRY, get_primitive

_MODULE_NATIVE_DISPATCH_OPS = frozenset(
    {
        "linear_proj",
        "linear_proj_down",
        "linear_proj_up",
        "rmsnorm",
        "layernorm",
        "conv1d_seq",
        "rwkv_channel",
        "swiglu_mlp",
        "gated_linear",
        "rwkv_time_mixing",
        "softmax_attention",
        "linear_attention",
        "gated_lane_blend",
        "route_lanes",
        "depth_gated_transform",
        "route_recursion",
        "depth_weighted_proj",
        "adaptive_recursion",
        "selective_scan",
        "state_space",
        "gated_delta",
    }
)


def _reshape_native_result_if_needed(
    result: torch.Tensor,
    *,
    inputs: tuple[torch.Tensor, ...],
    output_shape: ShapeInfo,
) -> torch.Tensor:
    if not inputs or not isinstance(result, torch.Tensor):
        return result

    input0 = inputs[0]
    if not isinstance(input0, torch.Tensor):
        return result
    if not output_shape.is_standard:
        return result

    target_shape = tuple(int(v) for v in input0.shape[:-1]) + (int(output_shape.dim),)
    if result.shape == target_shape:
        return result
    if result.numel() != int(torch.Size(target_shape).numel()):
        return result
    return result.reshape(target_shape)


def sanitize_mathspace_result(
    result: torch.Tensor,
    module: nn.Module,
    op_name: str,
    collect_telemetry: bool,
) -> torch.Tensor:
    if collect_telemetry:
        nonfinite = int((~torch.isfinite(result)).sum().item())
        if nonfinite > 0:
            result = torch.nan_to_num(result, nan=0.0, posinf=1e4, neginf=-1e4)
            telemetry = getattr(module, "mathspace_telemetry", {})
            if len(telemetry) < 256:
                stats = telemetry.get(op_name, {"calls": 0, "nonfinite": 0})
                stats["calls"] += 1
                stats["nonfinite"] += nonfinite
                telemetry[op_name] = stats
                setattr(module, "mathspace_telemetry", telemetry)
    else:
        result = torch.nan_to_num(result, nan=0.0, posinf=1e4, neginf=-1e4)
    return result


class CompiledOp(CompiledOpParamInitMixin, CompiledOpRuntimeMixin, nn.Module):
    """A single compiled primitive operation."""

    def __init__(
        self,
        op_name: str,
        config: Dict,
        input_shape: ShapeInfo,
        output_shape: ShapeInfo,
        model_dim: int,
    ):
        super().__init__()
        self.op_name = op_name
        self.config = config
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model_dim = model_dim
        self._cached_native_wrapper = None
        self._last_cast_dtype = None
        self._cached_dispatch_fn = OP_DISPATCH.get(op_name)
        op = get_primitive(op_name)
        self._is_math_op = op_name in MATHSPACE_OPS or op.category.value == "math_space"
        if op.has_params:
            self._init_params(op, config, input_shape)

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name == "_native_wrapper":
            super().__setattr__("_cached_native_wrapper", value)

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if inputs and inputs[0].is_floating_point():
            self._cast_params_to(inputs[0].dtype)

        profile = getattr(self, "collect_telemetry", False)
        if profile:
            import time as _time

            t0 = _time.perf_counter()

        wrapper = self._cached_native_wrapper
        if wrapper is not None:
            if self.op_name in _MODULE_NATIVE_DISPATCH_OPS:
                result = wrapper.dispatch(self.op_name, *inputs, module=self)
            else:
                result = wrapper.dispatch(self.op_name, *inputs)
            if result is not None:
                result = _reshape_native_result_if_needed(
                    result,
                    inputs=inputs,
                    output_shape=self.output_shape,
                )
                if profile:
                    self._record_op_timing(_time.perf_counter() - t0)
                return result

        dispatch_fn = self._cached_dispatch_fn
        if dispatch_fn is not None:
            result = dispatch_fn(self, inputs, self.config)
        else:
            prim = PRIMITIVE_REGISTRY.get(self.op_name)
            if (
                prim is not None
                and hasattr(prim, "execute_fn")
                and prim.execute_fn is not None
            ):
                result = prim.execute_fn(self, *inputs)
            else:
                raise ValueError(
                    f"Unknown op: {self.op_name} (no dispatch handler or execute_fn)"
                )
        if self._is_math_op:
            result = sanitize_mathspace_result(result, self, self.op_name, profile)
        if profile:
            self._record_op_timing(_time.perf_counter() - t0)
        return result
