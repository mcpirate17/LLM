from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import torch

from ._json_compat import dumps_json
from .compiler_op_utils import _get_stacked_params
from .graph import ComputationGraph
from .native_bound_common import (
    rows_for_bound_tensor,
    runtime_shape_key,
    supports_bound_input,
)
from .native_support import (
    BOUND_BACKWARD_OPS,
    BOUND_BINARY_OPS,
    BOUND_PARAM_OPS,
    BOUND_POINTWISE_OPS,
    BOUND_STRUCTURAL_OPS,
    BOUND_SUPPORTED_OPS,
)
from ..scientist.native.dispatch import (
    dispatch_graph_backward_native_cached,
    dispatch_graph_forward_saved_multi_input_cached,
    dispatch_graph_native_multi_input_cached,
)
from ..scientist.native.tensor_bridge import supports_host_array_bridge


def _resolve_attr(module: torch.nn.Module, attr_path: str) -> Any:
    value = module
    for attr in attr_path.split("."):
        value = getattr(value, attr)
    return value


@dataclass(slots=True)
class _PayloadSpec:
    node_id: int
    module: torch.nn.Module | None
    attr_path: str | None
    tensor: torch.Tensor | None = None
    stack_attr_name: str | None = None
    stack_count: int = 0


@dataclass(slots=True)
class _BoundGraphPlan:
    ir_json: str
    output_shape: tuple[int, ...]
    payload_specs: tuple[_PayloadSpec, ...]

    @property
    def input_node_ids(self) -> tuple[int, ...]:
        return tuple(spec.node_id for spec in self.payload_specs)

    def payloads(self, x: torch.Tensor) -> list[torch.Tensor]:
        out: list[torch.Tensor] = []
        for spec in self.payload_specs:
            if spec.tensor is not None:
                out.append(spec.tensor)
            elif spec.stack_attr_name is not None and spec.module is not None:
                out.append(
                    _get_stacked_params(
                        spec.module,
                        spec.stack_attr_name,
                        spec.stack_count,
                        x.dtype,
                    )
                )
            elif spec.module is None:
                out.append(x)
            else:
                out.append(_resolve_attr(spec.module, spec.attr_path))
        return out


class _BoundNativeGraphFunction:
    @staticmethod
    def make(dispatcher: "BoundNativeSubgraphDispatcher"):
        import torch

        class _Fn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, *params):
                plan = dispatcher._plan_for_input(x)
                payloads = [x, *params]
                result = dispatch_graph_forward_saved_multi_input_cached(
                    plan.ir_json,
                    payloads,
                    output_shape=plan.output_shape,
                )
                ctx._ir_json = plan.ir_json
                ctx._saved_activations = result["saved_activations"]
                ctx._input_node_ids = plan.input_node_ids
                ctx._input_shapes = tuple(
                    tuple(int(v) for v in tensor.shape) for tensor in payloads
                )
                return result["output"]

            @staticmethod
            def backward(ctx, grad_output):
                grads = dispatch_graph_backward_native_cached(
                    ctx._ir_json,
                    grad_output,
                    ctx._saved_activations,
                )
                outputs: list[torch.Tensor | None] = []
                for node_id, shape in zip(ctx._input_node_ids, ctx._input_shapes):
                    grad = grads.get(node_id)
                    if grad is None:
                        outputs.append(None)
                        continue
                    grad_tensor = torch.as_tensor(
                        grad,
                        dtype=grad_output.dtype,
                        device=grad_output.device,
                    ).reshape(shape)
                    outputs.append(grad_tensor)
                return tuple(outputs)

        return _Fn


class BoundNativeSubgraphDispatcher:
    def __init__(
        self,
        graph: ComputationGraph,
        *,
        flat_ops: list[torch.nn.Module | None],
        ir_node_ids: List[int],
        supported_ops: set[str],
    ):
        self._graph = graph
        self._flat_ops = flat_ops
        self._node_id_to_ir_idx = {
            int(node_id): idx for idx, node_id in enumerate(ir_node_ids)
        }
        self._supported_ops = frozenset(supported_ops)
        self._all_native = self._check_all_native()
        self._backward_native = self._all_native and self._check_backward_native()
        self._runtime_enabled = True
        self._plan_cache: Dict[tuple[int, ...], _BoundGraphPlan] = {}
        self._dispatch_count = 0
        self._fallback_count = 0
        self._autograd_fn = None
        self._last_refusal_reason: str | None = None

    @property
    def all_native(self) -> bool:
        return self._all_native and self._runtime_enabled

    def _check_all_native(self) -> bool:
        nodes = getattr(self._graph, "nodes", None)
        if not isinstance(nodes, dict) or not nodes:
            return False
        has_param_op = False
        for node in nodes.values():
            op_name = getattr(node, "op_name", "")
            if node.is_input:
                continue
            if op_name not in BOUND_SUPPORTED_OPS:
                return False
            if (
                op_name not in BOUND_STRUCTURAL_OPS
                and op_name != "conv_only"
                and op_name not in self._supported_ops
            ):
                return False
            if op_name in BOUND_PARAM_OPS:
                has_param_op = True
                ir_idx = self._node_id_to_ir_idx.get(node.id)
                if ir_idx is None or self._flat_ops[ir_idx] is None:
                    return False
        return has_param_op

    def _check_backward_native(self) -> bool:
        return all(
            node.op_name in BOUND_BACKWARD_OPS
            for node in self._graph.nodes.values()
            if not node.is_input and not node.is_output
        )

    def _runtime_shape_key(self, x: torch.Tensor) -> tuple[int, ...]:
        return runtime_shape_key(x)

    def _rows(self, x: torch.Tensor) -> int:
        return rows_for_bound_tensor(x, label="bound native input")

    def _supports_input(self, x: torch.Tensor) -> bool:
        return supports_bound_input(x)

    def _seq(self, x: torch.Tensor) -> int:
        return int(x.shape[1]) if x.ndim == 3 else 1

    def _shape_config(self, x: torch.Tensor, dim: int) -> dict:
        return {"batch": int(x.shape[0]), "seq": self._seq(x), "dim": int(dim)}

    def _append_tensor_input(
        self,
        *,
        next_id: int,
        payload_specs: list[_PayloadSpec],
        nodes: list[dict],
        edges: list[dict],
        target_inputs: list[int],
        tensor: torch.Tensor,
    ) -> int:
        param_id = next_id
        next_id += 1
        nodes.append(
            {
                "id": param_id,
                "op_name": "input",
                "input_ids": [],
                "config": {},
                "is_input": True,
                "is_output": False,
            }
        )
        payload_specs.append(_PayloadSpec(param_id, None, None, tensor))
        target_inputs.append(param_id)
        return next_id

    def _emit_param_inputs(
        self,
        *,
        next_id: int,
        payload_specs: list[_PayloadSpec],
        nodes: list[dict],
        edges: list[dict],
        target_inputs: list[int],
        module: torch.nn.Module,
        attr_paths: Iterable[str],
    ) -> int:
        for attr_path in attr_paths:
            param_id = next_id
            next_id += 1
            nodes.append(
                {
                    "id": param_id,
                    "op_name": "input",
                    "input_ids": [],
                    "config": {},
                    "is_input": True,
                    "is_output": False,
                }
            )
            payload_specs.append(_PayloadSpec(param_id, module, attr_path))
            target_inputs.append(param_id)
        return next_id

    def _emit_stacked_param_input(
        self,
        *,
        next_id: int,
        payload_specs: list[_PayloadSpec],
        nodes: list[dict],
        edges: list[dict],
        target_inputs: list[int],
        module: torch.nn.Module,
        attr_name: str,
        count: int,
    ) -> int:
        param_id = next_id
        next_id += 1
        nodes.append(
            {
                "id": param_id,
                "op_name": "input",
                "input_ids": [],
                "config": {},
                "is_input": True,
                "is_output": False,
            }
        )
        payload_specs.append(
            _PayloadSpec(
                param_id,
                module,
                None,
                stack_attr_name=attr_name,
                stack_count=count,
            )
        )
        target_inputs.append(param_id)
        return next_id

    # -- Parametric op lowering handlers --
    # Each returns (native_name, config, input_ids, next_id, early_return).
    # early_return=True means the handler already appended final nodes/edges.

    def _lower_linear_proj(
        self,
        x: torch.Tensor,
        module: torch.nn.Module,
        dim_in: int,
        dim_out: int,
        input_ids: list[int],
        next_id: int,
        payload_specs: list[_PayloadSpec],
        nodes: list[dict],
        edges: list[dict],
    ) -> tuple[str, dict, list[int], int]:
        config = {"batch": self._rows(x), "dim_in": dim_in, "dim_out": dim_out}
        next_id = self._emit_param_inputs(
            next_id=next_id,
            payload_specs=payload_specs,
            nodes=nodes,
            edges=edges,
            target_inputs=input_ids,
            module=module,
            attr_paths=("weight",),
        )
        return "linear", config, input_ids, next_id

    def _lower_conv1d_seq(
        self,
        x: torch.Tensor,
        module: torch.nn.Module,
        dim_in: int,
        input_ids: list[int],
        next_id: int,
        payload_specs: list[_PayloadSpec],
        nodes: list[dict],
        edges: list[dict],
    ) -> tuple[str, dict, list[int], int]:
        config = self._shape_config(x, dim_in)
        next_id = self._emit_param_inputs(
            next_id=next_id,
            payload_specs=payload_specs,
            nodes=nodes,
            edges=edges,
            target_inputs=input_ids,
            module=module,
            attr_paths=("conv_weight",),
        )
        conv_bias = getattr(module, "conv_bias", None)
        if conv_bias is None:
            conv_bias = module.conv_weight.new_zeros(module.conv_weight.shape[0])
        next_id = self._append_tensor_input(
            next_id=next_id,
            payload_specs=payload_specs,
            nodes=nodes,
            edges=edges,
            target_inputs=input_ids,
            tensor=conv_bias,
        )
        return "conv1d_seq", config, input_ids, next_id

    def _lower_normalization(
        self,
        x: torch.Tensor,
        module: torch.nn.Module,
        op_name: str,
        dim_in: int,
        input_ids: list[int],
        next_id: int,
        payload_specs: list[_PayloadSpec],
        nodes: list[dict],
        edges: list[dict],
    ) -> tuple[str, dict, list[int], int]:
        if op_name == "rmsnorm":
            config = {"batch": self._rows(x), "dim": dim_in, "eps": 1e-6}
            attrs: tuple[str, ...] = ("weight",)
        else:
            config = {"batch": self._rows(x), "dim": dim_in, "eps": 1e-5}
            attrs = ("weight", "bias")
        next_id = self._emit_param_inputs(
            next_id=next_id,
            payload_specs=payload_specs,
            nodes=nodes,
            edges=edges,
            target_inputs=input_ids,
            module=module,
            attr_paths=attrs,
        )
        return op_name, config, input_ids, next_id

    def _lower_gated_linear(
        self,
        x: torch.Tensor,
        module: torch.nn.Module,
        dim_in: int,
        dim_out: int,
        input_ids: list[int],
        next_id: int,
        payload_specs: list[_PayloadSpec],
        nodes: list[dict],
        edges: list[dict],
    ) -> tuple[str, dict, list[int], int]:
        config = {"batch": self._rows(x), "dim_in": dim_in, "dim_out": dim_out}
        next_id = self._emit_param_inputs(
            next_id=next_id,
            payload_specs=payload_specs,
            nodes=nodes,
            edges=edges,
            target_inputs=input_ids,
            module=module,
            attr_paths=("linear_weight", "linear_bias", "gate_weight", "gate_bias"),
        )
        return "gated_linear", config, input_ids, next_id

    def _lower_rwkv_time_mixing(
        self,
        x: torch.Tensor,
        module: torch.nn.Module,
        node: Any,
        node_id: int,
        dim_in: int,
        dim_out: int,
        input_ids: list[int],
        next_id: int,
        payload_specs: list[_PayloadSpec],
        nodes: list[dict],
        edges: list[dict],
    ) -> int:
        """Multi-node: mixing → linear projection. Returns next_id (early return)."""
        config = self._shape_config(x, dim_in)
        mixing_output_id = next_id
        next_id += 1
        mixing_inputs = list(input_ids)
        next_id = self._emit_param_inputs(
            next_id=next_id,
            payload_specs=payload_specs,
            nodes=nodes,
            edges=edges,
            target_inputs=mixing_inputs,
            module=module,
            attr_paths=("w_decay", "u_bonus", "W_k", "W_v", "W_r"),
        )
        nodes.append(
            {
                "id": mixing_output_id,
                "op_name": "rwkv_time_mixing",
                "input_ids": mixing_inputs,
                "config": config,
                "is_input": False,
                "is_output": False,
            }
        )
        for iid in mixing_inputs:
            edges.append({"source": iid, "target": mixing_output_id})

        linear_inputs = [mixing_output_id]
        next_id = self._emit_param_inputs(
            next_id=next_id,
            payload_specs=payload_specs,
            nodes=nodes,
            edges=edges,
            target_inputs=linear_inputs,
            module=module,
            attr_paths=("W_o",),
        )
        nodes.append(
            {
                "id": node_id,
                "op_name": "linear",
                "input_ids": linear_inputs,
                "config": {
                    "batch": self._rows(x),
                    "dim_in": dim_in,
                    "dim_out": dim_out,
                },
                "is_input": False,
                "is_output": node.is_output,
            }
        )
        for iid in linear_inputs:
            edges.append({"source": iid, "target": node_id})
        return next_id

    def _lower_rwkv_channel(
        self,
        x: torch.Tensor,
        module: torch.nn.Module,
        dim_in: int,
        input_ids: list[int],
        next_id: int,
        payload_specs: list[_PayloadSpec],
        nodes: list[dict],
        edges: list[dict],
    ) -> tuple[str, dict, list[int], int]:
        config = {
            **self._shape_config(x, dim_in),
            "hidden_dim": int(module.key_proj.weight.shape[0]),
        }
        next_id = self._emit_param_inputs(
            next_id=next_id,
            payload_specs=payload_specs,
            nodes=nodes,
            edges=edges,
            target_inputs=input_ids,
            module=module,
            attr_paths=(
                "mix_k",
                "mix_r",
                "key_proj.weight",
                "receptance_proj.weight",
                "value_proj.weight",
            ),
        )
        return "rwkv_channel", config, input_ids, next_id

    def _lower_swiglu_mlp(
        self,
        x: torch.Tensor,
        module: torch.nn.Module,
        dim_in: int,
        input_ids: list[int],
        next_id: int,
        payload_specs: list[_PayloadSpec],
        nodes: list[dict],
        edges: list[dict],
    ) -> tuple[str, dict, list[int], int]:
        hidden_dim = int(module.gate_proj.weight.shape[0])
        config = {"batch": self._rows(x), "dim": dim_in, "hidden_dim": hidden_dim}
        swiglu_inputs = list(input_ids)
        next_id = self._emit_param_inputs(
            next_id=next_id,
            payload_specs=payload_specs,
            nodes=nodes,
            edges=edges,
            target_inputs=swiglu_inputs,
            module=module,
            attr_paths=("gate_proj.weight", "up_proj.weight", "down_proj.weight"),
        )
        for linear_name in ("gate_proj", "up_proj", "down_proj"):
            linear = getattr(module, linear_name)
            bias = getattr(linear, "bias", None)
            if bias is None:
                bias = linear.weight.new_zeros(linear.weight.shape[0])
            next_id = self._append_tensor_input(
                next_id=next_id,
                payload_specs=payload_specs,
                nodes=nodes,
                edges=edges,
                target_inputs=swiglu_inputs,
                tensor=bias,
            )
        return "swiglu", config, swiglu_inputs, next_id

    def _lower_conv_only(
        self,
        x: torch.Tensor,
        module: torch.nn.Module,
        node: Any,
        node_id: int,
        dim_in: int,
        dim_out: int,
        input_ids: list[int],
        next_id: int,
        payload_specs: list[_PayloadSpec],
        nodes: list[dict],
        edges: list[dict],
    ) -> int:
        """Multi-node: conv → linear → add. Returns next_id (early return)."""
        config = self._shape_config(x, dim_in)
        conv_output_id = next_id
        next_id += 1
        conv_inputs = list(input_ids)
        next_id = self._emit_param_inputs(
            next_id=next_id,
            payload_specs=payload_specs,
            nodes=nodes,
            edges=edges,
            target_inputs=conv_inputs,
            module=module,
            attr_paths=("conv_dw.weight",),
        )
        conv_bias = module.conv_dw.bias
        if conv_bias is None:
            conv_bias = module.conv_dw.weight.new_zeros(module.conv_dw.weight.shape[0])
        next_id = self._append_tensor_input(
            next_id=next_id,
            payload_specs=payload_specs,
            nodes=nodes,
            edges=edges,
            target_inputs=conv_inputs,
            tensor=conv_bias,
        )
        nodes.append(
            {
                "id": conv_output_id,
                "op_name": "conv1d_seq",
                "input_ids": conv_inputs,
                "config": config,
                "is_input": False,
                "is_output": False,
            }
        )
        for iid in conv_inputs:
            edges.append({"source": iid, "target": conv_output_id})

        linear_output_id = next_id
        next_id += 1
        linear_inputs = [conv_output_id]
        next_id = self._emit_param_inputs(
            next_id=next_id,
            payload_specs=payload_specs,
            nodes=nodes,
            edges=edges,
            target_inputs=linear_inputs,
            module=module,
            attr_paths=("conv_proj.weight",),
        )
        nodes.append(
            {
                "id": linear_output_id,
                "op_name": "linear",
                "input_ids": linear_inputs,
                "config": {
                    "batch": self._rows(x),
                    "dim_in": dim_in,
                    "dim_out": dim_out,
                },
                "is_input": False,
                "is_output": False,
            }
        )
        for iid in linear_inputs:
            edges.append({"source": iid, "target": linear_output_id})

        nodes.append(
            {
                "id": node_id,
                "op_name": "add",
                "input_ids": [input_ids[0], linear_output_id],
                "config": {},
                "is_input": False,
                "is_output": node.is_output,
            }
        )
        edges.append({"source": input_ids[0], "target": node_id})
        edges.append({"source": linear_output_id, "target": node_id})
        return next_id

    def _lower_attention(
        self,
        x: torch.Tensor,
        module: torch.nn.Module,
        op_name: str,
        dim_in: int,
        input_ids: list[int],
        next_id: int,
        payload_specs: list[_PayloadSpec],
        nodes: list[dict],
        edges: list[dict],
    ) -> tuple[str, dict, list[int], int]:
        if op_name == "softmax_attention":
            config = {**self._shape_config(x, dim_in), "n_heads": int(module.n_heads)}
        else:
            config = self._shape_config(x, dim_in)
        next_id = self._emit_param_inputs(
            next_id=next_id,
            payload_specs=payload_specs,
            nodes=nodes,
            edges=edges,
            target_inputs=input_ids,
            module=module,
            attr_paths=(
                "q_proj.weight",
                "k_proj.weight",
                "v_proj.weight",
                "o_proj.weight",
            ),
        )
        return op_name, config, input_ids, next_id

    def _lower_depth_weighted(
        self,
        x: torch.Tensor,
        module: torch.nn.Module,
        op_name: str,
        dim_in: int,
        input_ids: list[int],
        next_id: int,
        payload_specs: list[_PayloadSpec],
        nodes: list[dict],
        edges: list[dict],
    ) -> tuple[str, dict, list[int], int]:
        if op_name in {"gated_lane_blend", "route_lanes"}:
            scorer_attr, stack_name = "lane_scorer", "lane_projs"
        elif op_name in {"depth_gated_transform", "route_recursion"}:
            scorer_attr, stack_name = "depth_scorer", "depth_projs"
        else:
            scorer_attr, stack_name = "depth_scorer", "step_projs"
        scorer = getattr(module, scorer_attr)
        max_depth = int(scorer.shape[0])
        config = {**self._shape_config(x, dim_in), "max_depth": max_depth}
        next_id = self._emit_param_inputs(
            next_id=next_id,
            payload_specs=payload_specs,
            nodes=nodes,
            edges=edges,
            target_inputs=input_ids,
            module=module,
            attr_paths=(scorer_attr,),
        )
        next_id = self._emit_stacked_param_input(
            next_id=next_id,
            payload_specs=payload_specs,
            nodes=nodes,
            edges=edges,
            target_inputs=input_ids,
            module=module,
            attr_name=stack_name,
            count=max_depth,
        )
        return "depth_weighted_proj", config, input_ids, next_id

    def _lower_ssm_family(
        self,
        x: torch.Tensor,
        module: torch.nn.Module,
        op_name: str,
        dim_in: int,
        input_ids: list[int],
        next_id: int,
        payload_specs: list[_PayloadSpec],
        nodes: list[dict],
        edges: list[dict],
    ) -> tuple[str, dict, list[int], int]:
        if op_name == "selective_scan":
            config = self._shape_config(x, dim_in)
            attrs: tuple[str, ...] = (
                "A_log",
                "dt_proj",
                "B_proj.weight",
                "C_proj.weight",
            )
        else:
            config = {
                **self._shape_config(x, dim_in),
                "state_dim": int(module.ssm_A.shape[1]),
            }
            attrs = (
                "ssm_A",
                "ssm_B.weight",
                "ssm_C.weight",
                "ssm_D",
                "ssm_dt.weight",
                "ssm_dt.bias",
            )
        next_id = self._emit_param_inputs(
            next_id=next_id,
            payload_specs=payload_specs,
            nodes=nodes,
            edges=edges,
            target_inputs=input_ids,
            module=module,
            attr_paths=attrs,
        )
        return op_name, config, input_ids, next_id

    def _lower_gated_delta(
        self,
        x: torch.Tensor,
        module: torch.nn.Module,
        dim_in: int,
        dim_out: int,
        input_ids: list[int],
        next_id: int,
        payload_specs: list[_PayloadSpec],
        nodes: list[dict],
        edges: list[dict],
    ) -> tuple[str, dict, list[int], int]:
        del dim_out
        config = {
            **self._shape_config(x, dim_in),
            "n_heads": int(getattr(module, "_gated_delta_heads", min(8, dim_in))),
        }
        next_id = self._emit_param_inputs(
            next_id=next_id,
            payload_specs=payload_specs,
            nodes=nodes,
            edges=edges,
            target_inputs=input_ids,
            module=module,
            attr_paths=(
                "q_proj.weight",
                "k_proj.weight",
                "v_proj.weight",
                "alpha_proj.weight",
                "beta_proj.weight",
                "o_proj.weight",
            ),
        )
        return "gated_delta", config, input_ids, next_id

    # -- Op dispatch table --

    _LOWER_DISPATCH_SIMPLE: dict[str, str] = {
        "linear_proj": "_lower_linear_proj",
        "linear_proj_down": "_lower_linear_proj",
        "linear_proj_up": "_lower_linear_proj",
        "gated_linear": "_lower_gated_linear",
        "gated_delta": "_lower_gated_delta",
    }
    _LOWER_DISPATCH_CONV: set[str] = {"conv1d_seq"}
    _LOWER_DISPATCH_NORM: set[str] = {"rmsnorm", "layernorm"}
    _LOWER_DISPATCH_ATTN: set[str] = {"softmax_attention", "linear_attention"}
    _LOWER_DISPATCH_DEPTH: set[str] = {
        "depth_weighted_proj",
        "adaptive_recursion",
        "gated_lane_blend",
        "route_lanes",
        "depth_gated_transform",
        "route_recursion",
    }
    _LOWER_DISPATCH_SSM: set[str] = {"selective_scan", "state_space"}
    _LOWER_DISPATCH_EARLY_RETURN: set[str] = {"rwkv_time_mixing", "conv_only"}
    _LOWER_DISPATCH_CHANNEL: set[str] = {"rwkv_channel"}
    _LOWER_DISPATCH_SWIGLU: set[str] = {"swiglu_mlp"}

    def _lower_node(
        self,
        *,
        node_id: int,
        x: torch.Tensor,
        next_id: int,
        payload_specs: list[_PayloadSpec],
        nodes: list[dict],
        edges: list[dict],
    ) -> int:
        node = self._graph.nodes[node_id]
        op_name = node.op_name
        input_ids = list(node.input_ids)

        # Fast path: pointwise, binary, input, output
        if node.is_input or op_name in BOUND_POINTWISE_OPS | BOUND_BINARY_OPS | {
            "output"
        }:
            config = dict(node.config)
            if op_name == "softmax":
                config = {"batch": self._rows(x), "dim": int(node.output_shape.dim)}
            nodes.append(
                {
                    "id": node_id,
                    "op_name": op_name,
                    "input_ids": input_ids,
                    "config": config,
                    "is_input": node.is_input,
                    "is_output": node.is_output,
                }
            )
            for iid in input_ids:
                edges.append({"source": iid, "target": node_id})
            return next_id

        # Parametric path setup
        ir_idx = self._node_id_to_ir_idx[node_id]
        module = self._flat_ops[ir_idx]
        dim_in = int(self._graph.nodes[input_ids[0]].output_shape.dim)
        dim_out = int(node.output_shape.dim)
        args = (x, module, dim_in, input_ids, next_id, payload_specs, nodes, edges)

        # Multi-node early-return ops
        if op_name == "rwkv_time_mixing":
            return self._lower_rwkv_time_mixing(
                x,
                module,
                node,
                node_id,
                dim_in,
                dim_out,
                input_ids,
                next_id,
                payload_specs,
                nodes,
                edges,
            )
        if op_name == "conv_only":
            return self._lower_conv_only(
                x,
                module,
                node,
                node_id,
                dim_in,
                dim_out,
                input_ids,
                next_id,
                payload_specs,
                nodes,
                edges,
            )

        # Standard single-node ops
        if op_name in self._LOWER_DISPATCH_SIMPLE:
            method_name = self._LOWER_DISPATCH_SIMPLE[op_name]
            method = getattr(self, method_name)
            if method_name in (
                "_lower_linear_proj",
                "_lower_gated_linear",
                "_lower_gated_delta",
            ):
                native_name, config, input_ids, next_id = method(
                    x,
                    module,
                    dim_in,
                    dim_out,
                    input_ids,
                    next_id,
                    payload_specs,
                    nodes,
                    edges,
                )
            else:
                native_name, config, input_ids, next_id = method(*args)
        elif op_name in self._LOWER_DISPATCH_CONV:
            native_name, config, input_ids, next_id = self._lower_conv1d_seq(*args)
        elif op_name in self._LOWER_DISPATCH_NORM:
            native_name, config, input_ids, next_id = self._lower_normalization(
                x,
                module,
                op_name,
                dim_in,
                input_ids,
                next_id,
                payload_specs,
                nodes,
                edges,
            )
        elif op_name in self._LOWER_DISPATCH_ATTN:
            native_name, config, input_ids, next_id = self._lower_attention(
                x,
                module,
                op_name,
                dim_in,
                input_ids,
                next_id,
                payload_specs,
                nodes,
                edges,
            )
        elif op_name in self._LOWER_DISPATCH_DEPTH:
            native_name, config, input_ids, next_id = self._lower_depth_weighted(
                x,
                module,
                op_name,
                dim_in,
                input_ids,
                next_id,
                payload_specs,
                nodes,
                edges,
            )
        elif op_name in self._LOWER_DISPATCH_SSM:
            native_name, config, input_ids, next_id = self._lower_ssm_family(
                x,
                module,
                op_name,
                dim_in,
                input_ids,
                next_id,
                payload_specs,
                nodes,
                edges,
            )
        elif op_name in self._LOWER_DISPATCH_CHANNEL:
            native_name, config, input_ids, next_id = self._lower_rwkv_channel(*args)
        elif op_name in self._LOWER_DISPATCH_SWIGLU:
            native_name, config, input_ids, next_id = self._lower_swiglu_mlp(*args)
        else:
            raise ValueError(f"Unsupported bound native op: {op_name}")

        nodes.append(
            {
                "id": node_id,
                "op_name": native_name,
                "input_ids": input_ids,
                "config": config,
                "is_input": False,
                "is_output": node.is_output,
            }
        )
        for iid in input_ids:
            edges.append({"source": iid, "target": node_id})
        return next_id

    def _plan_for_input(self, x: torch.Tensor) -> _BoundGraphPlan:
        cache_key = self._runtime_shape_key(x)
        cached = self._plan_cache.get(cache_key)
        if cached is not None:
            return cached

        nodes: list[dict] = []
        edges: list[dict] = []
        payload_specs = [_PayloadSpec(self._graph.input_node.id, None, None)]
        next_id = max(self._graph.nodes) + 1

        for node_id in self._graph.topological_order():
            next_id = self._lower_node(
                node_id=node_id,
                x=x,
                next_id=next_id,
                payload_specs=payload_specs,
                nodes=nodes,
                edges=edges,
            )

        plan = _BoundGraphPlan(
            ir_json=dumps_json(
                {
                    "schema_version": "native_ir.v1",
                    "model_dim": int(self._graph.model_dim),
                    "nodes": nodes,
                    "edges": edges,
                    "output_node_id": int(self._graph._output_node_id),
                }
            ),
            output_shape=tuple(int(v) for v in x.shape[:-1])
            + (int(self._graph.output_node.output_shape.dim),),
            payload_specs=tuple(sorted(payload_specs, key=lambda spec: spec.node_id)),
        )
        self._plan_cache[cache_key] = plan
        return plan

    def try_dispatch(self, x: torch.Tensor):
        if not self._all_native:
            self._last_refusal_reason = "graph_not_fully_native"
            return None
        if not self._runtime_enabled:
            self._last_refusal_reason = "bound_runtime_disabled"
            return None
        if not self._supports_input(x):
            self._last_refusal_reason = "unsupported_input_rank"
            return None
        if not supports_host_array_bridge(x):
            self._last_refusal_reason = "host_array_bridge_unsupported_device"
            return None
        try:
            plan = self._plan_for_input(x)
            payloads = plan.payloads(x)
            if not supports_host_array_bridge(*payloads):
                self._last_refusal_reason = "host_array_bridge_unsupported_device"
                return None
            use_native_autograd = torch.is_grad_enabled() and any(
                getattr(tensor, "requires_grad", False) for tensor in payloads
            )
            if use_native_autograd:
                if not self._backward_native:
                    self._fallback_count += 1
                    self._last_refusal_reason = "native_backward_unavailable"
                    return None
                if self._autograd_fn is None:
                    self._autograd_fn = _BoundNativeGraphFunction.make(self)
                result = self._autograd_fn.apply(*payloads)
            else:
                result = dispatch_graph_native_multi_input_cached(
                    plan.ir_json,
                    payloads,
                    output_shape=plan.output_shape,
                )
            self._dispatch_count += 1
            self._last_refusal_reason = None
            return result
        except Exception:
            self._runtime_enabled = False
            self._fallback_count += 1
            self._last_refusal_reason = "bound_dispatch_error"
            return None

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "all_native": self._all_native,
            "runtime_enabled": self._runtime_enabled,
            "subgraph_dispatches": self._dispatch_count,
            "subgraph_fallbacks": self._fallback_count,
            "native_backward_supported": self._backward_native,
            "last_refusal_reason": self._last_refusal_reason,
        }
