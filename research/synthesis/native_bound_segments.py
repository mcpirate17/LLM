from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch

from ._json_compat import dumps_json
from .compiler_op_utils import _get_stacked_params
from .graph import ComputationGraph
from .native_segments import NativeChainSegment
from .native_support import (
    BOUND_PARAM_OPS,
    BOUND_POINTWISE_OPS,
    BOUND_SUPPORTED_INPUT_RANKS,
)
from .primitives import PRIMITIVE_REGISTRY
from ..scientist.native.dispatch import dispatch_graph_native_multi_input_cached


@dataclass(slots=True)
class _BoundChainNode:
    op_name: str
    module: torch.nn.Module


class BoundNativeChainDispatcher:
    def __init__(self, chain_nodes: Iterable[_BoundChainNode]):
        self._chain_nodes = tuple(chain_nodes)
        self._ir_cache: Dict[tuple[int, ...], tuple[str, tuple[int, ...]]] = {}

    def _rows(self, x: torch.Tensor) -> int:
        if x.ndim == 3:
            return int(x.shape[0] * x.shape[1])
        if x.ndim == 2:
            return int(x.shape[0])
        raise ValueError(f"Unsupported tensor rank for bound native chain: {x.ndim}")

    def _supports_input(self, x: torch.Tensor) -> bool:
        return x.ndim in BOUND_SUPPORTED_INPUT_RANKS

    def _runtime_shape_key(self, x: torch.Tensor) -> tuple[int, ...]:
        return tuple(int(v) for v in x.shape)

    def _linear_bias(self, module: torch.nn.Module, out_dim: int) -> torch.Tensor:
        bias = getattr(module, "bias", None)
        if bias is not None:
            return bias
        return module.weight.new_zeros(out_dim)

    def _op_inputs_and_config(
        self,
        *,
        op_name: str,
        module: torch.nn.Module,
        current_dim: int,
        x: torch.Tensor,
    ) -> tuple[str, list[torch.Tensor], dict, int]:
        rows = self._rows(x)
        if op_name in BOUND_POINTWISE_OPS:
            return op_name, [], {}, current_dim
        if op_name in {"linear_proj", "linear_proj_down", "linear_proj_up"}:
            out_dim = int(module.weight.shape[0])
            return (
                "linear",
                [module.weight, self._linear_bias(module, out_dim)],
                {"batch": rows, "dim_in": current_dim, "dim_out": out_dim},
                out_dim,
            )
        if op_name == "conv1d_seq":
            conv_bias = getattr(module, "conv_bias", None)
            if conv_bias is None:
                conv_bias = module.conv_weight.new_zeros(module.conv_weight.shape[0])
            return (
                "conv1d_seq",
                [module.conv_weight, conv_bias],
                {
                    "batch": int(x.shape[0]),
                    "seq": int(x.shape[1]) if x.ndim == 3 else 1,
                    "dim": current_dim,
                },
                current_dim,
            )
        if op_name == "rmsnorm":
            return (
                "rmsnorm",
                [module.weight],
                {"batch": rows, "dim": current_dim, "eps": 1e-6},
                current_dim,
            )
        if op_name == "layernorm":
            return (
                "layernorm",
                [module.weight, module.bias],
                {"batch": rows, "dim": current_dim, "eps": 1e-5},
                current_dim,
            )
        if op_name == "gated_linear":
            out_dim = int(module.linear_weight.shape[0])
            return (
                "gated_linear",
                [
                    module.linear_weight,
                    module.linear_bias,
                    module.gate_weight,
                    module.gate_bias,
                ],
                {"batch": rows, "dim_in": current_dim, "dim_out": out_dim},
                out_dim,
            )
        if op_name == "rwkv_time_mixing":
            if x.ndim != 3:
                raise ValueError("rwkv_time_mixing native chain requires rank-3 input")
            return (
                "rwkv_time_mixing",
                [
                    module.w_decay,
                    module.u_bonus,
                    module.W_k,
                    module.W_v,
                    module.W_r,
                ],
                {
                    "batch": int(x.shape[0]),
                    "seq": int(x.shape[1]),
                    "dim": current_dim,
                },
                current_dim,
            )
        if op_name == "rwkv_channel":
            if x.ndim != 3:
                raise ValueError("rwkv_channel native chain requires rank-3 input")
            return (
                "rwkv_channel",
                [
                    module.mix_k,
                    module.mix_r,
                    module.key_proj.weight,
                    module.receptance_proj.weight,
                    module.value_proj.weight,
                ],
                {
                    "batch": int(x.shape[0]),
                    "seq": int(x.shape[1]),
                    "dim": current_dim,
                    "hidden_dim": int(module.key_proj.weight.shape[0]),
                },
                current_dim,
            )
        if op_name in {
            "depth_weighted_proj",
            "adaptive_recursion",
            "gated_lane_blend",
            "route_lanes",
            "depth_gated_transform",
            "route_recursion",
        }:
            if x.ndim != 3:
                raise ValueError(
                    "depth_weighted_proj native chain requires rank-3 input"
                )
            if op_name in {"gated_lane_blend", "route_lanes"}:
                scorer = module.lane_scorer
                stack_name = "lane_projs"
            elif op_name in {"depth_gated_transform", "route_recursion"}:
                scorer = module.depth_scorer
                stack_name = "depth_projs"
            else:
                scorer = module.depth_scorer
                stack_name = "step_projs"
            max_depth = int(scorer.shape[0])
            return (
                "depth_weighted_proj",
                [
                    scorer,
                    _get_stacked_params(module, stack_name, max_depth, x.dtype),
                ],
                {
                    "batch": int(x.shape[0]),
                    "seq": int(x.shape[1]),
                    "dim": current_dim,
                    "max_depth": max_depth,
                },
                current_dim,
            )
        if op_name == "swiglu_mlp":
            hidden_dim = int(module.gate_proj.weight.shape[0])
            return (
                "swiglu",
                [
                    module.gate_proj.weight,
                    module.up_proj.weight,
                    module.down_proj.weight,
                    self._linear_bias(module.gate_proj, hidden_dim),
                    self._linear_bias(module.up_proj, hidden_dim),
                    self._linear_bias(module.down_proj, current_dim),
                ],
                {
                    "batch": rows,
                    "dim": current_dim,
                    "hidden_dim": hidden_dim,
                },
                current_dim,
            )
        if op_name == "softmax_attention":
            return (
                "softmax_attention",
                [
                    module.q_proj.weight,
                    module.k_proj.weight,
                    module.v_proj.weight,
                    module.o_proj.weight,
                ],
                {
                    "batch": int(x.shape[0]),
                    "seq": int(x.shape[1]) if x.ndim == 3 else 1,
                    "dim": current_dim,
                    "n_heads": int(module.n_heads),
                },
                current_dim,
            )
        if op_name == "linear_attention":
            return (
                "linear_attention",
                [
                    module.q_proj.weight,
                    module.k_proj.weight,
                    module.v_proj.weight,
                    module.o_proj.weight,
                ],
                {
                    "batch": int(x.shape[0]),
                    "seq": int(x.shape[1]) if x.ndim == 3 else 1,
                    "dim": current_dim,
                },
                current_dim,
            )
        if op_name == "selective_scan":
            return (
                "selective_scan",
                [
                    module.A_log,
                    module.dt_proj,
                    module.B_proj.weight,
                    module.C_proj.weight,
                ],
                {
                    "batch": int(x.shape[0]),
                    "seq": int(x.shape[1]) if x.ndim == 3 else 1,
                    "dim": current_dim,
                },
                current_dim,
            )
        if op_name == "state_space":
            return (
                "state_space",
                [
                    module.ssm_A,
                    module.ssm_B.weight,
                    module.ssm_C.weight,
                    module.ssm_D,
                    module.ssm_dt.weight,
                    module.ssm_dt.bias,
                ],
                {
                    "batch": int(x.shape[0]),
                    "seq": int(x.shape[1]) if x.ndim == 3 else 1,
                    "dim": current_dim,
                    "state_dim": int(module.ssm_A.shape[1]),
                },
                current_dim,
            )
        if op_name == "gated_delta":
            return (
                "gated_delta",
                [
                    module.q_proj.weight,
                    module.k_proj.weight,
                    module.v_proj.weight,
                    module.alpha_proj.weight,
                    module.beta_proj.weight,
                    module.o_proj.weight,
                ],
                {
                    "batch": int(x.shape[0]),
                    "seq": int(x.shape[1]) if x.ndim == 3 else 1,
                    "dim": current_dim,
                    "n_heads": int(
                        getattr(module, "_gated_delta_heads", min(8, current_dim))
                    ),
                },
                current_dim,
            )
        raise ValueError(f"Unsupported bound native op: {op_name}")

    def _build_ir(self, x: torch.Tensor) -> tuple[str, tuple[int, ...]]:
        cache_key = self._runtime_shape_key(x)
        cached = self._ir_cache.get(cache_key)
        if cached is not None:
            return cached

        input_payloads: list[torch.Tensor] = [x]
        nodes: list[dict] = [
            {
                "id": 0,
                "op_name": "input",
                "input_ids": [],
                "config": {},
                "is_input": True,
                "is_output": False,
            }
        ]
        edges: list[dict] = []
        current_id = 0
        next_id = 1
        current_dim = int(x.shape[-1])

        for chain_node in self._chain_nodes:
            native_name, params, config, next_dim = self._op_inputs_and_config(
                op_name=chain_node.op_name,
                module=chain_node.module,
                current_dim=current_dim,
                x=x,
            )
            input_ids = [current_id]
            for param in params:
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
                input_payloads.append(param)
                input_ids.append(param_id)
            node_id = next_id
            next_id += 1
            nodes.append(
                {
                    "id": node_id,
                    "op_name": native_name,
                    "input_ids": input_ids,
                    "config": config,
                    "is_input": False,
                    "is_output": False,
                }
            )
            for input_id in input_ids:
                edges.append({"source": input_id, "target": node_id})
            current_id = node_id
            current_dim = next_dim

        output_id = next_id
        nodes.append(
            {
                "id": output_id,
                "op_name": "output",
                "input_ids": [current_id],
                "config": {},
                "is_input": False,
                "is_output": True,
            }
        )
        edges.append({"source": current_id, "target": output_id})
        ir_json = dumps_json(
            {
                "schema_version": "native_ir.v1",
                "model_dim": int(x.shape[-1]),
                "nodes": nodes,
                "edges": edges,
                "output_node_id": output_id,
            }
        )
        if x.ndim == 3:
            output_shape = (int(x.shape[0]), int(x.shape[1]), current_dim)
        else:
            output_shape = (self._rows(x), current_dim)
        self._ir_cache[cache_key] = (ir_json, output_shape)
        self._last_inputs = tuple(input_payloads)
        return ir_json, output_shape

    def _input_payloads(self, x: torch.Tensor) -> list[torch.Tensor]:
        payloads: list[torch.Tensor] = [x]
        current_dim = int(x.shape[-1])
        for chain_node in self._chain_nodes:
            _, params, _, next_dim = self._op_inputs_and_config(
                op_name=chain_node.op_name,
                module=chain_node.module,
                current_dim=current_dim,
                x=x,
            )
            payloads.extend(params)
            current_dim = next_dim
        return payloads

    def try_dispatch(self, x: torch.Tensor):
        if getattr(x, "requires_grad", False) or not self._supports_input(x):
            return None
        ir_json, output_shape = self._build_ir(x)
        return dispatch_graph_native_multi_input_cached(
            ir_json,
            self._input_payloads(x),
            output_shape=output_shape,
        )


def _unique_consumers(graph: ComputationGraph) -> Dict[int, list[int]]:
    consumers: Dict[int, list[int]] = {node_id: [] for node_id in graph.nodes}
    for node_id, node in graph.nodes.items():
        for parent_id in node.input_ids:
            if parent_id in consumers:
                consumers[parent_id].append(node_id)
    return consumers


def _bound_eligible_node(
    graph: ComputationGraph,
    flat_ops: list[torch.nn.Module | None],
    node_id_to_ir_idx: Dict[int, int],
    node_id: int,
    *,
    supported_ops: set[str],
) -> bool:
    node = graph.nodes[node_id]
    if node.is_input or len(node.input_ids) != 1 or not node.output_shape.is_standard:
        return False
    op_name = node.op_name
    if (
        op_name not in BOUND_POINTWISE_OPS | BOUND_PARAM_OPS
        or op_name not in supported_ops
    ):
        return False
    primitive = PRIMITIVE_REGISTRY.get(op_name)
    if primitive is None:
        return False
    ir_idx = node_id_to_ir_idx.get(node_id)
    module = None if ir_idx is None else flat_ops[ir_idx]
    if module is None:
        return False
    parent = graph.nodes.get(node.input_ids[0])
    return parent is not None and parent.output_shape.is_standard


def build_bound_native_chain_segments(
    graph: ComputationGraph,
    *,
    flat_ops: list[torch.nn.Module | None],
    ir_node_ids: List[int],
    exec_plan_node_indices: List[int],
    supported_ops: set[str],
) -> list[NativeChainSegment]:
    if not supported_ops:
        return []

    consumers = _unique_consumers(graph)
    topo = graph.topological_order()
    reachable_node_ids = set(ir_node_ids)
    node_id_to_ir_idx = {int(node_id): idx for idx, node_id in enumerate(ir_node_ids)}
    node_id_to_plan_idx = {
        int(ir_node_ids[node_idx]): plan_idx
        for plan_idx, node_idx in enumerate(exec_plan_node_indices)
    }
    visited: set[int] = set()
    segments: list[NativeChainSegment] = []

    for node_id in topo:
        if node_id in visited or node_id not in reachable_node_ids:
            continue
        if not _bound_eligible_node(
            graph, flat_ops, node_id_to_ir_idx, node_id, supported_ops=supported_ops
        ):
            continue

        parent_id = graph.nodes[node_id].input_ids[0]
        if parent_id in reachable_node_ids and _bound_eligible_node(
            graph,
            flat_ops,
            node_id_to_ir_idx,
            parent_id,
            supported_ops=supported_ops,
        ):
            continue

        chain_node_ids = [node_id]
        current_id = node_id
        has_param_op = graph.nodes[node_id].op_name in BOUND_PARAM_OPS
        while True:
            next_ids = consumers.get(current_id, [])
            if len(next_ids) != 1:
                break
            next_id = next_ids[0]
            if next_id in visited or next_id not in reachable_node_ids:
                break
            next_node = graph.nodes[next_id]
            if len(next_node.input_ids) != 1 or next_node.input_ids[0] != current_id:
                break
            if not _bound_eligible_node(
                graph,
                flat_ops,
                node_id_to_ir_idx,
                next_id,
                supported_ops=supported_ops,
            ):
                break
            chain_node_ids.append(next_id)
            has_param_op = has_param_op or next_node.op_name in BOUND_PARAM_OPS
            current_id = next_id

        if len(chain_node_ids) < 2 or not has_param_op:
            continue

        plan_indices = [node_id_to_plan_idx[current] for current in chain_node_ids]
        if plan_indices != list(range(plan_indices[0], plan_indices[-1] + 1)):
            continue

        dispatcher = BoundNativeChainDispatcher(
            _BoundChainNode(
                op_name=graph.nodes[current].op_name,
                module=flat_ops[node_id_to_ir_idx[current]],
            )
            for current in chain_node_ids
        )
        for current in chain_node_ids:
            visited.add(current)

        segments.append(
            NativeChainSegment(
                start_plan_index=plan_indices[0],
                end_plan_index=plan_indices[-1],
                input_ir_idx=node_id_to_ir_idx[parent_id],
                input_consume_count=1,
                output_ir_idx=node_id_to_ir_idx[chain_node_ids[-1]],
                release_ir_counts=tuple(
                    (node_id_to_ir_idx[current], 1) for current in chain_node_ids[:-1]
                ),
                dispatcher=dispatcher,
            )
        )

    return segments
