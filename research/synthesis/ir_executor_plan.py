from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch.nn as nn

from .compiled_op import CompiledOp
from .graph import ComputationGraphIR, ShapeInfo
from .primitives import REVERSE_OPCODE_MAP


@dataclass(slots=True)
class ExecutorPlan:
    consumer_counts: np.ndarray
    ops: nn.ModuleList
    idx_to_op_idx: dict[int, int]
    flat_ops: list[Optional[nn.Module]]
    n_nodes: int
    counts_original: list[int]
    counts_buf: list[int]
    input_node_indices: tuple[int, ...]
    exec_node_indices: tuple[int, ...]
    exec_in1_indices: tuple[int, ...]
    exec_in2_indices: tuple[int, ...]
    exec_ops: tuple[nn.Module, ...]


_LINEAR_OPS = frozenset(
    {
        "linear_proj",
        "linear_proj_down",
        "linear_proj_up",
        "fused_linear_gelu",
        "gated_linear",
        "nm_sparse_linear",
        "block_sparse_linear",
        "semi_structured_2_4_linear",
    }
)
_SCALAR_REDUCTION_OPS = frozenset(
    {
        "cosine_similarity",
        "sum_last",
        "mean_last",
        "max_last",
        "norm_last",
    }
)


def _infer_node_dims(ir: ComputationGraphIR) -> dict[int, int]:
    node_dims: dict[int, int] = {}
    for idx, opcode in enumerate(ir.op_codes):
        if opcode == 0:
            node_dims[idx] = ir.model_dim
            continue

        op_name = REVERSE_OPCODE_MAP.get(opcode)
        config = ir.configs[idx]
        in1 = ir.input_indices[idx, 0]
        in_dim = node_dims.get(in1, ir.model_dim) if in1 != -1 else ir.model_dim

        if op_name in _LINEAR_OPS:
            node_dims[idx] = config.get("out_dim", in_dim)
        elif op_name == "split2":
            node_dims[idx] = in_dim // 2
        elif op_name == "split3":
            node_dims[idx] = in_dim // 3
        elif op_name == "concat":
            in2 = ir.input_indices[idx, 1]
            d2 = node_dims.get(in2, ir.model_dim) if in2 != -1 else ir.model_dim
            node_dims[idx] = in_dim + d2
        elif op_name in _SCALAR_REDUCTION_OPS:
            node_dims[idx] = 1
        else:
            node_dims[idx] = in_dim
    return node_dims


def _build_consumer_counts(ir: ComputationGraphIR) -> np.ndarray:
    counts = np.zeros(len(ir.op_codes), dtype=np.int32)
    for idx, opcode in enumerate(ir.op_codes):
        if opcode == 0:
            continue
        in1 = ir.input_indices[idx, 0]
        in2 = ir.input_indices[idx, 1]
        if in1 != -1:
            counts[in1] += 1
        if in2 != -1:
            counts[in2] += 1
    return counts


def build_executor_plan(ir: ComputationGraphIR) -> ExecutorPlan:
    node_dims = _infer_node_dims(ir)
    n_nodes = len(ir.op_codes)
    ops = nn.ModuleList()
    idx_to_op_idx: dict[int, int] = {}
    flat_ops: list[Optional[nn.Module]] = [None] * n_nodes

    for idx, opcode in enumerate(ir.op_codes):
        if opcode == 0:
            continue
        op_name = REVERSE_OPCODE_MAP.get(opcode)
        if not op_name:
            continue

        in1 = ir.input_indices[idx, 0]
        in_dim = node_dims.get(in1, ir.model_dim) if in1 != -1 else ir.model_dim
        out_dim = node_dims.get(idx, ir.model_dim)
        op_mod = CompiledOp(
            op_name=op_name,
            config=ir.configs[idx],
            input_shape=ShapeInfo(batch="B", seq="S", dim=in_dim),
            output_shape=ShapeInfo(batch="B", seq="S", dim=out_dim),
            model_dim=ir.model_dim,
        )
        idx_to_op_idx[idx] = len(ops)
        ops.append(op_mod)
        flat_ops[idx] = op_mod

    if hasattr(ir.input_indices, "tolist"):
        input_indices_list = ir.input_indices.tolist()
    else:
        input_indices_list = [
            [int(ir.input_indices[idx, 0]), int(ir.input_indices[idx, 1])]
            for idx in range(n_nodes)
        ]

    consumer_counts = _build_consumer_counts(ir)
    counts_original = (
        consumer_counts.tolist()
        if hasattr(consumer_counts, "tolist")
        else list(consumer_counts)
    )
    exec_entries = tuple(
        (
            idx,
            input_indices_list[idx][0],
            input_indices_list[idx][1],
            flat_ops[idx],
        )
        for idx, opcode in enumerate(ir.op_codes)
        if opcode != 0 and flat_ops[idx] is not None
    )
    return ExecutorPlan(
        consumer_counts=consumer_counts,
        ops=ops,
        idx_to_op_idx=idx_to_op_idx,
        flat_ops=flat_ops,
        n_nodes=n_nodes,
        counts_original=counts_original,
        counts_buf=list(counts_original),
        input_node_indices=tuple(
            idx for idx, opcode in enumerate(ir.op_codes) if opcode == 0
        ),
        exec_node_indices=tuple(entry[0] for entry in exec_entries),
        exec_in1_indices=tuple(entry[1] for entry in exec_entries),
        exec_in2_indices=tuple(entry[2] for entry in exec_entries),
        exec_ops=tuple(entry[3] for entry in exec_entries),
    )
