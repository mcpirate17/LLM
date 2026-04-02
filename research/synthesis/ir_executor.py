"""
IR Executor

High-performance execution of ComputationGraphIR using torch.compile
and registry-based dispatch. Minimizes Python overhead by lowering
the entire IR traversal into a single compiled kernel.
"""

from __future__ import annotations

import os
import logging
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional

from .graph import ComputationGraphIR
from .primitives import REVERSE_OPCODE_MAP

logger = logging.getLogger(__name__)


class IRExecutor(nn.Module):
    """Executes ComputationGraphIR with minimal overhead."""

    def __init__(self, ir: ComputationGraphIR):
        super().__init__()
        self.model_dim = ir.model_dim
        self.op_codes = ir.op_codes
        self.input_indices = ir.input_indices
        self.output_node_idx = ir.output_node_idx
        self.configs = ir.configs

        # Z8: Pre-calculate reference counts for memory management
        self.consumer_counts = np.zeros(len(self.op_codes), dtype=np.int32)
        for i in range(len(self.op_codes)):
            if self.op_codes[i] == 0:
                continue
            in1 = self.input_indices[i, 0]
            in2 = self.input_indices[i, 1]
            if in1 != -1:
                self.consumer_counts[in1] += 1
            if in2 != -1:
                self.consumer_counts[in2] += 1
        if self.output_node_idx is not None:
            for i in range(len(self.op_codes)):
                if self.op_codes[i] == 0:
                    continue
                if i != int(self.output_node_idx) and self.consumer_counts[i] == 0:
                    op_name = REVERSE_OPCODE_MAP.get(self.op_codes[i], "unknown")
                    logger.warning(
                        "IRExecutor: node %d (%s) has zero consumers (possible dead branch)",
                        i,
                        op_name,
                    )

        self.ops = nn.ModuleList()
        # Map IR index to Module index in self.ops
        self.idx_to_op_idx = {}

        from .compiler import CompiledOp, ShapeInfo

        # Track output dim per node for shape-aware param init
        node_dims = {}
        for i in range(len(self.op_codes)):
            if self.op_codes[i] == 0:  # input
                node_dims[i] = self.model_dim
                continue
            op_name = REVERSE_OPCODE_MAP.get(self.op_codes[i])
            config = self.configs[i]
            in1 = self.input_indices[i, 0]
            in_dim = node_dims.get(in1, self.model_dim) if in1 != -1 else self.model_dim
            # Determine output dim based on op type and config
            if op_name in (
                "linear_proj",
                "linear_proj_down",
                "linear_proj_up",
                "fused_linear_gelu",
                "gated_linear",
                "nm_sparse_linear",
                "block_sparse_linear",
                "semi_structured_2_4_linear",
            ):
                node_dims[i] = config.get("out_dim", in_dim)
            elif op_name == "split2":
                node_dims[i] = in_dim // 2
            elif op_name == "split3":
                node_dims[i] = in_dim // 3
            elif op_name == "concat":
                in2 = self.input_indices[i, 1]
                d2 = node_dims.get(in2, self.model_dim) if in2 != -1 else self.model_dim
                node_dims[i] = in_dim + d2
            elif op_name in (
                "cosine_similarity",
                "sum_last",
                "mean_last",
                "max_last",
                "norm_last",
            ):
                node_dims[i] = 1
            else:
                node_dims[i] = in_dim

        # Build op modules with correct shapes
        for i in range(len(self.op_codes)):
            opcode = self.op_codes[i]
            if opcode == 0:  # input
                continue

            op_name = REVERSE_OPCODE_MAP.get(opcode)
            if not op_name:
                continue

            in1 = self.input_indices[i, 0]
            in_dim = node_dims.get(in1, self.model_dim) if in1 != -1 else self.model_dim
            out_dim = node_dims.get(i, self.model_dim)

            input_shape = ShapeInfo(batch="B", seq="S", dim=in_dim)
            output_shape = ShapeInfo(batch="B", seq="S", dim=out_dim)

            op_mod = CompiledOp(
                op_name=op_name,
                config=self.configs[i],
                input_shape=input_shape,
                output_shape=output_shape,
                model_dim=self.model_dim,
            )

            self.idx_to_op_idx[i] = len(self.ops)
            self.ops.append(op_mod)

        # Build flat op lookup array: O(1) list index instead of dict hash
        n_nodes = len(self.op_codes)
        self._flat_ops: List[Optional[nn.Module]] = [None] * n_nodes
        for ir_idx, op_idx in self.idx_to_op_idx.items():
            self._flat_ops[ir_idx] = self.ops[op_idx]

        # Pre-convert numpy arrays to Python lists (avoid per-element int() casts)
        self._op_codes_list = (
            self.op_codes.tolist()
            if hasattr(self.op_codes, "tolist")
            else list(self.op_codes)
        )
        self._in1_list = (
            self.input_indices[:, 0].tolist()
            if hasattr(self.input_indices, "tolist")
            else [int(self.input_indices[i, 0]) for i in range(n_nodes)]
        )
        self._in2_list = (
            self.input_indices[:, 1].tolist()
            if hasattr(self.input_indices, "tolist")
            else [int(self.input_indices[i, 1]) for i in range(n_nodes)]
        )
        _ccl = (
            self.consumer_counts.tolist()
            if hasattr(self.consumer_counts, "tolist")
            else list(self.consumer_counts)
        )
        self._counts_original = list(_ccl)
        self._counts_buf = list(_ccl)

        # torch.compile can dominate runtime for short-lived candidate models.
        # Keep it opt-in so screening throughput does not get bottlenecked by
        # per-architecture compile/recompile overhead.
        enable_compile = os.getenv(
            "RESEARCH_ENABLE_TORCH_COMPILE", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        if enable_compile:
            try:
                self.forward = torch.compile(self.forward)
            except Exception:
                pass

    def forward(
        self, x: torch.Tensor, capture_intermediates: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """Lowered execution loop. torch.compile fuses this into a single kernel."""
        n_nodes = len(self._op_codes_list)
        node_outputs: List[Optional[torch.Tensor]] = [None] * n_nodes
        captured = {} if capture_intermediates else None

        # Z8: Reset consumer counts in-place (no allocation)
        counts = self._counts_buf
        counts[:] = self._counts_original
        output_idx = int(self.output_node_idx)

        # Local refs avoid repeated attribute lookups in the loop
        op_codes = self._op_codes_list
        in1_list = self._in1_list
        in2_list = self._in2_list
        flat_ops = self._flat_ops

        for i in range(n_nodes):
            if op_codes[i] == 0:  # input
                node_outputs[i] = x
                continue

            in1_idx = in1_list[i]
            in2_idx = in2_list[i]

            op = flat_ops[i]
            if op is not None:
                t1 = node_outputs[in1_idx]
                if in2_idx != -1:
                    t2 = node_outputs[in2_idx]
                    node_outputs[i] = op(t1, t2)

                    counts[in2_idx] -= 1
                    if (
                        counts[in2_idx] <= 0
                        and in2_idx != output_idx
                        and captured is None
                    ):
                        node_outputs[in2_idx] = None
                else:
                    node_outputs[i] = op(t1)

                counts[in1_idx] -= 1
                if counts[in1_idx] <= 0 and in1_idx != output_idx and captured is None:
                    node_outputs[in1_idx] = None

                if captured is not None:
                    captured[i] = node_outputs[i].detach().clone()

        res = node_outputs[output_idx]
        if res is None:
            logger.warning(
                "IRExecutor: output node %d produced None, returning input", output_idx
            )
            res = x

        if captured is not None:
            return res, captured
        return res
