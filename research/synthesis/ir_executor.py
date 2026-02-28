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
from typing import List, Dict, Any, Tuple, Optional

from .graph import ComputationGraphIR
from .primitives import PrimitiveOp, get_primitive, REVERSE_OPCODE_MAP

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
            if self.op_codes[i] == 0: continue
            in1 = self.input_indices[i, 0]
            in2 = self.input_indices[i, 1]
            if in1 != -1: self.consumer_counts[in1] += 1
            if in2 != -1: self.consumer_counts[in2] += 1
        if self.output_node_idx is not None:
            for i in range(len(self.op_codes)):
                if self.op_codes[i] == 0:
                    continue
                if i != int(self.output_node_idx) and self.consumer_counts[i] == 0:
                    op_name = REVERSE_OPCODE_MAP.get(self.op_codes[i], "unknown")
                    logger.warning(
                        "IRExecutor: node %d (%s) has zero consumers (possible dead branch)",
                        i, op_name
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
            if op_name in ("linear_proj", "linear_proj_down", "linear_proj_up",
                           "fused_linear_gelu", "gated_linear", "nm_sparse_linear",
                           "block_sparse_linear", "semi_structured_2_4_linear"):
                node_dims[i] = config.get("out_dim", in_dim)
            elif op_name == "split2":
                node_dims[i] = in_dim // 2
            elif op_name == "split3":
                node_dims[i] = in_dim // 3
            elif op_name == "concat":
                in2 = self.input_indices[i, 1]
                d2 = node_dims.get(in2, self.model_dim) if in2 != -1 else self.model_dim
                node_dims[i] = in_dim + d2
            elif op_name in ("cosine_similarity", "sum_last", "mean_last",
                             "max_last", "norm_last"):
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
                model_dim=self.model_dim
            )
            
            self.idx_to_op_idx[i] = len(self.ops)
            self.ops.append(op_mod)
            
        # torch.compile can dominate runtime for short-lived candidate models.
        # Keep it opt-in so screening throughput does not get bottlenecked by
        # per-architecture compile/recompile overhead.
        enable_compile = os.getenv("RESEARCH_ENABLE_TORCH_COMPILE", "").strip().lower() in {
            "1", "true", "yes", "on"
        }
        if enable_compile:
            try:
                self.forward = torch.compile(self.forward)
            except Exception:
                pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Lowered execution loop. torch.compile fuses this into a single kernel."""
        # node_outputs[i] stores the output of the i-th IR node
        node_outputs: List[Optional[torch.Tensor]] = [None] * len(self.op_codes)
        
        # Z8: Copy consumer counts to track liveness per-forward
        # We use a list instead of numpy for better torch.compile compatibility
        counts = list(self.consumer_counts)
        output_idx = int(self.output_node_idx)
        is_cuda = x.is_cuda
        
        for i in range(len(self.op_codes)):
            opcode = self.op_codes[i]
            
            if opcode == 0: # input
                node_outputs[i] = x
                continue
            
            # Get inputs
            in1_idx = int(self.input_indices[i, 0])
            in2_idx = int(self.input_indices[i, 1])
            
            op_idx = self.idx_to_op_idx.get(i)
            if op_idx is not None:
                op = self.ops[op_idx]
                
                t1 = node_outputs[in1_idx]
                if in2_idx != -1:
                    t2 = node_outputs[in2_idx]
                    node_outputs[i] = op(t1, t2)
                    
                    # Decrement and reclaim in2 if done
                    counts[in2_idx] -= 1
                    if counts[in2_idx] <= 0 and in2_idx != output_idx:
                        node_outputs[in2_idx] = None
                else:
                    node_outputs[i] = op(t1)
                
                # Decrement and reclaim in1 if done
                counts[in1_idx] -= 1
                if counts[in1_idx] <= 0 and in1_idx != output_idx:
                    node_outputs[in1_idx] = None
        
        res = node_outputs[output_idx]
        if res is None:
            return x # Fallback
            
        return res
