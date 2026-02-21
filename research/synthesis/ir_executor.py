"""
IR Executor

High-performance execution of ComputationGraphIR using torch.compile
and registry-based dispatch. Minimizes Python overhead by lowering 
the entire IR traversal into a single compiled kernel.
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from .graph import ComputationGraphIR
from .primitives import PrimitiveOp, get_primitive, REVERSE_OPCODE_MAP


class IRExecutor(nn.Module):
    """Executes ComputationGraphIR with minimal overhead."""
    
    def __init__(self, ir: ComputationGraphIR):
        super().__init__()
        self.model_dim = ir.model_dim
        self.op_codes = ir.op_codes
        self.input_indices = ir.input_indices
        self.output_node_idx = ir.output_node_idx
        self.configs = ir.configs
        
        self.ops = nn.ModuleList()
        # Map IR index to Module index in self.ops
        self.idx_to_op_idx = {}
        
        from .compiler import CompiledOp, ShapeInfo
        
        # Build op modules
        for i in range(len(self.op_codes)):
            opcode = self.op_codes[i]
            if opcode == 0: # input
                continue
                
            op_name = REVERSE_OPCODE_MAP.get(opcode)
            if not op_name:
                continue
                
            # Note: IRExecutor currently assumes standard shapes for simplicity
            # but CompiledOp handles internal shape logic.
            # We'll need to improve shape tracking in IRExecutor for multi-scale ops.
            shape = ShapeInfo(batch="B", seq="S", dim=self.model_dim)
            
            op_mod = CompiledOp(
                op_name=op_name,
                config=self.configs[i],
                input_shape=shape,
                output_shape=shape,
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
        # Using a list instead of a dict for better compile behavior
        # Initializing with a dummy tensor to help torch.compile track types
        node_outputs: List[Optional[torch.Tensor]] = [None] * len(self.op_codes)
        
        for i in range(len(self.op_codes)):
            opcode = self.op_codes[i]
            
            if opcode == 0: # input
                node_outputs[i] = x
                continue
            
            # Get inputs
            in1_idx = self.input_indices[i, 0]
            in2_idx = self.input_indices[i, 1]
            
            # This is the 'op switch' - torch.compile will try to optimize this
            # but we use ModuleList indexing which is usually well-supported.
            op_idx = self.idx_to_op_idx.get(i)
            if op_idx is not None:
                op = self.ops[op_idx]
                
                # Fetch actual tensors
                t1 = node_outputs[in1_idx]
                if in2_idx != -1:
                    t2 = node_outputs[in2_idx]
                    node_outputs[i] = op(t1, t2)
                else:
                    node_outputs[i] = op(t1)
        
        res = node_outputs[self.output_node_idx]
        if res is None:
            return x # Fallback
        return res
