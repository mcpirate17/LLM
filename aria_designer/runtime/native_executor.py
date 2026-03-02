"""
Native Graph Executor

High-performance execution of ComputationGraph by lowering it to 
aria_core.GraphExecutor (C++). This eliminates Python overhead in 
the evaluation loop.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple
import aria_core

from research.synthesis.graph import ComputationGraph, OpNode
from research.synthesis.compiler import CompiledLayer

# Map primitive names to aria_core op types (from bindings.cpp)
OP_TYPE_MAP = {
    "relu": 0,
    "gelu": 1,
    "silu": 2,
    "add": 3,
    "mul": 4,
    "sub": 5,
    "rmsnorm": 6,
    "layernorm": 7,
    "matmul": 8,
    "linear": 9,
    "linear_proj": 9,
    "linear_proj_down": 9,
    "linear_proj_up": 9,
    "softmax": 10,
}

class NativeGraphExecutor(nn.Module):
    """Executes a ComputationGraph using the C++ GraphExecutor."""

    def __init__(self, graph: ComputationGraph, compiled_layer: Optional[CompiledLayer] = None):
        super().__init__()
        self.graph = graph
        self.model_dim = graph.model_dim
        
        if compiled_layer is None:
            self.compiled_layer = CompiledLayer(graph)
        else:
            self.compiled_layer = compiled_layer
            
        self.topo = graph.topological_order()
        self.node_to_idx = {}
        self.param_map = {}
        self._tensor_pool = {} 
        
        current_idx = 0
        # Input node ALWAYS mapped to 0
        input_node_id = graph._input_node_id
        self.node_to_idx[input_node_id] = 0
        current_idx += 1
        
        for nid in self.topo:
            node = graph.nodes[nid]
            if node.is_input: continue
            
            # Outputs for this node
            idx = current_idx
            self.node_to_idx[nid] = idx
            
            shape = (node.output_shape.batch, node.output_shape.seq, node.output_shape.dim)
            actual_shape = [1 if d == 'B' else (16 if d == 'S' else int(d)) for d in shape]
            self._tensor_pool[idx] = torch.zeros(actual_shape, dtype=torch.float32, device="cpu").contiguous()
            current_idx += 1
            
            # Parameters for this node
            op_name = node.op_name
            if nid in self.compiled_layer.ops:
                op_mod = self.compiled_layer.ops[nid]
                if op_name == "rmsnorm":
                    self.param_map[(nid, "weight")] = current_idx
                    self._tensor_pool[current_idx] = torch.zeros(node.output_shape.dim, dtype=torch.float32, device="cpu").contiguous()
                    current_idx += 1
                elif op_name in ("linear", "linear_proj", "linear_proj_down", "linear_proj_up"):
                    target = op_mod
                    if not hasattr(target, "weight") and hasattr(target, "linear"):
                        target = target.linear
                    self.param_map[(nid, "weight")] = current_idx
                    self._tensor_pool[current_idx] = torch.zeros(target.weight.shape, dtype=torch.float32, device="cpu").contiguous() 
                    current_idx += 1
                    self.param_map[(nid, "bias")] = current_idx
                    self._tensor_pool[current_idx] = torch.zeros(target.weight.shape[0], dtype=torch.float32, device="cpu").contiguous()
                    current_idx += 1
                elif op_name == "layernorm":
                    self.param_map[(nid, "weight")] = current_idx
                    self._tensor_pool[current_idx] = torch.zeros(node.output_shape.dim, dtype=torch.float32, device="cpu").contiguous()
                    current_idx += 1
                    self.param_map[(nid, "bias")] = current_idx
                    self._tensor_pool[current_idx] = torch.zeros(node.output_shape.dim, dtype=torch.float32, device="cpu").contiguous()
                    current_idx += 1
                    
        self.executor = aria_core.GraphExecutor(current_idx)
        # CRITICAL: We must set EVERY tensor in the pool,
        # including pre-allocated output buffers.
        for idx in range(current_idx):
            if idx in self._tensor_pool:
                self.executor.set_tensor(idx, self._tensor_pool[idx])
            else:
                # Placeholder for unexpected holes
                t = torch.zeros(1, dtype=torch.float32, device="cpu").contiguous()
                self.executor.set_tensor(idx, t)
                self._tensor_pool[idx] = t
            
        self._bake_graph()
        self._sync_parameters()

    def _bake_graph(self):
        nodes_to_bake = []
        for nid in self.topo:
            node = self.graph.nodes[nid]
            if node.is_input: continue
            
            op_name = node.op_name
            if op_name not in OP_TYPE_MAP:
                continue
                
            op_type = OP_TYPE_MAP[op_name]
            inputs = [self.node_to_idx[iid] for iid in node.input_ids]
            
            if nid in self.compiled_layer.ops:
                if op_name == "rmsnorm" and (nid, "weight") in self.param_map:
                    inputs.append(self.param_map[(nid, "weight")])
                elif op_name in ("linear", "linear_proj", "linear_proj_down", "linear_proj_up"):
                    if (nid, "weight") in self.param_map:
                        inputs.append(self.param_map[(nid, "weight")])
                    if (nid, "bias") in self.param_map:
                        inputs.append(self.param_map[(nid, "bias")])
                elif op_name == "layernorm":
                    if (nid, "weight") in self.param_map:
                        inputs.append(self.param_map[(nid, "weight")])
                    if (nid, "bias") in self.param_map:
                        inputs.append(self.param_map[(nid, "bias")])

            nodes_to_bake.append({
                "type": op_type,
                "inputs": inputs,
                "outputs": [self.node_to_idx[nid]],
                "params": [node.config.get("eps", 1e-6)] if "eps" in node.config else []
            })
            
        self.executor.bake(nodes_to_bake)

    def _sync_parameters(self):
        """Upload current torch parameters to the C++ tensor pool."""
        for nid in self.topo:
            node = self.graph.nodes[nid]
            if nid not in self.compiled_layer.ops: continue
            
            op_mod = self.compiled_layer.ops[nid]
            op_name = node.op_name
            
            if op_name == "rmsnorm":
                if hasattr(op_mod, "weight"):
                    t = op_mod.weight.data.detach().cpu().float().contiguous().reshape(-1)
                    idx = self.param_map[(nid, "weight")]
                    self._tensor_pool[idx] = t
                    self.executor.set_tensor(idx, t)
            elif op_name in ("linear", "linear_proj", "linear_proj_down", "linear_proj_up"):
                target = op_mod
                if not hasattr(target, "weight") and hasattr(target, "linear"):
                    target = target.linear
                
                if hasattr(target, "weight"):
                    tw = target.weight.data.detach().cpu().float().contiguous()
                    idx_w = self.param_map[(nid, "weight")]
                    self._tensor_pool[idx_w] = tw
                    self.executor.set_tensor(idx_w, tw)
                    
                    idx_b = self.param_map[(nid, "bias")]
                    if hasattr(target, "bias") and target.bias is not None:
                        tb = target.bias.data.detach().cpu().float().contiguous().reshape(-1)
                    else:
                        out_dim = target.weight.shape[0]
                        tb = torch.zeros(out_dim, device="cpu", dtype=torch.float32).contiguous()
                    self._tensor_pool[idx_b] = tb
                    self.executor.set_tensor(idx_b, tb)
            elif op_name == "layernorm":
                if hasattr(op_mod, "weight"):
                    tw = op_mod.weight.data.detach().cpu().float().contiguous().reshape(-1)
                    idx_w = self.param_map[(nid, "weight")]
                    self._tensor_pool[idx_w] = tw
                    self.executor.set_tensor(idx_w, tw)
                    
                    idx_b = self.param_map[(nid, "bias")]
                    if hasattr(op_mod, "bias") and op_mod.bias is not None:
                        tb = op_mod.bias.data.detach().cpu().float().contiguous().reshape(-1)
                    else:
                        out_dim = op_mod.weight.shape[0]
                        tb = torch.zeros(out_dim, device="cpu", dtype=torch.float32).contiguous()
                    self._tensor_pool[idx_b] = tb
                    self.executor.set_tensor(idx_b, tb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Set input tensor (must be CPU contiguous)
        x_cpu = x.detach().cpu().float().contiguous()
        # Input is ALWAYS at index 0
        self.executor.set_tensor(0, x_cpu)
        
        # 2. Execute C++ loop
        self.executor.execute()
        
        # 3. Get output tensor
        out_node_id = self.graph._output_node_id
        out_idx = self.node_to_idx[out_node_id]
        
        # We need to return the output of the EXECUTOR
        out_native = self.executor.get_tensor(out_idx)
        return out_native.clone()
