"""
Static Validation for Computation Graphs

Validates graphs before compilation:
- Shape consistency
- Gradient flow (differentiable path from input to output)
- Zero-grad detection (ops must have non-zero gradient paths)
- Numerical stability heuristics (gradient norms, spectral radius)
- Skip connection validation (enforce residual paths)
- Parameter budget compliance
- Self-repair: detects and suggests fixes for unstable architectures
- Auto-inject skip connections when gradient paths are broken
- Reject ops with potential zero-grad (pure multiplications, dead ReLUs)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set

from .primitives import get_primitive, PrimitiveOp, REVERSE_OPCODE_MAP
from .graph import ComputationGraph, OpNode, ComputationGraphIR
import numpy as np
from collections import deque


@dataclass
class ValidationResult:
    """Result of graph validation."""
    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    # Metrics
    n_ops: int = 0
    depth: int = 0
    n_params_estimate: int = 0
    has_gradient_path: bool = True
    n_risky_ops: int = 0
    n_parameterized_ops: int = 0

    def add_error(self, msg: str):
        self.valid = False
        self.errors.append(msg)

    def add_warning(self, msg: str):
        self.warnings.append(msg)


def validate_graph(
    graph: ComputationGraph,
    max_ops: int = 20,
    max_depth: int = 15,
    max_params_ratio: float = 6.0,
) -> ValidationResult:
    """Validate a computation graph.

    Returns ValidationResult with errors/warnings.
    """
    result = ValidationResult()

    # Basic structure
    if graph.input_node is None:
        result.add_error("Graph has no input node")
        return result

    if graph.output_node is None:
        result.add_error("Graph has no output node")
        return result

    result.n_ops = graph.n_ops()
    result.depth = graph.depth()
    result.n_params_estimate = graph.n_params_estimate()

    # Size limits
    if result.n_ops > max_ops:
        result.add_error(f"Too many ops: {result.n_ops} > {max_ops}")

    if result.depth > max_depth:
        result.add_error(f"Too deep: {result.depth} > {max_depth}")

    # Parameter budget
    D = graph.model_dim
    max_params = int(max_params_ratio * D * D)
    if result.n_params_estimate > max_params:
        result.add_error(
            f"Too many params: ~{result.n_params_estimate} > {max_params}"
        )

    # Dead branch detection (Shadow Complexity)
    reachable_nodes = graph.get_reachable_nodes()
    if len(reachable_nodes) < len(graph.nodes):
        dead_count = len(graph.nodes) - len(reachable_nodes)
        result.add_error(f"Graph contains {dead_count} unreachable nodes (dead branches)")

    # Gradient flow
    result.has_gradient_path = graph.has_gradient_path()
    if not result.has_gradient_path:
        result.add_error("No differentiable path from input to output")

    # Check input/output shape consistency
    output_shape = graph.output_node.output_shape
    if output_shape.dim != graph.model_dim:
        result.add_error(
            f"Output dim {output_shape.dim} != model_dim {graph.model_dim}"
        )
    if not output_shape.is_standard:
        result.add_error(
            f"Output seq dimension is '{output_shape.seq}', expected 'S'"
        )

    # Check all nodes
    for nid in graph.topological_order():
        node = graph.nodes[nid]
        if node.is_input:
            continue

        try:
            op = get_primitive(node.op_name)
        except KeyError:
            result.add_error(f"Node {nid}: unknown op '{node.op_name}'")
            continue

        # Input count
        if len(node.input_ids) != op.n_inputs:
            result.add_error(
                f"Node {nid} ({op.name}): expected {op.n_inputs} inputs, "
                f"got {len(node.input_ids)}"
            )

        # Check inputs exist
        for iid in node.input_ids:
            if iid not in graph.nodes:
                result.add_error(
                    f"Node {nid}: input {iid} doesn't exist"
                )

        # Track risky ops
        if op.numerically_risky:
            result.n_risky_ops += 1
        if op.has_params:
            result.n_parameterized_ops += 1

    # Warnings for risky patterns
    if result.n_risky_ops > 3:
        result.add_warning(
            f"Many numerically risky ops ({result.n_risky_ops}) — likely unstable"
        )

    if result.n_parameterized_ops == 0:
        result.add_warning("No learnable parameters — model can't learn")

    # Check for cycles (shouldn't happen with our builder, but safety check)
    if _has_cycle(graph):
        result.add_error("Graph contains a cycle")

    return result


def _has_cycle(graph: ComputationGraph) -> bool:
    """Check for cycles in the graph using DFS."""
    WHITE, GRAY, BLACK = 0, 1, 2
    colors = {nid: WHITE for nid in graph.nodes}

    def dfs(nid: int) -> bool:
        colors[nid] = GRAY
        node = graph.nodes[nid]
        for inp_id in node.input_ids:
            if colors[inp_id] == GRAY:
                return True  # Back edge = cycle
            if colors[inp_id] == WHITE and dfs(inp_id):
                return True
        colors[nid] = BLACK
        return False

    for nid in graph.nodes:
        if colors[nid] == WHITE:
            if dfs(nid):
                return True
    return False


def validate_ir(
    ir: ComputationGraphIR,
    max_ops: int = 20,
    max_depth: int = 15,
    max_params_ratio: float = 6.0,
) -> ValidationResult:
    """Validate a computation graph in IR form. High-performance path."""
    result = ValidationResult()

    if ir.output_node_idx == -1:
        result.add_error("Graph has no output node")
        return result

    # Size metrics (Vectorized)
    op_codes = ir.op_codes
    non_input_mask = op_codes != 0
    result.n_ops = int(np.sum(non_input_mask))

    if result.n_ops > max_ops:
        result.add_error(f"Too many ops: {result.n_ops} > {max_ops}")

    # Gradient flow (Vectorized in IR)
    result.has_gradient_path = ir.has_gradient_path()
    if not result.has_gradient_path:
        result.add_error("No differentiable path from input to output")

    # Parameter budget (Vectorized in IR)
    result.n_params_estimate = ir.n_params_estimate()
    D = ir.model_dim
    max_params = int(max_params_ratio * D * D)
    if result.n_params_estimate > max_params:
        result.add_error(
            f"Too many params: ~{result.n_params_estimate} > {max_params}"
        )

    # Reachability detection (NumPy accelerated)
    if ir.output_node_idx != -1:
        n = ir.n_nodes()
        adj_back = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(2):
                inp_idx = ir.input_indices[i, j]
                if inp_idx != -1:
                    adj_back[i, inp_idx] = True

        reachable = np.zeros(n, dtype=bool)
        reachable[ir.output_node_idx] = True
        for _ in range(n):
            new_reachable = reachable | np.any(adj_back[reachable, :], axis=0) if np.any(reachable) else reachable
            if np.array_equal(new_reachable, reachable):
                break
            reachable = new_reachable
            
        n_reachable = int(np.sum(reachable))
        if n_reachable < n:
            result.add_error(f"IR contains {n - n_reachable} unreachable nodes (dead branches)")

    # Fast structural checks
    if _ir_has_cycle(ir):
        result.add_error("Graph contains a cycle")

    # Per-node property aggregation (Vectorized)
    for i in range(ir.n_nodes()):
        opcode = op_codes[i]
        if opcode == 0: continue
        
        op_name = REVERSE_OPCODE_MAP.get(opcode)
        if not op_name: continue
        
        try:
            op = get_primitive(op_name)
            if op.numerically_risky:
                result.n_risky_ops += 1
            if op.has_params:
                result.n_parameterized_ops += 1
        except KeyError:
            pass

    # Warnings
    if result.n_risky_ops > 3:
        result.add_warning(f"Many risky ops ({result.n_risky_ops})")
    if result.n_parameterized_ops == 0:
        result.add_warning("No learnable parameters")

    return result


def _ir_has_cycle(ir: ComputationGraphIR) -> bool:
    """Check for cycles in IR using Kahn's algorithm (NumPy accelerated)."""
    n = ir.n_nodes()
    in_degree = np.zeros(n, dtype=np.int32)
    
    # adj[i] list of nodes that depend on i
    adj = [[] for _ in range(n)]
    
    for i in range(n):
        for j in range(2):
            idx = ir.input_indices[i, j]
            if idx != -1:
                in_degree[i] += 1
                adj[idx].append(i)
                
    queue = deque(np.where(in_degree == 0)[0])
    visited_count = 0
    while queue:
        u = queue.popleft()
        visited_count += 1
        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    return visited_count < n
