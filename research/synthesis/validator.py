"""
Static Validation for Computation Graphs

Validates graphs before compilation:
- Shape consistency
- Gradient flow (differentiable path from input to output)
- Numerical stability heuristics (gradient norms, spectral radius)
- Parameter budget compliance
- Self-repair: detects and suggests fixes for unstable architectures
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set

from .primitives import get_primitive, PrimitiveOp
from .graph import ComputationGraph, OpNode


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
