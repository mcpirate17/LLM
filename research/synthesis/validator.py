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
from typing import List

import numpy as np

from .context_rules import find_graph_context_violations
from .graph_validator import validate_dim_flow
from .native_analysis import analyze_ir_runtime_first
from .primitives import OPCODE_MAP, REVERSE_OPCODE_MAP, get_primitive
from .graph import ComputationGraph, ComputationGraphIR
from .native_validation import effective_depth_natively, summarize_validation
from .validation_opcode_tables import validation_opcode_tables


_NORM_OPS = frozenset({"rmsnorm", "layernorm", "batchnorm"})


def _compute_effective_depth_graph(graph: ComputationGraph) -> float:
    if "effective_depth" in graph._cache:
        return float(graph._cache["effective_depth"])

    if not graph.nodes:
        graph._cache["effective_depth"] = 0.0
        return 0.0

    tables = validation_opcode_tables()
    weights = tables.effective_depth_weight
    discount_successor = tables.discount_successor
    scores: dict[int, float] = {}
    for node_id in graph.topological_order():
        node = graph.nodes[node_id]
        if node.is_input:
            scores[node_id] = 0.0
            continue
        opcode = OPCODE_MAP.get(node.op_name, 0)
        weight = float(weights[opcode]) if opcode < len(weights) else 0.0
        parent_score = 0.0
        discounted = False
        for parent_id in node.input_ids:
            parent_score = max(parent_score, scores.get(parent_id, 0.0))
            parent = graph.nodes.get(parent_id)
            if parent is None or parent.is_input:
                continue
            parent_opcode = OPCODE_MAP.get(parent.op_name, 0)
            if (
                parent_opcode < discount_successor.shape[0]
                and opcode < discount_successor.shape[1]
                and discount_successor[parent_opcode, opcode]
            ):
                discounted = True
        if discounted and weight > 0.20:
            weight = 0.20
        scores[node_id] = parent_score + weight

    effective_depth = max(scores.values(), default=0.0)
    graph._cache["effective_depth"] = effective_depth
    return effective_depth


def _compute_effective_depth_ir(ir: ComputationGraphIR) -> float:
    cached = ir.analysis_cache.get("effective_depth")
    if cached is not None:
        return float(cached)

    n_nodes = ir.n_nodes()
    if n_nodes <= 0:
        ir.analysis_cache["effective_depth"] = 0.0
        return 0.0

    tables = validation_opcode_tables()
    native_depth = effective_depth_natively(
        op_codes=ir.op_codes,
        input_indices=ir.input_indices,
        effective_depth_weights=tables.effective_depth_weight,
        discount_successor_u8=tables.discount_successor_u8,
    )
    if native_depth is not None:
        ir.analysis_cache["effective_depth"] = native_depth
        return native_depth

    weights = tables.effective_depth_weight
    discount_successor = tables.discount_successor
    op_codes = ir.op_codes
    input_indices = ir.input_indices
    scores = np.zeros(n_nodes, dtype=np.float32)
    for idx in range(n_nodes):
        opcode = int(op_codes[idx])
        if opcode == 0:
            continue
        if opcode >= len(weights):
            continue
        weight = float(weights[opcode])
        parent_score = 0.0
        discounted = False
        for parent_idx_raw in input_indices[idx]:
            parent_idx = int(parent_idx_raw)
            if parent_idx < 0:
                continue
            parent_score = max(parent_score, float(scores[parent_idx]))
            parent_opcode = int(op_codes[parent_idx])
            if (
                parent_opcode != 0
                and parent_opcode < discount_successor.shape[0]
                and opcode < discount_successor.shape[1]
                and discount_successor[parent_opcode, opcode]
            ):
                discounted = True
        if discounted and weight > 0.20:
            weight = 0.20
        scores[idx] = parent_score + weight

    effective_depth = float(scores.max(initial=0.0))
    ir.analysis_cache["effective_depth"] = effective_depth
    return effective_depth


def compute_effective_depth(
    graph_or_ir: ComputationGraph | ComputationGraphIR,
) -> float:
    if isinstance(graph_or_ir, ComputationGraph):
        return _compute_effective_depth_graph(graph_or_ir)
    if isinstance(graph_or_ir, ComputationGraphIR):
        return _compute_effective_depth_ir(graph_or_ir)
    analyze_structure = getattr(graph_or_ir, "analyze_structure", None)
    if analyze_structure is not None:
        return float(analyze_structure(include_reachable=True).depth)
    raise TypeError(f"Unsupported graph type: {type(graph_or_ir)!r}")


@dataclass
class ValidationResult:
    """Result of graph validation."""

    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    # Metrics
    n_ops: int = 0
    depth: int = 0
    effective_depth: float = 0.0
    n_params_estimate: int = 0
    has_gradient_path: bool = True
    n_risky_ops: int = 0
    n_parameterized_ops: int = 0

    def add_error(self, msg: str):
        self.valid = False
        self.errors.append(msg)

    def add_warning(self, msg: str):
        self.warnings.append(msg)


def _summarize_graph_validation(graph: ComputationGraph) -> tuple[object, list[int]]:
    topo = graph.topological_order()
    known_op_flags = np.zeros(len(topo), dtype=np.int32)
    risky_op_flags = np.zeros(len(topo), dtype=np.int32)
    parameterized_op_flags = np.zeros(len(topo), dtype=np.int32)
    norm_op_flags = np.zeros(len(topo), dtype=np.int32)
    linear_op_flags = np.zeros(len(topo), dtype=np.int32)

    for idx, node_id in enumerate(topo):
        node = graph.nodes[node_id]
        if node.is_input:
            continue
        try:
            op = get_primitive(node.op_name)
        except KeyError:
            continue
        known_op_flags[idx] = 1
        risky_op_flags[idx] = int(op.numerically_risky)
        parameterized_op_flags[idx] = int(op.has_params)
        norm_op_flags[idx] = int(node.op_name in _NORM_OPS)
        linear_op_flags[idx] = int(op.has_params and op.shape_rule == "linear")

    return (
        summarize_validation(
            known_op_flags=known_op_flags,
            risky_op_flags=risky_op_flags,
            parameterized_op_flags=parameterized_op_flags,
            norm_op_flags=norm_op_flags,
            linear_op_flags=linear_op_flags,
        ),
        topo,
    )


def _summarize_ir_validation(ir: ComputationGraphIR):
    op_codes = ir.op_codes
    tables = validation_opcode_tables()

    return summarize_validation(
        known_op_flags=tables.known[op_codes],
        risky_op_flags=tables.risky[op_codes],
        parameterized_op_flags=tables.parameterized[op_codes],
        norm_op_flags=tables.norm[op_codes],
        linear_op_flags=tables.linear[op_codes],
    )


def validate_graph(
    graph: ComputationGraph,
    max_ops: int = 20,
    max_depth: int = 15,
    min_splits: int = 0,
    max_params: int | None = None,
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

    analysis_ir = graph._analysis_ir()
    analysis = analyze_ir_runtime_first(analysis_ir, include_reachable=True)

    result.n_ops = graph.n_ops()
    result.depth = analysis.depth
    result.effective_depth = compute_effective_depth(analysis_ir)
    result.n_params_estimate = analysis.param_estimate

    # Size limits
    if result.n_ops > max_ops:
        result.add_error(f"Too many ops: {result.n_ops} > {max_ops}")

    depth_limit = float(max_depth) + float(min_splits) * 0.5
    if result.effective_depth > depth_limit + 1e-9:
        result.add_error(
            "Too deep: "
            f"effective {result.effective_depth:.2f} > {depth_limit:.2f} "
            f"(raw={result.depth})"
        )

    # Dead branch detection (Shadow Complexity)
    if analysis.reachable_count < len(graph.nodes):
        dead_count = len(graph.nodes) - int(analysis.reachable_count)
        result.add_error(
            f"Graph contains {dead_count} unreachable nodes (dead branches)"
        )

    # Gradient flow and structural analysis come from the shared IR/native path.
    result.has_gradient_path = analysis.has_gradient_path
    if not result.has_gradient_path:
        result.add_error("No differentiable path from input to output")

    # Check input/output shape consistency
    output_shape = graph.output_node.output_shape
    if output_shape.dim != graph.model_dim:
        result.add_error(
            f"Output dim {output_shape.dim} != model_dim {graph.model_dim}"
        )
    if not output_shape.is_standard:
        result.add_error(f"Output seq dimension is '{output_shape.seq}', expected 'S'")

    if hasattr(analysis_ir, "op_codes"):
        validation_summary = _summarize_ir_validation(analysis_ir)
    else:
        validation_summary, _ = _summarize_graph_validation(graph)
    result.n_risky_ops = validation_summary.risky_op_count
    result.n_parameterized_ops = validation_summary.parameterized_op_count

    # Check all nodes
    for nid in sorted(graph.nodes):
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
                result.add_error(f"Node {nid}: input {iid} doesn't exist")

    for violation in find_graph_context_violations(graph):
        result.add_error(violation)

    dim_flow = validate_dim_flow(
        graph,
        max_params=max_params,
        analysis_ir=analysis_ir,
        analysis=analysis,
    )
    for error in dim_flow.errors:
        if error not in result.errors:
            result.add_error(error)
    for warning in dim_flow.warnings:
        if warning not in result.warnings:
            result.add_warning(warning)

    # Warnings for risky patterns
    if result.n_risky_ops > 3:
        result.add_warning(
            f"Many numerically risky ops ({result.n_risky_ops}) — likely unstable"
        )

    if result.n_parameterized_ops == 0:
        result.add_warning("No learnable parameters — model can't learn")

    # Deep projection chain detection: stacked linear projections without
    # intermediate normalization inflate output magnitudes, causing
    # initial_loss of 100–250 (vs ~12 for normalized architectures).
    # 75% of S1 failures come from this pattern (diagnosis 2026-03-20).
    if validation_summary.max_projection_chain_depth > 3:
        result.add_warning(
            "Deep projection chain without normalization "
            f"(depth={validation_summary.max_projection_chain_depth}): "
            f"likely high initial loss"
        )

    if analysis.has_cycle:
        result.add_error("Graph contains a cycle")

    return result


def validate_ir(
    ir: ComputationGraphIR,
    max_ops: int = 20,
    max_depth: int = 15,
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

    analysis = analyze_ir_runtime_first(ir, include_reachable=True)
    result.depth = int(analysis.depth)
    result.effective_depth = compute_effective_depth(ir)

    # Gradient flow (native-backed when available)
    result.has_gradient_path = analysis.has_gradient_path
    if not result.has_gradient_path:
        result.add_error("No differentiable path from input to output")

    result.n_params_estimate = analysis.param_estimate
    if analysis.reachable_count < ir.n_nodes():
        result.add_error(
            "IR contains "
            f"{ir.n_nodes() - int(analysis.reachable_count)} unreachable nodes "
            "(dead branches)"
        )

    # Fast structural checks
    if analysis.has_cycle:
        result.add_error("Graph contains a cycle")

    validation_summary = _summarize_ir_validation(ir)
    result.n_risky_ops = validation_summary.risky_op_count
    result.n_parameterized_ops = validation_summary.parameterized_op_count

    depth_limit = float(max_depth)
    if result.effective_depth > depth_limit + 1e-9:
        result.add_error(
            "Too deep: "
            f"effective {result.effective_depth:.2f} > {depth_limit:.2f} "
            f"(raw={result.depth})"
        )

    for i in range(ir.n_nodes()):
        opcode = op_codes[i]
        if opcode == 0:
            continue
        op_name = REVERSE_OPCODE_MAP.get(opcode)
        if op_name is None:
            continue
        try:
            get_primitive(op_name)
        except KeyError:
            result.add_warning(f"IR contains unknown op '{op_name}'")

    # Warnings
    if result.n_risky_ops > 3:
        result.add_warning(f"Many risky ops ({result.n_risky_ops})")
    if result.n_parameterized_ops == 0:
        result.add_warning("No learnable parameters")

    return result
