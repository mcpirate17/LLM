"""
Dimension-Flow Validator for Computation Graphs

Post-construction static analysis that walks the DAG and reports:
1. Dimension mismatches at edges (producer dim != consumer expectation)
2. Sequence-domain mismatches (freq-domain output into non-freq consumer)
3. Skip-only paths (template caught ValueError and fell back to input_id)
4. Reduce-to-1 outputs feeding ops expecting full dim
5. Unreachable parameterized nodes (dead learned weights)
6. Parameter budget enforcement (reject before expensive eval)
7. KV-cache compatibility flag (ops that break incremental decoding)

This runs AFTER graph construction, catching graphs that look valid but
still have skip-only or broken paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, FrozenSet, List, Optional

import numpy as np

from .dim_flow_opcode_tables import KV_CACHE_BREAKING_OPS
from .dim_flow_support import build_dim_flow_inputs, ensure_dim_flow_flags
from .graph import ComputationGraph
from .native_analysis import (
    analyze_ir_runtime_first,
    summarize_dim_flow_in_python,
    validate_edges,
    validate_packed_ir_natively,
)
from .native_dim_flow import dead_parameterized_mask_in_python
from .primitives import PRIMITIVE_REGISTRY


@dataclass(slots=True)
class DimFlowResult:
    """Result of dimension-flow validation."""

    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    reachable_param_count: int = 0
    reachable_param_estimate: int = 0
    reachable_nontrivial_ops: int = 0
    reachable_ops: int = 0

    @property
    def valid(self) -> bool:
        return len(self.errors) == 0

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


_OP_KIND_DEFAULT = 0
_OP_KIND_IRFFT = 1
_OP_KIND_IDENTITY = 2
_OP_KIND_BINARY_BROADCAST = 3


def _edge_error_indices(edge_validation: Any) -> np.ndarray:
    combined = (
        edge_validation.freq_mismatch_bits
        | edge_validation.reduce_full_dim_bits
        | edge_validation.binary_dim_mismatch
        | edge_validation.full_dim_input_bits
    )
    return np.flatnonzero(combined)


def _add_edge_validation_errors(
    *,
    result: DimFlowResult,
    graph: ComputationGraph,
    model_dim: int,
    analysis_node_ids: np.ndarray,
    edge_validation: Any,
    flagged_indices: np.ndarray,
) -> None:
    for raw_idx in flagged_indices:
        analysis_idx = int(raw_idx)
        nid = int(analysis_node_ids[analysis_idx])
        node = graph.nodes[nid]

        for i, pid in enumerate(node.input_ids):
            parent = graph.nodes.get(pid)
            if parent is None:
                result.add_error(
                    f"Node {nid} ({node.op_name}): input[{i}] id={pid} missing"
                )
                continue

            p_shape = parent.output_shape
            if edge_validation.freq_mismatch_bits[analysis_idx] & (1 << i):
                result.add_error(
                    f"Node {nid} ({node.op_name}): input[{i}] is freq-domain "
                    f"(seq={p_shape.seq}) but op expects time-domain"
                )

            if edge_validation.reduce_full_dim_bits[analysis_idx] & (1 << i):
                result.add_error(
                    f"Node {nid} ({node.op_name}): input[{i}] has dim=1 "
                    f"(from reduce) but op requires full dim"
                )

            if edge_validation.full_dim_input_bits[analysis_idx] & (1 << i):
                result.add_error(
                    f"Node {nid} ({node.op_name}): input[{i}] has "
                    f"dim={parent.output_shape.dim}, needs model_dim={model_dim}"
                )

        if edge_validation.binary_dim_mismatch[analysis_idx]:
            d0 = graph.nodes[node.input_ids[0]].output_shape.dim
            d1 = graph.nodes[node.input_ids[1]].output_shape.dim
            result.add_error(
                f"Node {nid} ({node.op_name}): dim mismatch {d0} vs {d1} at binary edge"
            )


def build_dim_flow_validation_inputs(
    graph: ComputationGraph,
    *,
    analysis_ir: Any | None = None,
    analysis: Any | None = None,
    compute_analysis: bool = True,
    build_flags: bool = True,
) -> Any:
    return build_dim_flow_inputs(
        graph,
        op_kind_default=_OP_KIND_DEFAULT,
        op_kind_irfft=_OP_KIND_IRFFT,
        op_kind_identity=_OP_KIND_IDENTITY,
        op_kind_binary_broadcast=_OP_KIND_BINARY_BROADCAST,
        analysis_ir=analysis_ir,
        analysis=analysis,
        compute_analysis=compute_analysis,
        build_flags=build_flags,
    )


def try_packed_dim_flow_validation(
    *,
    graph: ComputationGraph,
    analysis_ir: Any,
    dim_flow_inputs: Any,
    effective_depth_weights: Any | None = None,
    discount_successor_u8: Any | None = None,
) -> Any | None:
    if (
        not hasattr(analysis_ir, "op_codes")
        or not hasattr(analysis_ir, "input_indices")
        or not hasattr(analysis_ir, "output_node_idx")
    ):
        return None
    input_node_idx = dim_flow_inputs.node_id_to_analysis_idx.get(
        graph._input_node_id, -1
    )
    if not dim_flow_inputs.flags_ready:
        ensure_dim_flow_flags(
            dim_flow_inputs,
            op_kind_default=_OP_KIND_DEFAULT,
            op_kind_irfft=_OP_KIND_IRFFT,
            op_kind_identity=_OP_KIND_IDENTITY,
            op_kind_binary_broadcast=_OP_KIND_BINARY_BROADCAST,
        )
    return validate_packed_ir_natively(
        op_codes=analysis_ir.op_codes,
        input_indices=analysis_ir.input_indices,
        output_node_idx=int(analysis_ir.output_node_idx),
        param_estimates=dim_flow_inputs.param_estimates,
        has_params_flags=dim_flow_inputs.has_params_flags,
        nontrivial_flags=dim_flow_inputs.nontrivial_flags,
        kv_breaking_flags=dim_flow_inputs.kv_breaking_flags,
        node_dims=dim_flow_inputs.node_dims,
        node_seq_flags=dim_flow_inputs.node_seq_flags,
        op_kind_flags=dim_flow_inputs.op_kind_flags,
        full_dim_flags=dim_flow_inputs.full_dim_flags,
        model_dim=graph.model_dim,
        input_node_idx=int(input_node_idx),
        effective_depth_weights=effective_depth_weights,
        discount_successor_u8=discount_successor_u8,
    )


def validate_dim_flow(
    graph: ComputationGraph,
    max_params: Optional[int] = None,
    analysis_ir: Any | None = None,
    analysis: Any | None = None,
    dim_flow_inputs: Any | None = None,
    packed_validation: Any | None = None,
) -> DimFlowResult:
    """Walk the DAG and validate dimension flow at every edge.

    Returns DimFlowResult with errors (hard failures) and warnings.
    """
    result = DimFlowResult()
    caller_supplied_analysis = analysis is not None

    if graph.input_node is None or graph.output_node is None:
        result.add_error("Graph missing input or output node")
        return result

    model_dim = graph.model_dim
    if dim_flow_inputs is None:
        dim_flow_inputs = build_dim_flow_validation_inputs(
            graph,
            analysis_ir=analysis_ir,
            analysis=analysis,
            compute_analysis=caller_supplied_analysis,
        )
    elif analysis is not None and dim_flow_inputs.analysis is None:
        dim_flow_inputs.analysis = analysis
    analysis_ir = dim_flow_inputs.analysis_ir
    analysis = dim_flow_inputs.analysis
    analysis_node_ids = dim_flow_inputs.analysis_node_ids
    node_id_to_analysis_idx = dim_flow_inputs.node_id_to_analysis_idx

    if packed_validation is None and not caller_supplied_analysis:
        packed_validation = try_packed_dim_flow_validation(
            graph=graph,
            analysis_ir=analysis_ir,
            dim_flow_inputs=dim_flow_inputs,
        )

    if packed_validation is not None:
        reachable_mask = packed_validation.reachable_mask
        summary = packed_validation.dim_flow
        edge_validation = packed_validation.edge_validation
        dead_parameterized_mask = packed_validation.dead_parameterized_mask
        edge_error_count = packed_validation.edge_error_count
        dead_parameterized_count = packed_validation.dead_parameterized_count
    else:
        ensure_dim_flow_flags(
            dim_flow_inputs,
            op_kind_default=_OP_KIND_DEFAULT,
            op_kind_irfft=_OP_KIND_IRFFT,
            op_kind_identity=_OP_KIND_IDENTITY,
            op_kind_binary_broadcast=_OP_KIND_BINARY_BROADCAST,
        )
        if analysis is None:
            analysis = analyze_ir_runtime_first(analysis_ir, include_reachable=True)
        reachable_mask = getattr(analysis, "reachable_mask", None)
        if reachable_mask is None:
            reachable_mask = np.ones(analysis_ir.n_nodes(), dtype=np.int32)

        summary_reachable_mask = np.asarray(reachable_mask).astype(np.int32, copy=True)
        input_node_id = graph._input_node_id
        if input_node_id is not None and input_node_id in node_id_to_analysis_idx:
            summary_reachable_mask[node_id_to_analysis_idx[input_node_id]] = 0

        summary = summarize_dim_flow_in_python(
            reachable_mask=summary_reachable_mask,
            has_params_flags=dim_flow_inputs.has_params_flags,
            param_estimates=dim_flow_inputs.param_estimates,
            nontrivial_flags=dim_flow_inputs.nontrivial_flags,
            kv_breaking_flags=dim_flow_inputs.kv_breaking_flags,
        )
        edge_validation = validate_edges(
            reachable_mask=np.asarray(reachable_mask).astype(np.int32, copy=False),
            input_indices=analysis_ir.input_indices,
            node_dims=dim_flow_inputs.node_dims,
            node_seq_flags=dim_flow_inputs.node_seq_flags,
            op_kind_flags=dim_flow_inputs.op_kind_flags,
            full_dim_flags=dim_flow_inputs.full_dim_flags,
            model_dim=model_dim,
        )
        dead_parameterized_mask = None
        edge_error_count = -1
        dead_parameterized_count = -1
    result.reachable_param_count = summary.reachable_param_count
    result.reachable_param_estimate = summary.reachable_param_estimate
    result.reachable_nontrivial_ops = summary.reachable_nontrivial_ops
    result.reachable_ops = summary.reachable_ops

    # ── 1. Check reachable edge errors only when the native/vectorized masks
    # report something to format. The common valid path should not walk every
    # edge again just to rediscover an all-zero mask.
    flagged_edge_indices = (
        _edge_error_indices(edge_validation)
        if edge_error_count != 0
        else np.empty(0, dtype=np.int64)
    )
    if flagged_edge_indices.size:
        _add_edge_validation_errors(
            result=result,
            graph=graph,
            model_dim=model_dim,
            analysis_node_ids=analysis_node_ids,
            edge_validation=edge_validation,
            flagged_indices=flagged_edge_indices,
        )

    # ── 2. Detect skip-only paths ─────────────────────────────────
    # A graph where the output node is the input node, or the output is
    # an `add` whose only real input is the input node (the other path
    # is also just input_id).
    input_id = graph._input_node_id
    output_id = graph._output_node_id

    if output_id == input_id:
        result.add_error("Graph is skip-only: output == input (no computation)")
    else:
        # Check if the graph is effectively a no-op: output is `add(input, input)`
        # or a trivially thin chain.
        out_node = graph.nodes[output_id]
        if out_node.op_name == "add" and set(out_node.input_ids) == {input_id}:
            result.add_error(
                "Graph is skip-only: output is add(input, input) — "
                "all template ops were bypassed"
            )

    if result.reachable_param_count == 0:
        result.add_warning(
            "No parameterized ops on reachable path — model cannot learn"
        )

    # Minimum effective depth: reject graphs where most slots silently skipped.
    # At least 3 effective ops, or at least 30% of reachable ops must be non-trivial
    min_effective = min(3, max(1, int(result.reachable_ops * 0.3)))
    if result.reachable_nontrivial_ops < min_effective:
        result.add_error(
            f"Too few effective ops: {result.reachable_nontrivial_ops} < {min_effective} — "
            f"likely all slots fell back to skip"
        )

    if max_params is not None and result.reachable_param_estimate > max_params:
        result.add_error(
            "Parameter budget exceeded: "
            f"{result.reachable_param_estimate:,} > {max_params:,}"
        )

    # ── 3. Dead parameterized nodes (unreachable learned weights) ─
    if dead_parameterized_mask is None:
        dead_params = dead_parameterized_mask_in_python(
            reachable_mask=np.asarray(reachable_mask).astype(np.int32, copy=False),
            parameterized_flags=dim_flow_inputs.has_params_flags,
        )
        dead_mask = dead_params.mask
    elif dead_parameterized_count == 0:
        dead_mask = np.empty(0, dtype=np.int32)
    else:
        dead_mask = dead_parameterized_mask
    for idx in np.flatnonzero(dead_mask):
        nid = int(analysis_node_ids[idx])
        node = graph.nodes[nid]
        op = PRIMITIVE_REGISTRY.get(node.op_name)
        if op and op.has_params:
            result.add_warning(
                f"Node {nid} ({node.op_name}): parameterized but unreachable "
                f"(dead weight)"
            )

    # ── 4. Output dim must match model_dim ────────────────────────
    out_shape = graph.output_node.output_shape
    if out_shape.dim != model_dim:
        result.add_error(f"Output dim {out_shape.dim} != model_dim {model_dim}")
    if not out_shape.is_standard:
        result.add_error(
            f"Output in freq domain (seq={out_shape.seq}), must be time-domain"
        )

    return result


# ── 5.4: KV-cache compatibility ──────────────────────────────────

# Ops that break incremental decoding (KV-cache). These either:
# - Reorder/drop tokens (breaking positional alignment)
# - Require full-sequence context at every step (no incremental mode)
# - Use FFT across sequence dim (needs full sequence)
_KV_CACHE_BREAKING_OPS: FrozenSet[str] = KV_CACHE_BREAKING_OPS


def compute_kv_cacheable(graph: ComputationGraph) -> bool:
    """Check if a graph is compatible with KV-cache incremental decoding.

    Returns True if no ops in the reachable path break KV-cache.
    """
    dim_flow_inputs = build_dim_flow_inputs(
        graph,
        op_kind_default=_OP_KIND_DEFAULT,
        op_kind_irfft=_OP_KIND_IRFFT,
        op_kind_identity=_OP_KIND_IDENTITY,
        op_kind_binary_broadcast=_OP_KIND_BINARY_BROADCAST,
    )
    analysis_ir = dim_flow_inputs.analysis_ir
    analysis = dim_flow_inputs.analysis
    analysis_node_ids = dim_flow_inputs.analysis_node_ids

    summary_reachable_mask = analysis.reachable_mask.astype(np.int32, copy=True)
    input_node_id = graph._input_node_id
    if input_node_id is not None:
        input_idx = int(np.where(analysis_node_ids == input_node_id)[0][0])
        summary_reachable_mask[input_idx] = 0

    summary = summarize_dim_flow_in_python(
        reachable_mask=summary_reachable_mask,
        has_params_flags=np.zeros(analysis_ir.n_nodes(), dtype=np.int32),
        param_estimates=np.zeros(analysis_ir.n_nodes(), dtype=np.int64),
        nontrivial_flags=np.zeros(analysis_ir.n_nodes(), dtype=np.int32),
        kv_breaking_flags=dim_flow_inputs.kv_breaking_flags,
    )
    return summary.kv_cacheable


def annotate_kv_cacheable(graph: ComputationGraph) -> None:
    """Set the kv_cacheable flag in graph.metadata."""
    graph.metadata["kv_cacheable"] = compute_kv_cacheable(graph)
