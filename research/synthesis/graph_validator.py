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

This runs AFTER graph construction (where add_op silently falls back to
input_id on ValueError), catching graphs that look valid but have
skip-only or broken paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List

from .graph import ComputationGraph
from .primitives import PRIMITIVE_REGISTRY, estimate_op_params


@dataclass(slots=True)
class DimFlowResult:
    """Result of dimension-flow validation."""

    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return len(self.errors) == 0

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


# Ops that do internal Q/K/V projection and need full model_dim.
_FULL_DIM_OPS: FrozenSet[str] = frozenset(
    {
        "softmax_attention",
        "linear_attention",
        "graph_attention",
        "diff_attention",
        "state_space",
        "selective_scan",
        "rwkv_channel",
        "rwkv_time_mixing",
        "moe_topk",
        "moe_2expert",
        "swiglu_mlp",
        "gated_linear",
        "gated_delta",
    }
)


def validate_dim_flow(graph: ComputationGraph) -> DimFlowResult:
    """Walk the DAG and validate dimension flow at every edge.

    Returns DimFlowResult with errors (hard failures) and warnings.
    """
    result = DimFlowResult()

    if graph.input_node is None or graph.output_node is None:
        result.add_error("Graph missing input or output node")
        return result

    model_dim = graph.model_dim
    topo = graph.topological_order()
    reachable = graph.get_reachable_nodes()

    # Build successor map for skip-path detection.
    successors: Dict[int, List[int]] = {nid: [] for nid in graph.nodes}
    for nid, node in graph.nodes.items():
        for pid in node.input_ids:
            if pid in successors:
                successors[pid].append(nid)

    # ── 1. Check every edge for dim/seq compatibility ─────────────
    for nid in topo:
        node = graph.nodes[nid]
        if node.is_input:
            continue
        if nid not in reachable:
            continue

        op = PRIMITIVE_REGISTRY.get(node.op_name)
        if op is None:
            result.add_error(f"Node {nid}: unknown op '{node.op_name}'")
            continue

        for i, pid in enumerate(node.input_ids):
            parent = graph.nodes.get(pid)
            if parent is None:
                result.add_error(
                    f"Node {nid} ({node.op_name}): input[{i}] id={pid} missing"
                )
                continue

            p_shape = parent.output_shape

            # Seq-domain mismatch: freq-domain into non-freq consumer.
            if p_shape.is_freq_domain and op.shape_rule not in ("irfft", "identity"):
                result.add_error(
                    f"Node {nid} ({node.op_name}): input[{i}] is freq-domain "
                    f"(seq={p_shape.seq}) but op expects time-domain"
                )

            # Dim=1 from reduce feeding op that expects full dim.
            if p_shape.dim == 1 and op.name in _FULL_DIM_OPS:
                result.add_error(
                    f"Node {nid} ({node.op_name}): input[{i}] has dim=1 "
                    f"(from reduce) but op requires full dim"
                )

            # Binary ops: check dim compatibility at this edge.
            if op.shape_rule == "binary_broadcast" and len(node.input_ids) == 2:
                other_pid = node.input_ids[1 - i]
                other = graph.nodes.get(other_pid)
                if other is not None:
                    d0, d1 = p_shape.dim, other.output_shape.dim
                    if d0 != d1 and d0 != 1 and d1 != 1:
                        # Only report once (when i==0).
                        if i == 0:
                            result.add_error(
                                f"Node {nid} ({node.op_name}): dim mismatch "
                                f"{d0} vs {d1} at binary edge"
                            )

        # Full-dim ops receiving non-model-dim input.
        if node.op_name in _FULL_DIM_OPS:
            for i, pid in enumerate(node.input_ids):
                parent = graph.nodes.get(pid)
                if parent and parent.output_shape.dim != model_dim:
                    result.add_error(
                        f"Node {nid} ({node.op_name}): input[{i}] has "
                        f"dim={parent.output_shape.dim}, needs model_dim={model_dim}"
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

    # Count how many parameterized ops are on the reachable path.
    n_reachable_params = 0
    for nid in reachable:
        node = graph.nodes[nid]
        if node.is_input:
            continue
        op = PRIMITIVE_REGISTRY.get(node.op_name)
        if op and op.has_params:
            n_reachable_params += 1

    if n_reachable_params == 0:
        result.add_error(
            "No parameterized ops on reachable path — model cannot learn"
        )

    # Minimum effective depth: reject graphs where most slots silently skipped.
    _TRIVIAL_OPS = {"identity", "rmsnorm", "layernorm"}
    n_reachable_nontrivial = sum(
        1
        for nid in reachable
        if not graph.nodes[nid].is_input
    )
    n_effective_ops = sum(
        1
        for nid in reachable
        if not graph.nodes[nid].is_input
        and graph.nodes[nid].op_name not in _TRIVIAL_OPS
    )
    # At least 3 effective ops, or at least 30% of reachable ops must be non-trivial
    min_effective = min(3, max(1, int(n_reachable_nontrivial * 0.3)))
    if n_effective_ops < min_effective:
        result.add_error(
            f"Too few effective ops: {n_effective_ops} < {min_effective} — "
            f"likely all slots fell back to skip"
        )

    # ── 3. Dead parameterized nodes (unreachable learned weights) ─
    for nid, node in graph.nodes.items():
        if nid in reachable or node.is_input:
            continue
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


# ── 5.3: Parameter budget enforcement ─────────────────────────────


def check_param_budget(
    graph: ComputationGraph,
    max_params: int,
) -> DimFlowResult:
    """Reject graphs that exceed a parameter budget before eval.

    Args:
        graph: The computation graph to check.
        max_params: Maximum allowed learnable parameters.

    Returns:
        DimFlowResult with error if budget exceeded.
    """
    result = DimFlowResult()
    total = 0
    model_dim = graph.model_dim
    reachable = graph.get_reachable_nodes()

    for nid in reachable:
        node = graph.nodes[nid]
        if node.is_input:
            continue
        op = PRIMITIVE_REGISTRY.get(node.op_name)
        if op is None or not op.has_params:
            continue
        d_in = node.output_shape.dim or model_dim
        total += estimate_op_params(op, d_in)

    if total > max_params:
        result.add_error(f"Parameter budget exceeded: {total:,} > {max_params:,}")

    return result


# ── 5.4: KV-cache compatibility ──────────────────────────────────

# Ops that break incremental decoding (KV-cache). These either:
# - Reorder/drop tokens (breaking positional alignment)
# - Require full-sequence context at every step (no incremental mode)
# - Use FFT across sequence dim (needs full sequence)
_KV_CACHE_BREAKING_OPS: FrozenSet[str] = frozenset(
    {
        "adjacent_token_merge",  # drops tokens (was: token_merge)
        "depth_token_mask",  # routes subset of tokens (was: mod_topk)
        "spectral_filter",  # FFT over full sequence
        "rfft",  # frequency domain
        "irfft",  # frequency domain
        "sort_seq",  # reorders tokens
        "unsort_seq",  # reorders tokens
        "cumsum",  # depends on full prefix (marginal, but flagged)
        "cumprod_safe",  # depends on full prefix
    }
)


def compute_kv_cacheable(graph: ComputationGraph) -> bool:
    """Check if a graph is compatible with KV-cache incremental decoding.

    Returns True if no ops in the reachable path break KV-cache.
    """
    reachable = graph.get_reachable_nodes()
    for nid in reachable:
        node = graph.nodes[nid]
        if node.op_name in _KV_CACHE_BREAKING_OPS:
            return False
    return True


def annotate_kv_cacheable(graph: ComputationGraph) -> None:
    """Set the kv_cacheable flag in graph.metadata."""
    graph.metadata["kv_cacheable"] = compute_kv_cacheable(graph)
