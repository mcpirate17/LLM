"""Context rules — graph validation and byte-safety enforcement."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Dict, FrozenSet, Iterable, List, Optional

if TYPE_CHECKING:
    from .graph import ComputationGraph
from .primitives import PRIMITIVE_REGISTRY

from ._context_op_sets import (
    _LOCAL_WINDOW_VALID_PREDS,
    _MASK_VALID_SUCCESSORS,
    _REDUCTION_RESTORE_OPS,
    _RESTRICTED_LINEAR_SUCCESSORS,
    _ROUTER_VALID_SUCCESSORS,
    _STABILIZER_SUCCESSORS,
    _TROPICAL_BRIDGE_PREDS,
)
from ._context_registry import CONTEXT_RULES


# ── Graph traversal helpers ──────────────────────────────────────


def _child_map(graph: ComputationGraph) -> Dict[int, List[int]]:
    children: Dict[int, List[int]] = {nid: [] for nid in graph.nodes}
    for nid, node in graph.nodes.items():
        for inp_id in node.input_ids:
            if inp_id in children:
                children[inp_id].append(nid)
    return children


def _has_descendant_op(
    graph: ComputationGraph,
    start_id: int,
    allowed_ops: Iterable[str],
    children: Optional[Dict[int, List[int]]] = None,
) -> bool:
    children = children or _child_map(graph)
    q = deque(children.get(start_id, ()))
    seen: set = set()
    while q:
        nid = q.popleft()
        if nid in seen:
            continue
        seen.add(nid)
        node = graph.nodes.get(nid)
        if node is None:
            continue
        if node.op_name in allowed_ops:
            return True
        q.extend(children.get(nid, ()))
    return False


def _has_ancestor_op(
    graph: ComputationGraph,
    start_id: int,
    allowed_ops: Iterable[str],
) -> bool:
    start_node = graph.nodes.get(start_id)
    if start_node is None:
        return False
    q = deque(start_node.input_ids)
    seen: set = set()
    while q:
        nid = q.popleft()
        if nid in seen:
            continue
        seen.add(nid)
        node = graph.nodes.get(nid)
        if node is None:
            continue
        if node.op_name in allowed_ops:
            return True
        q.extend(node.input_ids)
    return False


def _has_immediate_predecessor_op(
    graph: ComputationGraph,
    node_id: int,
    allowed_ops: Iterable[str],
) -> bool:
    """Check if any direct parent of node_id has op_name in allowed_ops."""
    node = graph.nodes.get(node_id)
    if node is None:
        return False
    for pid in node.input_ids:
        parent = graph.nodes.get(pid)
        if parent is not None and parent.op_name in allowed_ops:
            return True
    return False


def _has_immediate_successor_op(
    children: Dict[int, List[int]],
    graph: ComputationGraph,
    node_id: int,
    allowed_ops: Iterable[str],
) -> bool:
    """Check if any direct child of node_id has op_name in allowed_ops."""
    for cid in children.get(node_id, ()):
        child = graph.nodes.get(cid)
        if child is not None and child.op_name in allowed_ops:
            return True
    return False


def _nearest_split2_ancestors(graph: ComputationGraph, start_id: int) -> list[int]:
    """Return nearest split2 ancestors on the path(s) above start_id.

    Traversal stops at the first split2 encountered on each upstream branch so
    downstream branch-local transforms don't hide the original split contract.
    """
    q = deque([start_id])
    seen: set[int] = set()
    found: list[int] = []
    while q:
        nid = q.popleft()
        if nid in seen:
            continue
        seen.add(nid)
        node = graph.nodes.get(nid)
        if node is None or node.is_input:
            continue
        if node.op_name == "split2":
            found.append(nid)
            continue
        q.extend(node.input_ids)
    return found


def _check_split_branch_restore_contracts(
    graph: ComputationGraph,
    violations: List[str],
) -> None:
    """Reject split2 sibling branches that rejoin through concat asymmetrically.

    The only current split2→concat templates are `parallel_split` and
    `dual_axis_block`. Both expect each branch to remain at half-width until the
    concat join. Historical runtime failures frequently restored only one branch
    to full width, producing 256+128 or 256+256 concat scaffolds that later
    fail far downstream and falsely blame generic ops.
    """
    for nid, node in graph.nodes.items():
        if node.op_name != "concat" or len(node.input_ids) != 2:
            continue
        branch_split_ids = [
            sorted(set(_nearest_split2_ancestors(graph, input_id)))
            for input_id in node.input_ids
        ]
        if any(len(split_ids) != 1 for split_ids in branch_split_ids):
            continue
        left_split_id = branch_split_ids[0][0]
        right_split_id = branch_split_ids[1][0]
        if left_split_id == right_split_id:
            continue
        left_split = graph.nodes.get(left_split_id)
        right_split = graph.nodes.get(right_split_id)
        if left_split is None or right_split is None:
            continue
        if (
            left_split.op_name != "split2"
            or right_split.op_name != "split2"
            or len(left_split.input_ids) != 1
            or len(right_split.input_ids) != 1
            or left_split.input_ids[0] != right_split.input_ids[0]
        ):
            continue
        parts = {
            int(left_split.config.get("part", 0)),
            int(right_split.config.get("part", 0)),
        }
        if parts != {0, 1}:
            continue

        expected_dim = left_split.output_shape.dim
        input_dims = [
            graph.nodes[input_id].output_shape.dim for input_id in node.input_ids
        ]
        if input_dims[0] != expected_dim or input_dims[1] != expected_dim:
            violations.append(
                "split2 branch restore mismatch: sibling split2 branches must "
                f"rejoin concat at half-width {expected_dim}, got {input_dims[0]} and {input_dims[1]}"
            )
            continue

        source_parent = graph.nodes.get(left_split.input_ids[0])
        if source_parent is None:
            continue
        if node.output_shape.dim != source_parent.output_shape.dim:
            violations.append(
                "split2 concat restore mismatch: sibling split2 concat must "
                f"restore source width {source_parent.output_shape.dim}, got {node.output_shape.dim}"
            )


# ── Per-op structural checks ────────────────────────────────────
# Extracted from find_graph_context_violations to keep functions <100 lines.


def _check_op_structural_rules(
    graph: ComputationGraph,
    nid: int,
    op_name: str,
    child_ops: set,
    parent_ops: set,
    children: Dict[int, List[int]],
    non_input_count: int,
    violations: List[str],
) -> None:
    """Check op-specific structural placement rules (appends to violations)."""
    if op_name == "local_window_attn":
        if not parent_ops & _LOCAL_WINDOW_VALID_PREDS:
            violations.append(
                "local_window_attn requires rmsnorm/layernorm predecessor"
            )
        if "linear_proj" not in child_ops:
            violations.append(
                "local_window_attn requires immediate linear_proj successor"
            )
        if not _has_descendant_op(graph, nid, {"add"}, children):
            violations.append(
                "local_window_attn must remain inside a residual attention block"
            )
    elif op_name in {"causal_mask", "sliding_window_mask"}:
        if not child_ops & _MASK_VALID_SUCCESSORS:
            violations.append(
                f"{op_name} must feed attention/projection, not stand alone"
            )
    elif op_name in {"sum_last", "mean_last", "max_last", "norm_last"}:
        if not child_ops & _REDUCTION_RESTORE_OPS:
            violations.append(
                f"{op_name} must rejoin through projection/merge, not stand alone"
            )
    elif op_name == "identity":
        if (
            graph.output_node is not None
            and graph.output_node.id == nid
            and non_input_count <= 2
        ):
            violations.append("identity cannot be the primary learning carrier")
    elif op_name in ("split2", "split3"):
        if not _has_descendant_op(graph, nid, {"concat", "add"}, children):
            violations.append(
                f"{op_name} must rejoin through concat or add before output"
            )
    elif op_name == "confidence_token_gate":
        if not _has_descendant_op(graph, nid, {"add"}, children):
            violations.append(
                "confidence_token_gate must sit inside a residual/routing block"
            )
    elif op_name == "depth_token_mask":
        if not _has_immediate_successor_op(
            children,
            graph,
            nid,
            {"linear_proj", "linear_proj_down", "rmsnorm", "layernorm"},
        ):
            violations.append(
                "depth_token_mask requires immediate projection/norm successor"
            )
        if not _has_descendant_op(graph, nid, {"add"}, children):
            violations.append(
                "depth_token_mask must remain inside a residual routing block"
            )
    elif op_name == "selective_scan":
        if not _has_immediate_predecessor_op(
            graph,
            nid,
            {"rmsnorm", "layernorm", "conv1d_seq", "silu"},
        ):
            violations.append(
                "selective_scan requires immediate norm/conv/silu predecessor context"
            )
        if not _has_descendant_op(graph, nid, {"add"}, children):
            violations.append(
                "selective_scan must remain inside a residual refinement block"
            )
    elif op_name == "hybrid_token_gate":
        if not _has_descendant_op(
            graph, nid, {"sparse_span_builder", "hybrid_sparse_router"}, children
        ):
            violations.append(
                "hybrid_token_gate must feed sparse_span_builder or hybrid_sparse_router"
            )
        if not _has_descendant_op(
            graph, nid, {"add", "calibrated_branch_merge"}, children
        ):
            violations.append(
                "hybrid_token_gate must remain inside a residual routing block"
            )
    elif op_name == "sparse_span_builder":
        if not _has_immediate_predecessor_op(graph, nid, {"hybrid_token_gate"}):
            violations.append(
                "sparse_span_builder requires immediate hybrid_token_gate predecessor"
            )
    elif op_name == "hybrid_sparse_router":
        if not _has_immediate_predecessor_op(
            graph, nid, {"sparse_span_builder", "hybrid_token_gate"}
        ):
            violations.append(
                "hybrid_sparse_router requires immediate sparse_span_builder or hybrid_token_gate predecessor"
            )
        if not _has_descendant_op(
            graph,
            nid,
            {"lane_conditioned_block", "add", "calibrated_branch_merge"},
            children,
        ):
            violations.append(
                "hybrid_sparse_router must remain inside a residual routing block"
            )
    elif op_name == "lane_conditioned_block":
        if not _has_immediate_predecessor_op(graph, nid, {"hybrid_sparse_router"}):
            violations.append(
                "lane_conditioned_block requires immediate hybrid_sparse_router predecessor"
            )
        if not _has_descendant_op(
            graph, nid, {"add", "calibrated_branch_merge"}, children
        ):
            violations.append(
                "lane_conditioned_block must rejoin through a residual merge"
            )
    elif op_name == "default_path":
        if not _has_immediate_successor_op(
            children, graph, nid, {"add", "calibrated_branch_merge"}
        ):
            violations.append(
                "default_path must feed a residual/calibrated branch merge"
            )
    elif op_name == "calibrated_branch_merge":
        if not parent_ops & {
            "default_path",
            "lane_conditioned_block",
            "hybrid_sparse_router",
            "calibrated_branch_merge",
        }:
            violations.append(
                "calibrated_branch_merge requires routed/default-path branch inputs"
            )
    elif op_name == "grade_mix":
        if not parent_ops & {
            "clifford_attention",
            "rotor_transform",
            "grade_select",
            "geometric_product",
        }:
            violations.append("grade_mix requires Clifford predecessor context")
        if not _has_descendant_op(
            graph,
            nid,
            {"linear_proj", "linear_proj_down", "add", "rmsnorm", "layernorm"},
            children,
        ):
            violations.append("grade_mix must feed projection/norm/residual context")
    elif op_name == "lif_neuron":
        if not _has_descendant_op(
            graph,
            nid,
            {"spike_rate_code", "stdp_attention", "tropical_gate"},
            children,
        ):
            violations.append("lif_neuron requires spiking successor context")
    elif op_name == "sparse_threshold":
        if not (child_ops & {"stdp_attention", "tropical_gate"}):
            violations.append(
                "sparse_threshold requires stdp_attention or tropical_gate successor"
            )
    elif op_name == "stdp_attention":
        if not (parent_ops & {"sparse_threshold", "spike_rate_code", "lif_neuron"}):
            violations.append("stdp_attention requires spiking predecessor context")
    elif op_name in {"geometric_product", "tropical_matmul"}:
        if not _has_ancestor_op(graph, nid, _LOCAL_WINDOW_VALID_PREDS):
            violations.append(f"{op_name} requires normalized predecessor context")
        if not _has_descendant_op(graph, nid, _RESTRICTED_LINEAR_SUCCESSORS, children):
            violations.append(f"{op_name} must feed projection/residual context")
    elif op_name in {"n_way_sparse_router", "sparse_bottleneck_moe"}:
        if not _has_ancestor_op(graph, nid, _LOCAL_WINDOW_VALID_PREDS):
            violations.append(
                "n_way_sparse_router requires normalized predecessor context"
            )
        if not child_ops & _ROUTER_VALID_SUCCESSORS:
            violations.append(
                "n_way_sparse_router must feed rmsnorm/layernorm/linear_proj, not stand alone"
            )
    elif op_name == "tropical_center":
        if not parent_ops & _TROPICAL_BRIDGE_PREDS:
            violations.append(
                "tropical_center requires tropical or normalized predecessor context"
            )
        if not _has_descendant_op(graph, nid, _RESTRICTED_LINEAR_SUCCESSORS, children):
            violations.append("tropical_center must feed projection/residual context")
    elif op_name == "early_exit":
        if not _has_descendant_op(graph, nid, {"add"}, children):
            violations.append("early_exit must sit inside a residual/routing block")
    else:
        _check_op_numerical_rules(graph, nid, op_name, children, violations)


def _check_op_numerical_rules(
    graph: ComputationGraph,
    nid: int,
    op_name: str,
    children: Dict[int, List[int]],
    violations: List[str],
) -> None:
    """Check numerically-sensitive op placement rules.

    Dispatches to elementwise or linear-algebra sub-checks.
    """
    if op_name in _ELEMENTWISE_NUMERICAL_OPS:
        _check_elementwise_numerical(graph, nid, op_name, children, violations)
    elif op_name in _LINALG_NUMERICAL_OPS:
        _check_linalg_numerical(graph, nid, op_name, children, violations)


_ELEMENTWISE_NUMERICAL_OPS = frozenset(
    {
        "reciprocal",
        "exp",
        "log",
        "div_safe",
        "sign_ste",
        "sub",
        "minimum",
        "cumsum",
    }
)

_LINALG_NUMERICAL_OPS = frozenset(
    {
        "matmul",
        "cosine_similarity",
        "outer_product",
        "tropical_matmul",
    }
)


def _check_elementwise_numerical(
    graph: ComputationGraph,
    nid: int,
    op_name: str,
    children: Dict[int, List[int]],
    violations: List[str],
) -> None:
    """Placement rules for elementwise numerically-sensitive ops."""
    if op_name == "reciprocal":
        if not _has_immediate_predecessor_op(
            graph, nid, {"rmsnorm", "layernorm", "sigmoid", "tanh"}
        ):
            violations.append(
                "reciprocal requires immediate bounded predecessor (norm/sigmoid/tanh)"
            )
    elif op_name == "exp":
        if not _has_immediate_predecessor_op(
            graph, nid, {"rmsnorm", "layernorm", "sigmoid", "tanh"}
        ):
            violations.append(
                "exp requires immediate bounded predecessor (norm/sigmoid/tanh)"
            )
        if not _has_immediate_successor_op(
            children, graph, nid, {"rmsnorm", "layernorm", "sigmoid", "tanh", "mul"}
        ):
            violations.append(
                "exp requires immediate stabilizer successor (norm/sigmoid/tanh/mul)"
            )
    elif op_name == "log":
        if not _has_immediate_predecessor_op(
            graph,
            nid,
            {"rmsnorm", "layernorm", "sigmoid", "softmax_last", "exp", "abs"},
        ):
            violations.append(
                "log requires immediate positive-bounded predecessor (norm/sigmoid/exp/abs)"
            )
        if not _has_immediate_successor_op(
            children,
            graph,
            nid,
            {"rmsnorm", "layernorm", "sigmoid", "tanh", "mul", "linear_proj"},
        ):
            violations.append(
                "log requires immediate stabilizer successor (norm/sigmoid/tanh/mul/linear_proj)"
            )
    elif op_name == "div_safe":
        if not _has_immediate_predecessor_op(
            graph, nid, {"rmsnorm", "layernorm", "sigmoid", "softmax_last"}
        ):
            violations.append(
                "div_safe requires immediate normalized predecessor (norm/sigmoid/softmax)"
            )
        if not _has_immediate_successor_op(
            children, graph, nid, _STABILIZER_SUCCESSORS | {"add"}
        ):
            violations.append(
                "div_safe requires immediate stabilizer/merge successor (proj/norm/mul/add)"
            )
    elif op_name == "sign_ste":
        if not _has_immediate_successor_op(
            children, graph, nid, {"mul", "linear_proj"}
        ):
            violations.append(
                "sign_ste must immediately feed mul/linear_proj (STE gradient flow)"
            )
    elif op_name == "sub":
        if not _has_immediate_successor_op(
            children, graph, nid, _STABILIZER_SUCCESSORS
        ):
            violations.append(
                "sub requires immediate stabilizer successor (proj/norm/mul)"
            )
    elif op_name == "minimum":
        if not _has_descendant_op(graph, nid, _RESTRICTED_LINEAR_SUCCESSORS, children):
            violations.append("minimum must feed projection/residual context")
    elif op_name == "cumsum":
        if not _has_immediate_successor_op(
            children, graph, nid, {"rmsnorm", "layernorm"}
        ):
            violations.append(
                "cumsum requires immediate norm successor (running sum grows unbounded)"
            )


def _check_linalg_numerical(
    graph: ComputationGraph,
    nid: int,
    op_name: str,
    children: Dict[int, List[int]],
    violations: List[str],
) -> None:
    """Placement rules for linear algebra ops (matmul, outer_product, etc.)."""
    if op_name == "matmul":
        if not _has_immediate_predecessor_op(
            graph, nid, {"linear_proj", "linear_proj_up", "linear_proj_down"}
        ):
            violations.append(
                "matmul requires immediate projection predecessor (68% fail without)"
            )
        if not _has_immediate_successor_op(
            children, graph, nid, _STABILIZER_SUCCESSORS | {"add"}
        ):
            violations.append(
                "matmul requires immediate stabilizer/merge successor (proj/norm/mul/add)"
            )
    elif op_name == "cosine_similarity":
        if not _has_immediate_predecessor_op(
            graph, nid, {"linear_proj", "linear_proj_up", "linear_proj_down"}
        ):
            violations.append(
                "cosine_similarity requires immediate projection predecessor (95% fail without)"
            )
    elif op_name == "outer_product":
        if not _has_immediate_successor_op(
            children, graph, nid, _STABILIZER_SUCCESSORS | {"add"}
        ):
            violations.append(
                "outer_product requires immediate stabilizer/merge successor (proj/norm/mul/add)"
            )
    elif op_name == "tropical_matmul":
        if not _has_immediate_successor_op(
            children,
            graph,
            nid,
            {"linear_proj", "linear_proj_down", "rmsnorm", "layernorm"},
        ):
            violations.append(
                "tropical_matmul requires immediate projection/norm successor"
            )


# ── Main graph validation ────────────────────────────────────────


def find_graph_context_violations(graph: ComputationGraph) -> List[str]:
    violations: List[str] = []
    children: Dict[int, List[int]] = _child_map(graph)
    non_input_nodes = [node for node in graph.nodes.values() if not node.is_input]
    non_input_count = len(non_input_nodes)

    for nid in graph.topological_order():
        node = graph.nodes[nid]
        if node.is_input:
            continue

        # Check forbidden predecessor/successor from CONTEXT_RULES registry
        rule = CONTEXT_RULES.get(node.op_name)
        if rule is not None:
            if rule.forbidden_predecessors:
                for pid in node.input_ids:
                    parent = graph.nodes.get(pid)
                    if (
                        parent is not None
                        and parent.op_name in rule.forbidden_predecessors
                    ):
                        violations.append(
                            f"Context rule: {parent.op_name} (id={pid}) -> {node.op_name} (id={nid}) is forbidden"
                        )
            if rule.forbidden_successors:
                for sid in children.get(nid, ()):
                    succ = graph.nodes.get(sid)
                    if succ is not None and succ.op_name in rule.forbidden_successors:
                        violations.append(
                            f"Context rule: {node.op_name} (id={nid}) -> {succ.op_name} (id={sid}) is forbidden"
                        )

        # Compute neighbor op sets for structural checks
        child_ops = {
            graph.nodes[cid].op_name
            for cid in children.get(nid, ())
            if cid in graph.nodes
        }
        parent_ops = {
            graph.nodes[parent_id].op_name
            for parent_id in node.input_ids
            if parent_id in graph.nodes and not graph.nodes[parent_id].is_input
        }

        # Delegate to per-op structural/numerical checks
        _check_op_structural_rules(
            graph,
            nid,
            node.op_name,
            child_ops,
            parent_ops,
            children,
            non_input_count,
            violations,
        )

    _check_split_branch_restore_contracts(graph, violations)

    return violations


def validate_context_rules(graph: ComputationGraph) -> Optional[str]:
    violations = find_graph_context_violations(graph)
    return violations[0] if violations else None


# ── Byte-safety enforcement ──────────────────────────────────────

# Lazily computed set of ops with byte_safe=False in the registry.
_BYTE_UNSAFE_OPS: FrozenSet[str] = frozenset(
    name for name, op in PRIMITIVE_REGISTRY.items() if not op.byte_safe
)


def find_byte_safety_violations(graph: ComputationGraph) -> List[str]:
    """Check that no byte-unsafe ops appear in the graph.

    Call this when the graph will run in native or quantized execution
    modes where token reordering/merging breaks tensor layout assumptions.
    """
    violations: List[str] = []
    for nid, node in graph.nodes.items():
        if node.is_input:
            continue
        if node.op_name in _BYTE_UNSAFE_OPS:
            violations.append(
                f"Byte-unsafe op '{node.op_name}' (node {nid}) is not "
                f"allowed in native/quantized execution mode"
            )
    return violations
