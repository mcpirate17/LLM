"""
Component Context Rules — Enforcement Layer

Encodes placement constraints for ops that have been audited as
context-sensitive. These rules prevent the grammar from generating
graphs where ops appear in invalid predecessor/successor chains,
and classify ops by search-mode so niche/restricted ops are not
sprayed into default search blindly.

Sources:
  - artifacts/component_context_rules.md
  - artifacts/component_context_rules.json
  - artifacts/low_s1_root_cause_audit.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, Iterable, List, Optional

from .graph import ComputationGraph
from .motifs import MOTIFS_BY_CLASS, Motif


# ── Search-mode classification ────────────────────────────────────


class SearchMode(Enum):
    __slots__ = ()
    GENERAL = "general"
    RESTRICTED = "restricted"
    NICHE = "niche"


# ── Per-op context rule ───────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class ContextRule:
    """Machine-actionable placement constraint for a single op."""

    search_mode: SearchMode
    # Concrete op names that are forbidden as direct predecessors.
    forbidden_predecessors: FrozenSet[str] = field(default_factory=frozenset)
    # Concrete op names that are forbidden as direct successors.
    forbidden_successors: FrozenSet[str] = field(default_factory=frozenset)
    # If True, the op must sit inside a residual bypass (add consuming same input).
    requires_residual_context: bool = False


CONTEXT_CLASS_GENERAL = "general-use"
CONTEXT_CLASS_RESTRICTED = "restricted-use"
CONTEXT_CLASS_STRUCTURAL = "structural"
CONTEXT_CLASS_REHAB = "rehab"


# ── Shared forbidden-predecessor sets ─────────────────────────────
# Many ops share the same "not after reduce" constraint.

_REDUCE_OPS: FrozenSet[str] = frozenset(
    {
        "sum_last",
        "mean_last",
        "max_last",
        "norm_last",
    }
)

_STRUCTURAL_SPLIT_OPS: FrozenSet[str] = frozenset(
    {
        "split2",
        "split3",
    }
)

# ── Context Rules Registry ────────────────────────────────────────
# Only ops with non-trivial constraints are listed.
# Ops not listed here are treated as general-use with no extra constraints.

CONTEXT_RULES: Dict[str, ContextRule] = {
    # ── Attention/position ops: promoted to GENERAL with dedicated templates ──
    "local_window_attn": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}),
        requires_residual_context=True,
    ),
    "sliding_window_mask": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=frozenset(),
        forbidden_successors=frozenset({"output_head"}),
        requires_residual_context=True,
    ),
    "causal_mask": ContextRule(
        search_mode=SearchMode.RESTRICTED,
        forbidden_predecessors=frozenset(),
        forbidden_successors=frozenset({"output_head"}),
        requires_residual_context=True,
    ),
    # ── Restricted-use: structural ops ──────────────────────────────
    "split2": ContextRule(
        search_mode=SearchMode.RESTRICTED,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}),
    ),
    "split3": ContextRule(
        search_mode=SearchMode.RESTRICTED,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}),
    ),
    "concat": ContextRule(
        search_mode=SearchMode.RESTRICTED,
        forbidden_predecessors=frozenset(),
        forbidden_successors=frozenset({"output_head"}),
    ),
    "identity": ContextRule(
        search_mode=SearchMode.RESTRICTED,
        forbidden_predecessors=frozenset(),
        forbidden_successors=frozenset({"output_head"}),
    ),
    # ── Restricted-use: reduce ops (must not chain or feed output directly) ─
    "norm_last": ContextRule(
        search_mode=SearchMode.RESTRICTED,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}) | _STRUCTURAL_SPLIT_OPS,
    ),
    "max_last": ContextRule(
        search_mode=SearchMode.RESTRICTED,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}) | _STRUCTURAL_SPLIT_OPS,
    ),
    "sum_last": ContextRule(
        search_mode=SearchMode.RESTRICTED,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}) | _STRUCTURAL_SPLIT_OPS,
    ),
    "mean_last": ContextRule(
        search_mode=SearchMode.RESTRICTED,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}) | _STRUCTURAL_SPLIT_OPS,
    ),
    # ── General-use ops with forbidden predecessor constraints ──────
    "graph_attention": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=frozenset(),
        forbidden_successors=frozenset({"output_head"}),
        requires_residual_context=True,
    ),
    "softmax_attention": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=frozenset(),
        forbidden_successors=frozenset({"output_head"}),
        requires_residual_context=True,
    ),
    "linear_attention": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=frozenset(),
        forbidden_successors=frozenset({"output_head"}),
        requires_residual_context=True,
    ),
    "state_space": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=_STRUCTURAL_SPLIT_OPS,
    ),
    "diff_attention": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=_STRUCTURAL_SPLIT_OPS,
    ),
    "gated_delta": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=_STRUCTURAL_SPLIT_OPS,
    ),
    "fused_linear_gelu": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=_STRUCTURAL_SPLIT_OPS,
    ),
    "rwkv_time_mixing": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=_STRUCTURAL_SPLIT_OPS,
    ),
    "integral_kernel": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=_STRUCTURAL_SPLIT_OPS,
    ),
    "fixed_point_iter": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=_STRUCTURAL_SPLIT_OPS,
    ),
    # ── Promoted from NICHE to GENERAL: have dedicated templates + MATH_SPACE_RULES ──
    "tropical_center": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=frozenset(),
        forbidden_successors=frozenset({"output_head"}),
    ),
    "tropical_matmul": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=_STRUCTURAL_SPLIT_OPS,
    ),
    "lif_neuron": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=_STRUCTURAL_SPLIT_OPS,
    ),
    "sparse_threshold": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=frozenset(),
        forbidden_successors=frozenset({"output_head"}),
    ),
    "stdp_attention": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=frozenset(),
        forbidden_successors=frozenset({"output_head"}),
    ),
    "n_way_sparse_router": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=_STRUCTURAL_SPLIT_OPS,
    ),
    "early_exit": ContextRule(
        search_mode=SearchMode.RESTRICTED,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=_STRUCTURAL_SPLIT_OPS,
        requires_residual_context=True,
    ),
    "cascade": ContextRule(
        search_mode=SearchMode.RESTRICTED,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=_STRUCTURAL_SPLIT_OPS,
    ),
    "hyp_linear": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=frozenset(),
        forbidden_successors=frozenset({"output_head"}),
    ),
    "hyp_tangent_nonlinear": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=frozenset(),
        forbidden_successors=frozenset({"output_head"}),
    ),
    # ── Gradient-sensitive ops: require bounded input, must feed projection ──
    "div_safe": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}) | _STRUCTURAL_SPLIT_OPS,
    ),
    "log": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}) | _STRUCTURAL_SPLIT_OPS,
    ),
    "reciprocal": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}) | _STRUCTURAL_SPLIT_OPS,
    ),
    # ── Elementwise ops: require normalized input, must feed projection ──
    "minimum": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}) | _STRUCTURAL_SPLIT_OPS,
    ),
    "sub": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}) | _STRUCTURAL_SPLIT_OPS,
    ),
    "exp": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}) | _STRUCTURAL_SPLIT_OPS,
    ),
    "sign_ste": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}) | _STRUCTURAL_SPLIT_OPS,
    ),
    # ── Still NICHE: no dedicated template, needs further investigation ──
    "geometric_product": ContextRule(
        search_mode=SearchMode.NICHE,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=_STRUCTURAL_SPLIT_OPS,
    ),
    "progressive_compression_gate": ContextRule(
        search_mode=SearchMode.NICHE,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=_STRUCTURAL_SPLIT_OPS,
    ),
}

_OP_CONTEXT_CLASS: Dict[str, str] = {
    # Structural ops: scaffolding, not standalone learners
    "causal_mask": CONTEXT_CLASS_STRUCTURAL,
    "split2": CONTEXT_CLASS_STRUCTURAL,
    "split3": CONTEXT_CLASS_STRUCTURAL,
    "concat": CONTEXT_CLASS_STRUCTURAL,
    "identity": CONTEXT_CLASS_STRUCTURAL,
    # Restricted: template-confined (UNSAFE role)
    "geometric_product": CONTEXT_CLASS_RESTRICTED,
    # Phase 3: promoted to GENERAL — have proven templates and MATH_SPACE_RULES
    # (previously RESTRICTED, now with dedicated template paths)
    # lif_neuron, sparse_threshold, stdp_attention → spiking_residual_block
    # tropical_center, tropical_matmul → tropical_residual / tropical_matmul_block
    # n_way_sparse_router → n_way_moe_block
    # early_exit → cascaded_early_exit
}

_MOTIF_TEMPLATE_ALLOWLIST: Dict[str, FrozenSet[str]] = {
    "attn_causal_mask": frozenset(
        {
            "residual_block",
            "transformer_block",
            "hybrid_parallel",
            "gated_residual",
            "difficulty_routed_block",
            "three_lane_adaptive",
        }
    ),
    "attn_local_window": frozenset(
        {
            "residual_block",
            "transformer_block",
            "hybrid_parallel",
            "gated_residual",
            "difficulty_routed_block",
            "three_lane_adaptive",
            "local_attention_block",
        }
    ),
    "attn_sliding_window": frozenset(
        {
            "residual_block",
            "transformer_block",
            "hybrid_parallel",
            "gated_residual",
            "difficulty_routed_block",
            "three_lane_adaptive",
            "windowed_attention",
        }
    ),
    "attn_graph": frozenset(
        {
            "residual_block",
            "transformer_block",
            "hybrid_parallel",
            "gated_residual",
            "graph_attention_block",
        }
    ),
    "ssm_state_space": frozenset(
        {
            "residual_block",
            "transformer_block",
            "gated_residual",
            "state_space_block",
        }
    ),
    "mix_integral_kernel": frozenset(
        {
            "residual_block",
            "gated_residual",
            "integral_kernel_block",
        }
    ),
    "ffn_fused_gelu": frozenset(
        {
            "residual_block",
            "transformer_block",
            "gated_residual",
            "fused_gelu_ffn",
        }
    ),
    "reduce_sum": frozenset({"parallel_split", "three_way_split", "reduce_attend"}),
    "reduce_mean": frozenset({"parallel_split", "three_way_split", "reduce_attend"}),
    "reduce_max": frozenset({"parallel_split", "three_way_split", "reduce_attend"}),
    "reduce_norm": frozenset({"parallel_split", "three_way_split", "reduce_attend"}),
    "route_identity": frozenset(
        {
            "difficulty_routed_block",
            "three_lane_adaptive",
            "cascaded_early_exit",
            "recursive_depth_router",
            "conditional_compute",
        }
    ),
    "route_early_exit": frozenset(
        {
            "difficulty_routed_block",
            "three_lane_adaptive",
            "cascaded_early_exit",
            "recursive_depth_router",
            "conditional_compute",
        }
    ),
    "spiking_lif_rate": frozenset(
        {
            "residual_block",
            "transformer_block",
            "gated_residual",
            "spiking_residual_block",
            "spiking_moe_block",
        }
    ),
    "spiking_threshold_stdp": frozenset(
        {
            "residual_block",
            "transformer_block",
            "gated_residual",
            "spiking_residual_block",
            "spiking_moe_block",
        }
    ),
    "spiking_tropical_gate": frozenset(
        {"spiking_moe_block", "residual_block", "gated_residual"}
    ),
    "spiking_rate_tropical_gate": frozenset(
        {"spiking_moe_block", "residual_block", "gated_residual"}
    ),
    "spiking_threshold_tropical_gate": frozenset(
        {"spiking_moe_block", "residual_block", "gated_residual"}
    ),
    "tropical_center_norm": frozenset(
        {
            "residual_block",
            "transformer_block",
            "gated_residual",
            "tropical_center_block",
        }
    ),
    "hyperbolic_residual_bridge": frozenset(
        {"residual_block", "hyperbolic_bridge_block"}
    ),
    "poincare_add_bridge": frozenset({"residual_block", "hyperbolic_bridge_block"}),
    "conv_only_block": frozenset(
        {"residual_block", "gated_residual", "conv_residual_block"}
    ),
    "attn_gated_delta": frozenset(
        {"residual_block", "transformer_block", "recurrent_delta_block"}
    ),
    "mix_fixed_point": frozenset({"residual_block", "iterative_refinement"}),
    "n_way_routing": frozenset({"moe", "n_way_moe_block"}),
    "proj_bottleneck": frozenset({"residual_block", "bottleneck", "gated_residual"}),
    "proj_low_rank": frozenset({"residual_block", "bottleneck", "gated_residual"}),
    "route_adaptive_recursion": frozenset(
        {"recursive_depth_router", "difficulty_routed_block", "conditional_compute"}
    ),
}

_CONTEXT_CLASS_PRIORS = {
    CONTEXT_CLASS_GENERAL: 1.0,
    CONTEXT_CLASS_RESTRICTED: 0.55,
    CONTEXT_CLASS_STRUCTURAL: 0.30,
    CONTEXT_CLASS_REHAB: 0.15,
}

_LOCAL_WINDOW_VALID_PREDS = frozenset({"rmsnorm", "layernorm"})
_MASK_VALID_SUCCESSORS = frozenset(
    {"softmax_attention", "linear_attention", "linear_proj"}
)
_REDUCTION_RESTORE_OPS = frozenset({"linear_proj", "linear_proj_up", "concat", "add"})
_TROPICAL_BRIDGE_PREDS = frozenset(
    {"rmsnorm", "layernorm", "tropical_attention", "tropical_gate"}
)
_RESTRICTED_LINEAR_SUCCESSORS = frozenset({"linear_proj", "linear_proj_up", "add"})
_ROUTER_VALID_SUCCESSORS = frozenset({"rmsnorm", "layernorm", "linear_proj"})
# Ops that stabilize unbounded output: projection rescales, norm bounds, mul gates
_STABILIZER_SUCCESSORS = frozenset(
    {
        "linear_proj",
        "linear_proj_down",
        "linear_proj_up",
        "rmsnorm",
        "layernorm",
        "mul",
    }
)


# ── Derived sets (frozen at import time) ──────────────────────────

NICHE_OPS: FrozenSet[str] = frozenset(
    name for name, rule in CONTEXT_RULES.items() if rule.search_mode == SearchMode.NICHE
)

RESTRICTED_OPS: FrozenSet[str] = frozenset(
    name
    for name, rule in CONTEXT_RULES.items()
    if rule.search_mode == SearchMode.RESTRICTED
)

STRUCTURAL_OPS: FrozenSet[str] = frozenset(
    {
        "identity",
        "split2",
        "split3",
        "concat",
    }
)

# Ops exempt from per-op S1 attribution: scaffolding (splits, concat,
# identity), positional masks (causal_mask, sliding_window_mask),
# dimension-reduction ops (norm_last, sum_last, mean_last, max_last),
# and parameter-free elementwise/sequence transforms (minimum, maximum,
# sub, cumprod_safe, cumsum). None have learnable parameters — they
# should not be judged as standalone learning carriers.
S1_EXEMPT_OPS: FrozenSet[str] = frozenset(
    {
        "identity",
        "split2",
        "split3",
        "concat",
        "causal_mask",
        "sliding_window_mask",
        "norm_last",
        "sum_last",
        "mean_last",
        "max_last",
        # Parameter-free elementwise transforms
        "minimum",
        "maximum",
        "sub",
        # Parameter-free sequence transforms
        "cumprod_safe",
        "cumsum",
    }
)

# Ops that require residual bypass context per audit.
REQUIRES_RESIDUAL_CONTEXT: FrozenSet[str] = frozenset(
    name for name, rule in CONTEXT_RULES.items() if rule.requires_residual_context
)


# ── Query helpers ─────────────────────────────────────────────────


def get_search_mode(op_name: str) -> SearchMode:
    """Return the search-mode classification for an op."""
    rule = CONTEXT_RULES.get(op_name)
    if rule is not None:
        return rule.search_mode
    return SearchMode.GENERAL


def is_niche(op_name: str) -> bool:
    return op_name in NICHE_OPS


def is_restricted(op_name: str) -> bool:
    return op_name in RESTRICTED_OPS


def is_structural(op_name: str) -> bool:
    return op_name in STRUCTURAL_OPS


def get_op_context_class(op_name: str) -> str:
    return _OP_CONTEXT_CLASS.get(op_name, CONTEXT_CLASS_GENERAL)


def _motif_context_class(motif: Motif) -> str:
    classes = {get_op_context_class(step.op_name) for step in motif.steps}
    if CONTEXT_CLASS_REHAB in classes:
        return CONTEXT_CLASS_REHAB
    if CONTEXT_CLASS_STRUCTURAL in classes:
        return CONTEXT_CLASS_STRUCTURAL
    if CONTEXT_CLASS_RESTRICTED in classes:
        return CONTEXT_CLASS_RESTRICTED
    return CONTEXT_CLASS_GENERAL


def apply_context_rule_priors(
    motif_weights: Optional[Dict[str, float]],
    exploration_targets: Optional[Iterable[str]] = None,
) -> Optional[Dict[str, float]]:
    weights = dict(motif_weights) if motif_weights else {}
    targeted_ops = set(exploration_targets or ())
    seen_motifs: set[str] = set()
    for motifs in MOTIFS_BY_CLASS.values():
        for motif in motifs:
            if motif.name in seen_motifs:
                continue
            seen_motifs.add(motif.name)
            motif_ops = {step.op_name for step in motif.steps}
            if motif_ops & targeted_ops:
                continue
            factor = _CONTEXT_CLASS_PRIORS[_motif_context_class(motif)]
            if motif.name == "attn_local_window":
                factor *= 0.7
            base = weights.get(motif.name, motif.lift)
            weights[motif.name] = max(0.05, base * factor)
    return weights or None


def motif_allowed_in_template(motif: Motif, template_name: Optional[str]) -> bool:
    if template_name is None:
        return True
    allowed_templates = _MOTIF_TEMPLATE_ALLOWLIST.get(motif.name)
    if allowed_templates is None:
        return True
    return template_name in allowed_templates


def _child_map(graph: ComputationGraph) -> Dict[int, List[int]]:
    children = {nid: [] for nid in graph.nodes}
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
    allowed = set(allowed_ops)
    queue = list(children.get(start_id, ()))
    seen = set()
    while queue:
        nid = queue.pop()
        if nid in seen:
            continue
        seen.add(nid)
        node = graph.nodes.get(nid)
        if node is None:
            continue
        if node.op_name in allowed:
            return True
        queue.extend(children.get(nid, ()))
    return False


def _has_ancestor_op(
    graph: ComputationGraph,
    start_id: int,
    allowed_ops: Iterable[str],
) -> bool:
    allowed = set(allowed_ops)
    start_node = graph.nodes.get(start_id)
    if start_node is None:
        return False
    queue = list(start_node.input_ids)
    seen = set()
    while queue:
        nid = queue.pop()
        if nid in seen:
            continue
        seen.add(nid)
        node = graph.nodes.get(nid)
        if node is None:
            continue
        if node.op_name in allowed:
            return True
        queue.extend(node.input_ids)
    return False


def _has_immediate_predecessor_op(
    graph: ComputationGraph,
    node_id: int,
    allowed_ops: Iterable[str],
) -> bool:
    """Check if any direct parent of node_id has op_name in allowed_ops."""
    allowed = set(allowed_ops)
    node = graph.nodes.get(node_id)
    if node is None:
        return False
    for pid in node.input_ids:
        parent = graph.nodes.get(pid)
        if parent is not None and parent.op_name in allowed:
            return True
    return False


def _has_immediate_successor_op(
    children: Dict[int, List[int]],
    graph: ComputationGraph,
    node_id: int,
    allowed_ops: Iterable[str],
) -> bool:
    """Check if any direct child of node_id has op_name in allowed_ops."""
    allowed = set(allowed_ops)
    for cid in children.get(node_id, ()):
        child = graph.nodes.get(cid)
        if child is not None and child.op_name in allowed:
            return True
    return False


def find_graph_context_violations(graph: ComputationGraph) -> List[str]:
    violations: List[str] = []
    successors: Dict[int, List[int]] = {nid: [] for nid in graph.nodes}
    for nid, node in graph.nodes.items():
        if node.is_input:
            continue
        for pid in node.input_ids:
            if pid in successors:
                successors[pid].append(nid)

    children = _child_map(graph)
    non_input_nodes = [node for node in graph.nodes.values() if not node.is_input]

    for nid in graph.topological_order():
        node = graph.nodes[nid]
        if node.is_input:
            continue

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
                for sid in successors.get(nid, ()):
                    succ = graph.nodes.get(sid)
                    if succ is not None and succ.op_name in rule.forbidden_successors:
                        violations.append(
                            f"Context rule: {node.op_name} (id={nid}) -> {succ.op_name} (id={sid}) is forbidden"
                        )

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
        if node.op_name == "local_window_attn":
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
        elif node.op_name in {"causal_mask", "sliding_window_mask"}:
            if not child_ops & _MASK_VALID_SUCCESSORS:
                violations.append(
                    f"{node.op_name} must feed attention/projection, not stand alone"
                )
        elif node.op_name in {"sum_last", "mean_last", "max_last", "norm_last"}:
            if not child_ops & _REDUCTION_RESTORE_OPS:
                violations.append(
                    f"{node.op_name} must rejoin through projection/merge, not stand alone"
                )
        elif node.op_name == "identity":
            if (
                graph.output_node is not None
                and graph.output_node.id == nid
                and len(non_input_nodes) <= 2
            ):
                violations.append("identity cannot be the primary learning carrier")
        elif node.op_name == "split3":
            if not _has_descendant_op(graph, nid, {"concat", "add"}, children):
                violations.append(
                    "split3 must rejoin through concat or add before output"
                )
        elif node.op_name == "lif_neuron":
            if not _has_descendant_op(
                graph,
                nid,
                {"spike_rate_code", "stdp_attention", "tropical_gate"},
                children,
            ):
                violations.append("lif_neuron requires spiking successor context")
        elif node.op_name == "sparse_threshold":
            if not (child_ops & {"stdp_attention", "tropical_gate"}):
                violations.append(
                    "sparse_threshold requires stdp_attention or tropical_gate successor"
                )
        elif node.op_name == "stdp_attention":
            if not (parent_ops & {"sparse_threshold", "spike_rate_code", "lif_neuron"}):
                violations.append("stdp_attention requires spiking predecessor context")
        elif node.op_name in {"geometric_product", "tropical_matmul"}:
            if not _has_ancestor_op(graph, nid, _LOCAL_WINDOW_VALID_PREDS):
                violations.append(
                    f"{node.op_name} requires normalized predecessor context"
                )
            if not _has_descendant_op(
                graph, nid, _RESTRICTED_LINEAR_SUCCESSORS, children
            ):
                violations.append(
                    f"{node.op_name} must feed projection/residual context"
                )
        elif node.op_name == "n_way_sparse_router":
            if not _has_ancestor_op(graph, nid, _LOCAL_WINDOW_VALID_PREDS):
                violations.append(
                    "n_way_sparse_router requires normalized predecessor context"
                )
            if not child_ops & _ROUTER_VALID_SUCCESSORS:
                violations.append(
                    "n_way_sparse_router must feed rmsnorm/layernorm/linear_proj, not stand alone"
                )
        elif node.op_name == "tropical_center":
            if not parent_ops & _TROPICAL_BRIDGE_PREDS:
                violations.append(
                    "tropical_center requires tropical or normalized predecessor context"
                )
            if not _has_descendant_op(
                graph, nid, _RESTRICTED_LINEAR_SUCCESSORS, children
            ):
                violations.append(
                    "tropical_center must feed projection/residual context"
                )
        elif node.op_name == "early_exit":
            if not _has_descendant_op(graph, nid, {"add"}, children):
                violations.append("early_exit must sit inside a residual/routing block")
        elif node.op_name == "reciprocal":
            if not _has_immediate_predecessor_op(
                graph, nid, {"rmsnorm", "layernorm", "sigmoid", "tanh"}
            ):
                violations.append(
                    "reciprocal requires immediate bounded predecessor (norm/sigmoid/tanh)"
                )
        elif node.op_name == "exp":
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
        elif node.op_name == "log":
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
        elif node.op_name == "div_safe":
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
        elif node.op_name == "sign_ste":
            if not _has_immediate_successor_op(
                children, graph, nid, {"mul", "linear_proj"}
            ):
                violations.append(
                    "sign_ste must immediately feed mul/linear_proj (STE gradient flow)"
                )
        elif node.op_name == "sub":
            if not _has_immediate_successor_op(
                children, graph, nid, _STABILIZER_SUCCESSORS
            ):
                violations.append(
                    "sub requires immediate stabilizer successor (proj/norm/mul)"
                )
        elif node.op_name == "minimum":
            if not _has_descendant_op(
                graph, nid, _RESTRICTED_LINEAR_SUCCESSORS, children
            ):
                violations.append("minimum must feed projection/residual context")
        elif node.op_name == "cumsum":
            if not _has_immediate_successor_op(
                children, graph, nid, {"rmsnorm", "layernorm"}
            ):
                violations.append(
                    "cumsum requires immediate norm successor (running sum grows unbounded)"
                )
        elif node.op_name == "outer_product":
            if not _has_immediate_successor_op(
                children, graph, nid, _STABILIZER_SUCCESSORS | {"add"}
            ):
                violations.append(
                    "outer_product requires immediate stabilizer/merge successor (proj/norm/mul/add)"
                )
        elif node.op_name == "tropical_matmul":
            if not _has_immediate_successor_op(
                children,
                graph,
                nid,
                {"linear_proj", "linear_proj_down", "rmsnorm", "layernorm"},
            ):
                violations.append(
                    "tropical_matmul requires immediate projection/norm successor"
                )

    return violations


# ── Graph validation ──────────────────────────────────────────────


def validate_context_rules(graph: ComputationGraph) -> Optional[str]:
    violations = find_graph_context_violations(graph)
    return violations[0] if violations else None
