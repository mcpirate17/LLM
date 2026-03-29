from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Tuple

from .component_identity import canonicalize_component_id, component_leaf


_INTENT_KEYWORDS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "refine_compression",
        ("compression", "compress", "bottleneck", "low rank", "sparse"),
    ),
    (
        "improve_stability",
        ("stability", "stable", "gradient", "nan", "brittle", "explode"),
    ),
    (
        "expand_capacity",
        ("capacity", "depth", "layer", "width", "hidden", "benchmark", "beat"),
    ),
    (
        "preserve_fingerprint",
        ("fingerprint", "preserve", "minimal", "surgical", "incremental"),
    ),
)

_COMPONENT_GROUPS: Dict[str, Tuple[str, ...]] = {
    "io": ("io",),
    "math": ("activation", "math"),
    "normalization": ("normalization",),
    "mixing": ("mixing",),
    "routing": ("routing",),
    "structural": ("structural",),
    "linear_algebra": ("projection",),
    "math_space": ("projection", "compression"),
    "sequence": ("mixing",),
    "functional": ("mixing",),
}

_LEAF_GROUPS: Dict[str, Tuple[str, ...]] = {
    "relu": ("activation",),
    "gelu": ("activation",),
    "silu": ("activation",),
    "swish": ("activation",),
    "sigmoid": ("activation",),
    "tanh": ("activation",),
    "rmsnorm": ("normalization",),
    "layernorm": ("normalization",),
    "linear_proj": ("projection",),
    "linear_proj_up": ("projection",),
    "linear_proj_down": ("projection",),
    "low_rank_proj": ("projection", "compression"),
    "bottleneck_proj": ("projection", "compression"),
    "grouped_linear": ("projection", "compression"),
    "nm_sparse_linear": ("projection", "compression", "sparse"),
    "block_sparse_linear": ("projection", "compression", "sparse"),
    "semi_structured_2_4_linear": ("projection", "compression", "sparse"),
    "softmax_attention": ("mixing", "attention"),
    "linear_attention": ("mixing", "attention"),
    "graph_attention": ("mixing", "attention"),
    "local_window_attn": ("mixing", "attention"),
    "state_space": ("mixing",),
    "selective_scan": ("mixing",),
    "conv1d_seq": ("mixing",),
    "conv_only": ("mixing",),
    "moe_topk": ("routing",),
    "depth_token_mask": ("routing",),
    "relu_gated_moe": ("routing",),
    "topk_gate": ("routing",),
    "split2": ("structural", "routing"),
    "split3": ("structural", "routing"),
    "split4": ("structural", "routing"),
    "concat": ("structural",),
}

_INTENT_PRESETS: Dict[str, "IntentConstraints"] = {}


@dataclass(slots=True, frozen=True)
class IntentConstraints:
    intent_key: str
    allowed_mutations: Tuple[str, ...]
    preferred_component_groups: Tuple[str, ...]
    blocked_component_groups: Tuple[str, ...]
    replacement_components: Tuple[str, ...]
    target_param_names: Tuple[str, ...]
    param_direction: int
    max_param_delta: float
    max_nodes_touched: int
    min_retention_ratio: float
    preserve_novelty: bool


def component_groups(component_type: str) -> Tuple[str, ...]:
    token = canonicalize_component_id(str(component_type or "").strip().lower())
    if not token:
        return ()
    prefix, _, leaf = token.partition("/")
    groups = list(_COMPONENT_GROUPS.get(prefix, ()))
    for group in _LEAF_GROUPS.get(leaf or prefix, ()):
        if group not in groups:
            groups.append(group)
    if not groups and prefix:
        groups.append(prefix)
    return tuple(groups)


def compute_insertion_point(
    nodes: List[Dict[str, Any]] | None,
    edges: List[Dict[str, Any]] | None,
    component_type: str | None,
) -> Dict[str, str | None]:
    ordered_nodes = _topological_nodes(nodes or [], edges or [])
    if not ordered_nodes:
        return {"after_node_id": None, "before_node_id": None}

    target_groups = set(component_groups(str(component_type or "")))
    output_idx = _find_first_index(
        ordered_nodes, lambda node: "output" in _leaf_token(node)
    )
    if output_idx is None:
        output_idx = len(ordered_nodes)

    non_output = ordered_nodes[:output_idx] if output_idx >= 0 else ordered_nodes

    if "routing" in target_groups:
        before_idx = _find_first_index(
            non_output,
            lambda node: (
                "mixing" in set(component_groups(node.get("component_type", "")))
            ),
        )
        if before_idx is not None:
            return _between(ordered_nodes, before_idx - 1, before_idx)
        after_idx = _find_last_index(
            non_output,
            lambda node: (
                "projection" in set(component_groups(node.get("component_type", "")))
            ),
        )
        return _after_with_output_fallback(ordered_nodes, after_idx, output_idx)

    if "normalization" in target_groups:
        after_idx = _find_last_index(
            non_output,
            lambda node: bool(
                {"projection", "activation"}
                & set(component_groups(node.get("component_type", "")))
            ),
        )
        before_idx = _find_first_index_after(
            non_output,
            (after_idx if after_idx is not None else -1) + 1,
            lambda node: bool(
                {"mixing", "routing"}
                & set(component_groups(node.get("component_type", "")))
            ),
        )
        if before_idx is not None:
            return _between(ordered_nodes, after_idx, before_idx)
        return _after_with_output_fallback(ordered_nodes, after_idx, output_idx)

    if "activation" in target_groups:
        after_idx = _find_last_index(
            non_output,
            lambda node: (
                "projection" in set(component_groups(node.get("component_type", "")))
            ),
        )
        if after_idx is not None:
            before_idx = _find_first_index_after(
                non_output,
                after_idx + 1,
                lambda node: bool(
                    {"normalization", "mixing", "routing"}
                    & set(component_groups(node.get("component_type", "")))
                ),
            )
            if before_idx is not None:
                return _between(ordered_nodes, after_idx, before_idx)
            return _after_with_output_fallback(ordered_nodes, after_idx, output_idx)

    if "mixing" in target_groups:
        after_idx = _find_last_index(
            non_output,
            lambda node: bool(
                {"normalization", "projection", "activation"}
                & set(component_groups(node.get("component_type", "")))
            ),
        )
        return _after_with_output_fallback(ordered_nodes, after_idx, output_idx)

    if "projection" in target_groups or "compression" in target_groups:
        after_idx = _find_last_index(
            non_output,
            lambda node: (
                "input" in _leaf_token(node)
                or "projection" in set(component_groups(node.get("component_type", "")))
            ),
        )
        return _after_with_output_fallback(ordered_nodes, after_idx, output_idx)

    return _after_with_output_fallback(
        ordered_nodes, len(non_output) - 1 if non_output else None, output_idx
    )


def _topological_nodes(
    nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    nodes_by_id = {
        str(node.get("id")): node for node in nodes if node.get("id") is not None
    }
    indegree = {node_id: 0 for node_id in nodes_by_id}
    outgoing: Dict[str, List[str]] = {node_id: [] for node_id in nodes_by_id}
    order_index = {str(node.get("id")): idx for idx, node in enumerate(nodes)}

    for edge in edges:
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if source in outgoing and target in indegree:
            outgoing[source].append(target)
            indegree[target] += 1

    queue = sorted(
        (node_id for node_id, degree in indegree.items() if degree == 0),
        key=lambda node_id: order_index[node_id],
    )
    ordered: List[Dict[str, Any]] = []
    cursor = 0
    while cursor < len(queue):
        node_id = queue[cursor]
        cursor += 1
        ordered.append(nodes_by_id[node_id])
        for target in sorted(
            outgoing.get(node_id, ()), key=lambda item: order_index.get(item, 0)
        ):
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)

    if len(ordered) != len(nodes_by_id):
        seen = {str(node.get("id")) for node in ordered}
        ordered.extend(node for node in nodes if str(node.get("id")) not in seen)
    return ordered


def _leaf_token(node: Dict[str, Any]) -> str:
    return component_leaf(node.get("component_type") or "")


def _find_first_index(nodes: List[Dict[str, Any]], predicate) -> int | None:
    for idx, node in enumerate(nodes):
        if predicate(node):
            return idx
    return None


def _find_first_index_after(
    nodes: List[Dict[str, Any]], start: int, predicate
) -> int | None:
    for idx in range(max(0, start), len(nodes)):
        if predicate(nodes[idx]):
            return idx
    return None


def _find_last_index(nodes: List[Dict[str, Any]], predicate) -> int | None:
    for idx in range(len(nodes) - 1, -1, -1):
        if predicate(nodes[idx]):
            return idx
    return None


def _between(
    nodes: List[Dict[str, Any]], after_idx: int | None, before_idx: int | None
) -> Dict[str, str | None]:
    after_node_id = (
        str(nodes[after_idx].get("id"))
        if after_idx is not None and after_idx >= 0
        else None
    )
    before_node_id = (
        str(nodes[before_idx].get("id"))
        if before_idx is not None and before_idx < len(nodes)
        else None
    )
    return {"after_node_id": after_node_id, "before_node_id": before_node_id}


def _after_with_output_fallback(
    nodes: List[Dict[str, Any]],
    after_idx: int | None,
    output_idx: int | None,
) -> Dict[str, str | None]:
    before_idx = (
        output_idx if output_idx is not None and output_idx < len(nodes) else None
    )
    return _between(nodes, after_idx, before_idx)


def parse_intent_constraints(
    intent: str | None,
    parent_scores: Dict[str, Any] | None = None,
) -> IntentConstraints:
    preset_key = _resolve_intent_key(intent)
    constraints = _INTENT_PRESETS[preset_key]
    return _apply_parent_guardrails(constraints, parent_scores or {})


def _resolve_intent_key(intent: str | None) -> str:
    text = str(intent or "").strip().lower()
    if not text:
        return "balanced_refine"
    for intent_key, keywords in _INTENT_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return intent_key
    return "balanced_refine"


def _apply_parent_guardrails(
    constraints: IntentConstraints,
    parent_scores: Dict[str, Any],
) -> IntentConstraints:
    tier = str(parent_scores.get("tier") or "").lower()
    composite = float(parent_scores.get("composite_score") or 0.0)
    high_tier = (
        tier in {"investigation", "validation", "breakthrough"} or composite >= 100.0
    )
    if not high_tier:
        return constraints
    narrowed = tuple(m for m in constraints.allowed_mutations if m != "remove_node")
    return replace(
        constraints,
        allowed_mutations=narrowed or constraints.allowed_mutations,
        max_param_delta=min(constraints.max_param_delta, 0.12),
        max_nodes_touched=min(constraints.max_nodes_touched, 2),
        min_retention_ratio=max(constraints.min_retention_ratio, 0.85),
        preserve_novelty=True,
    )


_INTENT_PRESETS.update(
    {
        "balanced_refine": IntentConstraints(
            intent_key="balanced_refine",
            allowed_mutations=("mutate_param", "replace_activation", "add_layer"),
            preferred_component_groups=("projection", "activation", "mixing"),
            blocked_component_groups=("io",),
            replacement_components=("math/silu", "math/gelu", "linear_algebra/rmsnorm"),
            target_param_names=(
                "out_dim",
                "hidden_dim",
                "ff_dim",
                "rank",
                "heads",
                "expansion",
            ),
            param_direction=1,
            max_param_delta=0.18,
            max_nodes_touched=2,
            min_retention_ratio=0.75,
            preserve_novelty=False,
        ),
        "refine_compression": IntentConstraints(
            intent_key="refine_compression",
            allowed_mutations=("mutate_param", "replace_activation"),
            preferred_component_groups=("projection", "compression", "normalization"),
            blocked_component_groups=("io", "routing"),
            replacement_components=("linear_algebra/rmsnorm", "math/silu"),
            target_param_names=("out_dim", "hidden_dim", "ff_dim", "rank", "heads"),
            param_direction=-1,
            max_param_delta=0.22,
            max_nodes_touched=2,
            min_retention_ratio=0.82,
            preserve_novelty=True,
        ),
        "improve_stability": IntentConstraints(
            intent_key="improve_stability",
            allowed_mutations=("replace_activation", "mutate_param"),
            preferred_component_groups=("activation", "normalization", "projection"),
            blocked_component_groups=("io", "routing"),
            replacement_components=("linear_algebra/rmsnorm", "math/silu", "math/gelu"),
            target_param_names=("out_dim", "hidden_dim", "ff_dim", "heads"),
            param_direction=-1,
            max_param_delta=0.12,
            max_nodes_touched=2,
            min_retention_ratio=0.9,
            preserve_novelty=True,
        ),
        "expand_capacity": IntentConstraints(
            intent_key="expand_capacity",
            allowed_mutations=("mutate_param", "add_layer", "replace_activation"),
            preferred_component_groups=("projection", "mixing", "activation"),
            blocked_component_groups=("io",),
            replacement_components=("math/silu", "math/gelu"),
            target_param_names=(
                "out_dim",
                "hidden_dim",
                "ff_dim",
                "heads",
                "expansion",
            ),
            param_direction=1,
            max_param_delta=0.25,
            max_nodes_touched=3,
            min_retention_ratio=0.8,
            preserve_novelty=False,
        ),
        "preserve_fingerprint": IntentConstraints(
            intent_key="preserve_fingerprint",
            allowed_mutations=("mutate_param", "replace_activation"),
            preferred_component_groups=("projection", "activation", "normalization"),
            blocked_component_groups=("io", "routing", "structural"),
            replacement_components=("math/silu", "math/gelu"),
            target_param_names=("out_dim", "hidden_dim", "ff_dim", "rank"),
            param_direction=0,
            max_param_delta=0.08,
            max_nodes_touched=1,
            min_retention_ratio=0.92,
            preserve_novelty=True,
        ),
    }
)
