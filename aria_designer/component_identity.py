from __future__ import annotations

from typing import Any, Dict, Iterable, List, Set

# Base leaf to category mapping.
# Every key is a primitive leaf name; value is its canonical category/id.
_CANONICAL_MAP: Dict[str, str] = {
    "relu": "math/relu",
    "gelu": "math/gelu",
    "silu": "math/silu",
    "sigmoid": "math/sigmoid",
    "tanh": "math/tanh",
    "abs": "math/abs",
    "add": "math/add",
    "sub": "math/sub",
    "mul": "math/mul",
    "div_safe": "math/div_safe",
    "exp": "math/exp",
    "log": "math/log",
    "sqrt": "math/sqrt",
    "square": "math/square",
    "rmsnorm": "linear_algebra/rmsnorm",
    "layernorm": "normalization/layernorm",
    "linear_proj": "linear_algebra/linear_proj",
    "linear_proj_up": "linear_algebra/linear_proj_up",
    "linear_proj_down": "linear_algebra/linear_proj_down",
    "low_rank_proj": "math_space/low_rank_proj",
    "bottleneck_proj": "math_space/bottleneck_proj",
    "softmax_attention": "mixing/softmax_attention",
    "linear_attention": "mixing/linear_attention",
    "graph_attention": "mixing/graph_attention",
    "local_window_attn": "sequence/local_window_attn",
    "ultrametric_attention": "math_space/ultrametric_attention",
    "tropical_attention": "math_space/tropical_attention",
    "selective_scan": "linear_algebra/selective_scan",
    "conv1d_seq": "linear_algebra/conv1d_seq",
    "moe_topk": "channel_mixing/moe_topk",
    "token_difficulty_proj": "routing/token_difficulty_proj",
    "split2": "structural/split2",
    "split3": "structural/split3",
    "concat": "structural/concat",
    "conditional_dispatch": "structural/conditional_dispatch",
    "conditional_gather": "structural/conditional_gather",
    "input": "io/input",
    "graph_input": "io/input",
    "output": "io/output_head",
    "output_head": "io/output_head",
    "graph_output": "io/output",
    "conv_only": "mixing/conv_only",
    "swiglu_mlp": "channel_mixing/swiglu_mlp",
    "pq_embedding_moe_block": "channel_mixing/pq_embedding_moe_block",
    "confidence_token_gate": "routing/confidence_token_gate",
    "matmul": "linear_algebra/matmul",
    "topk_gate": "linear_algebra/topk_gate",
    "rope_rotate": "positional/rope_rotate",
    "identity": "identity",
    "identity_skip": "identity",
    "spectral_filter": "frequency/spectral_filter",
}


def _normalize_token(raw_id: str | None) -> str:
    return str(raw_id or "").strip().lower()


def _build_registry_maps(
    registry_ids: Iterable[str] | None,
) -> tuple[Set[str], Dict[str, str]]:
    registry_set = {
        _normalize_token(component_type)
        for component_type in (registry_ids or [])
        if _normalize_token(component_type)
    }
    leaf_to_canonical: Dict[str, str] = {}
    for component_type in sorted(registry_set):
        leaf = component_type.split("/")[-1]
        leaf_to_canonical.setdefault(leaf, component_type)
    return registry_set, leaf_to_canonical


def canonicalize_component_id(
    raw_id: str, registry_ids: Iterable[str] | None = None
) -> str:
    """Resolve a leaf name to a canonical category/id string."""
    token = _normalize_token(raw_id)
    if not token:
        return token

    registry_set, registry_leaf_map = _build_registry_maps(registry_ids)
    if token in registry_set:
        return token

    leaf = token.split("/")[-1] if "/" in token else token

    if leaf in registry_leaf_map:
        return registry_leaf_map[leaf]

    if leaf in _CANONICAL_MAP:
        canonical = _CANONICAL_MAP[leaf]
        if not registry_set or canonical in registry_set:
            return canonical

    if "/" in token:
        if token in registry_set:
            return token
        if leaf in registry_leaf_map:
            return registry_leaf_map[leaf]
        if leaf in _CANONICAL_MAP:
            return _CANONICAL_MAP[leaf]
        return token

    return _CANONICAL_MAP.get(token, token)


def canonicalize_workflow_ids(
    workflow: Dict[str, Any],
    registry_ids: Iterable[str] | None = None,
    *,
    preserve_raw_ids: bool = False,
) -> Dict[str, Any]:
    """In-place canonicalization of all node component_types in a workflow."""
    metadata = workflow.setdefault("metadata", {})
    original_ids: Dict[str, str] = {}
    nodes = workflow.get("nodes", [])
    for node in nodes:
        ct = node.get("component_type")
        if not ct:
            continue
        canonical = canonicalize_component_id(ct, registry_ids)
        if preserve_raw_ids and canonical != ct and node.get("id"):
            original_ids[str(node["id"])] = str(ct)
        node["component_type"] = canonical
    if original_ids:
        metadata["original_component_types"] = original_ids
    return workflow


def canonicalize_workflow(
    workflow: Dict[str, Any],
    registry_ids: Iterable[str] | None = None,
    *,
    preserve_raw_ids: bool = False,
) -> Dict[str, Any]:
    return canonicalize_workflow_ids(
        workflow,
        registry_ids=registry_ids,
        preserve_raw_ids=preserve_raw_ids,
    )


def collect_unresolved_component_ids(
    workflow: Dict[str, Any], registry_ids: Iterable[str]
) -> List[str]:
    """Find component_types that are not present in the live registry."""
    registry_set, _ = _build_registry_maps(registry_ids)
    unresolved = []
    for node in workflow.get("nodes", []):
        ct = _normalize_token(node.get("component_type", ""))
        if ct and ct not in registry_set:
            unresolved.append(ct)
    return unresolved


def component_leaf(component_type: str) -> str:
    """Extract the leaf portion of a component type string, lowercased."""
    token = str(component_type or "").strip()
    return token.rsplit("/", 1)[-1].lower() if token else ""


def discover_concepts(message: str) -> List[Dict[str, str]]:
    """Extract component concepts from natural language."""
    lower = message.lower()
    found: List[Dict[str, str]] = []
    seen_canonical: Set[str] = set()

    all_terms = sorted(_CANONICAL_MAP.keys(), key=len, reverse=True)

    for term in all_terms:
        if term in lower:
            canonical = canonicalize_component_id(term)
            if canonical not in seen_canonical:
                found.append({"concept": term, "component_type": canonical})
                seen_canonical.add(canonical)

    return found


__all__ = [
    "canonicalize_component_id",
    "canonicalize_workflow",
    "canonicalize_workflow_ids",
    "collect_unresolved_component_ids",
    "component_leaf",
    "discover_concepts",
]
