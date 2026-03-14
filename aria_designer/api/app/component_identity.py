from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Set

logger = logging.getLogger(__name__)

# Base leaf to category mapping.
# This is the source of truth for "leaf -> canonical ID".
_CANONICAL_MAP: Dict[str, str] = {
    "relu": "math/relu",
    "gelu": "math/gelu",
    "silu": "math/silu",
    "swish": "math/silu",
    "sigmoid": "math/sigmoid",
    "tanh": "math/tanh",
    "abs": "math/abs",
    "add": "math/add",
    "sub": "math/sub",
    "mul": "math/mul",
    "div": "math/div_safe",
    "exp": "math/exp",
    "log": "math/log",
    "sqrt": "math/sqrt",
    "square": "math/square",
    
    "rmsnorm": "linear_algebra/rmsnorm",
    "rmsnorm_pre": "normalization/rmsnorm_pre",
    "layernorm": "normalization/layernorm_pre",
    "layernorm_pre": "normalization/layernorm_pre",
    
    "linear_proj": "linear_algebra/linear_proj",
    "linear_proj_up": "linear_algebra/linear_proj_up",
    "linear_proj_down": "linear_algebra/linear_proj_down",
    "dense": "linear_algebra/linear_proj",
    
    "low_rank_proj": "math_space/low_rank_proj",
    "bottleneck_proj": "math_space/bottleneck_proj",
    
    "softmax_attention": "mixing/softmax_attention",
    "linear_attention": "mixing/linear_attention",
    "graph_attention": "mixing/graph_attention",
    "local_window_attn": "mixing/local_window_attn",
    "ultrametric_attention": "math_space/ultrametric_attention",
    "tropical_attention": "math_space/tropical_attention",
    
    "selective_scan": "linear_algebra/selective_scan",
    "state_space": "linear_algebra/selective_scan",
    "conv1d_seq": "linear_algebra/conv1d_seq",
    
    "moe_topk": "channel_mixing/moe_topk",
    "difficulty_scorer": "routing/difficulty_scorer",
    "lane_router": "routing/lane_router",
    
    "split2": "structural/split2",
    "split3": "structural/split3",
    "concat": "structural/concat",
    "conditional_dispatch": "structural/conditional_dispatch",
    "conditional_gather": "structural/conditional_gather",
    
    "input": "io/input",
    "output": "io/output_head",
    "output_head": "io/output_head",
}

# Conversation aliases (many-to-one)
_ALIAS_MAP: Dict[str, str] = {
    "difficulty": "difficulty_scorer",
    "scorer": "difficulty_scorer",
    "score": "difficulty_scorer",
    "gate": "difficulty_scorer",
    "gating": "difficulty_scorer",
    "router": "lane_router",
    "routing": "lane_router",
    "dispatch": "conditional_dispatch",
    "gather": "conditional_gather",
    "moe": "moe_topk",
    "mixture of experts": "moe_topk",
    "expert": "moe_topk",
    "split": "split2",
    "branch": "split2",
    "fork": "split2",
    "merge": "concat",
    "combine": "concat",
    "join": "concat",
    "concatenate": "concat",
    "fuse": "concat",
    "attention": "softmax_attention",
    "self-attention": "softmax_attention",
    "self attention": "softmax_attention",
    "multi-head": "softmax_attention",
    "multihead": "softmax_attention",
    "mha": "softmax_attention",
    "linear attention": "linear_attention",
    "efficient attention": "linear_attention",
    "graph attention": "graph_attention",
    "local attention": "local_window_attn",
    "window attention": "local_window_attn",
    "ultrametric": "ultrametric_attention",
    "ssm": "selective_scan",
    "mamba": "selective_scan",
    "state space": "selective_scan",
    "scan": "selective_scan",
    "conv": "conv1d_seq",
    "convolution": "conv1d_seq",
    "linear": "linear_proj",
    "projection": "linear_proj",
    "ffn": "linear_proj",
    "compress": "bottleneck_proj",
    "compression": "bottleneck_proj",
    "bottleneck": "bottleneck_proj",
    "low rank": "low_rank_proj",
    "low-rank": "low_rank_proj",
    "norm": "rmsnorm",
    "normalize": "rmsnorm",
    "normalization": "rmsnorm",
    "residual": "add",
    "skip connection": "add",
}

def _normalize_token(raw_id: str | None) -> str:
    return str(raw_id or "").strip().lower()


def _build_registry_maps(registry_ids: Iterable[str] | None) -> tuple[Set[str], Dict[str, str]]:
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


def canonicalize_component_id(raw_id: str, registry_ids: Iterable[str] | None = None) -> str:
    """Resolve any alias or leaf name to a canonical category/id string."""
    token = _normalize_token(raw_id)
    if not token:
        return token

    registry_set, registry_leaf_map = _build_registry_maps(registry_ids)
    if token in registry_set:
        return token

    alias_target = _ALIAS_MAP.get(token, token)
    if alias_target in registry_leaf_map:
        return registry_leaf_map[alias_target]

    if alias_target in _CANONICAL_MAP:
        canonical = _CANONICAL_MAP[alias_target]
        if not registry_set or canonical in registry_set:
            return canonical

    if "/" in alias_target:
        _, _, leaf = alias_target.partition("/")
        if alias_target in registry_set:
            return alias_target
        if leaf in registry_leaf_map:
            return registry_leaf_map[leaf]
        if leaf in _CANONICAL_MAP:
            return _CANONICAL_MAP[leaf]
        return alias_target

    if alias_target in registry_leaf_map:
        return registry_leaf_map[alias_target]

    return _CANONICAL_MAP.get(alias_target, alias_target)


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


def collect_unresolved_component_ids(workflow: Dict[str, Any], registry_ids: Iterable[str]) -> List[str]:
    """Find component_types that are not present in the live registry."""
    registry_set, _ = _build_registry_maps(registry_ids)
    unresolved = []
    for node in workflow.get("nodes", []):
        ct = _normalize_token(node.get("component_type", ""))
        if ct and ct not in registry_set:
            unresolved.append(ct)
    return unresolved

def discover_concepts(message: str) -> List[Dict[str, str]]:
    """Extract component concepts from natural language.
    
    Returns list of {concept, component_type} dicts for each match,
    with component_type being canonicalized.
    """
    lower = message.lower()
    found: List[Dict[str, str]] = []
    seen_canonical: Set[str] = set()

    # Search for aliases (longer phrases first for better matching)
    all_terms = sorted(list(_ALIAS_MAP.keys()) + list(_CANONICAL_MAP.keys()), key=len, reverse=True)
    
    for term in all_terms:
        if term in lower:
            canonical = canonicalize_component_id(term)
            if canonical not in seen_canonical:
                found.append({"concept": term, "component_type": canonical})
                seen_canonical.add(canonical)
                
    return found
