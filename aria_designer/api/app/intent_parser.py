from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Tuple


_INTENT_KEYWORDS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("refine_compression", ("compression", "compress", "bottleneck", "low rank", "sparse")),
    ("improve_stability", ("stability", "stable", "gradient", "nan", "brittle", "explode")),
    ("expand_capacity", ("capacity", "depth", "layer", "width", "hidden", "benchmark", "beat")),
    ("preserve_fingerprint", ("fingerprint", "preserve", "minimal", "surgical", "incremental")),
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
    "rmsnorm_pre": ("normalization",),
    "layernorm": ("normalization",),
    "layernorm_pre": ("normalization",),
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
    "mod_topk": ("routing",),
    "relu_gate_routing": ("routing",),
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
    token = str(component_type or "").strip().lower()
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
    high_tier = tier in {"investigation", "validation", "breakthrough"} or composite >= 100.0
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
            replacement_components=("math/silu", "math/gelu", "normalization/rmsnorm_pre"),
            target_param_names=("out_dim", "hidden_dim", "ff_dim", "rank", "heads", "expansion"),
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
            replacement_components=("normalization/rmsnorm_pre", "math/silu"),
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
            replacement_components=("normalization/rmsnorm_pre", "math/silu", "math/gelu"),
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
            target_param_names=("out_dim", "hidden_dim", "ff_dim", "heads", "expansion"),
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
