from __future__ import annotations

import random
from typing import Dict, Sequence, Tuple

from ..scientist.shared_utils import clamp
from ..synthesis.graph import ComputationGraph
from ..synthesis.grammar import GrammarConfig
from ..synthesis.primitives import get_primitive, graph_binding_range_class


_SPARSE_ROUTING_OPS = frozenset(
    {
        "nm_sparse_linear",
        "block_sparse_linear",
        "semi_structured_2_4_linear",
        "ternary_projection",
        "token_entropy",
        "moe_topk",
        "moe_2expert",
        "adjacent_token_merge",
    }
)
_FULL_RANGE_TEMPLATE_KEYS = ("transformer_block", "state_space_block")
_FULL_RANGE_OPS = frozenset(
    {
        "softmax_attention",
        "linear_attention",
        "graph_attention",
        "diff_attention",
        "state_space",
        "selective_scan",
        "rwkv_time_mixing",
        "gated_delta",
    }
)
_PRIMITIVE_CATEGORY_CACHE: Dict[str, str] = {}


def derive_mutation_grammar(
    graph: ComputationGraph,
    base: GrammarConfig,
    rng: random.Random,
) -> GrammarConfig:
    hard_max_depth = 18
    hard_max_ops = 28
    parent_depth = max(1, graph.depth())
    parent_ops = max(1, graph.n_ops())
    category_weights = _mutated_category_weights(
        base.category_weights, category_histogram(graph), rng
    )
    template_weights, motif_weights, op_weights, sparsity_bias = _derive_weight_biases(
        graphs=(graph,),
        base=base,
    )
    return GrammarConfig(
        model_dim=graph.model_dim,
        max_depth=_bounded_target(
            parent_depth, base.max_depth, hard_max_depth, 2, 4, 3
        ),
        max_width=base.max_width,
        max_ops=_bounded_target(
            parent_ops, base.max_ops, hard_max_ops, 2, 8, parent_ops + 2
        ),
        residual_prob=clamp(base.residual_prob + rng.uniform(-0.1, 0.1), 0.0, 1.0),
        split_prob=clamp(base.split_prob + rng.uniform(-0.08, 0.08), 0.0, 1.0),
        merge_prob=clamp(base.merge_prob + rng.uniform(-0.08, 0.08), 0.0, 1.0),
        risky_op_prob=clamp(base.risky_op_prob + rng.uniform(-0.05, 0.05), 0.0, 1.0),
        freq_domain_prob=clamp(
            base.freq_domain_prob + rng.uniform(-0.05, 0.05), 0.0, 1.0
        ),
        category_weights=category_weights,
        op_weights=op_weights,
        template_weights=template_weights,
        motif_weights=motif_weights,
        structured_sparsity_bias=sparsity_bias,
        routing_mandatory=base.routing_mandatory,
    )


def derive_crossover_grammar(
    g1: ComputationGraph,
    g2: ComputationGraph,
    base: GrammarConfig,
    rng: random.Random,
) -> GrammarConfig:
    hard_max_depth = 18
    hard_max_ops = 28
    target_depth = max(
        2,
        int(
            round(
                (max(1, g1.depth()) + max(1, g2.depth())) / 2 + rng.choice([-1, 0, 1])
            )
        ),
    )
    target_ops = max(
        3,
        int(
            round(
                (max(1, g1.n_ops()) + max(1, g2.n_ops())) / 2
                + rng.choice([-2, -1, 0, 1, 2])
            )
        ),
    )
    category_weights = _crossover_category_weights(
        base.category_weights, category_histogram(g1), category_histogram(g2), rng
    )
    template_weights, motif_weights, op_weights, sparsity_bias = _derive_weight_biases(
        graphs=(g1, g2),
        base=base,
    )
    return GrammarConfig(
        model_dim=g1.model_dim,
        max_depth=_bounded_target(
            target_depth, base.max_depth, hard_max_depth, 2, 4, 3
        ),
        max_width=max(base.max_width, 2),
        max_ops=_bounded_target(
            target_ops, base.max_ops, hard_max_ops, 2, 10, target_ops + 2
        ),
        residual_prob=clamp(
            (base.residual_prob + 0.65) / 2 + rng.uniform(-0.08, 0.08), 0.0, 1.0
        ),
        split_prob=clamp(
            (base.split_prob + 0.35) / 2 + rng.uniform(-0.06, 0.06), 0.0, 1.0
        ),
        merge_prob=clamp(
            (base.merge_prob + 0.45) / 2 + rng.uniform(-0.06, 0.06), 0.0, 1.0
        ),
        risky_op_prob=clamp(base.risky_op_prob + rng.uniform(-0.04, 0.04), 0.0, 1.0),
        freq_domain_prob=clamp(
            base.freq_domain_prob + rng.uniform(-0.04, 0.04), 0.0, 1.0
        ),
        category_weights=category_weights,
        op_weights=op_weights,
        template_weights=template_weights,
        motif_weights=motif_weights,
        structured_sparsity_bias=sparsity_bias,
        routing_mandatory=base.routing_mandatory,
    )


def category_histogram(graph: ComputationGraph) -> Dict[str, int]:
    hist: Dict[str, int] = {}
    for node in graph.nodes.values():
        if node.is_input:
            continue
        cat = primitive_category_value(node.op_name)
        if cat is None:
            continue
        hist[cat] = hist.get(cat, 0) + 1
    return hist


def primitive_category_value(op_name: str) -> str | None:
    cached = _PRIMITIVE_CATEGORY_CACHE.get(op_name)
    if cached is not None:
        return cached
    try:
        category = get_primitive(op_name).category.value
    except KeyError:
        return None
    _PRIMITIVE_CATEGORY_CACHE[op_name] = category
    return category


def _bounded_target(
    target: int,
    base_value: int,
    hard_cap: int,
    min_delta: int,
    max_delta: int,
    floor_value: int,
) -> int:
    return min(
        hard_cap,
        max(floor_value, min(max(base_value, target + min_delta), target + max_delta)),
    )


def _mutated_category_weights(
    base_weights: Dict[str, float],
    category_histogram: Dict[str, int],
    rng: random.Random,
) -> Dict[str, float]:
    category_weights = dict(base_weights)
    for cat_name in category_weights:
        if category_histogram.get(cat_name, 0) > 0:
            category_weights[cat_name] *= 1.25
        else:
            category_weights[cat_name] = max(0.1, category_weights[cat_name] * 0.9)
        category_weights[cat_name] = max(
            0.1, category_weights[cat_name] * rng.uniform(0.9, 1.1)
        )
    return category_weights


def _crossover_category_weights(
    base_weights: Dict[str, float],
    left_histogram: Dict[str, int],
    right_histogram: Dict[str, int],
    rng: random.Random,
) -> Dict[str, float]:
    category_weights = dict(base_weights)
    for cat_name, weight in category_weights.items():
        used = left_histogram.get(cat_name, 0) + right_histogram.get(cat_name, 0)
        if used > 0:
            category_weights[cat_name] = max(0.1, weight * 1.2)
        else:
            category_weights[cat_name] = max(0.1, weight * 0.85)
        category_weights[cat_name] = max(
            0.1, category_weights[cat_name] * rng.uniform(0.92, 1.08)
        )
    return category_weights


def _derive_weight_biases(
    *,
    graphs: Sequence[ComputationGraph],
    base: GrammarConfig,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], float]:
    template_weights = dict(base.template_weights) if base.template_weights else {}
    motif_weights = dict(base.motif_weights) if base.motif_weights else {}
    op_weights = dict(base.op_weights)

    sparsity_bias = base.structured_sparsity_bias
    if any(
        node.op_name in _SPARSE_ROUTING_OPS
        for graph in graphs
        for node in graph.nodes.values()
        if not node.is_input
    ):
        sparsity_bias = max(sparsity_bias, 0.6)

    binding_classes = [graph_binding_range_class(graph) for graph in graphs]
    if all(binding in ("local", "none") for binding in binding_classes):
        for tpl_key in _FULL_RANGE_TEMPLATE_KEYS:
            if tpl_key in template_weights:
                template_weights[tpl_key] *= 2.5
        for op_name in _FULL_RANGE_OPS:
            op_weights[op_name] = op_weights.get(op_name, 1.0) * 3.0

    return template_weights, motif_weights, op_weights, sparsity_bias
