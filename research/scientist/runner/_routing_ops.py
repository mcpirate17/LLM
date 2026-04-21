"""Shared routing-op policy for runner helpers.

This module owns the routing/specialization op sets used across screening,
benchmarking, and observability helpers so the policy does not drift between
split helper modules.
"""

from __future__ import annotations

ROUTING_FAST_LANE_OPS: frozenset[str] = frozenset(
    {
        "moe_topk",
        "hetero_moe",
        "arch_router",
        "compute_budget_router",
        "signal_conditioned_compression",
    }
)

ROUTING_OBSERVED_OPS: frozenset[str] = frozenset(
    set(ROUTING_FAST_LANE_OPS)
    | {
        "hybrid_token_gate",
        "hybrid_sparse_router",
        "sparse_span_builder",
        "adjacent_token_merge",
        "cheap_verify_blend",
        "adaptive_lane_mixer",
        "route_lanes",
        "block_sparse_linear",
        "semi_structured_2_4_linear",
        "adaptive_recursion",
        "route_recursion",
        "moe_2expert",
        "token_class_proj",
    }
)
