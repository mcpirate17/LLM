from __future__ import annotations

from collections import Counter
from typing import Any


MEDIUM_MECHANISM_FAMILIES = {
    "conv_only": (
        "local_convolutional_mixing",
        "Pure local convolutional token mixing over the merged multiscale route features.",
    ),
    "conv1d_seq": (
        "local_convolutional_mixing",
        "Sequence convolution with local receptive-field token mixing.",
    ),
    "rwkv_time_mixing": (
        "recurrent_time_mixing",
        "RWKV-style recurrent or time-mixing dynamics over routed features.",
    ),
    "block_sparse_linear": (
        "structured_sparse_linear",
        "Block-structured sparse linear transform applied to routed features.",
    ),
    "semi_structured_2_4_linear": (
        "structured_sparse_linear",
        "Semi-structured 2:4 sparse linear transform.",
    ),
    "nm_sparse_linear": (
        "structured_sparse_linear",
        "N:M structured sparse linear transform.",
    ),
    "cheap_verify_blend": (
        "verification_hybrid",
        "Cheap-first compute path with a verification correction path.",
    ),
    "default_path": (
        "cheap_fallback",
        "Cheap residual-style fallback compute with minimal specialization pressure.",
    ),
    "adaptive_lane_mixer": (
        "hybrid_lane_mixing",
        "Difficulty-conditioned lane mixing across routed medium features.",
    ),
    "route_lanes": (
        "hybrid_lane_mixing",
        "Explicit multi-lane routed blending over medium features.",
    ),
    "hybrid_sparse_router": (
        "nested_hybrid_routing",
        "A nested router reused as a medium operator, adding another routing stage.",
    ),
}

HARD_MECHANISM_FAMILIES = {
    "route_recursion": (
        "recursion_adaptive_depth",
        "Depth-conditioned recursive transforms for hard tokens.",
    ),
    "adaptive_recursion": (
        "recursion_adaptive_depth",
        "Adaptive recursion depth weighted by difficulty.",
    ),
    "mixed_recursion_gate": (
        "recursion_adaptive_depth",
        "Gated mixed-depth recursion with per-token difficulty control.",
    ),
    "moe_topk": (
        "moe_expert_routing",
        "Top-k expert routing for hard-token specialization.",
    ),
    "moe_2expert": (
        "moe_expert_routing",
        "Lightweight two-expert routing for hard-token specialization.",
    ),
    "dual_compression_blend": (
        "compression_first_routing",
        "Compression-first hard path with multiple compressed expert transforms.",
    ),
    "routing_conditioned_compression": (
        "compression_first_routing",
        "Signal-conditioned compression before hard compute.",
    ),
    "n_way_sparse_router": (
        "bottleneck_sparse_routing",
        "Sparse bottleneck routing across a fixed set of hard-token paths.",
    ),
    "state_space": (
        "state_space_recurrent",
        "State-space or recurrent sequence dynamics on hard tokens.",
    ),
}


def classify_medium_mechanism(op_name: str) -> dict[str, str]:
    family, description = MEDIUM_MECHANISM_FAMILIES[op_name]
    return {"family": family, "description": description}


def classify_hard_mechanism(op_name: str) -> dict[str, str]:
    family, description = HARD_MECHANISM_FAMILIES[op_name]
    return {"family": family, "description": description}


def classify_mechanism(op_name: str, slot: str) -> dict[str, str]:
    if slot == "medium":
        return classify_medium_mechanism(op_name)
    if slot == "hard":
        return classify_hard_mechanism(op_name)
    raise ValueError(f"Unsupported slot: {slot}")


def build_mechanism_coverage(rows: list[dict[str, Any]], slot: str) -> dict[str, Any]:
    members: dict[str, list[dict[str, Any]]] = {}
    classifier = (
        classify_medium_mechanism if slot == "medium" else classify_hard_mechanism
    )
    for row in rows:
        payload = classifier(row["dispatch_name"])
        family = payload["family"]
        members.setdefault(family, []).append(
            {
                "slot_ref": row["slot_ref"],
                "canonical_name": row["canonical_name"],
                "dispatch_name": row["dispatch_name"],
                "description": payload["description"],
            }
        )
    counts = Counter({family: len(entries) for family, entries in members.items()})
    total = sum(counts.values())
    sorted_families = sorted(
        (
            {
                "family": family,
                "count": count,
                "share": 0.0 if total <= 0 else round(count / total, 4),
                "members": members[family],
            }
            for family, count in counts.items()
        ),
        key=lambda row: (-row["count"], row["family"]),
    )
    return {
        "slot": slot,
        "family_count": len(sorted_families),
        "largest_family_share": 0.0
        if total <= 0
        else round(max(counts.values()) / total, 4),
        "families": sorted_families,
    }
