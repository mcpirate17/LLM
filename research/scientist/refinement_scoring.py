from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple


_RISKY_EXACT_OPS = {
    "exp",
    "mul",
    "outer_product",
    "gated_linear",
    "moe_topk",
    "moe_2expert",
    "route_topk",
    "route_lanes",
    "route_recursion",
    "adaptive_recursion",
    "mixed_recursion_gate",
    "fixed_point_iter",
    "implicit_fixed_point",
    "ultrametric_attention",
    "tropical_attention",
    "clifford_attention",
    "poincare_add",
}

_RISKY_TOKENS = (
    "route",
    "moe",
    "gate",
    "recursion",
    "fixed_point",
)

_NORM_OPS = {
    "layernorm",
    "rmsnorm",
    "norm_last",
    "dynamic_norm",
    "group_norm",
    "layernorm_pre",
    "rmsnorm_pre",
}


def oscillation_risk_score(graph: Any) -> Tuple[float, Dict[str, float]]:
    """Cheap structural prior for fingerprints that often produce sawtooth loss curves."""
    ops: List[str] = [
        str(node.op_name)
        for node in graph.nodes.values()
        if not getattr(node, "is_input", False)
    ]
    n_ops = max(1, len(ops))
    depth = max(1, int(graph.depth()))
    has_residual = bool(graph.has_residual_path()) if hasattr(graph, "has_residual_path") else False

    norm_count = sum(1 for op in ops if op in _NORM_OPS)
    risky_count = 0
    routing_count = 0
    for op in ops:
        lowered = op.lower()
        if op in _RISKY_EXACT_OPS or any(tok in lowered for tok in _RISKY_TOKENS):
            risky_count += 1
        if "route" in lowered or "moe" in lowered:
            routing_count += 1

    risky_density = min(1.0, risky_count / n_ops)
    routing_density = min(1.0, routing_count / n_ops)
    no_residual_risk = 1.0 if n_ops >= 3 and risky_density >= 0.34 and not has_residual else 0.0
    no_norm_risk = 1.0 if n_ops >= 3 and risky_density >= 0.34 and norm_count == 0 else 0.0
    serial_depth_risk = min(1.0, max(0.0, depth - 4.0) / 6.0) * (0.5 if has_residual else 1.0)

    risk = min(
        1.0,
        0.40 * no_residual_risk
        + 0.20 * no_norm_risk
        + 0.25 * risky_density
        + 0.10 * routing_density
        + 0.05 * serial_depth_risk,
    )
    return risk, {
        "oscillation_risk": float(risk),
        "risky_density": float(risky_density),
        "routing_density": float(routing_density),
        "has_residual": 1.0 if has_residual else 0.0,
        "norm_count": float(norm_count),
        "no_residual_risk": float(no_residual_risk),
        "no_norm_risk": float(no_norm_risk),
        "serial_depth_risk": float(serial_depth_risk),
    }


def rank_synthesis_candidates_by_stability(graphs: Iterable[Any]) -> List[Any]:
    """Mildly prefer structurally stable candidates before expensive screening."""
    decorated = []
    for idx, graph in enumerate(graphs):
        risk, details = oscillation_risk_score(graph)
        decorated.append((
            risk,
            -float(details.get("has_residual", 0.0)),
            float(details.get("norm_count", 0.0) == 0.0),
            idx,
            graph,
        ))
    decorated.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return [graph for *_rest, graph in decorated]
