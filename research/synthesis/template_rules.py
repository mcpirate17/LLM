"""Post-synthesis graph validator for template-level structural invariants.

Checks whole-graph properties that per-op context_rules.py cannot enforce:
  1. Final norm: graph must end with rmsnorm/layernorm before output
  2. Lane diversity: multi-lane splits must have distinct op categories per lane
  3. Bottleneck dimension: no full-dim parametric op after a bottleneck until up-proj

Usage:
    from research.synthesis.template_rules import validate_template_graph
    errors = validate_template_graph(graph)
    if errors:
        raise ValueError(f"Template rule violations: {errors}")
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Set

from .graph import ComputationGraph
from .primitives import PRIMITIVE_REGISTRY


def validate_template_graph(graph: ComputationGraph) -> List[str]:
    """Validate template-level invariants on a completed graph.

    Returns list of error strings. Empty list = valid.
    """
    errors: List[str] = []
    children = _build_children_map(graph)
    errors.extend(_check_final_norm(graph))
    errors.extend(_check_lane_diversity(graph, children))
    errors.extend(_check_bottleneck_dimension(graph, children))
    return errors


def _check_final_norm(graph: ComputationGraph) -> List[str]:
    """Graph must end with a normalization op (rmsnorm or layernorm)."""
    out = graph.output_node
    if out is None:
        return ["No output node set"]
    if out.op_name not in ("rmsnorm", "layernorm"):
        return [
            f"Final op is '{out.op_name}', expected rmsnorm/layernorm. "
            f"Uncontrolled logit scale will harm perplexity."
        ]
    return []


def _build_children_map(graph: ComputationGraph) -> Dict[int, List[int]]:
    """Build parent→children adjacency list in O(E)."""
    return graph.children_map()


def _check_lane_diversity(
    graph: ComputationGraph,
    children: Dict[int, List[int]] | None = None,
) -> List[str]:
    """Multi-lane splits must have distinct op categories per lane."""
    errors = []
    if children is None:
        children = _build_children_map(graph)

    for nid, node in graph.nodes.items():
        if node.op_name != "split3":
            continue

        consumers = children.get(nid, [])
        if len(consumers) < 3:
            continue

        categories = set()
        for cid in consumers:
            cn = graph.nodes[cid]
            op = PRIMITIVE_REGISTRY.get(cn.op_name)
            if op is not None:
                categories.add(op.category)

        if len(categories) <= 1:
            errors.append(
                f"split3 (id={nid}) has {len(consumers)} lanes but only "
                f"category '{next(iter(categories))}' — no lane diversity"
            )

    return errors


_BOTTLENECK_OPS = frozenset({"bottleneck_proj", "low_rank_proj", "linear_proj_down"})
_UPPROJ_OPS = frozenset({"linear_proj_up", "linear_proj"})
# Norm ops are expected inside bottlenecks (normalize before up-proj) — skip them.
_BOTTLENECK_EXEMPT_OPS = frozenset({"rmsnorm", "layernorm"})
import functools


@functools.lru_cache(maxsize=1)
def _get_parametric_ops() -> frozenset:
    """Lazily build set of ops with learnable parameters."""
    return frozenset(name for name, op in PRIMITIVE_REGISTRY.items() if op.has_params)


def _check_bottleneck_dimension(
    graph: ComputationGraph,
    children: Dict[int, List[int]] | None = None,
) -> List[str]:
    """After a bottleneck (D→D/4), no full-dim parametric op should appear
    until an up-projection restores dimensionality.
    """
    errors = []
    parametric = _get_parametric_ops()
    if children is None:
        children = _build_children_map(graph)

    # Kahn's algorithm with adjacency list — O(V+E)
    in_degree: Dict[int, int] = {nid: 0 for nid in graph.nodes}
    for nid, node in graph.nodes.items():
        for pid in node.input_ids:
            if pid in in_degree:
                in_degree[nid] += 1

    queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
    in_bottleneck: Set[int] = set()

    while queue:
        nid = queue.popleft()
        node = graph.nodes[nid]

        if not node.is_input:
            parents_in_bottleneck = any(pid in in_bottleneck for pid in node.input_ids)

            if node.op_name in _BOTTLENECK_OPS:
                in_bottleneck.add(nid)
            elif node.op_name not in _UPPROJ_OPS and parents_in_bottleneck:
                in_bottleneck.add(nid)
                if (
                    node.op_name in parametric
                    and node.op_name not in _BOTTLENECK_OPS | _BOTTLENECK_EXEMPT_OPS
                ):
                    out_dim = node.output_shape.dim if node.output_shape else None
                    if out_dim and out_dim == graph.model_dim:
                        errors.append(
                            f"Op '{node.op_name}' (id={nid}) has full dim={out_dim} "
                            f"inside bottleneck — wastes params on reduced-rank input"
                        )

        for child_id in children[nid]:
            in_degree[child_id] -= 1
            if in_degree[child_id] == 0:
                queue.append(child_id)

    return errors
