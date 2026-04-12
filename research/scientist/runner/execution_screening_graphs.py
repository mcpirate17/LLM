"""Hot-loop graph screening helpers.

These helpers keep per-graph structural analysis in one focused place so the
screening loop can reuse a single pass over graph nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, Tuple

from ...eval._eval_native import load_eval_native


@dataclass(frozen=True, slots=True)
class ScreeningGraphAnalysis:
    """Cached graph facts used repeatedly during screening."""

    op_names: FrozenSet[str]
    counted_ops: Tuple[str, ...]
    toxic_bigrams: Tuple[str, ...]
    has_parameterized_op: bool


def analyze_graph_for_screening(
    graph: Any,
    get_primitive: Callable[[str], Any] | None,
) -> ScreeningGraphAnalysis:
    """Collect reusable graph facts in one pass.

    The screening loop checks parameterized ops, routing ops, toxic bigrams,
    and stage-0 op accounting for nearly every candidate. Doing those in a
    single scan avoids repeated dictionary iteration on large batches.
    """

    nodes = graph.nodes
    ordered_nodes = list(nodes.items())
    has_params_flags = []

    if get_primitive is not None:
        primitive_cache: Dict[str, bool] = {}
        for _node_id, node in ordered_nodes:
            op_name = node.op_name
            if not op_name or getattr(node, "is_input", False):
                has_params_flags.append(False)
                continue
            if op_name not in primitive_cache:
                try:
                    primitive = get_primitive(op_name)
                except (KeyError, ValueError):
                    primitive = None
                primitive_cache[op_name] = bool(
                    primitive is not None and getattr(primitive, "has_params", False)
                )
            has_params_flags.append(primitive_cache[op_name])
    else:
        has_params_flags = [False] * len(ordered_nodes)

    try:
        native = load_eval_native()
        analysis = native.screening_graph_analysis_native(
            [int(node_id) for node_id, _ in ordered_nodes],
            [str(node.op_name or "") for _, node in ordered_nodes],
            [
                [int(parent_id) for parent_id in node.input_ids]
                for _, node in ordered_nodes
            ],
            [bool(getattr(node, "is_input", False)) for _, node in ordered_nodes],
            [bool(getattr(node, "is_output", False)) for _, node in ordered_nodes],
            [bool(flag) for flag in has_params_flags],
        )
        return ScreeningGraphAnalysis(
            op_names=frozenset(analysis["op_names"]),
            counted_ops=tuple(analysis["counted_ops"]),
            toxic_bigrams=tuple(analysis["toxic_bigrams"]),
            has_parameterized_op=bool(analysis["has_parameterized_op"]),
        )
    except Exception:
        pass

    op_names = set()
    counted_ops = []
    toxic_bigrams = set()
    has_parameterized_op = False

    for idx, (_node_id, node) in enumerate(ordered_nodes):
        if node.is_input:
            continue

        op_name = node.op_name
        if op_name:
            counted_ops.append(op_name)

        if getattr(node, "is_output", False):
            continue

        op_names.add(op_name)
        if not has_parameterized_op and has_params_flags[idx]:
            has_parameterized_op = True

        for parent_id in node.input_ids:
            parent = nodes.get(parent_id)
            if (
                parent is not None
                and not parent.is_input
                and not getattr(parent, "is_output", False)
            ):
                toxic_bigrams.add(f"{parent.op_name}->{op_name}")

    return ScreeningGraphAnalysis(
        op_names=frozenset(op_names),
        counted_ops=tuple(counted_ops),
        toxic_bigrams=tuple(sorted(toxic_bigrams)),
        has_parameterized_op=has_parameterized_op,
    )


def structural_gate_failure(
    graph: Any,
    *,
    routing_mandatory: bool,
    efficiency_ops: FrozenSet[str],
    analysis: ScreeningGraphAnalysis,
) -> str | None:
    """Return the first failing structural gate code, or ``None``."""

    if graph.n_ops() <= 7:
        return "gate1_min_ops"
    if not graph.has_gradient_path():
        return "gate2_no_grad"
    if not graph.has_residual_path():
        return "gate3_no_residual"
    if not analysis.has_parameterized_op:
        return "gate4_no_params"
    if routing_mandatory and not (analysis.op_names & efficiency_ops):
        return "gate5_no_routing"
    return None


def toxic_failure_ratio(
    failure_blocklist: Dict[str, float],
    analysis: ScreeningGraphAnalysis,
) -> float:
    """Compute toxic bigram ratio from cached graph analysis."""

    if not failure_blocklist or not analysis.toxic_bigrams:
        return 0.0
    toxic_weight = sum(
        1.0 - failure_blocklist[bigram]
        for bigram in analysis.toxic_bigrams
        if bigram in failure_blocklist
    )
    return toxic_weight / len(analysis.toxic_bigrams)
