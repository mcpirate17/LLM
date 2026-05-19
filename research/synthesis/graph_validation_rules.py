"""Generated graph validation rules used by the synthesis grammar."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping

from .graph import ComputationGraph, OpNode
from .grammar_support import (
    ROUTING_COMPRESSION_MOE_OPS,
    check_graph_space_consistency,
)
from .primitives import (
    REQUIRES_RESIDUAL_BYPASS,
    get_wiring_rule,
    validate_wiring,
)
from .template_rules import validate_template_graph
from .validator import validate_graph

logger = logging.getLogger(__name__)


def validate_generated_graph(
    graph: ComputationGraph,
    config: Any,
    *,
    dim_flow_inputs: object | None = None,
    packed_validation: object | None = None,
) -> None:
    """Validate a generated graph and raise ValueError if invalid."""
    _validate_runtime_limits(graph, config, dim_flow_inputs, packed_validation)
    _validate_space_consistency(graph)
    _validate_routing_requirement(graph, config)
    _validate_depth_constraints(graph)
    successors = graph.children_map()
    _validate_residual_bypass(graph)
    _validate_residual_context(graph, successors)
    _validate_wiring_constraints(graph)
    _validate_activation_constraints(graph, successors)
    _validate_math_space_constraints(graph, successors)
    _validate_template_rules(graph)


def _validate_runtime_limits(
    graph: ComputationGraph,
    config: Any,
    dim_flow_inputs: object | None,
    packed_validation: object | None,
) -> None:
    max_params = 12 * 4 * config.model_dim * config.model_dim
    validation_kwargs = {
        "max_ops": config.max_ops,
        "max_depth": config.max_depth + 2,
        "min_splits": config.min_splits,
        "max_params": max_params,
    }
    if dim_flow_inputs is not None:
        validation_kwargs["dim_flow_inputs"] = dim_flow_inputs
    if packed_validation is not None:
        validation_kwargs["packed_validation"] = packed_validation
    result = validate_graph(graph, **validation_kwargs)
    if not result.valid:
        raise ValueError(
            result.errors[0] if result.errors else "Graph validation failed"
        )


def _validate_space_consistency(graph: ComputationGraph) -> None:
    space_err = check_graph_space_consistency(graph)
    if space_err is not None:
        raise ValueError(space_err)


def _validate_routing_requirement(graph: ComputationGraph, config: Any) -> None:
    skip_routing_check = config.forced_template or graph.metadata.get(
        "_template_exploration_used"
    )
    if not config.routing_mandatory or skip_routing_check:
        return
    op_names = {node.op_name for node in graph.nodes.values() if not node.is_input}
    if op_names & ROUTING_COMPRESSION_MOE_OPS:
        return
    raise ValueError(
        "routing_mandatory=True but graph has no routing/compression/MoE ops"
    )


def _validate_depth_constraints(graph: ComputationGraph) -> None:
    for nid, node in graph.nodes.items():
        if node.is_input:
            continue
        depth_rule = get_wiring_rule(node.op_name) or {}
        min_layer_depth = int(depth_rule.get("min_layer_depth", 0))
        if min_layer_depth > 0 and node.depth < min_layer_depth:
            raise ValueError(
                f"{node.op_name} (id={nid}) placed at depth {node.depth} "
                f"before min_layer_depth={min_layer_depth}"
            )


def _add_inputs_by_source(graph: ComputationGraph) -> Dict[int, set[int]]:
    add_inputs_by_source: Dict[int, set[int]] = {}
    for other_node in graph.nodes.values():
        if other_node.op_name == "add":
            add_inputs = set(other_node.input_ids)
            for source_id in add_inputs:
                add_inputs_by_source.setdefault(source_id, set()).update(add_inputs)
    return add_inputs_by_source


def _validate_residual_bypass(graph: ComputationGraph) -> None:
    add_inputs_by_source = _add_inputs_by_source(graph)
    for nid, node in graph.nodes.items():
        if node.is_input or node.op_name not in REQUIRES_RESIDUAL_BYPASS:
            continue
        node_inputs = set(node.input_ids)
        if not (add_inputs_by_source.get(nid, set()) & node_inputs):
            raise ValueError(
                f"{node.op_name} (id={nid}) requires residual bypass but none found"
            )


def _reaches_any(
    start_id: int,
    targets: set[int],
    successors: Dict[int, List[int]],
) -> bool:
    seen: set[int] = {start_id}
    queue: List[int] = [start_id]
    while queue:
        cur = queue.pop()
        if cur in targets:
            return True
        for child in successors.get(cur, ()):
            if child not in seen:
                seen.add(child)
                queue.append(child)
    return False


def _validate_residual_context(
    graph: ComputationGraph,
    successors: Dict[int, List[int]],
) -> None:
    from ._context_registry import REQUIRES_RESIDUAL_CONTEXT_OPS

    if not REQUIRES_RESIDUAL_CONTEXT_OPS:
        return
    add_consumers = {nid for nid, node in graph.nodes.items() if node.op_name == "add"}
    for nid, node in graph.nodes.items():
        if node.is_input or node.op_name in REQUIRES_RESIDUAL_BYPASS:
            continue
        if node.op_name not in REQUIRES_RESIDUAL_CONTEXT_OPS:
            continue
        if _reaches_any(nid, add_consumers, successors):
            continue
        raise ValueError(
            f"{node.op_name} (id={nid}) requires residual context but "
            f"no downstream add is reachable from its output"
        )


def _validate_wiring_constraints(graph: ComputationGraph) -> None:
    wiring_errors = validate_wiring(graph)
    if wiring_errors:
        raise ValueError(f"Wiring constraint violated: {wiring_errors[0]}")


def _validate_activation_successors(
    graph: ComputationGraph,
    nid: int,
    node: OpNode,
    before: object,
    successors: Dict[int, List[int]],
) -> None:
    if before is None:
        return
    for other_nid in successors.get(nid, ()):
        other_node = graph.nodes[other_nid]
        if other_node.is_input or other_node.op_name in before:
            continue
        raise ValueError(
            f"Activation constraint: {node.op_name} (id={nid}) "
            f"→ {other_node.op_name} (id={other_nid}) is not allowed; "
            f"valid successors: {before}"
        )


def _validate_activation_predecessors(
    graph: ComputationGraph,
    nid: int,
    node: OpNode,
    after: object,
) -> None:
    if after is None:
        return
    from .op_roles import get_role

    for parent_id in node.input_ids:
        parent = graph.nodes.get(parent_id)
        if parent is None or parent.is_input:
            continue
        parent_role = get_role(parent.op_name)
        if parent.op_name in after or parent_role in after:
            continue
        raise ValueError(
            f"Activation constraint: {parent.op_name} (id={parent_id}) "
            f"→ {node.op_name} (id={nid}) is not allowed; "
            f"valid predecessors: {after}"
        )


def _validate_activation_constraints(
    graph: ComputationGraph,
    successors: Dict[int, List[int]],
) -> None:
    from .motifs import ACTIVATION_RULES

    for nid in sorted(graph.nodes):
        node = graph.nodes[nid]
        if node.is_input:
            continue
        rules = ACTIVATION_RULES.get(node.op_name)
        if rules is None:
            continue
        _validate_activation_successors(
            graph, nid, node, rules.get("before"), successors
        )
        _validate_activation_predecessors(graph, nid, node, rules.get("after"))


def _node_has_parent_in(
    graph: ComputationGraph,
    node: OpNode,
    allowed_ops: object,
) -> bool:
    return any(
        (parent := graph.nodes.get(parent_id)) is not None
        and parent.op_name in allowed_ops
        for parent_id in node.input_ids
    )


def _validate_math_parent_rule(
    graph: ComputationGraph,
    nid: int,
    node: OpNode,
    rules: Mapping[str, object],
    rule_name: str,
    message: str,
) -> None:
    allowed_ops = rules.get(rule_name)
    if allowed_ops is None or _node_has_parent_in(graph, node, allowed_ops):
        return
    raise ValueError(
        f"Math-space constraint: {node.op_name} (id={nid}) {message} {allowed_ops}"
    )


def _validate_math_successor_rule(
    graph: ComputationGraph,
    nid: int,
    node: OpNode,
    rules: Mapping[str, object],
    successors: Dict[int, List[int]],
) -> None:
    required_ops = rules.get("must_follow_with")
    if required_ops is None:
        return
    if any(
        not graph.nodes[other_nid].is_input
        and graph.nodes[other_nid].op_name in required_ops
        for other_nid in successors.get(nid, ())
    ):
        return
    raise ValueError(
        f"Math-space constraint: {node.op_name} (id={nid}) "
        f"must be followed by one of {required_ops}"
    )


def _validate_math_space_constraints(
    graph: ComputationGraph,
    successors: Dict[int, List[int]],
) -> None:
    from .motifs import MATH_SPACE_RULES

    for nid in sorted(graph.nodes):
        node = graph.nodes[nid]
        if node.is_input:
            continue
        ms_rules = MATH_SPACE_RULES.get(node.op_name)
        if ms_rules is None:
            continue
        _validate_math_parent_rule(
            graph, nid, node, ms_rules, "must_precede", "requires predecessor from"
        )
        _validate_math_parent_rule(
            graph, nid, node, ms_rules, "must_follow", "must follow one of"
        )
        _validate_math_successor_rule(graph, nid, node, ms_rules, successors)


def _validate_template_rules(graph: ComputationGraph) -> None:
    tpl_errors = validate_template_graph(graph)
    if tpl_errors:
        for err in tpl_errors:
            logger.debug("template_rule: %s", err)
        graph.metadata["template_rule_warnings"] = tpl_errors
        raise ValueError(f"Template rule violations: {tpl_errors}")
