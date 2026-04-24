from __future__ import annotations

from ._context_registry import CONTEXT_RULES


def context_pair_allowed(prev_op: str | None, next_op: str | None) -> bool:
    if prev_op is None or next_op is None:
        return True
    prev_rule = CONTEXT_RULES.get(prev_op)
    if prev_rule is not None and next_op in prev_rule.forbidden_successors:
        return False
    next_rule = CONTEXT_RULES.get(next_op)
    if next_rule is not None and prev_op in next_rule.forbidden_predecessors:
        return False
    return True


def with_local_wildcard_probability(
    graph,
    callback,
    *,
    wildcard_prob: float,
):
    previous = graph.metadata.get("_wildcard_slot_prob", 0.0)
    graph.metadata["_wildcard_slot_prob"] = wildcard_prob
    try:
        return callback()
    finally:
        graph.metadata["_wildcard_slot_prob"] = previous
