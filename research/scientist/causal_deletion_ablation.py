"""Deletion-ablation primitives — generate per-op deletion children of a graph.

Single source of truth shared by:

* ``tools/focused_op_deletion_ablation.py`` — manual driver that walks
  top-K leaderboard parents and runs deletion sweeps.
* ``scientist/runner/_helpers_benchmark.py::_record_investigation_result`` —
  auto-ablation hook that fires at the end of every investigation tier
  when ``RunConfig.enable_investigation_auto_ablation`` is True.

Both call ``run_ablation_suite`` (in ``causal_attribution.py``) for the
actual evaluation/persistence/leaderboard-promotion — this module only
constructs the candidate set so the two paths can't drift on dedup or
scaffold-skipping logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

from research.scientist.native_runner import compile_model_native_first as compile_model
from research.synthesis.graph import ComputationGraph
from research.synthesis.validator import validate_graph

from .causal_attribution import CausalAblationCandidate, SCAFFOLD_OPS
from .construction_priors import assess_local_edit_prior


@dataclass(slots=True)
class DeletionChild:
    """A deletion ablation child — graph with one op removed via bypass."""

    graph: ComputationGraph
    node_id: int
    op_name: str
    bypass_input_id: int
    fingerprint: str
    pruned_nodes: int


def _recompute_depths(graph: ComputationGraph) -> None:
    graph._cache.clear()
    for node_id in graph.topological_order():
        node = graph.nodes[node_id]
        if node.is_input or not node.input_ids:
            node.depth = 0
        else:
            node.depth = 1 + max(
                graph.nodes[input_id].depth for input_id in node.input_ids
            )
    graph._cache.clear()


def delete_node_by_bypass(
    graph: ComputationGraph,
    node_id: int,
) -> tuple[Optional[ComputationGraph], dict[str, Any]]:
    """Generate a deletion child by routing the deleted op's consumers to its first input.

    Returns ``(child_graph, meta)``.  ``child_graph`` is None when the
    deletion is structurally impossible — input node, no bypass input, or
    the bypass would change the output shape.  ``meta`` always carries a
    ``reason`` string in the failure case.
    """
    if node_id not in graph.nodes:
        return None, {"reason": "node_missing", "node_id": node_id}
    source = graph.nodes[node_id]
    if source.is_input:
        return None, {"reason": "input_node", "node_id": node_id}
    if not source.input_ids:
        return None, {"reason": "no_bypass_input", "node_id": node_id}

    child = graph.copy()
    source = child.nodes[node_id]
    bypass_input_id = int(source.input_ids[0])
    if bypass_input_id not in child.nodes:
        return None, {"reason": "bypass_input_missing", "node_id": node_id}

    children = child.children_map().get(node_id, [])
    if source.is_output:
        bypass = child.nodes[bypass_input_id]
        if (
            bypass.output_shape.dim != child.model_dim
            or not bypass.output_shape.is_standard
        ):
            return None, {
                "reason": "output_bypass_shape_mismatch",
                "node_id": node_id,
                "bypass_input_id": bypass_input_id,
            }
        bypass.is_output = True
        child._output_node_id = bypass_input_id

    for child_id in children:
        consumer = child.nodes[child_id]
        consumer.input_ids = [
            bypass_input_id if input_id == node_id else input_id
            for input_id in consumer.input_ids
        ]
    del child.nodes[node_id]
    pruned = child.prune_unreachable_nodes()
    _recompute_depths(child)
    return child, {
        "node_id": node_id,
        "op_name": source.op_name,
        "bypass_input_id": bypass_input_id,
        "consumer_count": len(children),
        "pruned_nodes": int(pruned),
    }


def _attempt_one_deletion(
    *,
    graph: ComputationGraph,
    node_id: int,
    op_name: str,
    parent_fingerprint: str,
    max_ops: int,
    max_depth: int,
    min_splits: Any,
    vocab_size: int,
    max_seq_len: int,
    global_seen: set[str],
) -> tuple[Optional[DeletionChild], dict[str, Any]]:
    """Build, validate, compile, fingerprint, and dedup a single deletion child.

    Returns ``(child, meta)``.  ``child`` is None when the deletion was
    rejected; ``meta['reason']`` carries the rejection cause.  On success
    ``global_seen`` is mutated to include the new fingerprint.
    """
    child, meta = delete_node_by_bypass(graph, node_id)
    meta.update(
        {
            "parent_fingerprint": parent_fingerprint,
            "node_id": node_id,
            "op_name": op_name,
        }
    )
    if child is None:
        return None, meta
    try:
        validation = validate_graph(
            child,
            max_ops=max(1, int(max_ops)),
            max_depth=max(1, int(max_depth)),
            min_splits=min_splits,
        )
        if not validation.valid:
            meta["reason"] = "validation_failed"
            meta["errors"] = list(validation.errors)
            return None, meta
        compile_model(
            [child],
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
        )
        fingerprint = child.fingerprint()
    except (RuntimeError, ValueError, TypeError) as exc:
        meta["reason"] = "compile_failed"
        meta["error"] = str(exc)
        return None, meta
    if fingerprint == parent_fingerprint:
        meta["reason"] = "duplicate_parent_fingerprint"
        meta["fingerprint"] = fingerprint
        return None, meta
    if fingerprint in global_seen:
        meta["reason"] = "duplicate_planned_fingerprint"
        meta["fingerprint"] = fingerprint
        return None, meta
    global_seen.add(fingerprint)
    return (
        DeletionChild(
            graph=child,
            node_id=node_id,
            op_name=op_name,
            bypass_input_id=int(meta["bypass_input_id"]),
            fingerprint=fingerprint,
            pruned_nodes=int(meta.get("pruned_nodes") or 0),
        ),
        meta,
    )


def build_deletion_children(
    *,
    graph: ComputationGraph,
    parent_fingerprint: str,
    max_ops: int,
    max_depth: int,
    min_splits: Any,
    vocab_size: int,
    max_seq_len: int,
    global_seen: set[str],
    skip_scaffold: bool = True,
    rule_keys: Optional[Iterable[str]] = None,
) -> tuple[list[DeletionChild], list[dict[str, Any]]]:
    """Enumerate deletion-eligible ops, build & validate children.

    Dedup invariants:

    * Children whose fingerprint is in ``global_seen`` (mutated in place
      across calls) are rejected — keeps multi-parent driver runs from
      paying compute on architecturally-equivalent siblings.
    * Children whose fingerprint matches ``parent_fingerprint`` are rejected
      — those are no-op deletions that prune away orphan nodes without
      changing the live computation.
    * SCAFFOLD_OPS (add, linear_proj, normalisations, pointwise activations)
      are skipped when ``skip_scaffold=True``: deleting them mostly fails
      validation and burns compute proving load-bearing structure is
      load-bearing.

    Returns ``(children, rejected)``.  ``rejected`` carries per-op
    diagnostic dicts (``reason`` field) for observability.
    """
    children: list[DeletionChild] = []
    rejected: list[dict[str, Any]] = []
    rule_key_set = set(rule_keys) if rule_keys is not None else None
    reachable = graph.get_reachable_nodes()
    for node_id in graph.topological_order():
        node = graph.nodes[node_id]
        if node.is_input or node_id not in reachable:
            continue
        if skip_scaffold and node.op_name in SCAFFOLD_OPS:
            continue
        rule_key = f"{node_id}:{node.op_name}"
        if rule_key_set is not None and rule_key not in rule_key_set:
            continue
        child, meta = _attempt_one_deletion(
            graph=graph,
            node_id=node_id,
            op_name=node.op_name,
            parent_fingerprint=parent_fingerprint,
            max_ops=max_ops,
            max_depth=max_depth,
            min_splits=min_splits,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            global_seen=global_seen,
        )
        if child is None:
            rejected.append(meta)
        else:
            children.append(child)
    return children, rejected


def make_deletion_candidate(
    *,
    child: DeletionChild,
    parent_experiment_id: str,
    parent_result_id: str,
    parent_fingerprint: str,
    parent_loss_ratio: Optional[float],
    parent_graph: ComputationGraph,
    parent_composite_score: Optional[float] = None,
    active_prior: Optional[dict[str, Any]] = None,
) -> CausalAblationCandidate:
    """Wrap a DeletionChild as a CausalAblationCandidate for run_ablation_suite."""
    rule_key = f"{child.node_id}:{child.op_name}"
    prior_assessment = assess_local_edit_prior(
        active_prior, rule_type="node_delete", rule_key=rule_key
    )
    return CausalAblationCandidate(
        parent_experiment_id=parent_experiment_id,
        parent_result_id=parent_result_id,
        parent_fingerprint=parent_fingerprint,
        parent_loss_ratio=parent_loss_ratio,
        graph=parent_graph,
        rule_type="node_delete",
        rule_key=rule_key,
        hypothesis=f"node_delete:{rule_key}",
        context={
            "node_id": child.node_id,
            "deleted_op": child.op_name,
            "bypass_input_id": child.bypass_input_id,
            "child_fingerprint": child.fingerprint,
            "pruned_nodes": child.pruned_nodes,
            "parent_composite_score": parent_composite_score,
            "prior_assessment": prior_assessment,
        },
    )
