#!/usr/bin/env python
"""Factory-scale, GPU-FREE structural capability gate — the static analog of long_range_reach.

The measured probe (`measured_descriptors.py`) needs a forward/backward pass (~0.4s, GPU). At the
million-graph factory (`generate_novel_screened`) that's too slow. But the *necessary* part of the
filter is a pure graph property: induction (and any cross-position skill) requires a sequence-mixing
op on a path from input to output — if every op on the input→output cut is position-wise, no
information crosses positions and the design is structurally incapable, provable without running it.

Validated (2026-05-26, n=11225 induction-labeled graphs, CPU): `n_mixers_on_path` ROC **0.888** at
**~7300 graphs/s** — beats GPU long_range_reach (0.762) and has_attention_motif (0.846); the binary
"has a mixer on path" keeps 95.7% of capable designs. So this is the factory-scale first stage of the
cascade: CPU static gate (millions) → GPU mechanism probe on survivors (thousands) → real probe (tens).

Pure CPU, no torch — safe to import in the hot generation loop.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from research.synthesis.op_roles import OpRole, get_role
from research.tools.spectral_path_features import _topo


def on_path_op_names(nodes: Dict[str, Any] | List[Any]) -> List[str]:
    """Op names on any input→output path through the DAG (excludes inputs). CPU, ~microseconds."""
    node_list = list(nodes.values()) if isinstance(nodes, dict) else list(nodes)
    _, preds, succs = _topo(node_list)
    by_id = {n["id"]: n for n in node_list}
    sources = [i for i in by_id if not preds[i]]
    sinks = [i for i in by_id if not succs[i] and not by_id[i].get("is_input")] or [
        i for i in by_id if not succs[i]
    ]
    on_path = _reachable(sources, succs) & _reachable(sinks, preds)
    return [str(by_id[i]["op_name"]) for i in on_path if not by_id[i].get("is_input")]


def mixer_reach(nodes: Dict[str, Any] | List[Any]) -> Tuple[bool, int]:
    """(has_seq_mixer_on_path, n_mixers_on_path) over input→output paths. CPU, ~microseconds."""
    n_mix = sum(1 for op in on_path_op_names(nodes) if get_role(op) is OpRole.MIX)
    return n_mix > 0, n_mix


def mixer_chain_depth(nodes: Dict[str, Any] | List[Any]) -> int:
    """Longest chain of sequence-mixers along an input→output path (the ROUTING depth).

    Distinct from raw graph depth (size-confounded) and from the mixer COUNT: it measures how many
    times information is re-routed *in sequence*. The induction circuit needs depth>=2 (prev-token
    head → induction head). Validated: capable graphs median 3 (88% have >=2), incapable median 1
    (19% have >=2); ROC 0.897 vs induction at only 0.43 correlation with op-count. CPU DP, ~µs.
    """
    node_list = list(nodes.values()) if isinstance(nodes, dict) else list(nodes)
    order, preds, succs = _topo(node_list)
    by_id = {n["id"]: n for n in node_list}
    is_mix = {
        i: (
            not by_id[i].get("is_input")
            and get_role(str(by_id[i]["op_name"])) is OpRole.MIX
        )
        for i in by_id
    }
    depth: Dict[int, int] = {}
    for nid in order:
        depth[nid] = max((depth[p] for p in preds[nid]), default=0) + (
            1 if is_mix[nid] else 0
        )
    sinks = [i for i in by_id if not succs[i] and not by_id[i].get("is_input")] or list(
        by_id
    )
    return max((depth[s] for s in sinks), default=0)


def _reachable(starts: List[int], adj: Dict[int, List[int]]) -> set:
    """All nodes reachable from ``starts`` along ``adj`` (inclusive of starts)."""
    seen = set(starts)
    stack = list(starts)
    while stack:
        for nxt in adj[stack.pop()]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def passes_gate(graph: Any, min_mixers_on_path: int = 1) -> bool:
    """True iff the graph has >= ``min_mixers_on_path`` sequence-mixers on an input→output path.

    Accepts a ComputationGraph (uses .to_dict()) or a node dict/list. Necessary-condition gate:
    keeps 95.7% of induction-capable designs at min=1; raise the threshold to bias harder.
    """
    nodes = graph.to_dict()["nodes"] if hasattr(graph, "to_dict") else graph
    _, n_mix = mixer_reach(nodes)
    return n_mix >= min_mixers_on_path
