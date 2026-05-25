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


def mixer_reach(nodes: Dict[str, Any] | List[Any]) -> Tuple[bool, int]:
    """(has_seq_mixer_on_path, n_mixers_on_path) over input→output paths. CPU, ~microseconds."""
    node_list = list(nodes.values()) if isinstance(nodes, dict) else list(nodes)
    _, preds, succs = _topo(node_list)
    by_id = {n["id"]: n for n in node_list}
    sources = [i for i in by_id if not preds[i]]
    sinks = [i for i in by_id if not succs[i] and not by_id[i].get("is_input")] or [
        i for i in by_id if not succs[i]
    ]
    on_path = _reachable(sources, succs) & _reachable(sinks, preds)
    n_mix = sum(
        1
        for i in on_path
        if not by_id[i].get("is_input")
        and get_role(str(by_id[i]["op_name"])) is OpRole.MIX
    )
    return n_mix > 0, n_mix


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
