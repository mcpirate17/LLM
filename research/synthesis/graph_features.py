"""Graph-structure feature extraction for ML prediction.

Extracts a feature dict from a ComputationGraph's serialized dict (graph_json)
for use by the GBM pre-screener. All features are available BEFORE eval
(no forward pass needed), enabling cheap rejection of hopeless graphs.

Performance target: <5ms per graph.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from collections import Counter, deque
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Op categories matching synthesis/primitives.py registry
_CATEGORY_NAMES = (
    "mixing",
    "linear_algebra",
    "normalization",
    "structural",
    "sequence",
    "io",
    "channel_mixing",
    "routing",
    "parameterized",
    "math_space",
    "activation",
)

# Top ops by frequency — used as individual binary/count features
_TOP_OPS = (
    "linear_proj",
    "linear_proj_up",
    "linear_proj_down",
    "layer_norm",
    "rms_norm",
    "gelu",
    "swiglu",
    "attention",
    "rotary_embed",
    "causal_mask",
    "moe_topk",
    "moe_2expert",
    "selective_scan",
    "add",
    "concat",
    "split",
    "token_entropy",
    "route_topk",
    "route_lanes",
    "bottleneck_proj",
)

# Sentinel ops for boolean features
_ATTENTION_OPS = frozenset({"attention", "local_attention", "sliding_window_attention"})
_SSM_OPS = frozenset({"selective_scan", "state_space", "mamba_block"})
_MOE_OPS = frozenset({"moe_topk", "moe_2expert", "sparse_bottleneck_moe"})
_NORM_OPS = frozenset(
    {"layer_norm", "rms_norm", "group_norm", "batch_norm", "dynamic_norm"}
)
_RESIDUAL_OPS = frozenset({"add"})
_ROPE_OPS = frozenset({"rotary_embed", "rope"})
_CAUSAL_OPS = frozenset({"causal_mask"})


def _build_adjacency(
    nodes: Dict[str, dict],
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Build forward and reverse adjacency lists from node dict."""
    fwd: Dict[str, List[str]] = {}
    rev: Dict[str, List[str]] = {}
    for nid, node in nodes.items():
        fwd.setdefault(nid, [])
        rev.setdefault(nid, [])
        for inp in node.get("input_ids") or []:
            inp_str = str(inp)
            fwd.setdefault(inp_str, []).append(nid)
            rev[nid].append(inp_str)
    return fwd, rev


def _longest_path(nodes: Dict[str, dict], fwd: Dict[str, List[str]]) -> int:
    """Compute longest path (depth) via iterative topological DP.

    Uses Kahn's algorithm to process nodes in topological order,
    avoiding stack overflow on deep graphs.
    """
    # Compute in-degrees
    in_degree: Dict[str, int] = {nid: 0 for nid in nodes}
    for nid, children in fwd.items():
        for c in children:
            if c in in_degree:
                in_degree[c] += 1

    # Kahn's: start from roots (in_degree == 0)
    queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
    dist: Dict[str, int] = {nid: 0 for nid in queue}

    while queue:
        nid = queue.popleft()
        for c in fwd.get(nid, []):
            new_d = dist[nid] + 1
            if new_d > dist.get(c, 0):
                dist[c] = new_d
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)

    return max(dist.values(), default=0)


def _compute_depth_map(
    nodes: Dict[str, dict], fwd: Dict[str, List[str]]
) -> Dict[str, int]:
    """BFS depth assignment from root nodes. Shared by width/skip-connection features."""
    roots = [nid for nid, node in nodes.items() if not (node.get("input_ids") or [])]
    depth_map: Dict[str, int] = {}
    queue = list(roots)
    for r in queue:
        depth_map[r] = 0
    i = 0
    while i < len(queue):
        nid = queue[i]
        i += 1
        for child in fwd.get(nid, []):
            if child not in depth_map:
                depth_map[child] = depth_map[nid] + 1
                queue.append(child)
    return depth_map


def _width_at_depths(depth_map: Dict[str, int]) -> int:
    """Max number of nodes at the same depth level."""
    if not depth_map:
        return 1
    counts = Counter(depth_map.values())
    return max(counts.values())


def _count_skip_connections(nodes: Dict[str, dict], depth_map: Dict[str, int]) -> int:
    """Count 'add' nodes whose inputs come from different depths (skip connections)."""
    count = 0
    for nid, node in nodes.items():
        if node.get("op_name") != "add":
            continue
        inputs = node.get("input_ids") or []
        if len(inputs) < 2:
            continue
        depths = {depth_map.get(str(inp), 0) for inp in inputs}
        if len(depths) > 1:
            count += 1
    return count


def extract_graph_features(graph_json: Any) -> Dict[str, float]:
    """Extract feature dict from graph structure for ML prediction.

    Args:
        graph_json: Either a dict (graph.to_dict()) or a JSON string.

    Returns:
        Dict of feature_name → float value. All values are numeric.
        Returns empty dict on parse failure.
    """
    if isinstance(graph_json, str):
        try:
            graph_json = json.loads(graph_json)
        except (json.JSONDecodeError, TypeError):
            return {}

    if not isinstance(graph_json, dict):
        return {}

    nodes = graph_json.get("nodes") or {}
    metadata = graph_json.get("metadata") or {}

    if not nodes:
        return {}

    # Collect all op names
    op_names: List[str] = []
    for node in nodes.values():
        op = node.get("op_name", "")
        if op and op != "input":
            op_names.append(op)

    op_counter = Counter(op_names)
    op_set = set(op_names)
    n_nodes = len(nodes)
    n_ops = len(op_names)

    # Build adjacency for topology features
    fwd, rev = _build_adjacency(nodes)
    n_edges = sum(len(children) for children in fwd.values())

    features: Dict[str, float] = {}

    # ── Topology ──
    features["n_nodes"] = float(n_nodes)
    features["n_edges"] = float(n_edges)
    features["n_ops"] = float(n_ops)
    depth_map = _compute_depth_map(nodes, fwd)
    features["depth"] = float(_longest_path(nodes, fwd))
    features["width"] = float(_width_at_depths(depth_map))
    features["n_unique_ops"] = float(len(op_set))
    features["n_skip_connections"] = float(_count_skip_connections(nodes, depth_map))
    features["edge_density"] = n_edges / max(n_nodes, 1)

    # ── Op histogram (top ops as individual features) ──
    for op in _TOP_OPS:
        features[f"op_{op}"] = float(op_counter.get(op, 0))

    # ── Category histogram ──
    # Infer category from op name prefix or known mappings
    try:
        from .primitives import get_primitive

        cat_counter: Counter = Counter()
        for op in op_names:
            try:
                prim = get_primitive(op)
                cat = prim.category
                cat_name = cat.value if hasattr(cat, "value") else str(cat)
                cat_counter[cat_name] += 1
            except (KeyError, Exception):
                cat_counter["unknown"] += 1
    except ImportError:
        cat_counter = Counter()

    for cat in _CATEGORY_NAMES:
        features[f"cat_{cat}"] = float(cat_counter.get(cat, 0))

    # ── Boolean sentinel features ──
    features["has_attention"] = float(bool(op_set & _ATTENTION_OPS))
    features["has_ssm"] = float(bool(op_set & _SSM_OPS))
    features["has_moe"] = float(bool(op_set & _MOE_OPS))
    features["has_norm"] = float(bool(op_set & _NORM_OPS))
    features["has_residual"] = float(bool(op_set & _RESIDUAL_OPS))
    features["has_rope"] = float(bool(op_set & _ROPE_OPS))
    features["has_causal_mask"] = float(bool(op_set & _CAUSAL_OPS))

    # ── Metadata features ──
    templates_used = metadata.get("templates_used") or []
    features["n_templates_used"] = float(len(templates_used))

    # Model dim
    model_dim = graph_json.get("model_dim", 0)
    features["model_dim"] = float(model_dim) if model_dim else 0.0

    return features


_op_stats_cache: Dict[str, Tuple[Dict[str, Tuple[float, float]], float]] = {}
_OP_STATS_TTL: float = 60.0  # seconds


def load_op_stats(
    db_path: str = "research/lab_notebook.db",
) -> Dict[str, Tuple[float, float]]:
    """Load op_stats table with TTL cache (60s). Returns dict of op_name → (s1_rate, mean_loss)."""
    import time

    now = time.monotonic()
    cached = _op_stats_cache.get(db_path)
    if cached is not None:
        data, ts = cached
        if now - ts < _OP_STATS_TTL:
            return data

    conn: Optional[sqlite3.Connection] = None
    try:
        conn = sqlite3.connect(db_path, timeout=2.0)
        conn.execute("PRAGMA busy_timeout=2000")
        rows = conn.execute(
            "SELECT op_name, eval_count, s1_pass_count, mean_loss FROM op_stats"
        ).fetchall()
    except sqlite3.Error as exc:
        logger.debug("load_op_stats failed for %s: %s", db_path, exc)
        return {}
    finally:
        if conn is not None:
            conn.close()

    stats: Dict[str, Tuple[float, float]] = {}
    for op_name, eval_count, s1_count, mean_loss in rows:
        ec = max(eval_count, 1)
        s1_rate = s1_count / ec
        ml = (
            float(mean_loss)
            if mean_loss is not None and math.isfinite(mean_loss)
            else 1.0
        )
        stats[op_name] = (s1_rate, ml)
    _op_stats_cache[db_path] = (stats, now)
    return stats


def enrich_with_op_stats(
    features: Dict[str, float],
    op_names: List[str],
    db_path: str = "research/lab_notebook.db",
    *,
    preloaded: Optional[Dict[str, Tuple[float, float]]] = None,
) -> None:
    """Add historical op performance features from op_stats table (in-place).

    Adds: mean_op_s1_rate, min_op_s1_rate, mean_op_loss.
    Pass preloaded=load_op_stats() to avoid repeated DB queries in bulk.
    """
    if not op_names:
        return

    stats = preloaded if preloaded is not None else load_op_stats(db_path)

    s1_rates: List[float] = []
    losses: List[float] = []
    for op in set(op_names):
        if op in stats:
            s1_rates.append(stats[op][0])
            losses.append(stats[op][1])

    features["mean_op_s1_rate"] = sum(s1_rates) / len(s1_rates) if s1_rates else 0.0
    features["min_op_s1_rate"] = min(s1_rates) if s1_rates else 0.0
    features["mean_op_loss"] = sum(losses) / len(losses) if losses else 1.0
