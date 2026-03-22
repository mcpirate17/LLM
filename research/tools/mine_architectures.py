#!/usr/bin/env python3
"""Architecture Data Mining — extract patterns from 7K+ program_results.

Extracts graph-level features from every program_result, builds a feature
matrix, and runs statistical analysis to find what predicts S1 success
and low loss_ratio.

Analyses:
  1. Op presence → S1 correlation (point-biserial)
  2. Op pair synergy/toxicity (lift analysis)
  3. Graph topology → loss_ratio correlation (Spearman)
  4. Feature importance via gradient boosting
  5. Failure mode clustering by op pattern
  6. Template/motif effectiveness ranking
  7. Op interaction network (which ops amplify/cancel each other)

Usage:
    python -m research.tools.mine_architectures                  # full analysis
    python -m research.tools.mine_architectures --section ops    # just op analysis
    python -m research.tools.mine_architectures --top 30         # top 30 results per section
    python -m research.tools.mine_architectures --csv output.csv # export feature matrix
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)


# ── Feature extraction ───────────────────────────────────────────────


@dataclass(slots=True)
class GraphFeatures:
    """Features extracted from a single program_result entry."""

    result_id: str
    fingerprint: str

    # Outcome
    s0_passed: bool
    s1_passed: bool
    loss_ratio: Optional[float]
    initial_loss: Optional[float]
    final_loss: Optional[float]

    # Graph structure
    n_ops: int
    n_unique_ops: int
    depth: int
    n_edges: int
    has_residual: bool
    n_residuals: int  # count of 'add' ops used as residuals
    model_dim: int

    # Op census
    ops: frozenset  # set of unique op names
    op_counts: Dict[str, int]  # op_name → count

    # Op categories
    n_norm_ops: int
    n_linear_ops: int
    n_activation_ops: int
    n_routing_ops: int
    n_attention_ops: int
    n_recurrence_ops: int
    n_math_space_ops: int

    # Template/motif metadata
    template: Optional[str]
    motifs_used: List[str]

    # Numeric metrics (from program_results)
    param_count: Optional[int]
    forward_time_ms: Optional[float]
    peak_memory_mb: Optional[float]
    wikitext_score: Optional[float]
    error_type: Optional[str]
    failure_op: Optional[str]


_NORM_OPS = frozenset({"rmsnorm", "layernorm"})
_LINEAR_OPS = frozenset(
    {
        "linear_proj",
        "linear_proj_up",
        "linear_proj_down",
        "fused_linear_gelu",
        "gated_linear",
        "tied_proj",
        "bottleneck_proj",
        "low_rank_proj",
        "grouped_linear",
        "shared_basis_proj",
    }
)
_ACTIVATION_OPS = frozenset(
    {
        "relu",
        "gelu",
        "silu",
        "sigmoid",
        "tanh",
        "sin",
        "cos",
        "exp",
        "log",
        "sqrt",
        "square",
        "abs",
        "sign_ste",
        "softmax_last",
    }
)
_ROUTING_OPS = frozenset(
    {
        "early_exit",
        "cascade",
        "route_lanes",
        "route_recursion",
        "adaptive_recursion",
        "adaptive_lane_mixer",
        "n_way_sparse_router",
        "token_merge",
        "moe_topk",
        "moe_2expert",
        "topk_gate",
        "relu_gate_routing",
        "compression_mixture_experts",
        "routing_conditioned_compression",
        "speculative",
    }
)
_ATTENTION_OPS = frozenset(
    {
        "softmax_attention",
        "linear_attention",
        "graph_attention",
        "diff_attention",
        "local_window_attn",
        "stdp_attention",
    }
)
_RECURRENCE_OPS = frozenset(
    {
        "state_space",
        "rwkv_time_mixing",
        "rwkv_channel",
        "gated_delta",
        "selective_scan",
        "conv_only",
        "conv1d_seq",
    }
)
_MATH_SPACE_OPS = frozenset(
    {
        "tropical_add",
        "tropical_matmul",
        "tropical_center",
        "tropical_gate",
        "tropical_attention",
        "tropical_moe",
        "tropical_router",
        "geometric_product",
        "rotor_transform",
        "grade_select",
        "grade_mix",
        "lif_neuron",
        "spike_rate_code",
        "sparse_threshold",
        "exp_map",
        "log_map",
        "hyp_linear",
        "hyp_tangent_nonlinear",
        "poincare_add",
        "padic_expand",
        "padic_gate",
        "padic_residual",
    }
)


def _extract_features(row: sqlite3.Row) -> Optional[GraphFeatures]:
    """Extract features from a single program_results row."""
    graph_json_str = row["graph_json"]
    if not graph_json_str:
        return None

    try:
        g = json.loads(graph_json_str)
    except (json.JSONDecodeError, TypeError):
        return None

    nodes = g.get("nodes", {})
    if not isinstance(nodes, dict):
        return None

    # Parse ops
    op_counts: Dict[str, int] = Counter()
    n_edges = 0
    has_residual = False
    n_residuals = 0
    max_depth = 0

    for nid, node in nodes.items():
        if not isinstance(node, dict):
            continue
        op_name = node.get("op_name", "")
        if not op_name or op_name == "input":
            continue
        op_counts[op_name] += 1
        inputs = node.get("input_ids", [])
        n_edges += len(inputs)
        # Compute depth (longest path from input)
        # Simple: count inputs recursively (approximate)
        if op_name == "add" and len(inputs) >= 2:
            n_residuals += 1
            has_residual = True

    ops = frozenset(op_counts.keys())
    n_ops = sum(op_counts.values())

    # Approximate depth from topological structure
    # Use the longest chain heuristic
    node_depth: Dict[str, int] = {}
    for nid, node in sorted(
        nodes.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0
    ):
        if not isinstance(node, dict):
            continue
        if node.get("op_name") == "input" or node.get("is_input"):
            node_depth[nid] = 0
        else:
            parent_depths = [
                node_depth.get(str(pid), 0) for pid in node.get("input_ids", [])
            ]
            node_depth[nid] = max(parent_depths, default=0) + 1
    max_depth = max(node_depth.values()) if node_depth else 0

    # Metadata
    meta = g.get("metadata", {})
    templates = meta.get("templates_used", [])
    template = templates[0] if templates else None
    motifs = meta.get("motifs_used", [])

    return GraphFeatures(
        result_id=row["result_id"],
        fingerprint=row["graph_fingerprint"] or "",
        s0_passed=bool(row["stage0_passed"]),
        s1_passed=bool(row["stage1_passed"]),
        loss_ratio=row["loss_ratio"],
        initial_loss=row["initial_loss"],
        final_loss=row["final_loss"],
        n_ops=n_ops,
        n_unique_ops=len(ops),
        depth=max_depth,
        n_edges=n_edges,
        has_residual=has_residual,
        n_residuals=n_residuals,
        model_dim=g.get("model_dim", 64),
        ops=ops,
        op_counts=dict(op_counts),
        n_norm_ops=sum(1 for o in ops if o in _NORM_OPS),
        n_linear_ops=sum(1 for o in ops if o in _LINEAR_OPS),
        n_activation_ops=sum(1 for o in ops if o in _ACTIVATION_OPS),
        n_routing_ops=sum(1 for o in ops if o in _ROUTING_OPS),
        n_attention_ops=sum(1 for o in ops if o in _ATTENTION_OPS),
        n_recurrence_ops=sum(1 for o in ops if o in _RECURRENCE_OPS),
        n_math_space_ops=sum(1 for o in ops if o in _MATH_SPACE_OPS),
        template=template,
        motifs_used=motifs if isinstance(motifs, list) else [],
        param_count=row["param_count"],
        forward_time_ms=row["forward_time_ms"],
        peak_memory_mb=row["peak_memory_mb"],
        wikitext_score=row["wikitext_score"],
        error_type=row["error_type"],
        failure_op=row["failure_op"],
    )


# ── Analysis functions ───────────────────────────────────────────────


def analyze_op_correlations(features: List[GraphFeatures], top_n: int = 30) -> None:
    """Op presence → S1 pass rate and loss_ratio correlation."""
    print("\n" + "=" * 80)
    print("  OP PRESENCE → S1 PASS RATE  (point-biserial correlation)")
    print("=" * 80)

    # Build op → (s1_count, total_count, loss_ratios) map
    op_data: Dict[str, Dict[str, Any]] = {}
    s0_entries = [f for f in features if f.s0_passed]
    global_s1_rate = sum(1 for f in s0_entries if f.s1_passed) / max(len(s0_entries), 1)

    for f in s0_entries:
        for op in f.ops:
            if op not in op_data:
                op_data[op] = {"s1": 0, "total": 0, "lrs": []}
            op_data[op]["total"] += 1
            if f.s1_passed:
                op_data[op]["s1"] += 1
            if f.loss_ratio is not None:
                op_data[op]["lrs"].append(f.loss_ratio)

    # Compute point-biserial correlation and effect size
    results = []
    for op, d in op_data.items():
        if d["total"] < 10:
            continue
        s1_rate = d["s1"] / d["total"]
        # Effect vs global baseline
        lift = s1_rate / max(global_s1_rate, 1e-6)
        avg_lr = sum(d["lrs"]) / len(d["lrs"]) if d["lrs"] else None
        best_lr = min(d["lrs"]) if d["lrs"] else None
        results.append((op, s1_rate, d["total"], d["s1"], lift, avg_lr, best_lr))

    results.sort(key=lambda x: -x[1])

    print(
        f"\nGlobal S1 rate: {global_s1_rate:.1%} ({sum(1 for f in s0_entries if f.s1_passed)}/{len(s0_entries)})"
    )
    print(
        f"\n{'Op':25s} {'S1%':>6s} {'N':>5s} {'S1':>4s} {'Lift':>5s} {'AvgLR':>7s} {'BestLR':>7s}"
    )
    print("-" * 80)
    for op, s1_rate, n, s1, lift, avg_lr, best_lr in results[:top_n]:
        lr_str = f"{avg_lr:.3f}" if avg_lr is not None else "-"
        best_str = f"{best_lr:.3f}" if best_lr is not None else "-"
        marker = " ★" if lift > 2.0 else (" ▼" if lift < 0.5 else "")
        print(
            f"{op:25s} {s1_rate:5.1%} {n:5d} {s1:4d} {lift:5.1f}x {lr_str:>7s} {best_str:>7s}{marker}"
        )

    print("\n--- Bottom (worst S1 rate, min 20 obs) ---")
    bottom = [r for r in results if r[2] >= 20]
    bottom.sort(key=lambda x: x[1])
    for op, s1_rate, n, s1, lift, avg_lr, best_lr in bottom[:15]:
        lr_str = f"{avg_lr:.3f}" if avg_lr is not None else "-"
        best_str = f"{best_lr:.3f}" if best_lr is not None else "-"
        print(
            f"{op:25s} {s1_rate:5.1%} {n:5d} {s1:4d} {lift:5.1f}x {lr_str:>7s} {best_str:>7s}"
        )


def analyze_op_pairs(features: List[GraphFeatures], top_n: int = 25) -> None:
    """Op pair synergy analysis via lift."""
    print("\n" + "=" * 80)
    print("  OP PAIR SYNERGY  (lift = pair_s1_rate / expected_if_independent)")
    print("=" * 80)

    s0_entries = [f for f in features if f.s0_passed]
    len(s0_entries)

    # Per-op S1 rates
    op_s1: Dict[str, Tuple[int, int]] = {}  # op → (s1, total)
    for f in s0_entries:
        for op in f.ops:
            if op not in op_s1:
                op_s1[op] = [0, 0]
            op_s1[op][1] += 1
            if f.s1_passed:
                op_s1[op][0] += 1

    # Pair analysis
    pair_data: Dict[Tuple[str, str], List[int]] = {}  # (a,b) → [s1, total]
    for f in s0_entries:
        sorted_ops = sorted(f.ops)
        for i, a in enumerate(sorted_ops):
            for b in sorted_ops[i + 1 :]:
                pair = (a, b)
                if pair not in pair_data:
                    pair_data[pair] = [0, 0]
                pair_data[pair][1] += 1
                if f.s1_passed:
                    pair_data[pair][0] += 1

    # Compute lift: observed_pair_s1 / (p(s1|a) * p(s1|b) * expected_joint)
    results = []
    for (a, b), (s1, total) in pair_data.items():
        if total < 10:
            continue
        pair_s1_rate = s1 / total
        # Independent expectation
        rate_a = op_s1[a][0] / max(op_s1[a][1], 1)
        rate_b = op_s1[b][0] / max(op_s1[b][1], 1)
        expected = rate_a * rate_b
        lift = pair_s1_rate / max(expected, 1e-6) if expected > 0 else 0
        results.append((a, b, pair_s1_rate, total, s1, lift))

    # Top synergistic pairs (high lift)
    results.sort(key=lambda x: -x[5])
    print(f"\n{'Op A':20s} {'Op B':20s} {'S1%':>6s} {'N':>5s} {'S1':>4s} {'Lift':>6s}")
    print("-" * 75)
    shown = 0
    for a, b, s1r, n, s1, lift in results:
        if shown >= top_n:
            break
        # Skip trivial pairs (both are universal ops)
        if a in ("add", "layernorm", "rmsnorm", "linear_proj") and b in (
            "add",
            "layernorm",
            "rmsnorm",
            "linear_proj",
        ):
            continue
        marker = " ★★" if lift > 5.0 else (" ★" if lift > 2.0 else "")
        print(f"{a:20s} {b:20s} {s1r:5.1%} {n:5d} {s1:4d} {lift:5.1f}x{marker}")
        shown += 1

    # Bottom: toxic pairs (low lift, meaning they cancel each other)
    results.sort(key=lambda x: x[5])
    print("\n--- Toxic pairs (lift < 0.5, min 15 obs) ---")
    print(f"{'Op A':20s} {'Op B':20s} {'S1%':>6s} {'N':>5s} {'S1':>4s} {'Lift':>6s}")
    print("-" * 75)
    shown = 0
    for a, b, s1r, n, s1, lift in results:
        if n < 15 or lift >= 0.5:
            continue
        if shown >= 15:
            break
        print(f"{a:20s} {b:20s} {s1r:5.1%} {n:5d} {s1:4d} {lift:5.1f}x")
        shown += 1


def analyze_topology_correlations(
    features: List[GraphFeatures], top_n: int = 20
) -> None:
    """Graph topology features → loss_ratio Spearman correlation."""
    print("\n" + "=" * 80)
    print("  TOPOLOGY → LOSS_RATIO  (Spearman rank correlation)")
    print("=" * 80)

    # Filter to entries with loss_ratio
    entries = [f for f in features if f.loss_ratio is not None and f.s0_passed]
    if len(entries) < 30:
        print("  Insufficient data for correlation analysis")
        return

    # Build feature vectors
    topo_features = {
        "n_ops": [f.n_ops for f in entries],
        "n_unique_ops": [f.n_unique_ops for f in entries],
        "depth": [f.depth for f in entries],
        "n_edges": [f.n_edges for f in entries],
        "n_residuals": [f.n_residuals for f in entries],
        "n_norm_ops": [f.n_norm_ops for f in entries],
        "n_linear_ops": [f.n_linear_ops for f in entries],
        "n_activation_ops": [f.n_activation_ops for f in entries],
        "n_routing_ops": [f.n_routing_ops for f in entries],
        "n_attention_ops": [f.n_attention_ops for f in entries],
        "n_recurrence_ops": [f.n_recurrence_ops for f in entries],
        "n_math_space_ops": [f.n_math_space_ops for f in entries],
        "has_residual": [int(f.has_residual) for f in entries],
        "ops_per_depth": [f.n_ops / max(f.depth, 1) for f in entries],
        "residual_density": [f.n_residuals / max(f.n_ops, 1) for f in entries],
    }
    loss_ratios = [f.loss_ratio for f in entries]

    # Spearman rank correlation
    def _spearman(xs, ys):
        n = len(xs)
        if n < 3:
            return 0.0
        rx = _ranks(xs)
        ry = _ranks(ys)
        d_sq = sum((a - b) ** 2 for a, b in zip(rx, ry))
        return 1 - 6 * d_sq / (n * (n * n - 1))

    def _ranks(vals):
        indexed = sorted(enumerate(vals), key=lambda x: x[1])
        ranks = [0.0] * len(vals)
        for rank, (idx, _) in enumerate(indexed):
            ranks[idx] = float(rank)
        return ranks

    results = []
    for feat_name, feat_vals in topo_features.items():
        rho = _spearman(feat_vals, loss_ratios)
        results.append((feat_name, rho))

    results.sort(key=lambda x: x[1])  # Most negative = best predictor of LOW loss

    print(f"\nN = {len(entries)} entries with loss_ratio")
    print(f"\n{'Feature':25s} {'Spearman ρ':>11s}  Interpretation")
    print("-" * 75)
    for feat_name, rho in results:
        if abs(rho) < 0.02:
            interp = "no effect"
        elif rho < -0.1:
            interp = "MORE → LOWER loss (good)"
        elif rho < 0:
            interp = "weak negative"
        elif rho > 0.1:
            interp = "MORE → HIGHER loss (bad)"
        else:
            interp = "weak positive"
        marker = " ★" if abs(rho) > 0.1 else ""
        print(f"{feat_name:25s} {rho:+10.4f}  {interp}{marker}")


def analyze_templates(features: List[GraphFeatures], top_n: int = 30) -> None:
    """Template effectiveness ranking."""
    print("\n" + "=" * 80)
    print("  TEMPLATE EFFECTIVENESS")
    print("=" * 80)

    s0_entries = [f for f in features if f.s0_passed and f.template]

    tpl_data: Dict[str, Dict[str, Any]] = {}
    for f in s0_entries:
        t = f.template
        if t not in tpl_data:
            tpl_data[t] = {"s1": 0, "total": 0, "lrs": []}
        tpl_data[t]["total"] += 1
        if f.s1_passed:
            tpl_data[t]["s1"] += 1
        if f.loss_ratio is not None:
            tpl_data[t]["lrs"].append(f.loss_ratio)

    results = []
    for t, d in tpl_data.items():
        if d["total"] < 3:
            continue
        s1_rate = d["s1"] / d["total"]
        avg_lr = sum(d["lrs"]) / len(d["lrs"]) if d["lrs"] else None
        best_lr = min(d["lrs"]) if d["lrs"] else None
        results.append((t, s1_rate, d["total"], d["s1"], avg_lr, best_lr))

    results.sort(key=lambda x: -x[1])

    print(
        f"\n{'Template':40s} {'S1%':>6s} {'N':>5s} {'S1':>4s} {'AvgLR':>7s} {'BestLR':>7s}"
    )
    print("-" * 80)
    for t, s1_rate, n, s1, avg_lr, best_lr in results[:top_n]:
        lr_str = f"{avg_lr:.3f}" if avg_lr is not None else "-"
        best_str = f"{best_lr:.3f}" if best_lr is not None else "-"
        print(f"{t:40s} {s1_rate:5.1%} {n:5d} {s1:4d} {lr_str:>7s} {best_str:>7s}")


def analyze_failure_modes(features: List[GraphFeatures], top_n: int = 20) -> None:
    """Failure mode analysis — what ops cause which errors."""
    print("\n" + "=" * 80)
    print("  FAILURE MODE ANALYSIS")
    print("=" * 80)

    failed = [f for f in features if not f.s1_passed and f.error_type]

    # Error type distribution
    error_counts = Counter(f.error_type for f in failed)
    print(f"\nError type distribution ({len(failed)} failures):")
    for err, n in error_counts.most_common(10):
        print(f"  {err:30s}: {n:5d} ({n / len(failed) * 100:.1f}%)")

    # Ops most associated with each error type
    print(f"\n{'Error Type':25s} {'Top Correlated Ops':50s}")
    print("-" * 80)
    for err_type, _ in error_counts.most_common(8):
        err_entries = [f for f in failed if f.error_type == err_type]
        # Count op frequency in this error vs overall
        op_freq = Counter()
        for f in err_entries:
            for op in f.ops:
                op_freq[op] += 1
        # Normalize by total frequency
        all_op_freq = Counter()
        for f in failed:
            for op in f.ops:
                all_op_freq[op] += 1
        # Lift: freq_in_error / freq_overall
        op_lifts = []
        for op, count in op_freq.items():
            if all_op_freq[op] < 5:
                continue
            expected = all_op_freq[op] * len(err_entries) / max(len(failed), 1)
            lift = count / max(expected, 1e-6)
            if lift > 1.3:
                op_lifts.append((op, lift, count))
        op_lifts.sort(key=lambda x: -x[1])
        top_ops = ", ".join(f"{op}({lift:.1f}x)" for op, lift, _ in op_lifts[:4])
        print(f"{err_type:25s} {top_ops:50s}")

    # Failure ops (directly attributed)
    fail_ops = Counter(f.failure_op for f in failed if f.failure_op)
    if fail_ops:
        print("\nDirectly attributed failure ops:")
        for op, n in fail_ops.most_common(15):
            print(f"  {op:25s}: {n:5d}")


def analyze_feature_importance(features: List[GraphFeatures], top_n: int = 20) -> None:
    """Feature importance for S1 prediction via simple decision stump ensemble."""
    print("\n" + "=" * 80)
    print("  FEATURE IMPORTANCE  (information gain ranking)")
    print("=" * 80)

    s0_entries = [f for f in features if f.s0_passed]
    if len(s0_entries) < 100:
        print("  Insufficient data")
        return

    # Build binary op-presence features + numeric topology features
    all_ops = sorted({op for f in s0_entries for op in f.ops})
    # Filter to ops with >= 20 occurrences
    op_freq = Counter(op for f in s0_entries for op in f.ops)
    all_ops = [op for op in all_ops if op_freq[op] >= 20]

    labels = [int(f.s1_passed) for f in s0_entries]
    pos_rate = sum(labels) / len(labels)

    # Information gain for each binary feature
    def _entropy(p):
        if p <= 0 or p >= 1:
            return 0.0
        return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

    base_entropy = _entropy(pos_rate)

    results = []

    # Op presence features
    for op in all_ops:
        present = [int(op in f.ops) for f in s0_entries]
        n1 = sum(present)
        n0 = len(present) - n1
        if n0 < 10 or n1 < 10:
            continue
        # S1 rate when present vs absent
        s1_when_present = sum(l for l, p in zip(labels, present) if p) / max(n1, 1)
        s1_when_absent = sum(l for l, p in zip(labels, present) if not p) / max(n0, 1)
        # Information gain
        h_present = _entropy(s1_when_present)
        h_absent = _entropy(s1_when_absent)
        ig = base_entropy - (n1 / len(labels) * h_present + n0 / len(labels) * h_absent)
        direction = "+" if s1_when_present > s1_when_absent else "-"
        results.append((f"op:{op}", ig, direction, s1_when_present, s1_when_absent, n1))

    # Topology features
    topo = {
        "n_ops": [f.n_ops for f in s0_entries],
        "depth": [f.depth for f in s0_entries],
        "n_residuals": [f.n_residuals for f in s0_entries],
        "n_routing_ops": [f.n_routing_ops for f in s0_entries],
        "n_attention_ops": [f.n_attention_ops for f in s0_entries],
        "n_recurrence_ops": [f.n_recurrence_ops for f in s0_entries],
        "has_residual": [int(f.has_residual) for f in s0_entries],
    }

    for feat_name, vals in topo.items():
        # Split at median
        median = sorted(vals)[len(vals) // 2]
        above = [int(v > median) for v in vals]
        n1 = sum(above)
        n0 = len(above) - n1
        if n0 < 10 or n1 < 10:
            continue
        s1_high = sum(l for l, a in zip(labels, above) if a) / max(n1, 1)
        s1_low = sum(l for l, a in zip(labels, above) if not a) / max(n0, 1)
        h_high = _entropy(s1_high)
        h_low = _entropy(s1_low)
        ig = base_entropy - (n1 / len(labels) * h_high + n0 / len(labels) * h_low)
        direction = "+" if s1_high > s1_low else "-"
        results.append((f"topo:{feat_name}>median", ig, direction, s1_high, s1_low, n1))

    results.sort(key=lambda x: -x[1])

    print(f"\nBase S1 rate: {pos_rate:.1%}")
    print(
        f"\n{'Feature':35s} {'IG':>7s} {'Dir':>3s} {'S1%|yes':>8s} {'S1%|no':>8s} {'N(yes)':>7s}"
    )
    print("-" * 80)
    for feat, ig, direction, s1_yes, s1_no, n_yes in results[:top_n]:
        print(
            f"{feat:35s} {ig:.4f}  {direction:>2s}  {s1_yes:7.1%}  {s1_no:7.1%}  {n_yes:6d}"
        )


def print_summary(features: List[GraphFeatures]) -> None:
    """Print dataset summary."""
    print("=" * 80)
    print("  ARCHITECTURE DATA MINING — DATASET SUMMARY")
    print("=" * 80)

    n = len(features)
    n_s0 = sum(1 for f in features if f.s0_passed)
    n_s1 = sum(1 for f in features if f.s1_passed)
    n_lr = sum(1 for f in features if f.loss_ratio is not None)
    n_fp = len({f.fingerprint for f in features if f.fingerprint})
    n_ops = len({op for f in features for op in f.ops})

    print(f"\n  Entries:          {n:,d}")
    print(f"  Unique archs:     {n_fp:,d}")
    print(f"  S0 passed:        {n_s0:,d} ({n_s0 / n:.1%})")
    print(f"  S1 passed:        {n_s1:,d} ({n_s1 / max(n_s0, 1):.1%} of S0)")
    print(f"  Has loss_ratio:   {n_lr:,d}")
    print(f"  Unique ops:       {n_ops:d}")

    # Loss ratio distribution for S1 passers
    s1_lrs = sorted(
        f.loss_ratio for f in features if f.s1_passed and f.loss_ratio is not None
    )
    if s1_lrs:
        print("\n  S1 loss_ratio distribution:")
        print(
            f"    min={s1_lrs[0]:.4f}  p25={s1_lrs[len(s1_lrs) // 4]:.4f}  "
            f"median={s1_lrs[len(s1_lrs) // 2]:.4f}  p75={s1_lrs[3 * len(s1_lrs) // 4]:.4f}  "
            f"max={s1_lrs[-1]:.4f}"
        )


def export_csv(features: List[GraphFeatures], path: str) -> None:
    """Export feature matrix to CSV for external analysis."""
    import csv

    all_ops = sorted({op for f in features for op in f.ops})

    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        header = [
            "result_id",
            "fingerprint",
            "s0_passed",
            "s1_passed",
            "loss_ratio",
            "n_ops",
            "n_unique_ops",
            "depth",
            "n_edges",
            "n_residuals",
            "n_norm_ops",
            "n_linear_ops",
            "n_activation_ops",
            "n_routing_ops",
            "n_attention_ops",
            "n_recurrence_ops",
            "n_math_space_ops",
            "template",
            "param_count",
            "forward_time_ms",
            "error_type",
            "failure_op",
        ] + [f"op:{op}" for op in all_ops]

        writer.writerow(header)
        for f in features:
            row = [
                f.result_id,
                f.fingerprint,
                int(f.s0_passed),
                int(f.s1_passed),
                f.loss_ratio,
                f.n_ops,
                f.n_unique_ops,
                f.depth,
                f.n_edges,
                f.n_residuals,
                f.n_norm_ops,
                f.n_linear_ops,
                f.n_activation_ops,
                f.n_routing_ops,
                f.n_attention_ops,
                f.n_recurrence_ops,
                f.n_math_space_ops,
                f.template or "",
                f.param_count,
                f.forward_time_ms,
                f.error_type or "",
                f.failure_op or "",
            ] + [int(op in f.ops) for op in all_ops]
            writer.writerow(row)

    logger.info("Exported %d rows × %d columns to %s", len(features), len(header), path)


# ── Main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Mine architecture patterns from program_results"
    )
    parser.add_argument(
        "--db", default="research/lab_notebook.db", help="Database path"
    )
    parser.add_argument(
        "--section",
        choices=[
            "ops",
            "pairs",
            "topology",
            "templates",
            "failures",
            "importance",
            "all",
        ],
        default="all",
    )
    parser.add_argument("--top", type=int, default=30, help="Top N results per section")
    parser.add_argument("--csv", help="Export feature matrix to CSV")
    args = parser.parse_args()

    db_path = args.db
    if not Path(db_path).exists():
        logger.error("Database not found: %s", db_path)
        sys.exit(1)

    t0 = time.time()
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row

    logger.info("Loading program_results...")
    rows = db.execute(
        "SELECT * FROM program_results WHERE graph_json IS NOT NULL"
    ).fetchall()
    logger.info("Loaded %d rows in %.1fs", len(rows), time.time() - t0)

    t1 = time.time()
    features = []
    for row in rows:
        f = _extract_features(row)
        if f is not None:
            features.append(f)
    logger.info(
        "Extracted features from %d entries in %.1fs", len(features), time.time() - t1
    )

    if args.csv:
        export_csv(features, args.csv)

    print_summary(features)

    sections = {
        "ops": analyze_op_correlations,
        "pairs": analyze_op_pairs,
        "topology": analyze_topology_correlations,
        "templates": analyze_templates,
        "failures": analyze_failure_modes,
        "importance": analyze_feature_importance,
    }

    if args.section == "all":
        for name, fn in sections.items():
            fn(features, top_n=args.top)
    else:
        sections[args.section](features, top_n=args.top)

    print(f"\nTotal analysis time: {time.time() - t0:.1f}s")
    db.close()


if __name__ == "__main__":
    main()
