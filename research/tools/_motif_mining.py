#!/usr/bin/env python3
"""Motif mining analysis for judgment engine plan.

Mines the lab notebook database to find op combinations, motifs, and patterns
statistically associated with top-performing neural architectures.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any

import numpy as np

DB_PATH = "/home/tim/Projects/LLM/research/lab_notebook.db"


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def extract_ops_from_graph(graph_json: str) -> list[str]:
    """Extract ordered op sequence from graph JSON (topo order by node id)."""
    try:
        g = json.loads(graph_json)
    except (json.JSONDecodeError, TypeError):
        return []
    nodes = g.get("nodes", {})
    # Sort by node id (numeric order = topological order for these DAGs)
    sorted_ids = sorted(nodes.keys(), key=lambda x: int(x))
    ops = []
    for nid in sorted_ids:
        node = nodes[nid]
        op = node.get("op_name", "")
        if op and op != "input":
            ops.append(op)
    return ops


def extract_edges(graph_json: str) -> list[tuple[str, str]]:
    """Extract (src_op, dst_op) edges from graph JSON."""
    try:
        g = json.loads(graph_json)
    except (json.JSONDecodeError, TypeError):
        return []
    nodes = g.get("nodes", {})
    edges = []
    for nid, node in nodes.items():
        dst_op = node.get("op_name", "")
        if dst_op == "input":
            continue
        for inp_id in node.get("input_ids", []):
            src_node = nodes.get(str(inp_id), {})
            src_op = src_node.get("op_name", "")
            if src_op and src_op != "input":
                edges.append((src_op, dst_op))
    return edges


def main() -> dict[str, Any]:
    conn = connect()

    # ── 1. Population overview ──────────────────────────────────────────
    print("=" * 70)
    print("SECTION 1: POPULATION OVERVIEW")
    print("=" * 70)

    row = conn.execute("""
        SELECT COUNT(*) as total,
               SUM(CASE WHEN stage0_passed=1 THEN 1 ELSE 0 END) as s0,
               SUM(CASE WHEN stage05_passed=1 THEN 1 ELSE 0 END) as s05,
               SUM(CASE WHEN stage1_passed=1 THEN 1 ELSE 0 END) as s1
        FROM program_results
    """).fetchone()
    total, s0, s05, s1 = row["total"], row["s0"], row["s05"], row["s1"]
    print(f"Total programs: {total}")
    print(f"Stage 0 passed: {s0} ({100*s0/total:.1f}%)")
    print(f"Stage 0.5 passed: {s05} ({100*s05/total:.1f}%)")
    print(f"Stage 1 passed: {s1} ({100*s1/total:.1f}%)")

    # Leaderboard tiers
    tiers = conn.execute("""
        SELECT tier, COUNT(*) as cnt FROM leaderboard
        GROUP BY tier ORDER BY cnt DESC
    """).fetchall()
    print("\nLeaderboard tier distribution:")
    for t in tiers:
        print(f"  {t['tier']:20s}: {t['cnt']}")

    # Loss ratio distribution for stage1 passers
    lr_stats = conn.execute("""
        SELECT MIN(loss_ratio) as mn, MAX(loss_ratio) as mx,
               AVG(loss_ratio) as av,
               COUNT(*) as n
        FROM program_results WHERE stage1_passed=1 AND loss_ratio IS NOT NULL
    """).fetchone()
    print(f"\nLoss ratio (stage1 passers): n={lr_stats['n']}, "
          f"min={lr_stats['mn']:.4f}, max={lr_stats['mx']:.4f}, avg={lr_stats['av']:.4f}")

    # Stability distribution
    stab_stats = conn.execute("""
        SELECT MIN(stability_score) as mn, MAX(stability_score) as mx,
               AVG(stability_score) as av
        FROM program_results WHERE stability_score IS NOT NULL
    """).fetchone()
    print(f"Stability score: min={stab_stats['mn']:.4f}, max={stab_stats['mx']:.4f}, avg={stab_stats['av']:.4f}")

    # Loss ratio percentiles for stage1 passers
    all_lr = [r[0] for r in conn.execute(
        "SELECT loss_ratio FROM program_results WHERE stage1_passed=1 AND loss_ratio IS NOT NULL"
    ).fetchall()]
    if all_lr:
        all_lr_arr = np.array(all_lr)
        pcts = [10, 15, 25, 50, 75, 90]
        print("\nLoss ratio percentiles (stage1 passers):")
        for p in pcts:
            print(f"  P{p}: {np.percentile(all_lr_arr, p):.4f}")
        top15_threshold = np.percentile(all_lr_arr, 15)
        print(f"\nTop 15% threshold (lower is better): {top15_threshold:.4f}")

    # ── 2. Top performer identification ─────────────────────────────────
    print("\n" + "=" * 70)
    print("SECTION 2: TOP PERFORMER IDENTIFICATION")
    print("=" * 70)

    # Top performers: investigation/validation tier OR (stage1_passed AND loss_ratio in top 15%)
    # Get investigation/validation result_ids from leaderboard
    top_result_ids_lb = {r[0] for r in conn.execute("""
        SELECT result_id FROM leaderboard
        WHERE tier IN ('investigation', 'validation', 'breakthrough')
          AND result_id IS NOT NULL
    """).fetchall()}

    # Get top 15% stage1 passers by loss_ratio
    top_result_ids_lr = {r[0] for r in conn.execute(f"""
        SELECT result_id FROM program_results
        WHERE stage1_passed=1 AND loss_ratio IS NOT NULL AND loss_ratio <= ?
    """, (top15_threshold,)).fetchall()}

    top_result_ids = top_result_ids_lb | top_result_ids_lr
    print(f"Top performers from leaderboard (inv/val): {len(top_result_ids_lb)}")
    print(f"Top performers from top 15% loss_ratio: {len(top_result_ids_lr)}")
    print(f"Combined (union): {len(top_result_ids)}")

    # ── 3. Load all graphs ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SECTION 3: OP FREQUENCY ANALYSIS")
    print("=" * 70)

    # Load all programs with their graphs
    all_programs = conn.execute("""
        SELECT result_id, graph_json, stage1_passed, loss_ratio, stability_score
        FROM program_results
        WHERE graph_json IS NOT NULL
    """).fetchall()

    # Parse ops for all programs
    all_ops_by_id: dict[str, list[str]] = {}
    all_edges_by_id: dict[str, list[tuple[str, str]]] = {}
    for prog in all_programs:
        rid = prog["result_id"]
        ops = extract_ops_from_graph(prog["graph_json"])
        if ops:
            all_ops_by_id[rid] = ops
            all_edges_by_id[rid] = extract_edges(prog["graph_json"])

    general_ids = set(all_ops_by_id.keys())
    top_ids_with_ops = top_result_ids & general_ids

    print(f"Programs with parseable graphs: {len(general_ids)}")
    print(f"Top performers with parseable graphs: {len(top_ids_with_ops)}")

    # Op frequency in top performers vs general population
    gen_op_counts: Counter = Counter()
    top_op_counts: Counter = Counter()

    for rid in general_ids:
        for op in set(all_ops_by_id[rid]):  # unique ops per program
            gen_op_counts[op] += 1

    for rid in top_ids_with_ops:
        for op in set(all_ops_by_id[rid]):
            top_op_counts[op] += 1

    n_gen = len(general_ids)
    n_top = len(top_ids_with_ops)

    # Compute lift
    op_lift = {}
    for op in gen_op_counts:
        gen_rate = gen_op_counts[op] / n_gen
        top_rate = top_op_counts.get(op, 0) / n_top if n_top > 0 else 0
        lift = top_rate / gen_rate if gen_rate > 0 else 0
        op_lift[op] = {
            "gen_count": gen_op_counts[op],
            "top_count": top_op_counts.get(op, 0),
            "gen_rate": gen_rate,
            "top_rate": top_rate,
            "lift": lift,
        }

    # Sort by lift (descending)
    sorted_ops = sorted(op_lift.items(), key=lambda x: x[1]["lift"], reverse=True)

    print(f"\n{'Op':<35} {'Gen%':>7} {'Top%':>7} {'Lift':>7} {'GenN':>6} {'TopN':>6}")
    print("-" * 78)
    for op, d in sorted_ops:
        if d["gen_count"] >= 5:  # minimum support
            print(f"{op:<35} {100*d['gen_rate']:6.1f}% {100*d['top_rate']:6.1f}% "
                  f"{d['lift']:6.2f}x {d['gen_count']:>6} {d['top_count']:>6}")

    # ── 4. Op-pair (edge-based) analysis ────────────────────────────────
    print("\n" + "=" * 70)
    print("SECTION 4: OP-PAIR (BIGRAM/EDGE) ANALYSIS")
    print("=" * 70)

    gen_pair_counts: Counter = Counter()
    top_pair_counts: Counter = Counter()

    for rid in general_ids:
        for edge in set(all_edges_by_id[rid]):
            gen_pair_counts[edge] += 1

    for rid in top_ids_with_ops:
        for edge in set(all_edges_by_id[rid]):
            top_pair_counts[edge] += 1

    # Compute pair lift
    pair_lift = {}
    for pair in gen_pair_counts:
        gen_rate = gen_pair_counts[pair] / n_gen
        top_rate = top_pair_counts.get(pair, 0) / n_top if n_top > 0 else 0
        lift = top_rate / gen_rate if gen_rate > 0 else 0
        pair_lift[pair] = {
            "gen_count": gen_pair_counts[pair],
            "top_count": top_pair_counts.get(pair, 0),
            "gen_rate": gen_rate,
            "top_rate": top_rate,
            "lift": lift,
        }

    sorted_pairs = sorted(pair_lift.items(), key=lambda x: x[1]["lift"], reverse=True)

    print(f"\nTotal unique edge types: {len(pair_lift)}")
    print(f"\n{'Pair':<50} {'Gen%':>7} {'Top%':>7} {'Lift':>7} {'TopN':>5}")
    print("-" * 85)
    for pair, d in sorted_pairs[:80]:
        if d["gen_count"] >= 5:  # minimum support
            label = f"{pair[0]} -> {pair[1]}"
            print(f"{label:<50} {100*d['gen_rate']:6.1f}% {100*d['top_rate']:6.1f}% "
                  f"{d['lift']:6.2f}x {d['top_count']:>5}")

    # ── 5. Op trigram analysis ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SECTION 5: OP TRIGRAM (3-OP PATH) ANALYSIS")
    print("=" * 70)

    def extract_paths(graph_json: str, length: int = 3) -> list[tuple[str, ...]]:
        """Extract all paths of given length from graph."""
        try:
            g = json.loads(graph_json)
        except (json.JSONDecodeError, TypeError):
            return []
        nodes = g.get("nodes", {})
        # Build adjacency: parent -> children
        children: dict[str, list[str]] = defaultdict(list)
        for nid, node in nodes.items():
            for inp_id in node.get("input_ids", []):
                children[str(inp_id)].append(nid)

        paths = []
        def dfs(nid: str, path: list[str]) -> None:
            op = nodes.get(nid, {}).get("op_name", "")
            if op == "input":
                cur_path = path
            else:
                cur_path = path + [op]

            if len(cur_path) == length:
                paths.append(tuple(cur_path))
                return
            if len(cur_path) > length:
                return
            for child in children.get(nid, []):
                dfs(child, cur_path)

        # Start DFS from all nodes
        for nid in nodes:
            op = nodes[nid].get("op_name", "")
            if op != "input":
                for child in children.get(nid, []):
                    dfs(child, [op])

        return paths

    gen_tri_counts: Counter = Counter()
    top_tri_counts: Counter = Counter()

    for rid in general_ids:
        prog = next((p for p in all_programs if p["result_id"] == rid), None)
        if prog:
            tris = set(extract_paths(prog["graph_json"], 3))
            for tri in tris:
                gen_tri_counts[tri] += 1

    for rid in top_ids_with_ops:
        prog = next((p for p in all_programs if p["result_id"] == rid), None)
        if prog:
            tris = set(extract_paths(prog["graph_json"], 3))
            for tri in tris:
                top_tri_counts[tri] += 1

    # Compute trigram lift
    tri_lift = {}
    for tri in gen_tri_counts:
        gen_rate = gen_tri_counts[tri] / n_gen
        top_rate = top_tri_counts.get(tri, 0) / n_top if n_top > 0 else 0
        lift = top_rate / gen_rate if gen_rate > 0 else 0
        tri_lift[tri] = {
            "gen_count": gen_tri_counts[tri],
            "top_count": top_tri_counts.get(tri, 0),
            "gen_rate": gen_rate,
            "top_rate": top_rate,
            "lift": lift,
        }

    sorted_tris = sorted(tri_lift.items(), key=lambda x: x[1]["lift"], reverse=True)

    print(f"\nTotal unique trigram types: {len(tri_lift)}")
    print(f"\n{'Trigram':<60} {'Gen%':>7} {'Top%':>7} {'Lift':>7} {'TopN':>5}")
    print("-" * 95)
    for tri, d in sorted_tris[:60]:
        if d["gen_count"] >= 3:
            label = " -> ".join(tri)
            print(f"{label:<60} {100*d['gen_rate']:6.1f}% {100*d['top_rate']:6.1f}% "
                  f"{d['lift']:6.2f}x {d['top_count']:>5}")

    # ── 6. Co-occurrence analysis in top performers ─────────────────────
    print("\n" + "=" * 70)
    print("SECTION 6: OP CO-OCCURRENCE IN TOP PERFORMERS")
    print("=" * 70)

    # Build co-occurrence matrix
    top_ops_flat = set()
    for rid in top_ids_with_ops:
        top_ops_flat.update(set(all_ops_by_id[rid]))

    top_ops_list = sorted(top_ops_flat)
    cooccur: Counter = Counter()
    for rid in top_ids_with_ops:
        ops_set = set(all_ops_by_id[rid])
        for a, b in combinations(sorted(ops_set), 2):
            cooccur[(a, b)] += 1

    # Top co-occurring pairs
    sorted_cooccur = cooccur.most_common(40)
    print(f"\n{'Op Pair':<55} {'Count':>6} {'% of Top':>8}")
    print("-" * 72)
    for (a, b), cnt in sorted_cooccur:
        print(f"{a} + {b:<40} {cnt:>6} {100*cnt/n_top:7.1f}%")

    # ── 7. Cluster analysis ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SECTION 7: CLUSTER ANALYSIS OF TOP PERFORMERS")
    print("=" * 70)

    # Build feature vectors: op presence (binary) for top performers
    all_unique_ops = sorted(set().union(*[set(all_ops_by_id[rid]) for rid in top_ids_with_ops]))
    op_to_idx = {op: i for i, op in enumerate(all_unique_ops)}
    n_features = len(all_unique_ops)

    top_ids_list = sorted(top_ids_with_ops)
    X = np.zeros((len(top_ids_list), n_features), dtype=np.float32)
    for i, rid in enumerate(top_ids_list):
        for op in set(all_ops_by_id[rid]):
            if op in op_to_idx:
                X[i, op_to_idx[op]] = 1.0

    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    # Try k=5..10, pick by silhouette
    from sklearn.metrics import silhouette_score

    best_k, best_sil = 5, -1
    for k in range(4, 12):
        if k >= len(top_ids_list):
            break
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        print(f"  k={k}: silhouette={sil:.3f}")
        if sil > best_sil:
            best_sil = sil
            best_k = k

    print(f"\nBest k={best_k} (silhouette={best_sil:.3f})")

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    # Describe each cluster
    cluster_results = {}
    for c in range(best_k):
        mask = labels == c
        cluster_size = mask.sum()
        cluster_ids = [top_ids_list[i] for i in range(len(top_ids_list)) if labels[i] == c]

        # Dominant ops in this cluster
        cluster_X = X[mask]
        op_presence = cluster_X.mean(axis=0)
        top_ops_in_cluster = sorted(
            [(all_unique_ops[j], float(op_presence[j])) for j in range(n_features) if op_presence[j] > 0.3],
            key=lambda x: x[1], reverse=True
        )

        # Get avg metrics for this cluster
        if cluster_ids:
            placeholders = ",".join("?" * len(cluster_ids))
            metrics = conn.execute(f"""
                SELECT AVG(loss_ratio) as avg_lr, AVG(stability_score) as avg_stab,
                       AVG(graph_n_ops) as avg_ops, AVG(graph_depth) as avg_depth,
                       MIN(loss_ratio) as best_lr
                FROM program_results
                WHERE result_id IN ({placeholders})
            """, cluster_ids).fetchone()
        else:
            metrics = None

        cluster_results[c] = {
            "size": int(cluster_size),
            "top_ops": top_ops_in_cluster[:15],
            "avg_lr": float(metrics["avg_lr"]) if metrics and metrics["avg_lr"] else None,
            "best_lr": float(metrics["best_lr"]) if metrics and metrics["best_lr"] else None,
            "avg_stab": float(metrics["avg_stab"]) if metrics and metrics["avg_stab"] else None,
            "avg_ops": float(metrics["avg_ops"]) if metrics and metrics["avg_ops"] else None,
            "avg_depth": float(metrics["avg_depth"]) if metrics and metrics["avg_depth"] else None,
        }

        print(f"\n--- Cluster {c} (n={cluster_size}) ---")
        if metrics:
            print(f"  Avg loss_ratio: {metrics['avg_lr']:.4f}" if metrics['avg_lr'] else "  Avg loss_ratio: N/A")
            print(f"  Best loss_ratio: {metrics['best_lr']:.4f}" if metrics['best_lr'] else "  Best loss_ratio: N/A")
            print(f"  Avg stability: {metrics['avg_stab']:.4f}" if metrics['avg_stab'] else "  Avg stability: N/A")
            print(f"  Avg n_ops: {metrics['avg_ops']:.1f}" if metrics['avg_ops'] else "  Avg n_ops: N/A")
        print(f"  Dominant ops (>30% presence):")
        for op, pct in top_ops_in_cluster[:15]:
            print(f"    {op:<35} {100*pct:5.1f}%")

    # ── 8. Loss/stability correlation ───────────────────────────────────
    print("\n" + "=" * 70)
    print("SECTION 8: OP-METRIC CORRELATIONS")
    print("=" * 70)

    # For each op, compute average loss_ratio and stability when present vs absent
    # among stage1 passers only
    s1_programs = conn.execute("""
        SELECT result_id, loss_ratio, stability_score, graph_json
        FROM program_results
        WHERE stage1_passed=1 AND loss_ratio IS NOT NULL
    """).fetchall()

    s1_ops: dict[str, list[str]] = {}
    for prog in s1_programs:
        ops = extract_ops_from_graph(prog["graph_json"])
        if ops:
            s1_ops[prog["result_id"]] = ops

    # Collect metrics per op
    op_metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"lr_present": [], "stab_present": []})
    all_lrs = []
    all_stabs = []
    s1_lookup = {p["result_id"]: p for p in s1_programs}

    for rid, ops in s1_ops.items():
        prog = s1_lookup[rid]
        lr = prog["loss_ratio"]
        stab = prog["stability_score"]
        all_lrs.append(lr)
        if stab is not None:
            all_stabs.append(stab)

        for op in set(ops):
            op_metrics[op]["lr_present"].append(lr)
            if stab is not None:
                op_metrics[op]["stab_present"].append(stab)

    avg_lr_global = np.mean(all_lrs) if all_lrs else 0
    avg_stab_global = np.mean(all_stabs) if all_stabs else 0

    print(f"\nGlobal avg loss_ratio (stage1): {avg_lr_global:.4f}")
    print(f"Global avg stability (stage1): {avg_stab_global:.4f}")

    print(f"\n{'Op':<35} {'AvgLR':>8} {'LR_diff':>8} {'AvgStab':>8} {'Stab_diff':>9} {'N':>5}")
    print("-" * 80)

    op_corr_results = []
    for op, m in op_metrics.items():
        if len(m["lr_present"]) >= 10:
            avg_lr = np.mean(m["lr_present"])
            avg_stab = np.mean(m["stab_present"]) if m["stab_present"] else 0
            lr_diff = avg_lr - avg_lr_global  # negative = better
            stab_diff = avg_stab - avg_stab_global  # positive = better
            op_corr_results.append((op, avg_lr, lr_diff, avg_stab, stab_diff, len(m["lr_present"])))

    # Sort by lr_diff (most beneficial ops first, i.e. most negative)
    op_corr_results.sort(key=lambda x: x[2])
    for op, avg_lr, lr_diff, avg_stab, stab_diff, n in op_corr_results:
        sign_lr = "+" if lr_diff >= 0 else ""
        sign_st = "+" if stab_diff >= 0 else ""
        print(f"{op:<35} {avg_lr:7.4f} {sign_lr}{lr_diff:7.4f} {avg_stab:7.4f} {sign_st}{stab_diff:8.4f} {n:>5}")

    # ── 9. Op-pair conditioned loss analysis ────────────────────────────
    print("\n" + "=" * 70)
    print("SECTION 9: OP-PAIR CONDITIONED LOSS ANALYSIS")
    print("=" * 70)

    pair_lr: dict[tuple[str, str], list[float]] = defaultdict(list)
    for rid, ops in s1_ops.items():
        prog = s1_lookup[rid]
        lr = prog["loss_ratio"]
        ops_set = sorted(set(ops))
        for a, b in combinations(ops_set, 2):
            pair_lr[(a, b)].append(lr)

    pair_lr_results = []
    for pair, lrs in pair_lr.items():
        if len(lrs) >= 15:
            avg = np.mean(lrs)
            pair_lr_results.append((pair, avg, avg - avg_lr_global, len(lrs)))

    pair_lr_results.sort(key=lambda x: x[1])

    print(f"\n{'Op Pair':<55} {'AvgLR':>8} {'Diff':>8} {'N':>5}")
    print("-" * 80)
    for pair, avg, diff, n in pair_lr_results[:40]:
        sign = "+" if diff >= 0 else ""
        print(f"{pair[0]} + {pair[1]:<40} {avg:7.4f} {sign}{diff:7.4f} {n:>5}")

    # ── 10. Graph structural patterns ───────────────────────────────────
    print("\n" + "=" * 70)
    print("SECTION 10: STRUCTURAL PATTERNS")
    print("=" * 70)

    # Graph size, depth, unique ops for top vs general
    struct_top = conn.execute(f"""
        SELECT AVG(graph_n_ops) as ops, AVG(graph_depth) as depth,
               AVG(graph_n_unique_ops) as uniq, AVG(graph_n_edges) as edges,
               AVG(param_count) as params
        FROM program_results
        WHERE result_id IN ({','.join('?' * len(top_ids_with_ops))})
    """, list(top_ids_with_ops)).fetchone()

    struct_gen = conn.execute("""
        SELECT AVG(graph_n_ops) as ops, AVG(graph_depth) as depth,
               AVG(graph_n_unique_ops) as uniq, AVG(graph_n_edges) as edges,
               AVG(param_count) as params
        FROM program_results
        WHERE graph_json IS NOT NULL
    """).fetchone()

    print(f"\n{'Metric':<25} {'General':>12} {'Top Perf':>12}")
    print("-" * 52)
    for metric in ["ops", "depth", "uniq", "edges", "params"]:
        g = struct_gen[metric]
        t = struct_top[metric]
        g_str = f"{g:.1f}" if g else "N/A"
        t_str = f"{t:.1f}" if t else "N/A"
        print(f"{metric:<25} {g_str:>12} {t_str:>12}")

    # Has residual connections (add op)?
    has_add_gen = sum(1 for rid in general_ids if "add" in set(all_ops_by_id[rid]))
    has_add_top = sum(1 for rid in top_ids_with_ops if "add" in set(all_ops_by_id[rid]))
    print(f"\nResidual connections (add op):")
    print(f"  General: {has_add_gen}/{n_gen} ({100*has_add_gen/n_gen:.1f}%)")
    print(f"  Top:     {has_add_top}/{n_top} ({100*has_add_top/n_top:.1f}%)")

    # Has normalization (layernorm/rmsnorm)?
    norm_ops = {"layernorm", "rmsnorm", "group_norm", "batch_norm"}
    has_norm_gen = sum(1 for rid in general_ids if norm_ops & set(all_ops_by_id[rid]))
    has_norm_top = sum(1 for rid in top_ids_with_ops if norm_ops & set(all_ops_by_id[rid]))
    print(f"\nNormalization ops:")
    print(f"  General: {has_norm_gen}/{n_gen} ({100*has_norm_gen/n_gen:.1f}%)")
    print(f"  Top:     {has_norm_top}/{n_top} ({100*has_norm_top/n_top:.1f}%)")

    # ── 11. Reference architecture analysis ─────────────────────────────
    print("\n" + "=" * 70)
    print("SECTION 11: REFERENCE ARCHITECTURE COMPARISON")
    print("=" * 70)

    refs = conn.execute("""
        SELECT l.reference_name, l.tier, l.composite_score,
               l.screening_loss_ratio, l.investigation_loss_ratio, l.validation_loss_ratio,
               p.graph_json, p.stability_score
        FROM leaderboard l
        JOIN program_results p ON l.result_id = p.result_id
        WHERE l.is_reference = 1
    """).fetchall()

    for ref in refs:
        ops = extract_ops_from_graph(ref["graph_json"])
        print(f"\n{ref['reference_name']} ({ref['tier']}):")
        print(f"  Ops: {' -> '.join(ops)}")
        print(f"  Screening LR: {ref['screening_loss_ratio']}")
        print(f"  Composite: {ref['composite_score']}")

    # ── Collect results for report ──────────────────────────────────────
    results = {
        "population": {"total": total, "s0": s0, "s05": s05, "s1": s1},
        "top15_threshold": top15_threshold,
        "n_top": n_top,
        "op_lift": sorted_ops,
        "pair_lift": sorted_pairs,
        "tri_lift": sorted_tris,
        "clusters": cluster_results,
        "op_corr": op_corr_results,
        "pair_lr": pair_lr_results,
        "cooccur": sorted_cooccur,
    }

    conn.close()
    return results


if __name__ == "__main__":
    main()
