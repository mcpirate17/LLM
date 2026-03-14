#!/usr/bin/env python3
"""
Op-Pair Mining Script (Task 2I)

Queries lab_notebook.db for top-performing graphs and extracts op bigrams
from their topological order. Outputs a CSV with:
pair, count, avg_loss_ratio, avg_stability_score
"""

import sqlite3
import json
import csv
import os
from collections import defaultdict
from typing import List, Dict, Any

# Path to the database
DB_PATH = "research/lab_notebook.db"
OUTPUT_CSV = "op_pairs.csv"

def get_top_performing_graphs(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Query for stage1-passing graphs in the top quartile of loss_ratio."""
    cursor = conn.cursor()
    
    # 1. Get the 25th percentile loss_ratio for stage1_passed programs
    cursor.execute("""
        SELECT loss_ratio FROM program_results 
        WHERE stage1_passed = 1 AND loss_ratio IS NOT NULL
        ORDER BY loss_ratio ASC
    """)
    losses = [row[0] for row in cursor.fetchall()]
    
    if not losses:
        print("No stage1-passing programs found.")
        return []
    
    threshold_idx = max(0, len(losses) // 4)
    loss_threshold = losses[threshold_idx]
    print(f"Top quartile loss_ratio threshold: {loss_threshold:.4f} (n={len(losses)})")
    
    # 2. Query for graphs below this threshold
    cursor.execute("""
        SELECT graph_json, loss_ratio, stability_score 
        FROM program_results 
        WHERE stage1_passed = 1 AND loss_ratio <= ?
    """, (loss_threshold,))
    
    return [
        {"graph_json": row[0], "loss_ratio": row[1], "stability_score": row[2]} 
        for row in cursor.fetchall()
    ]

def extract_bigrams(graph_json: str) -> List[tuple]:
    """Extract op bigrams from a graph's topological order."""
    try:
        data = json.loads(graph_json)
        nodes = data.get("nodes", [])
        if isinstance(nodes, dict):
            nodes = list(nodes.values())
        
        # Build adjacency list
        adj = defaultdict(list)
        node_map = {}
        for n in nodes:
            nid = n.get("id")
            node_map[nid] = n
            for inp_id in n.get("input_ids", []):
                adj[inp_id].append(nid)
        
        # Sort nodes by ID as a proxy for topological order if not provided
        # (Though canonical topo sort would be better, for bigrams, input->output pairs are what matters)
        bigrams = []
        for nid, node in node_map.items():
            if node.get("is_input"):
                op_a = "input"
            else:
                op_a = node.get("op_name") or node.get("op")
            
            if not op_a: continue
            
            for child_id in adj.get(nid, []):
                child = node_map.get(child_id)
                if not child: continue
                op_b = child.get("op_name") or child.get("op")
                if op_b:
                    bigrams.append((op_a, op_b))
        
        return bigrams
    except Exception as e:
        print(f"Error parsing graph: {e}")
        return []

def main():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    try:
        graphs = get_top_performing_graphs(conn)
        print(f"Mining {len(graphs)} top-performing graphs...")
        
        stats = defaultdict(lambda: {"count": 0, "total_loss": 0.0, "total_stability": 0.0})
        
        for g in graphs:
            bigrams = extract_bigrams(g["graph_json"])
            lr = g["loss_ratio"] or 1.0
            stab = g["stability_score"] or 0.0
            
            # Use set to avoid double-counting same pair in same graph if preferred, 
            # but usually we want frequency.
            for pair in bigrams:
                pair_str = f"{pair[0]} -> {pair[1]}"
                s = stats[pair_str]
                s["count"] += 1
                s["total_loss"] += lr
                s["total_stability"] += stab
        
        # Compile results
        results = []
        for pair, s in stats.items():
            count = s["count"]
            results.append({
                "pair": pair,
                "count": count,
                "avg_loss_ratio": s["total_loss"] / count,
                "avg_stability_score": s["total_stability"] / count
            })
        
        # Sort by count DESC
        results.sort(key=lambda x: x["count"], reverse=True)
        
        # Write to CSV
        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["pair", "count", "avg_loss_ratio", "avg_stability_score"])
            writer.writeheader()
            writer.writerows(results)
            
        print(f"Successfully mined {len(results)} unique op pairs. Results saved to {OUTPUT_CSV}")
        
        # Success criteria check
        if len(results) >= 50:
            print("Success: CSV has ≥50 op pairs.")
        else:
            print(f"Warning: Only {len(results)} op pairs found. Success criteria requires ≥50.")

    finally:
        conn.close()

if __name__ == "__main__":
    main()
