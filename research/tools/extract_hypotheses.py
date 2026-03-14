#!/usr/bin/env python3
"""
Automated Hypothesis Extraction (Task 3I)

Scans the leaderboard for clusters of high-performing architectures 
with the same architecture_family. Outputs a Markdown summary 
of "Winning Motifs".
"""

import sqlite3
import os
from collections import defaultdict
from typing import List, Dict, Any

# Path to the database
DB_PATH = "research/lab_notebook.db"
OUTPUT_MD = "winning_motifs.md"

def get_leaderboard_data(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Fetch all non-reference leaderboard entries with raw data for classification."""
    cursor = conn.cursor()
    # Join with program_results to get graph_json and routing_mode for classification
    cursor.execute("""
        SELECT pr.graph_json, pr.routing_mode, l.screening_loss_ratio, 
               l.validation_loss_ratio, l.composite_score, l.tier, 
               l.architecture_desc, l.result_id
        FROM leaderboard l
        JOIN program_results pr ON l.result_id = pr.result_id
        WHERE l.is_reference = 0
    """)
    
    entries = []
    for row in cursor.fetchall():
        graph_json, routing_mode = row[0], row[1]
        # Use simple classification logic similar to leaderboard_crud.py
        family = classify_family(graph_json, routing_mode)
        
        entries.append({
            "family": family, 
            "screening_lr": row[2], 
            "validation_lr": row[3],
            "score": row[4],
            "tier": row[5],
            "desc": row[6],
            "id": row[7]
        })
    return entries

def classify_family(graph_json: str, routing_mode: str) -> str:
    """Simple replica of _classify_architecture_family."""
    if routing_mode: return "Routed-MoE"
    if not graph_json: return "Unknown"
    try:
        data = json.loads(graph_json)
        nodes = data.get("nodes", [])
        if isinstance(nodes, dict): nodes = nodes.values()
        ops = {str(n.get("op_name") or n.get("op")).strip() for n in nodes}
        
        if not ops: return "Unknown"
        
        if any(o in ops for o in ["attention", "self_attention", "mha"]):
            return "Attention"
        if any(o in ops for o in ["state_space", "selective_scan"]):
            return "Mamba-SSM"
        if any(o in ops for o in ["fft", "fourier_mix"]):
            return "Spectral"
        if any(o in ops for o in ["conv1d", "depthwise_conv1d"]):
            return "Conv"
        return "Hybrid-Mixer"
    except:
        return "Unknown"

def analyze_families(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group entries by family and compute aggregate metrics."""
    families = defaultdict(lambda: {
        "count": 0, 
        "scores": [], 
        "screening_lrs": [], 
        "validation_lrs": [],
        "tiers": defaultdict(int),
        "ids": []
    })
    
    for e in entries:
        f = e["family"] or "unknown"
        f_data = families[f]
        f_data["count"] += 1
        if e["score"] is not None: f_data["scores"].append(e["score"])
        if e["screening_lr"] is not None: f_data["screening_lrs"].append(e["screening_lr"])
        if e["validation_lr"] is not None: f_data["validation_lrs"].append(e["validation_lr"])
        f_data["tiers"][e["tier"]] += 1
        f_data["ids"].append(e["id"])

    results = []
    for name, data in families.items():
        if data["count"] < 2: continue # Need a cluster
        
        avg_score = sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0
        avg_screening = sum(data["screening_lrs"]) / len(data["screening_lrs"]) if data["screening_lrs"] else 1.0
        
        # Heuristic for "Winning Motif": high score or low screening_lr
        # Compare vs baseline (assume baseline LR ~ 0.26 from schema)
        baseline_lr = 0.2646
        improvement = (baseline_lr - avg_screening) / baseline_lr
        
        results.append({
            "family": name,
            "count": data["count"],
            "avg_score": avg_score,
            "avg_screening_lr": avg_screening,
            "improvement_pct": improvement * 100,
            "top_tier": max(data["tiers"].keys(), key=lambda t: data["tiers"][t]),
            "n_validated": data["tiers"].get("validation", 0) + data["tiers"].get("breakthrough", 0)
        })
    
    # Sort by improvement
    results.sort(key=lambda x: x["improvement_pct"], reverse=True)
    return results

def main():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    try:
        entries = get_leaderboard_data(conn)
        motifs = analyze_families(entries)
        
        with open(OUTPUT_MD, "w") as f:
            f.write("# Winning Motifs — Automated Hypothesis Extraction\n\n")
            f.write("Analysis of architectural families showing consistent high performance.\n\n")
            
            f.write("| Family | Count | Avg Score | Screening LR | Improv vs Baseline | Top Tier |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
            
            for m in motifs:
                f.write(f"| **{m['family']}** | {m['count']} | {m['avg_score']:.1f} | {m['avg_screening_lr']:.4f} | {m['improvement_pct']:+.1f}% | {m['top_tier']} |\n")
            
            f.write("\n## Key Insights\n\n")
            for m in motifs[:3]:
                if m['improvement_pct'] > 10:
                    f.write(f"- **{m['family']}** shows significant potential with a average improvement of {m['improvement_pct']:.1f}% over baseline across {m['count']} candidates.\n")
            
        print(f"Successfully extracted {len(motifs)} motifs. Summary saved to {OUTPUT_MD}")

    finally:
        conn.close()

if __name__ == "__main__":
    main()
