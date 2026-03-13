#!/usr/bin/env python3
import json
import sqlite3
import time
from tqdm import tqdm

DB_PATH = "research/lab_notebook.db"

def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 1. Fetch all program results
    cursor.execute("SELECT graph_json, stage0_passed, stage05_passed, stage1_passed, loss_ratio, novelty_score FROM program_results")
    rows = cursor.fetchall()
    
    op_stats = {}

    for row in tqdm(rows, desc="Calculating op stats"):
        try:
            graph = json.loads(row["graph_json"])
            nodes = graph.get("nodes", {})
            unique_ops = set()
            for node in nodes.values():
                op_name = node.get("op_name")
                if op_name and op_name != "input":
                    unique_ops.add(op_name)
            
            for op_name in unique_ops:
                if op_name not in op_stats:
                    op_stats[op_name] = {
                        "n_used": 0, "n0": 0, "n05": 0, "n1": 0, "total_loss": 0.0, "total_nov": 0.0, "loss_count": 0
                    }
                
                stats = op_stats[op_name]
                stats["n_used"] += 1
                if row["stage0_passed"]: stats["n0"] += 1
                if row["stage05_passed"]: stats["n05"] += 1
                if row["stage1_passed"]: 
                    stats["n1"] += 1
                    if row["loss_ratio"] is not None:
                        stats["total_loss"] += row["loss_ratio"]
                        stats["loss_count"] += 1
                    if row["novelty_score"] is not None:
                        stats["total_nov"] += row["novelty_score"]
        except Exception as e:
            continue

    # 2. Update op_success_rates table
    now = time.time()
    for op_name, s in op_stats.items():
        avg_loss = s["total_loss"] / s["loss_count"] if s["loss_count"] > 0 else None
        avg_nov = s["total_nov"] / s["n1"] if s["n1"] > 0 else 0.0
        
        cursor.execute("""
            INSERT INTO op_success_rates (op_name, n_used, n_stage0_passed, n_stage05_passed, n_stage1_passed, avg_loss_ratio, avg_novelty, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(op_name) DO UPDATE SET
                n_used = excluded.n_used,
                n_stage0_passed = excluded.n_stage0_passed,
                n_stage05_passed = excluded.n_stage05_passed,
                n_stage1_passed = excluded.n_stage1_passed,
                avg_loss_ratio = excluded.avg_loss_ratio,
                avg_novelty = excluded.avg_novelty,
                last_updated = excluded.last_updated
        """, (op_name, s["n_used"], s["n0"], s["n05"], s["n1"], avg_loss, avg_nov, now))

    conn.commit()
    conn.close()
    print("Op success rates recalculated.")

if __name__ == "__main__":
    main()
