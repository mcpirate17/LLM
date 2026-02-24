
import os
import json
import sqlite3
import torch
import logging
from typing import Dict, Any, List

# Local imports
from research.synthesis.serializer import graph_from_json
from research.synthesis.compiler import compile_graph
from research.eval.sandbox import _stability_probe, SandboxResult
from research.scientist.notebook import LabNotebook

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("backfill")

DB_PATH = "research/lab_notebook.db"

def calculate_graph_metrics(graph) -> Dict[str, Any]:
    """Extracted from runner.py logic."""
    metrics = {}
    metrics["graph_n_ops"] = graph.n_ops()
    metrics["graph_depth"] = graph.depth()
    metrics["graph_n_params_estimate"] = graph.n_params_estimate()
    metrics["graph_has_gradient_path"] = graph.has_gradient_path()
    
    n_edges = sum(len(n.input_ids) for n in graph.nodes.values())
    metrics["graph_n_edges"] = n_edges
    
    ops_used = set()
    cat_counts = {}
    uses_math = False
    uses_freq = False
    
    # Simple category mapping since we don't want to import everything
    for node in graph.nodes.values():
        if node.is_input: continue
        ops_used.add(node.op_name)
        # We'll skip the detailed category histogram for now to keep it lightweight
        # but check for specific flags
        if "spectral" in node.op_name or "fft" in node.op_name:
            uses_freq = True
        if "math" in node.op_name or "poly" in node.op_name:
            uses_math = True
            
    metrics["graph_n_unique_ops"] = len(ops_used)
    metrics["graph_uses_math_spaces"] = int(uses_math)
    metrics["graph_uses_frequency_domain"] = int(uses_freq)
    return metrics

def backfill():
    nb = LabNotebook(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 1. Fetch records missing graph metrics
    cursor.execute("SELECT result_id, graph_json FROM program_results WHERE graph_n_ops IS NULL")
    rows = cursor.fetchall()
    logger.info(f"Found {len(rows)} records missing graph metrics.")

    for row in rows:
        try:
            graph = graph_from_json(row["graph_json"])
            metrics = calculate_graph_metrics(graph)
            
            set_clause = ", ".join([f"{k} = ?" for k in metrics.keys()])
            values = list(metrics.values()) + [row["result_id"]]
            conn.execute(f"UPDATE program_results SET {set_clause} WHERE result_id = ?", values)
        except Exception as e:
            logger.error(f"Failed graph metrics for {row['result_id']}: {e}")
    conn.commit()

    # 2. Backfill Stability (Prioritize Stage 1 Passed)
    cursor.execute("""
        SELECT result_id, graph_json 
        FROM program_results 
        WHERE stability_score IS NULL AND stage1_passed = 1
        LIMIT 50
    """)
    rows = cursor.fetchall()
    logger.info(f"Backfilling stability for {len(rows)} high-priority candidates.")
    
    if rows:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        for row in rows:
            try:
                graph = graph_from_json(row["graph_json"])
                model = compile_graph(graph)
                model.to(device)
                
                # Probes use batch_size=2, seq_len=128 by default
                stability = _stability_probe(model, torch.device(device), 2, 128, 32000)
                
                conn.execute("""
                    UPDATE program_results 
                    SET stability_score = ?, extreme_input_passed = ?, random_input_passed = ?
                    WHERE result_id = ?
                """, (stability["score"], int(stability["extreme_passed"]), int(stability["random_passed"]), row["result_id"]))
                logger.info(f"Updated stability for {row['result_id']}: {stability['score']}")
                
                del model
                if device == "cuda": torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"Failed stability for {row['result_id']}: {e}")
        conn.commit()

    # 3. Simplified Regression Gates (Loss-based)
    # Logic: Pass if loss_ratio is better (higher) than the average of its experiment
    logger.info("Applying loss-based regression gates.")
    cursor.execute("""
        UPDATE program_results
        SET regression_gate_pass = 1, regression_gate_reason = 'Passed: Better than experiment avg'
        WHERE loss_ratio > (
            SELECT AVG(p2.loss_ratio) 
            FROM program_results p2 
            WHERE p2.experiment_id = program_results.experiment_id
            AND p2.loss_ratio IS NOT NULL
        ) AND regression_gate_pass IS NULL;
    """)
    cursor.execute("""
        UPDATE program_results
        SET regression_gate_pass = 0, regression_gate_reason = 'Failed: Worse than experiment avg'
        WHERE loss_ratio <= (
            SELECT AVG(p2.loss_ratio) 
            FROM program_results p2 
            WHERE p2.experiment_id = program_results.experiment_id
            AND p2.loss_ratio IS NOT NULL
        ) AND regression_gate_pass IS NULL;
    """)
    conn.commit()
    conn.close()
    logger.info("Backfill complete.")

if __name__ == "__main__":
    backfill()
