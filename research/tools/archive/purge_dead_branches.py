#!/usr/bin/env python3
"""
Purge Zombie Models from lab_notebook.db (Phase 3 - Deep Clean)

Deletes models from program_results even if they aren't on the leaderboard,
if they meet zombie criteria based on raw training metrics.
"""

import os
import json
import sqlite3
import logging
from tqdm import tqdm
from research.synthesis.graph import ComputationGraph

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
LOGGER = logging.getLogger(__name__)

DB_PATH = "research/lab_notebook.db"

def main():
    if not os.path.exists(DB_PATH):
        LOGGER.error(f"Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 1. Fetch ALL program results
    LOGGER.info("Fetching all program results...")
    query = """
        SELECT p.result_id, p.graph_json, p.fp_jacobian_spectral_norm, 
               p.loss_ratio, p.baseline_loss_ratio,
               l.wikitext_perplexity, l.is_reference
        FROM program_results p
        LEFT JOIN leaderboard l ON p.result_id = l.result_id
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    LOGGER.info(f"Found {len(rows)} results to analyze.")

    purge_ids = []
    
    # 2. Analyze each model
    for row in tqdm(rows, desc="Analyzing models"):
        rid = row["result_id"]
        is_ref = bool(row["is_reference"])
        if is_ref: continue

        # Criteria 1: Dead Branches
        try:
            graph_dict = json.loads(row["graph_json"])
            graph = ComputationGraph.from_dict(graph_dict)
            reachable = graph.get_reachable_node_ids()
            if len(reachable) < len(graph.nodes):
                purge_ids.append(rid)
                continue
        except Exception as e:
            pass

        # Criteria 2: Numerical Collapse
        spec_norm = row["fp_jacobian_spectral_norm"]
        if spec_norm is not None and spec_norm < 0.005:
            purge_ids.append(rid)
            continue

        # Criteria 3: Generalization Failure (Leaderboard only metric usually)
        perp = row["wikitext_perplexity"]
        if perp is not None and perp > 1000000:
            purge_ids.append(rid)
            continue

        # Criteria 4: No improvement over baseline (CRITICAL FIX: check p.baseline_loss_ratio)
        # If we have a baseline ratio and it's >= 1.0, it's trash.
        # BUT: Reference models might have baseline_ratio = 1.0 (they ARE the baseline).
        # Since we skipped refs above, this is safe.
        base_ratio = row["baseline_loss_ratio"]
        if base_ratio is not None and base_ratio >= 1.0:
            purge_ids.append(rid)
            continue
            
        # Criteria 5: Suspiciously low loss with NO spectral norm (Phase 1 artifacts)
        # If loss < 0.05 and spec_norm is NULL, it's likely an old zombie.
        if row["loss_ratio"] is not None and row["loss_ratio"] < 0.05 and spec_norm is None:
            purge_ids.append(rid)
            continue

    if not purge_ids:
        LOGGER.info("No zombie models found. Database is clean!")
    else:
        LOGGER.info(f"Total unique models to purge: {len(purge_ids)}")

        # 3. Perform cascade deletion
        LOGGER.info("Starting deletion...")
        batch_size = 500
        for i in range(0, len(purge_ids), batch_size):
            batch = purge_ids[i:i+batch_size]
            placeholders = ', '.join(['?'] * len(batch))
            cursor.execute(f"DELETE FROM leaderboard WHERE result_id IN ({placeholders})", batch)
            cursor.execute(f"DELETE FROM training_curves WHERE result_id IN ({placeholders})", batch)
            cursor.execute(f"DELETE FROM program_results WHERE result_id IN ({placeholders})", batch)
        conn.commit()
        LOGGER.info("Purge complete!")

    # 4. Sync experiment aggregates and delete empty ones
    LOGGER.info("Syncing experiment table...")
    cursor.execute("SELECT experiment_id FROM experiments")
    experiments = [r["experiment_id"] for r in cursor.fetchall()]
    for eid in tqdm(experiments, desc="Syncing experiments"):
        cursor.execute("""
            SELECT COUNT(*) as n_s1, MIN(loss_ratio) as best_loss, MAX(novelty_score) as best_nov
            FROM program_results WHERE experiment_id = ? AND stage1_passed = 1
        """, (eid,))
        agg = cursor.fetchone()
        cursor.execute("""
            UPDATE experiments SET n_stage1_passed = ?, best_loss_ratio = ?, best_novelty_score = ?
            WHERE experiment_id = ?
        """, (agg["n_s1"] or 0, agg["best_loss"], agg["best_nov"], eid))
    
    cursor.execute("DELETE FROM experiments WHERE n_stage1_passed = 0 AND n_programs_generated > 0")
    LOGGER.info(f"Deleted {cursor.rowcount} orphaned experiments.")
    conn.commit()
    conn.close()

if __name__ == "__main__":
    main()
