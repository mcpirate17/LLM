#!/usr/bin/env python3
import sqlite3
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

DB_PATH = "research/lab_notebook.db"


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 1. Fetch all experiments that still have program results
    LOGGER.info("Fetching experiment list...")
    cursor.execute("SELECT experiment_id FROM experiments")
    experiments = [row["experiment_id"] for row in cursor.fetchall()]
    LOGGER.info(f"Syncing {len(experiments)} experiments...")

    for eid in tqdm(experiments):
        # Calculate true aggregates from surviving programs
        cursor.execute(
            """
            SELECT 
                COUNT(*) as n_s1,
                MIN(loss_ratio) as best_loss,
                MAX(novelty_score) as best_nov
            FROM program_results
            WHERE experiment_id = ? AND stage1_passed = 1
        """,
            (eid,),
        )
        agg = cursor.fetchone()

        # Update experiment record
        cursor.execute(
            """
            UPDATE experiments 
            SET n_stage1_passed = ?,
                best_loss_ratio = ?,
                best_novelty_score = ?
            WHERE experiment_id = ?
        """,
            (agg["n_s1"] or 0, agg["best_loss"], agg["best_nov"], eid),
        )

    conn.commit()
    LOGGER.info("Experiment aggregates synchronized.")

    # 2. Delete experiments that now have 0 survivors (if any were missed)
    cursor.execute(
        "DELETE FROM experiments WHERE n_stage1_passed = 0 AND n_programs_generated > 0"
    )
    if cursor.rowcount > 0:
        LOGGER.info(f"Deleted {cursor.rowcount} experiments that now have 0 survivors.")
        conn.commit()

    conn.close()


if __name__ == "__main__":
    main()
