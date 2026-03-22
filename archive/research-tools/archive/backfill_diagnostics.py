import json
import sqlite3
import torch
import logging

# Local imports
from research.synthesis.serializer import graph_from_json
from research.synthesis.compiler import compile_model
from research.eval.diagnostic_tasks import run_diagnostic_suite
from research.scientist.notebook import LabNotebook

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("backfill_diagnostics")

DB_PATH = "research/lab_notebook.db"


def backfill_diagnostics(limit: int = 50):
    LabNotebook(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Fetch top N records by loss_ratio that passed Stage 1 and lack diagnostic scores
    cursor.execute(
        """
        SELECT result_id, graph_json, loss_ratio
        FROM program_results 
        WHERE stage1_passed = 1 AND diagnostic_score IS NULL
        ORDER BY loss_ratio DESC
        LIMIT ?
    """,
        (limit,),
    )
    rows = cursor.fetchall()
    logger.info(f"Found {len(rows)} high-performing records missing diagnostics.")

    if not rows:
        logger.info("No records to backfill.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    for row in rows:
        rid = row["result_id"]
        logger.info(f"Evaluating {rid} (loss_ratio: {row['loss_ratio']:.4f})")

        try:
            graph = graph_from_json(row["graph_json"])

            # Use compile_model to get a full SynthesizedModel with embedding/head
            model = compile_model([graph])

            # Run the suite (this takes ~40-60s per model)
            suite_result = run_diagnostic_suite(model, device=device, n_steps=100)

            for task_res in suite_result.tasks:
                if task_res.error:
                    logger.error(
                        f"    Task {task_res.task_name} error: {task_res.error}"
                    )

            # Update DB
            conn.execute(
                """
                UPDATE program_results 
                SET diagnostic_score = ?, diagnostic_tasks_json = ?
                WHERE result_id = ?
            """,
                (
                    suite_result.diagnostic_score,
                    json.dumps(suite_result.to_dict()),
                    rid,
                ),
            )

            logger.info(f"  Score: {suite_result.diagnostic_score:.4f}")

            # Cleanup
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

            # Commit after each model to save progress
            conn.commit()

        except Exception as e:
            logger.error(f"  Failed diagnostics for {rid}: {e}")
            conn.rollback()

    conn.close()
    logger.info("Diagnostic backfill complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    backfill_diagnostics(limit=args.limit)
