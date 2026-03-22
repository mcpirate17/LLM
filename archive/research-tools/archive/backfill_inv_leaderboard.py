import sqlite3
import zlib
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backfill_leaderboard")

db_path = "research/lab_notebook.db"
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row

# 1. Fetch all investigation experiments
rows = conn.execute(
    "SELECT experiment_id, results_json FROM experiments WHERE experiment_type = 'investigation' AND status = 'completed'"
).fetchall()

print(f"Analyzing {len(rows)} investigation experiments for backfill...")

for row in rows:
    exp_id = row["experiment_id"]
    if row["results_json"]:
        try:
            data = zlib.decompress(row["results_json"])
            js = json.loads(data)
            inv_results = js.get("investigation_results", [])

            for res in inv_results:
                rid = res.get("result_id")
                robustness = res.get("robustness")
                best_lr = res.get("best_loss_ratio")

                if rid and robustness is not None:
                    # Check if leaderboard entry exists
                    lb_row = conn.execute(
                        "SELECT entry_id, investigation_robustness FROM leaderboard WHERE result_id = ?",
                        (rid,),
                    ).fetchone()
                    if lb_row:
                        if lb_row["investigation_robustness"] is None:
                            print(
                                f"Backfilling {rid}: robustness={robustness}, best_lr={best_lr}"
                            )
                            # We'll use a simplified update here
                            conn.execute(
                                """
                                UPDATE leaderboard SET
                                    investigation_loss_ratio = ?,
                                    investigation_robustness = ?,
                                    investigation_passed = ?
                                WHERE result_id = ?
                            """,
                                (
                                    best_lr,
                                    robustness,
                                    1
                                    if (robustness >= 0.5 and (best_lr or 1.0) < 0.5)
                                    else 0,
                                    rid,
                                ),
                            )
        except Exception as e:
            logger.error(f"Error processing experiment {exp_id}: {e}")

conn.commit()
conn.close()
print("Backfill complete.")
