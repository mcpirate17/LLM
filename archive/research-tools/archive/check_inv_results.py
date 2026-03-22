import sqlite3
import zlib
import json

db_path = "research/lab_notebook.db"
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row

experiments = ["42e16828-dcd", "309a555b-136"]

for exp_id in experiments:
    row = conn.execute(
        "SELECT results_json FROM experiments WHERE experiment_id = ?", (exp_id,)
    ).fetchone()
    if row and row["results_json"]:
        try:
            data = zlib.decompress(row["results_json"])
            js = json.loads(data)
            print(f"--- Experiment {exp_id} ---")
            inv_res = js.get("investigation_results", [])
            for r in inv_res:
                print(
                    f"Source ID: {r.get('result_id')}, Robustness: {r.get('robustness')}, Best LR: {r.get('best_loss_ratio')}"
                )
        except Exception as e:
            print(f"Error decoding {exp_id}: {e}")

conn.close()
