
import sqlite3
import os

db_path = "research/lab_notebook.db"
if not os.path.exists(db_path):
    print(f"Error: Database not found at {db_path}")
    exit(1)

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row

print("Fetching spectral norm values from program_results...")
results = conn.execute("SELECT result_id, fp_jacobian_spectral_norm FROM program_results WHERE fp_jacobian_spectral_norm IS NOT NULL").fetchall()
print(f"Found {len(results)} potential values to backfill.")

updated_count = 0
for row in results:
    rid = row["result_id"]
    val = row["fp_jacobian_spectral_norm"]
    
    cur = conn.execute(
        "UPDATE leaderboard SET fp_jacobian_spectral_norm = ? WHERE result_id = ? AND fp_jacobian_spectral_norm IS NULL",
        (val, rid)
    )
    updated_count += cur.rowcount

conn.commit()
conn.close()

print(f"Successfully backfilled {updated_count} leaderboard entries.")
