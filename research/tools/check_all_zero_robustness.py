
import sqlite3
import zlib
import json

db_path = "research/lab_notebook.db"
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row

rows = conn.execute("SELECT experiment_id, results_json FROM experiments WHERE experiment_type = 'investigation'").fetchall()

zero_robustness_ids = []

for row in rows:
    exp_id = row['experiment_id']
    if row['results_json']:
        try:
            data = zlib.decompress(row['results_json'])
            js = json.loads(data)
            inv_res = js.get('investigation_results', [])
            for r in inv_res:
                if r.get('robustness') == 0.0:
                    zero_robustness_ids.append(r.get('result_id'))
        except:
            pass

print(f"Found {len(zero_robustness_ids)} results with 0.0 robustness in experiments.")
for rid in zero_robustness_ids[:10]:
    row = conn.execute("SELECT investigation_robustness FROM leaderboard WHERE result_id = ?", (rid,)).fetchone()
    if row:
        print(f"ID: {rid}, Leaderboard Robustness: {row[0]}")
    else:
        print(f"ID: {rid}, NOT IN LEADERBOARD")

conn.close()
