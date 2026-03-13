
import sqlite3
import time
from research.scientist.notebook import LabNotebook

db_path = "research/lab_notebook.db"
nb = LabNotebook(db_path)

result_id = "test-repro-id"
model_source = "test"

# 1. Create entry
nb.upsert_leaderboard(
    result_id=result_id,
    model_source=model_source,
    screening_loss_ratio=0.5,
    screening_novelty=0.5,
    tier="screening"
)

print("Created entry. Checking robustness...")
row = nb.conn.execute("SELECT investigation_robustness FROM leaderboard WHERE result_id = ?", (result_id,)).fetchone()
print(f"Robustness (expected NULL): {row[0]}")

# 2. Update with robustness 0.0
nb.upsert_leaderboard(
    result_id=result_id,
    model_source=model_source,
    screening_loss_ratio=0.5,
    screening_novelty=0.5,
    investigation_loss_ratio=None,
    investigation_robustness=0.0,
    investigation_passed=False,
    tier="screening"
)

print("Updated entry. Checking robustness...")
row = nb.conn.execute("SELECT investigation_robustness, investigation_passed FROM leaderboard WHERE result_id = ?", (result_id,)).fetchone()
print(f"Robustness (expected 0.0): {row[0]}")
print(f"Passed (expected 0): {row[1]}")

# Cleanup
nb.conn.execute("DELETE FROM leaderboard WHERE result_id = ?", (result_id,))
nb.conn.commit()
nb.close()
