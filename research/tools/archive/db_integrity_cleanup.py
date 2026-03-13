import os
import sqlite3
from pathlib import Path
from research.scientist.notebook import LabNotebook

def cleanup():
    db_path = LabNotebook.resolve_db_path("research/lab_notebook.db")
    if not db_path.exists():
        print(f"Database not found at {db_path}")
        return

    print(f"Cleaning database: {db_path}")
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA busy_timeout=10000")
    
    # 1. Orphaned program results (no parent experiment)
    cur = conn.execute("DELETE FROM program_results WHERE experiment_id NOT IN (SELECT experiment_id FROM experiments)")
    if cur.rowcount > 0:
        print(f"  - Removed {cur.rowcount} orphaned program results")

    # 2. Empty/Aborted experiments with no programs
    cur = conn.execute("DELETE FROM experiments WHERE n_programs_generated = 0 AND status != 'running'")
    if cur.rowcount > 0:
        print(f"  - Removed {cur.rowcount} empty/stale experiments")

    # 3. Clean up orphaned training curves
    cur = conn.execute("DELETE FROM training_curves WHERE result_id NOT IN (SELECT result_id FROM program_results)")
    if cur.rowcount > 0:
        print(f"  - Removed {cur.rowcount} orphaned training curves")

    # 4. Clean up disconnected fingerprints in leaderboard
    cur = conn.execute("DELETE FROM leaderboard WHERE result_id NOT IN (SELECT result_id FROM program_results)")
    if cur.rowcount > 0:
        print(f"  - Removed {cur.rowcount} disconnected leaderboard entries")

    # 5. Fix corrupted screening robustness (from previous sessions)
    cur = conn.execute("""
        UPDATE leaderboard 
        SET investigation_robustness = NULL, 
            investigation_passed = NULL, 
            investigation_loss_ratio = NULL 
        WHERE tier = 'screening' AND (investigation_robustness = 0.0 OR investigation_robustness IS NOT NULL)
    """)
    if cur.rowcount > 0:
        print(f"  - Reset {cur.rowcount} stuck screening candidates")

    conn.commit()
    conn.execute("VACUUM")
    conn.close()
    print("Database cleanup complete.")

if __name__ == "__main__":
    cleanup()
