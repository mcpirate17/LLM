from research.scientist.notebook import LabNotebook
from research.scientist.analytics import ExperimentAnalytics


def debug_db():
    db_path = "/tmp/debug_brain.db"
    import os

    if os.path.exists(db_path):
        os.remove(db_path)

    nb = LabNotebook(db_path)
    exp_id = nb.start_experiment("test", {}, "test")

    r1 = nb.record_program_result(
        exp_id,
        "fp1",
        "{}",
        stage0_passed=1,
        stage1_passed=1,
        loss_ratio=0.2,
        param_count=1000,
    )
    print(f"Recorded r1: {r1}")

    # Check if it's in DB
    row = nb.conn.execute(
        "SELECT result_id, stage1_passed, loss_ratio, param_count FROM program_results"
    ).fetchone()
    print(f"DB Row: {dict(row) if row else 'None'}")

    analytics = ExperimentAnalytics(nb)
    ids = analytics.pareto_optimal_programs()
    print(f"Pareto IDs: {ids}")

    nb.close()


if __name__ == "__main__":
    debug_db()
