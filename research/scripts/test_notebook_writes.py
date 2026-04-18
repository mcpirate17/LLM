from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
import uuid

from research.scientist.notebook import LabNotebook


def _plain_sqlite_probe(db_path: str) -> None:
    exp_id = f"probe-sql-{uuid.uuid4().hex[:8]}"
    now = time.time()
    conn = sqlite3.connect(db_path, timeout=10.0)
    try:
        conn.execute(
            """INSERT INTO experiments
            (experiment_id, timestamp, experiment_type, status, hypothesis,
             research_question, preregistration_id, config_json, started_at)
            VALUES (?, ?, ?, 'running', ?, ?, ?, ?, ?)""",
            (
                exp_id,
                now,
                "probe_sqlite",
                "plain sqlite probe",
                None,
                None,
                json.dumps({"probe": "plain_sqlite", "code_version": "probe"}),
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    fresh = sqlite3.connect(db_path, timeout=10.0)
    try:
        row = fresh.execute(
            "SELECT status FROM experiments WHERE experiment_id = ?",
            (exp_id,),
        ).fetchone()
        if not row or row[0] != "running":
            raise RuntimeError(f"Plain insert not visible from fresh connection: {row}")
        fresh.execute(
            """UPDATE experiments
               SET status = 'completed',
                   completed_at = ?,
                   n_programs_generated = ?,
                   aria_summary = ?
               WHERE experiment_id = ?""",
            (time.time(), 1, "plain sqlite probe complete", exp_id),
        )
        fresh.commit()
    finally:
        fresh.close()

    verify = sqlite3.connect(db_path, timeout=10.0)
    try:
        row = verify.execute(
            "SELECT status, n_programs_generated FROM experiments WHERE experiment_id = ?",
            (exp_id,),
        ).fetchone()
        if not row or row[0] != "completed" or row[1] != 1:
            raise RuntimeError(
                f"Plain completion not visible from fresh connection: {row}"
            )
        verify.execute("DELETE FROM experiments WHERE experiment_id = ?", (exp_id,))
        verify.commit()
    finally:
        verify.close()


def _lab_notebook_probe(db_path: str, iterations: int) -> None:
    failures: list[str] = []
    for i in range(iterations):
        nb = None
        exp_id = None
        try:
            nb = LabNotebook(db_path, skip_migrate=True)
            exp_id = nb.start_experiment(
                "probe_notebook",
                {"probe_iteration": i},
                hypothesis="lab notebook probe",
            )
            row = sqlite3.connect(db_path, timeout=10.0).execute(
                "SELECT status FROM experiments WHERE experiment_id = ?",
                (exp_id,),
            ).fetchone()
            if not row or row[0] != "running":
                failures.append(f"iter {i}: start not visible: {row!r}")
                continue

            nb.complete_experiment(
                exp_id,
                {
                    "total": 1,
                    "stage0_passed": 1,
                    "stage05_passed": 1,
                    "stage1_passed": 1,
                    "best_loss_ratio": 0.1,
                    "best_novelty_score": 0.2,
                },
                aria_summary="lab notebook probe complete",
            )
            row = sqlite3.connect(db_path, timeout=10.0).execute(
                "SELECT status, n_programs_generated FROM experiments WHERE experiment_id = ?",
                (exp_id,),
            ).fetchone()
            if not row or row[0] != "completed" or row[1] != 1:
                failures.append(f"iter {i}: completion not visible: {row!r}")
        finally:
            if exp_id:
                cleanup = sqlite3.connect(db_path, timeout=10.0)
                try:
                    cleanup.execute(
                        "DELETE FROM entries WHERE experiment_id = ?",
                        (exp_id,),
                    )
                    cleanup.execute(
                        "DELETE FROM experiments WHERE experiment_id = ?",
                        (exp_id,),
                    )
                    cleanup.commit()
                finally:
                    cleanup.close()
            if nb is not None:
                nb.close()

    if failures:
        raise RuntimeError("\n".join(failures))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db-path",
        default="research/lab_notebook.db",
        help="Path to the notebook SQLite database.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of LabNotebook lifecycle iterations to run.",
    )
    args = parser.parse_args()

    conn = sqlite3.connect(args.db_path, timeout=10.0)
    try:
        print("integrity_check:", conn.execute("PRAGMA integrity_check").fetchone()[0])
        print("quick_check:", conn.execute("PRAGMA quick_check").fetchone()[0])
        print("journal_mode:", conn.execute("PRAGMA journal_mode").fetchone()[0])
        print("synchronous:", conn.execute("PRAGMA synchronous").fetchone()[0])
    finally:
        conn.close()

    _plain_sqlite_probe(args.db_path)
    print("plain_sqlite_probe: ok")
    _lab_notebook_probe(args.db_path, args.iterations)
    print(f"lab_notebook_probe: ok ({args.iterations} iterations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
