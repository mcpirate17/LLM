#!/usr/bin/env python3
"""Legacy compatibility wrapper for the unified rescore path."""

from __future__ import annotations

from research.tools._script_audit import (
    complete_script_experiment,
    start_script_experiment,
)
from research.tools.backfill import DB_PATH, rescore_all


def main() -> None:
    nb, exp_id = start_script_experiment(
        db_path=DB_PATH,
        experiment_type="score_backfill",
        config={"mode": "rescore"},
        source_script="rescore_all_v7",
        hypothesis="Bulk leaderboard rescore",
    )
    try:
        total, changed = rescore_all(nb)
        print(f"Rescored {total} entries, {changed} changed.")
        complete_script_experiment(
            nb,
            exp_id,
            results={"total": total, "changed": changed, "mode": "rescore"},
            summary=f"Bulk rescore complete: changed={changed}/{total}",
        )
    finally:
        nb.conn.close()


if __name__ == "__main__":
    main()
