#!/usr/bin/env python3
"""Rerun validation for leaderboard rows missing measured multi-seed std.

Uses the built-in ExperimentRunner validation path with ``force=True`` so
already-validated candidates can be revalidated under the current policy.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path


def _target_result_ids(db_path: str, limit: int | None = None) -> list[str]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        query = """
            SELECT l.result_id
            FROM leaderboard l
            JOIN program_results pr ON pr.result_id = l.result_id
            WHERE l.tier IN ('validation', 'breakthrough')
              AND l.validation_multi_seed_std IS NULL
              AND pr.graph_json IS NOT NULL
              AND pr.graph_json != ''
            ORDER BY l.composite_score DESC NULLS LAST, l.timestamp DESC
        """
        if limit is not None:
            query += f" LIMIT {int(limit)}"
        rows = conn.execute(query).fetchall()
        return [str(r["result_id"]) for r in rows if r["result_id"]]
    finally:
        conn.close()


def rerun_validation_missing_std(
    db_path: str,
    device: str = "cuda",
    batch_size: int = 10,
    limit: int | None = None,
    poll_seconds: float = 5.0,
    early_stop_patience: int = 300,
    early_stop_min_delta: float = 1e-3,
    early_stop_min_steps: int = 100,
    dry_run: bool = False,
) -> int:
    from research.scientist.runner import ExperimentRunner, RunConfig

    result_ids = _target_result_ids(db_path, limit=limit)
    if dry_run:
        print({"targets": len(result_ids), "result_ids": result_ids[:20]})
        return 0
    if not result_ids:
        print({"targets": 0, "status": "nothing_to_rerun"})
        return 0

    processed = 0
    batch_index = 0
    while processed < len(result_ids):
        batch = result_ids[processed : processed + batch_size]
        batch_index += 1
        print(
            {
                "batch_index": batch_index,
                "batch_size": len(batch),
                "remaining_after": max(0, len(result_ids) - (processed + len(batch))),
                "first_result_id": batch[0] if batch else None,
            }
        )
        runner = ExperimentRunner(db_path)
        try:
            config = RunConfig(
                device=device,
                early_stop_patience=early_stop_patience,
                early_stop_min_delta=early_stop_min_delta,
                early_stop_min_steps=early_stop_min_steps,
            )
            exp_id = runner.start_validation(
                result_ids=batch,
                config=config,
                hypothesis=(
                    "Validation rerun: fill missing multi-seed variance for "
                    f"{len(batch)} previously validated candidates."
                ),
                trigger="validation_backfill_missing_std",
                force=True,
            )
            print({"experiment_id": exp_id, "status": "started"})
            while runner.is_running:
                progress = runner.progress
            print(
                {
                    "experiment_id": exp_id,
                    "status": progress.status,
                    "current": progress.current_program,
                    "total": progress.total_programs,
                    "aria_message": progress.aria_message,
                    "early_stop_patience": early_stop_patience,
                    "early_stop_min_delta": early_stop_min_delta,
                    "early_stop_min_steps": early_stop_min_steps,
                }
            )
            time.sleep(poll_seconds)
        finally:
            pass
        processed += len(batch)

    print({"status": "completed", "processed": processed})
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="research/lab_notebook.db", help="Path to notebook SQLite DB")
    parser.add_argument("--device", default="cuda", help="Device to use for rerun validation")
    parser.add_argument("--batch-size", type=int, default=10, help="Validation rerun batch size")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on candidates")
    parser.add_argument("--poll-seconds", type=float, default=5.0, help="Progress polling interval")
    parser.add_argument("--early-stop-patience", type=int, default=300, help="Validation plateau patience in steps")
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-3, help="Minimum loss improvement to reset patience")
    parser.add_argument("--early-stop-min-steps", type=int, default=100, help="Minimum steps before early stopping can trigger")
    parser.add_argument("--dry-run", action="store_true", help="List targets without running validation")
    args = parser.parse_args()

    db_path = str(Path(args.db))
    if not Path(db_path).exists():
        print(f"Database not found: {db_path}", file=sys.stderr)
        return 1

    return rerun_validation_missing_std(
        db_path=db_path,
        device=args.device,
        batch_size=args.batch_size,
        limit=args.limit,
        poll_seconds=args.poll_seconds,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        early_stop_min_steps=args.early_stop_min_steps,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
