"""Persist a controlled-language S1.0 comparison report into program_results."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from research.eval.controlled_lang_probe import CONTROLLED_LANG_METRIC_VERSION
from research.tools.controlled_lang_backfill import _ensure_backfill_columns
from research.tools.db_health import backup_sqlite_db


def _latest_report() -> Path:
    reports = sorted(
        Path("research/reports").glob("controlled_lang_s10_vocab240_comparison_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not reports:
        raise FileNotFoundError(
            "no controlled_lang_s10_vocab240_comparison report found"
        )
    return reports[0]


def _checkpoint_json(row: dict) -> str:
    return json.dumps(
        row.get("checkpoints") or [],
        sort_keys=True,
        separators=(",", ":"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("research/lab_notebook.db"))
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    report_path = args.report or _latest_report()
    report = json.loads(report_path.read_text())
    rows = [row for row in report.get("rows", []) if row.get("status") == "ok"]
    print(f"report={report_path}")
    print(f"rows={len(rows)}")
    if args.dry_run:
        return 0

    if not args.no_backup:
        backup_path = backup_sqlite_db(args.db, suffix="pre_s10_vocab240_apply")
        print(f"backup={backup_path}")

    conn = sqlite3.connect(str(args.db), timeout=30.0)
    _ensure_backfill_columns(conn)
    updated = 0
    for row in rows:
        conn.execute(
            """
            UPDATE program_results
            SET controlled_lang_metric_version = ?,
                controlled_lang_s10_nb_score = ?,
                controlled_lang_s10_nb_order_acc = ?,
                controlled_lang_s10_checkpoints_json = ?
            WHERE result_id = ?
            """,
            (
                CONTROLLED_LANG_METRIC_VERSION,
                row.get("new_s10_nb"),
                row.get("new_s10_order"),
                _checkpoint_json(row),
                row["result_id"],
            ),
        )
        updated += conn.total_changes - updated
    conn.commit()
    conn.close()
    print(f"updated={updated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
