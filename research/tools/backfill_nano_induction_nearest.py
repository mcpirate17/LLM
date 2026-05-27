"""Import nano_induction_nearest JSONL results into graph_runs.

The source JSONL is produced by the cheap structural nearest-induction probe.
Only clean ``nearest_status == "ok"`` rows populate metric values; failed rows
record status/error/provenance without fake zero accuracies.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from research.defaults import RUNS_DB

NANO_INDUCTION_NEAREST_COLUMNS: dict[str, str] = {
    "nano_induction_nearest_max_accuracy": "REAL",
    "nano_induction_nearest_final_accuracy": "REAL",
    "nano_induction_nearest_status": "TEXT",
    "nano_induction_nearest_elapsed_ms": "REAL",
    "nano_induction_nearest_error": "TEXT",
    "nano_induction_nearest_accuracies_json": "TEXT",
    "nano_induction_nearest_train_steps": "INTEGER",
    "nano_induction_nearest_protocol_version": "TEXT",
}

DEFAULT_PROTOCOL_VERSION = "nano_induction_nearest_v1_steps120"


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})")}


def ensure_nano_induction_nearest_columns(conn: sqlite3.Connection) -> list[str]:
    """Ensure canonical persistence columns exist on program_results/graph_runs."""
    added: list[str] = []
    for table in ("program_results", "graph_runs"):
        if not _table_exists(conn, table):
            continue
        existing = _table_columns(conn, table)
        for name, col_type in NANO_INDUCTION_NEAREST_COLUMNS.items():
            if name in existing:
                continue
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {col_type}")
            added.append(f"{table}.{name}")
    return added


def updates_from_record(
    record: dict[str, Any],
    *,
    protocol_version: str = DEFAULT_PROTOCOL_VERSION,
) -> dict[str, Any]:
    status = str(record.get("nearest_status") or "missing")
    ok = status == "ok"
    updates: dict[str, Any] = {
        "nano_induction_nearest_status": status,
        "nano_induction_nearest_train_steps": 120,
        "nano_induction_nearest_protocol_version": protocol_version,
    }
    elapsed = record.get("nearest_elapsed_s")
    if elapsed is not None:
        updates["nano_induction_nearest_elapsed_ms"] = round(float(elapsed) * 1000.0, 3)
    if record.get("nearest_error") is not None:
        updates["nano_induction_nearest_error"] = str(record.get("nearest_error"))
    accuracies = record.get("nearest_accuracies")
    if isinstance(accuracies, list):
        updates["nano_induction_nearest_accuracies_json"] = json.dumps(
            accuracies, separators=(",", ":")
        )
    if ok:
        updates["nano_induction_nearest_max_accuracy"] = _finite_or_none(
            record.get("nearest_max_accuracy")
        )
        updates["nano_induction_nearest_final_accuracy"] = _finite_or_none(
            record.get("nearest_final_accuracy")
        )
    return updates


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    out = float(value)
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                yield (
                    line_no,
                    {
                        "nearest_status": "unparseable",
                        "nearest_error": str(exc),
                    },
                )


def apply_backfill(
    conn: sqlite3.Connection,
    jsonl_path: Path,
    *,
    force: bool = False,
    dry_run: bool = True,
) -> dict[str, int]:
    if not dry_run:
        ensure_nano_induction_nearest_columns(conn)
    columns = _table_columns(conn, "graph_runs")
    allowed = [c for c in NANO_INDUCTION_NEAREST_COLUMNS if c in columns or dry_run]
    has_status = "nano_induction_nearest_status" in columns
    seen = updated = skipped_missing = skipped_existing = values = 0
    for _line_no, record in iter_jsonl(jsonl_path):
        seen += 1
        result_id = str(record.get("result_id") or "")
        if not result_id:
            skipped_missing += 1
            continue
        if has_status:
            target = conn.execute(
                "SELECT result_id, nano_induction_nearest_status "
                "FROM graph_runs WHERE result_id = ?",
                (result_id,),
            ).fetchone()
        else:
            target = conn.execute(
                "SELECT result_id FROM graph_runs WHERE result_id = ?",
                (result_id,),
            ).fetchone()
        if target is None:
            skipped_missing += 1
            continue
        existing_status = (
            target["nano_induction_nearest_status"] if has_status else None
        )
        if not force and existing_status is not None:
            skipped_existing += 1
            continue
        raw_updates = updates_from_record(record)
        updates = {
            col: raw_updates[col]
            for col in allowed
            if col in raw_updates and raw_updates[col] is not None
        }
        if not updates:
            continue
        updated += 1
        values += len(updates)
        if dry_run:
            continue
        set_clause = ", ".join(f"{col} = ?" for col in updates)
        conn.execute(
            f"UPDATE graph_runs SET {set_clause} WHERE result_id = ?",
            (*updates.values(), result_id),
        )
    if not dry_run:
        conn.commit()
    return {
        "source_rows": seen,
        "updated_rows": updated,
        "updated_values": values,
        "skipped_missing_result": skipped_missing,
        "skipped_existing": skipped_existing,
        "dry_run": int(bool(dry_run)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path(RUNS_DB))
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    conn = _connect(args.db)
    try:
        summary = apply_backfill(
            conn,
            args.jsonl,
            force=bool(args.force),
            dry_run=not bool(args.apply),
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
