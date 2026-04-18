#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import time
import uuid
from pathlib import Path

from research.eval.native_induction import (
    INDUCTION_BATCH_SIZE,
    INDUCTION_EVAL_EXAMPLES,
    INDUCTION_GAPS,
    INDUCTION_METRIC_VERSION,
    INDUCTION_POOL_SIZE,
    INDUCTION_SPEED_MODE,
    INDUCTION_TRAIN_STEPS,
)
from research.scientist.notebook import LabNotebook
from research.tools.backfill import rescore_all

DB_PATH = "research/lab_notebook.db"
DEFAULT_CSV = "tasks/induction_native_probe/induction_auc_results.csv"
COHORT_PRIORITY = {
    "stage1_passed_all500": 0,
    "binding_only_s1_all500": 1,
    "public_reference_pool64": 2,
    "binding_only_s1_next100": 3,
    "binding_only_s1_first20": 4,
    "non_reference_top10": 5,
    "public_reference": 6,
}


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 10000")
    return conn


def ensure_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS induction_metrics_v2 (
            graph_fingerprint TEXT PRIMARY KEY,
            result_id TEXT,
            source_cohort TEXT NOT NULL,
            metric_version TEXT NOT NULL,
            speed_mode TEXT NOT NULL,
            train_steps INTEGER NOT NULL,
            eval_examples INTEGER NOT NULL,
            batch_size INTEGER NOT NULL,
            pool_size INTEGER NOT NULL,
            gaps_json TEXT NOT NULL,
            auc REAL NOT NULL,
            gap_4 REAL,
            gap_8 REAL,
            gap_16 REAL,
            gap_32 REAL,
            gap_64 REAL,
            wall_ms REAL,
            updated_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS induction_metrics_archive (
            archive_id TEXT PRIMARY KEY,
            archived_at REAL NOT NULL,
            source_table TEXT NOT NULL,
            source_key TEXT NOT NULL,
            result_id TEXT,
            graph_fingerprint TEXT,
            induction_auc REAL,
            induction_gap_accuracies_json TEXT,
            induction_probe_train_steps INTEGER,
            induction_probe_eval_examples INTEGER,
            induction_probe_batch_size INTEGER,
            induction_probe_gaps_json TEXT,
            induction_probe_elapsed_ms REAL,
            induction_probe_metric_version TEXT,
            induction_probe_speed_mode TEXT,
            induction_probe_pool_size INTEGER
        );
        """
    )
    existing = {
        row[1] for row in conn.execute("PRAGMA table_info(program_results)").fetchall()
    }
    for name, col_type in (
        ("induction_probe_metric_version", "TEXT"),
        ("induction_probe_speed_mode", "TEXT"),
        ("induction_probe_pool_size", "INTEGER"),
    ):
        if name not in existing:
            conn.execute(f"ALTER TABLE program_results ADD COLUMN {name} {col_type}")


def archive_existing(conn: sqlite3.Connection) -> tuple[int, int]:
    archived_at = time.time()
    pr_rows = conn.execute(
        """
        SELECT result_id, graph_fingerprint, induction_auc, induction_gap_accuracies_json,
               induction_probe_train_steps, induction_probe_eval_examples,
               induction_probe_batch_size, induction_probe_gaps_json,
               induction_probe_elapsed_ms, induction_probe_metric_version,
               induction_probe_speed_mode, induction_probe_pool_size
        FROM program_results
        WHERE induction_auc IS NOT NULL
           OR induction_gap_accuracies_json IS NOT NULL
        """
    ).fetchall()
    lb_rows = conn.execute(
        """
        SELECT l.entry_id, l.result_id, pr.graph_fingerprint, l.induction_auc
        FROM leaderboard l
        LEFT JOIN program_results pr ON pr.result_id = l.result_id
        WHERE l.induction_auc IS NOT NULL
        """
    ).fetchall()
    conn.executemany(
        """
        INSERT INTO induction_metrics_archive (
            archive_id, archived_at, source_table, source_key, result_id, graph_fingerprint,
            induction_auc, induction_gap_accuracies_json, induction_probe_train_steps,
            induction_probe_eval_examples, induction_probe_batch_size, induction_probe_gaps_json,
            induction_probe_elapsed_ms, induction_probe_metric_version,
            induction_probe_speed_mode, induction_probe_pool_size
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                str(uuid.uuid4()),
                archived_at,
                "program_results",
                str(row["result_id"]),
                row["result_id"],
                row["graph_fingerprint"],
                row["induction_auc"],
                row["induction_gap_accuracies_json"],
                row["induction_probe_train_steps"],
                row["induction_probe_eval_examples"],
                row["induction_probe_batch_size"],
                row["induction_probe_gaps_json"],
                row["induction_probe_elapsed_ms"],
                row["induction_probe_metric_version"],
                row["induction_probe_speed_mode"],
                row["induction_probe_pool_size"],
            )
            for row in pr_rows
        ]
        + [
            (
                str(uuid.uuid4()),
                archived_at,
                "leaderboard",
                str(row["entry_id"]),
                row["result_id"],
                row["graph_fingerprint"],
                row["induction_auc"],
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
            for row in lb_rows
        ],
    )
    return len(pr_rows), len(lb_rows)


def _row_rank(row: dict[str, str]) -> tuple[int, int, str]:
    return (
        COHORT_PRIORITY.get(row["cohort"], 99),
        0 if row["speed_mode"] == INDUCTION_SPEED_MODE else 1,
        row["result_id"],
    )


def load_canonical_rows(csv_path: Path) -> dict[str, dict[str, str]]:
    rows = list(csv.DictReader(csv_path.open(newline="")))
    chosen: dict[str, dict[str, str]] = {}
    for row in rows:
        if row.get("speed_mode") != INDUCTION_SPEED_MODE:
            continue
        if str(row.get("steps")) != str(INDUCTION_TRAIN_STEPS):
            continue
        fp = str(row.get("fingerprint_id") or "")
        if not fp:
            continue
        prev = chosen.get(fp)
        if prev is None or _row_rank(row) < _row_rank(prev):
            chosen[fp] = row
    return chosen


def populate_induction_v2(
    conn: sqlite3.Connection, canonical_rows: dict[str, dict[str, str]]
) -> int:
    ts = time.time()
    conn.executemany(
        """
        INSERT OR REPLACE INTO induction_metrics_v2 (
            graph_fingerprint, result_id, source_cohort, metric_version, speed_mode,
            train_steps, eval_examples, batch_size, pool_size, gaps_json,
            auc, gap_4, gap_8, gap_16, gap_32, gap_64, wall_ms, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                fp,
                row.get("result_id"),
                row.get("cohort"),
                INDUCTION_METRIC_VERSION,
                INDUCTION_SPEED_MODE,
                INDUCTION_TRAIN_STEPS,
                INDUCTION_EVAL_EXAMPLES,
                INDUCTION_BATCH_SIZE,
                INDUCTION_POOL_SIZE,
                json.dumps(list(INDUCTION_GAPS)),
                float(row["auc"]),
                float(row["ind_gap_4"] or 0.0),
                float(row["ind_gap_8"] or 0.0),
                float(row["ind_gap_16"] or 0.0),
                float(row["ind_gap_32"] or 0.0),
                float(row["ind_gap_64"] or 0.0),
                float(row["wall_ms"] or 0.0),
                ts,
            )
            for fp, row in canonical_rows.items()
        ],
    )
    return len(canonical_rows)


def overwrite_program_results(conn: sqlite3.Connection) -> int:
    rows = conn.execute(
        """
        SELECT graph_fingerprint, auc, gap_4, gap_8, gap_16, gap_32, gap_64, wall_ms
        FROM induction_metrics_v2
        """
    ).fetchall()
    before = conn.total_changes
    conn.executemany(
        """
        UPDATE program_results
        SET induction_auc = ?,
            induction_gap_accuracies_json = ?,
            induction_probe_train_steps = ?,
            induction_probe_eval_examples = ?,
            induction_probe_batch_size = ?,
            induction_probe_gaps_json = ?,
            induction_probe_elapsed_ms = ?,
            induction_probe_metric_version = ?,
            induction_probe_speed_mode = ?,
            induction_probe_pool_size = ?
        WHERE graph_fingerprint = ?
        """,
        [
            (
                row["auc"],
                json.dumps(
                    {
                        4: row["gap_4"],
                        8: row["gap_8"],
                        16: row["gap_16"],
                        32: row["gap_32"],
                        64: row["gap_64"],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                INDUCTION_TRAIN_STEPS,
                INDUCTION_EVAL_EXAMPLES,
                INDUCTION_BATCH_SIZE,
                json.dumps(list(INDUCTION_GAPS)),
                row["wall_ms"],
                INDUCTION_METRIC_VERSION,
                INDUCTION_SPEED_MODE,
                INDUCTION_POOL_SIZE,
                row["graph_fingerprint"],
            )
            for row in rows
        ],
    )
    return conn.total_changes - before


def overwrite_leaderboard(conn: sqlite3.Connection) -> int:
    before = conn.total_changes
    conn.execute(
        """
        UPDATE leaderboard
        SET induction_auc = (
            SELECT im.auc
            FROM program_results pr
            JOIN induction_metrics_v2 im ON im.graph_fingerprint = pr.graph_fingerprint
            WHERE pr.result_id = leaderboard.result_id
        )
        WHERE EXISTS (
            SELECT 1
            FROM program_results pr
            JOIN induction_metrics_v2 im ON im.graph_fingerprint = pr.graph_fingerprint
            WHERE pr.result_id = leaderboard.result_id
        )
        """
    )
    return conn.total_changes - before


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate canonical induction metrics into the notebook DB."
    )
    parser.add_argument("--db", default=DB_PATH)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument(
        "--apply", action="store_true", help="Mutate the target DB. Default is dry-run."
    )
    parser.add_argument("--skip-rescore", action="store_true")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    canonical_rows = load_canonical_rows(csv_path)
    print(f"canonical_fingerprints={len(canonical_rows)} csv={csv_path}")

    if not args.apply:
        print("dry_run=1")
        return

    conn = _connect(args.db)
    ensure_tables(conn)
    archived_pr, archived_lb = archive_existing(conn)
    loaded = populate_induction_v2(conn, canonical_rows)
    updated_pr = overwrite_program_results(conn)
    updated_lb = overwrite_leaderboard(conn)
    conn.commit()
    conn.close()

    rescored_total = 0
    rescored_changed = 0
    if not args.skip_rescore:
        nb = LabNotebook(args.db)
        rescored_total, rescored_changed = rescore_all(nb)

    print(
        "archived_program_results=%d archived_leaderboard=%d loaded_v2=%d "
        "updated_program_results=%d updated_leaderboard=%d rescored_total=%d rescored_changed=%d"
        % (
            archived_pr,
            archived_lb,
            loaded,
            updated_pr,
            updated_lb,
            rescored_total,
            rescored_changed,
        )
    )


if __name__ == "__main__":
    main()
