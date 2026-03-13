#!/usr/bin/env python3
"""Snapshot current novelty/fingerprint fields into legacy columns.

Use this before recalculating novelty under a new scoring policy so the old
scores remain available for comparison.
"""

from __future__ import annotations

import argparse
import sqlite3


LEGACY_POLICY_VERSION = "full_fp_legacy_v1"
_LEGACY_COLUMNS = {
    "novelty_score_legacy": "REAL",
    "structural_novelty_legacy": "REAL",
    "behavioral_novelty_legacy": "REAL",
    "novelty_confidence_legacy": "REAL",
    "novelty_raw_score_legacy": "REAL",
    "novelty_z_score_legacy": "REAL",
    "novelty_reference_version_legacy": "TEXT",
    "novelty_valid_for_promotion_legacy": "INTEGER",
    "novelty_validity_reason_legacy": "TEXT",
    "fingerprint_json_legacy": "TEXT",
}


def _ensure_legacy_columns(conn: sqlite3.Connection) -> None:
    existing = {
        row[1]
        for row in conn.execute("PRAGMA table_info(program_results)").fetchall()
    }
    for column_name, column_type in _LEGACY_COLUMNS.items():
        if column_name in existing:
            continue
        conn.execute(f"ALTER TABLE program_results ADD COLUMN {column_name} {column_type}")
    conn.commit()


def _snapshot_legacy_novelty_scores_where(
    conn: sqlite3.Connection,
    where_sql: str,
    where_params: tuple[object, ...] = (),
    dry_run: bool = False,
) -> int:
    _ensure_legacy_columns(conn)
    row = conn.execute(
        f"""
        SELECT COUNT(*)
        FROM program_results
        WHERE novelty_score IS NOT NULL
          AND novelty_score_legacy IS NULL
          AND ({where_sql})
        """,
        where_params,
    ).fetchone()
    count = int(row[0] if row else 0)
    if dry_run or count <= 0:
        return count

    conn.execute(
        f"""
        UPDATE program_results
        SET
            novelty_score_legacy = novelty_score,
            structural_novelty_legacy = structural_novelty,
            behavioral_novelty_legacy = behavioral_novelty,
            novelty_confidence_legacy = novelty_confidence,
            novelty_raw_score_legacy = novelty_raw_score,
            novelty_z_score_legacy = novelty_z_score,
            novelty_reference_version_legacy = novelty_reference_version,
            novelty_valid_for_promotion_legacy = novelty_valid_for_promotion,
            novelty_validity_reason_legacy = novelty_validity_reason,
            fingerprint_json_legacy = fingerprint_json,
            novelty_scoring_policy_version = COALESCE(
                novelty_scoring_policy_version,
                ?
            )
        WHERE novelty_score IS NOT NULL
          AND novelty_score_legacy IS NULL
          AND ({where_sql})
        """,
        (LEGACY_POLICY_VERSION, *where_params),
    )
    conn.commit()
    return count


def snapshot_legacy_novelty_scores(db_path: str, dry_run: bool = False) -> int:
    conn = sqlite3.connect(db_path)
    try:
        return _snapshot_legacy_novelty_scores_where(
            conn,
            where_sql="1 = 1",
            dry_run=dry_run,
        )
    finally:
        conn.close()


def snapshot_legacy_novelty_scores_for_result_ids(
    db_path: str,
    result_ids: list[str],
    dry_run: bool = False,
) -> int:
    if not result_ids:
        return 0
    placeholders = ", ".join("?" for _ in result_ids)
    conn = sqlite3.connect(db_path)
    try:
        return _snapshot_legacy_novelty_scores_where(
            conn,
            where_sql=f"result_id IN ({placeholders})",
            where_params=tuple(result_ids),
            dry_run=dry_run,
        )
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="Path to notebook SQLite DB")
    parser.add_argument("--dry-run", action="store_true", help="Show rows to snapshot without updating")
    args = parser.parse_args()

    count = snapshot_legacy_novelty_scores(args.db, dry_run=args.dry_run)
    if args.dry_run:
        print(f"Would snapshot {count} program_results rows")
    else:
        print(f"Snapshotted {count} program_results rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
