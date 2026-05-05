"""Apply controlled-language cascade gates to already-backfilled leaderboard rows."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from research.scientist.controlled_lang_gates import (
    CONTROLLED_LANG_SCORE_GATES,
    S05_NB_SCREENING_FAILURE_THRESHOLD,
    apply_controlled_lang_screening_failure,
)
from research.tools.db_health import backup_sqlite_db


def _candidate_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    return conn.execute(
        """
        SELECT l.entry_id,
               l.result_id,
               l.tier,
               pr.graph_fingerprint,
               pr.controlled_lang_s05_sa_score,
               pr.controlled_lang_s05_nb_order_acc,
               pr.controlled_lang_s05_nb_score,
               pr.controlled_lang_s10_sa_score,
               pr.controlled_lang_s10_nb_order_acc,
               pr.controlled_lang_s10_nb_score,
               pr.controlled_lang_inv_sa_score,
               pr.controlled_lang_inv_nb_order_acc,
               pr.controlled_lang_inv_nb_score
        FROM leaderboard l
        JOIN program_results pr ON pr.result_id = l.result_id
        WHERE (
              pr.controlled_lang_s05_sa_score < ?
           OR pr.controlled_lang_s05_nb_order_acc < ?
           OR pr.controlled_lang_s05_nb_score < ?
           OR pr.controlled_lang_s10_sa_score < ?
           OR pr.controlled_lang_s10_nb_order_acc < ?
           OR pr.controlled_lang_s10_nb_score < ?
           OR pr.controlled_lang_inv_sa_score < ?
           OR pr.controlled_lang_inv_nb_order_acc < ?
           OR pr.controlled_lang_inv_nb_score < ?
        )
          AND COALESCE(l.is_reference, 0) = 0
          AND COALESCE(l.tier, '') NOT IN ('screened_out', 'retired')
        ORDER BY l.tier, l.composite_score DESC
        """,
        (
            S05_NB_SCREENING_FAILURE_THRESHOLD,
            S05_NB_SCREENING_FAILURE_THRESHOLD,
            S05_NB_SCREENING_FAILURE_THRESHOLD,
            S05_NB_SCREENING_FAILURE_THRESHOLD,
            S05_NB_SCREENING_FAILURE_THRESHOLD,
            S05_NB_SCREENING_FAILURE_THRESHOLD,
            S05_NB_SCREENING_FAILURE_THRESHOLD,
            S05_NB_SCREENING_FAILURE_THRESHOLD,
            S05_NB_SCREENING_FAILURE_THRESHOLD,
        ),
    ).fetchall()


def _first_failed_gate(row: sqlite3.Row) -> dict[str, str] | None:
    for gate in CONTROLLED_LANG_SCORE_GATES:
        score = row[gate["score_key"]]
        if score is not None and float(score) < S05_NB_SCREENING_FAILURE_THRESHOLD:
            return gate
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("research/lab_notebook.db"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating a SQLite backup before applying changes.",
    )
    args = parser.parse_args()

    conn = sqlite3.connect(str(args.db), timeout=30.0)
    rows = _candidate_rows(conn)
    by_tier: dict[str, int] = {}
    for row in rows:
        by_tier[str(row["tier"] or "")] = by_tier.get(str(row["tier"] or ""), 0) + 1

    print(f"candidates={len(rows)} threshold={S05_NB_SCREENING_FAILURE_THRESHOLD:.2f}")
    for tier, count in sorted(by_tier.items(), key=lambda item: (-item[1], item[0])):
        print(f"{tier or 'unknown'}={count}")

    if args.dry_run or not rows:
        conn.close()
        return 0

    if not args.no_backup:
        backup_path = backup_sqlite_db(args.db, suffix="pre_controlled_lang_gates")
        print(f"backup={backup_path}")

    applied = 0
    for row in rows:
        gate = _first_failed_gate(row)
        if gate is None:
            continue
        if apply_controlled_lang_screening_failure(
            conn,
            result_id=str(row["result_id"]),
            gate=gate,
            score=row[gate["score_key"]],
            source="apply_controlled_lang_gates",
        ):
            applied += 1
    conn.commit()
    conn.close()
    print(f"applied={applied}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
