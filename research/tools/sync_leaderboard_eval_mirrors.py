"""Sync leaderboard eval mirror columns from program_results.

The leaderboard table stores denormalized metric columns for fast UI reads.
After the 2026-04-26 BPE/tiktoken backfill, program_results became the
authoritative source for Wikitext, TinyStories, HellaSwag, BLiMP, and metric
provenance. This tool copies only current BPE eval rows by default so stale
byte-era mirrors do not leak into the dashboard.

Usage:
    python -m research.tools.sync_leaderboard_eval_mirrors --dry-run
    python -m research.tools.sync_leaderboard_eval_mirrors --backup
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Any

from research.tools.db_health import assert_sqlite_health, backup_sqlite_db

BPE_VERSION = "bpe_eval_v1"

EVAL_MIRROR_COLUMNS = (
    "wikitext_perplexity",
    "wikitext_score",
    "wikitext_pre_perplexity",
    "wikitext_ppl_improvement",
    "screening_wikitext_metric_version",
    "screening_wikitext_status",
    "screening_wikitext_variant",
    "screening_wikitext_elapsed_ms",
    "screening_wikitext_budget_json",
    "tokenizer_mode",
    "corpus_path",
    "hellaswag_acc",
    "blimp_overall_accuracy",
    "blimp_n_subtasks",
    "blimp_status",
    "tinystories_perplexity",
    "tinystories_score",
)


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def _changed(old: Any, new: Any) -> bool:
    if old is None and new is None:
        return False
    if isinstance(old, float) or isinstance(new, float):
        try:
            return abs(float(old) - float(new)) > 1e-9
        except (TypeError, ValueError):
            return old != new
    return old != new


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        default="research/runs.db",
        help="Path to lab notebook DB.",
    )
    parser.add_argument(
        "--metric-version",
        default=BPE_VERSION,
        help=f"Program result metric version to copy. Default: {BPE_VERSION}.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Report changes without writing."
    )
    parser.add_argument(
        "--backup", action="store_true", help="Create a SQLite backup before writing."
    )
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="Skip pre/post SQLite quick_check gates.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit.")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not args.skip_health_check:
        assert_sqlite_health(db_path, label="pre-sync")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        lb_cols = _table_columns(conn, "leaderboard")
        pr_cols = _table_columns(conn, "program_results")
        copy_cols = [c for c in EVAL_MIRROR_COLUMNS if c in lb_cols and c in pr_cols]
        if not copy_cols:
            raise SystemExit("no shared eval mirror columns found")

        sql = (
            "SELECT lb.entry_id, lb.result_id, "
            + ", ".join(f"lb.{c} AS lb_{c}, pr.{c} AS pr_{c}" for c in copy_cols)
            + " FROM leaderboard lb "
            + "JOIN program_results_compat pr ON pr.result_id = lb.result_id "
            + "WHERE pr.screening_wikitext_metric_version = ? "
            + "ORDER BY lb.composite_score DESC"
        )
        params: list[Any] = [args.metric_version]
        if args.limit is not None:
            sql += " LIMIT ?"
            params.append(int(args.limit))
        rows = conn.execute(sql, params).fetchall()

        updates: list[tuple[str, dict[str, Any]]] = []
        for row in rows:
            changed = {
                col: row[f"pr_{col}"]
                for col in copy_cols
                if row[f"pr_{col}"] is not None
                and _changed(row[f"lb_{col}"], row[f"pr_{col}"])
            }
            if changed:
                updates.append((str(row["entry_id"]), changed))

        print(
            f"scanned={len(rows)} rows_with_changes={len(updates)} columns={','.join(copy_cols)}"
        )
        if args.dry_run:
            for entry_id, changed in updates[:20]:
                print(f"dry_run entry_id={entry_id} changed={','.join(changed)}")
            if len(updates) > 20:
                print(f"dry_run omitted={len(updates) - 20}")
            return

        if args.backup:
            backup_path = backup_sqlite_db(db_path, suffix="pre_eval_sync")
            print(f"backup={backup_path}")

        for entry_id, changed in updates:
            sets = ", ".join(f"{col} = ?" for col in changed)
            conn.execute(
                f"UPDATE leaderboard SET {sets} WHERE entry_id = ?",
                [*changed.values(), entry_id],
            )
        conn.commit()
        print(f"updated={len(updates)}")
    finally:
        conn.close()

    if not args.skip_health_check and not args.dry_run:
        assert_sqlite_health(db_path, label="post-sync")


if __name__ == "__main__":
    main()
