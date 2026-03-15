"""Reset failure statistics for protected ops so they get a fresh start.

Clears toxic bigram counts and zero-success failure signatures where either
member is a PROTECTED_OP. This undoes the historical penalty that accumulated
before root-cause bug fixes were applied.

Usage:
    python -m research.tools.reset_protected_op_stats [--dry-run]
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path

from ..synthesis.primitives import PROTECTED_OPS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "lab_notebook.db"


def reset_protected_op_stats(dry_run: bool = False) -> dict:
    """Reset failure counts for protected ops in the database."""
    if not DB_PATH.exists():
        log.error("Database not found: %s", DB_PATH)
        return {"error": "db_not_found"}

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    results = {
        "bigrams_reset": 0,
        "failure_sigs_reset": 0,
        "ops_affected": set(),
    }

    try:
        # 1. Reset toxic bigram failure counts
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='failure_signatures'"
        )
        if cur.fetchone():
            # Find rows where the signature mentions a protected op
            rows = conn.execute(
                "SELECT rowid, signature, count FROM failure_signatures"
            ).fetchall()
            reset_rowids = []
            for row in rows:
                sig = row["signature"] or ""
                # Check if any protected op appears in the signature
                for op in PROTECTED_OPS:
                    if op in sig:
                        reset_rowids.append(row["rowid"])
                        results["ops_affected"].add(op)
                        break

            if reset_rowids and not dry_run:
                placeholders = ",".join("?" * len(reset_rowids))
                conn.execute(
                    f"UPDATE failure_signatures SET count = 0 WHERE rowid IN ({placeholders})",
                    reset_rowids,
                )
                results["failure_sigs_reset"] = len(reset_rowids)
            elif reset_rowids:
                results["failure_sigs_reset"] = len(reset_rowids)
            log.info("Failure signatures to reset: %d", len(reset_rowids))

        # 2. Reset toxic bigram counts in op_bigram_stats if it exists
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='op_bigram_stats'"
        )
        if cur.fetchone():
            rows = conn.execute(
                "SELECT rowid, op_a, op_b FROM op_bigram_stats"
            ).fetchall()
            reset_rowids = []
            for row in rows:
                op_a = row["op_a"] or ""
                op_b = row["op_b"] or ""
                if op_a in PROTECTED_OPS or op_b in PROTECTED_OPS:
                    reset_rowids.append(row["rowid"])
                    if op_a in PROTECTED_OPS:
                        results["ops_affected"].add(op_a)
                    if op_b in PROTECTED_OPS:
                        results["ops_affected"].add(op_b)

            if reset_rowids and not dry_run:
                placeholders = ",".join("?" * len(reset_rowids))
                conn.execute(
                    f"UPDATE op_bigram_stats SET failure_count = 0 WHERE rowid IN ({placeholders})",
                    reset_rowids,
                )
                results["bigrams_reset"] = len(reset_rowids)
            elif reset_rowids:
                results["bigrams_reset"] = len(reset_rowids)
            log.info("Bigrams to reset: %d", len(reset_rowids))

        if not dry_run:
            conn.commit()
            log.info("Committed resets to database")
        else:
            log.info("DRY RUN — no changes written")

    finally:
        conn.close()

    results["ops_affected"] = sorted(results["ops_affected"])
    log.info(
        "Reset summary: %d failure_sigs, %d bigrams, %d ops affected",
        results["failure_sigs_reset"],
        results["bigrams_reset"],
        len(results["ops_affected"]),
    )
    return results


def main():
    parser = argparse.ArgumentParser(description="Reset failure stats for protected ops")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be reset without writing")
    args = parser.parse_args()
    reset_protected_op_stats(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
