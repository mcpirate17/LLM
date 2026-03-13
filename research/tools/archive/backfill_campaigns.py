#!/usr/bin/env python3
"""Backfill additional campaigns from historical experiments."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def backfill_campaigns(db_path: Path, target_campaigns: int = 3, chunk_size: int = 5) -> int:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    existing = cur.execute("SELECT COUNT(*) AS c FROM campaigns").fetchone()["c"]
    if existing >= target_campaigns:
        conn.close()
        return 0

    rows = cur.execute(
        "SELECT experiment_id FROM experiments WHERE campaign_id IS NULL OR TRIM(campaign_id) = '' "
        "ORDER BY timestamp ASC"
    ).fetchall()
    exp_ids = [r["experiment_id"] for r in rows if r["experiment_id"]]

    created = 0
    while existing + created < target_campaigns and exp_ids:
        batch = exp_ids[:chunk_size]
        exp_ids = exp_ids[chunk_size:]
        camp_id = f"backfill-{existing + created + 1:03d}"
        title = f"Backfilled Campaign {existing + created + 1}"
        objective = "Backfilled from historical experiments for campaign-level analytics."
        criteria = "Maintain stage1 survivor flow while increasing novelty diversity."
        now = __import__("time").time()
        cur.execute(
            "INSERT OR IGNORE INTO campaigns (campaign_id, timestamp, title, objective, success_criteria, status, parent_campaign_id, started_at) "
            "VALUES (?, ?, ?, ?, ?, 'completed', NULL, ?)",
            (camp_id, now, title, objective, criteria, now),
        )
        for exp_id in batch:
            cur.execute("UPDATE experiments SET campaign_id = ? WHERE experiment_id = ?", (camp_id, exp_id))
        cur.execute(
            "UPDATE campaigns SET completed_at = ?, findings_summary = ?, completion_reason = ? WHERE campaign_id = ?",
            (now, f"Backfilled {len(batch)} historical experiments.", "backfill", camp_id),
        )
        created += 1

    conn.commit()
    conn.close()
    return created


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", default="research/lab_notebook.db")
    parser.add_argument("--target-campaigns", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=5)
    args = parser.parse_args()

    created = backfill_campaigns(
        db_path=Path(args.db_path),
        target_campaigns=max(1, int(args.target_campaigns)),
        chunk_size=max(1, int(args.chunk_size)),
    )
    print(f"Created {created} backfilled campaign(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
