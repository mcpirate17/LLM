from __future__ import annotations

import sqlite3
from typing import Any, Sequence

from research.tools._fingerprint_selection import dedupe_records_by_fingerprint


def fetch_latest_unique_fingerprint_rows(
    conn: sqlite3.Connection,
    *,
    select_sql: str,
    extra_where_sql: str = "",
    params: Sequence[Any] = (),
    limit: int = 0,
    include_leaderboard: bool = True,
) -> list[sqlite3.Row]:
    """Fetch newest program rows and keep the first row per fingerprint.

    The shared contract is:
    - filter to valid, compiled graph rows that passed stage0/stage0.5
    - exclude reference rows
    - order by newest program timestamp first
    - collapse repeated ``graph_fingerprint`` values after ordering
    - apply the final limit after deduplication
    """

    leaderboard_join = (
        "LEFT JOIN leaderboard l ON l.result_id = pr.result_id"
        if include_leaderboard
        else ""
    )

    query = f"""
        SELECT
            {select_sql}
        FROM program_results pr
        JOIN experiments e ON e.experiment_id = pr.experiment_id
        {leaderboard_join}
        WHERE TRIM(COALESCE(pr.graph_json, '')) <> ''
          AND pr.graph_json <> '{{}}'
          AND pr.stage0_passed = 1
          AND pr.stage05_passed = 1
          AND COALESCE(pr.trust_label, '') <> 'reference'
          AND TRIM(COALESCE(pr.graph_fingerprint, '')) <> ''
          {extra_where_sql}
        ORDER BY pr.timestamp DESC, pr.result_id DESC
    """
    rows = list(conn.execute(query, tuple(params)).fetchall())
    rows = dedupe_records_by_fingerprint(rows)
    if limit > 0:
        rows = rows[: int(limit)]
    return rows
