"""Canonical leaderboard rescore helpers shared by API and maintenance scripts."""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, Optional, Tuple

from .leaderboard_scoring import (
    build_score_kwargs_from_prefetch,
    compute_composite,
    get_scoring_version,
    prefetch_program_results,
)
from .notebook import LabNotebook


def rescore_entry(
    nb: LabNotebook,
    entry_id: str,
    result_id: str,
    is_ref: bool,
    pr_cache: Dict[str, Dict],
    pr_updates: Optional[Dict[str, Any]] = None,
    *,
    reason: str = "canonical_rescore",
) -> Tuple[float, float]:
    """Recompute and persist one leaderboard composite score."""
    existing = nb.conn.execute(
        "SELECT * FROM leaderboard WHERE entry_id = ?",
        (entry_id,),
    ).fetchone()
    if not existing:
        return 0.0, 0.0

    current_version = get_scoring_version()
    row = dict(existing)
    old_score = float(row.get("composite_score") or 0.0)
    old_version = str(row.get("scoring_version") or "").strip()
    pr_dict = dict(pr_cache.get(result_id, {}))
    if pr_updates:
        pr_dict.update(pr_updates)
    score_kw = build_score_kwargs_from_prefetch(pr_dict, row, is_ref)
    new_score = float(compute_composite(**score_kw) or 0.0)

    if new_score != old_score or old_version != current_version:
        columns = nb._get_leaderboard_columns()
        sets = ["composite_score = ?"]
        params: list[Any] = [new_score]
        if "scoring_version" in columns:
            sets.append("scoring_version = ?")
            params.append(current_version)
        if "rescore_status" in columns:
            sets.append("rescore_status = 'rescored'")
        if "rescore_timestamp" in columns:
            sets.append("rescore_timestamp = ?")
            params.append(time.time())
        if "old_composite_score" in columns:
            sets.append("old_composite_score = ?")
            params.append(old_score)
        if "rescore_reason" in columns:
            sets.append("rescore_reason = ?")
            params.append(reason)
        params.append(entry_id)
        nb.conn.execute(
            f"UPDATE leaderboard SET {', '.join(sets)} WHERE entry_id = ?",
            params,
        )
    return new_score, old_score


def rescore_leaderboard(
    nb: LabNotebook,
    *,
    result_ids: Optional[Iterable[str]] = None,
    only_stale: bool = False,
    reason: str = "canonical_rescore",
) -> Tuple[int, int]:
    """Bulk rescore leaderboard rows against the active backend scoring version."""
    params: list[Any] = []
    where: list[str] = []
    normalized_ids = [str(result_id) for result_id in (result_ids or []) if result_id]
    if normalized_ids:
        placeholders = ",".join("?" for _ in normalized_ids)
        where.append(f"result_id IN ({placeholders})")
        params.extend(normalized_ids)
    if only_stale:
        where.append("(scoring_version IS NULL OR scoring_version != ?)")
        params.append(get_scoring_version())

    current_version = get_scoring_version()
    query = (
        "SELECT entry_id, result_id, is_reference, composite_score, scoring_version "
        "FROM leaderboard"
    )
    if where:
        query += " WHERE " + " AND ".join(where)
    query += " ORDER BY composite_score DESC"

    rows = nb.conn.execute(query, tuple(params)).fetchall()
    all_ids = [str(row["result_id"]) for row in rows if row["result_id"]]
    pr_cache = prefetch_program_results(nb.conn, all_ids)

    changed = 0
    for row in rows:
        new_score, old_score = rescore_entry(
            nb,
            str(row["entry_id"]),
            str(row["result_id"]),
            bool(row["is_reference"]),
            pr_cache,
            reason=reason,
        )
        old_version = str(row["scoring_version"] or "").strip()
        if new_score != old_score or old_version != current_version:
            changed += 1
    nb.conn.commit()
    return len(rows), changed
