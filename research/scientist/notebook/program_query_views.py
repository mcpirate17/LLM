from __future__ import annotations

"""Focused, performance-sensitive program selection queries."""

from typing import Dict, Iterable, List

from ..trust_policy import sql_trusted_clause

VALID_PROGRAM_SORTS = frozenset(
    {
        "novelty_score",
        "loss_ratio",
        "structural_novelty",
        "behavioral_novelty",
        "validation_loss_ratio",
        "discovery_loss_ratio",
    }
)


def normalize_program_sort(sort_by: str, *, default: str) -> str:
    return sort_by if sort_by in VALID_PROGRAM_SORTS else default


def program_sort_order(sort_by: str) -> str:
    if sort_by in {"novelty_score", "structural_novelty", "behavioral_novelty"}:
        return "DESC"
    return "ASC"


def trusted_program_filter(trusted_only: bool, *, table_alias: str = "") -> str:
    if not trusted_only:
        return ""
    alias = table_alias.rstrip(".")
    return f" AND {sql_trusted_clause(table_alias=alias or None)}"


def ensure_architecture_family(nb, rows: Iterable[Dict]) -> List[Dict]:
    rows_dicts = [dict(r) for r in rows]
    for row in rows_dicts:
        if not row.get("architecture_family"):
            row["architecture_family"] = nb._classify_architecture_family(
                graph_json=row.get("graph_json"),
                routing_mode=row.get("routing_mode"),
            )
    return rows_dicts


def fetch_top_programs(
    nb,
    *,
    n: int,
    sort_by: str,
    trusted_only: bool,
) -> List[Dict]:
    sort_key = normalize_program_sort(sort_by, default="novelty_score")
    order = program_sort_order(sort_key)
    rows = nb.conn.execute(
        f"""SELECT * FROM program_results
            WHERE stage1_passed = 1{trusted_program_filter(trusted_only)}
            ORDER BY {sort_key} {order} NULLS LAST
            LIMIT ?""",
        (n,),
    ).fetchall()
    return ensure_architecture_family(nb, rows)


def fetch_report_top_programs_grouped_by_fingerprint(
    nb,
    *,
    n: int,
    sort_by: str,
    trusted_only: bool,
) -> List[Dict]:
    sort_key = normalize_program_sort(sort_by, default="loss_ratio")
    order = program_sort_order(sort_key)
    trust_filter = trusted_program_filter(trusted_only)
    candidate_rows = nb.conn.execute(
        f"""
        SELECT * FROM program_results
        WHERE stage1_passed = 1
          AND graph_fingerprint IS NOT NULL
          AND TRIM(graph_fingerprint) != ''
          {trust_filter}
        ORDER BY {sort_key} {order} NULLS LAST, timestamp DESC
        LIMIT ?
        """,
        (max(n * 12, 200),),
    ).fetchall()
    grouped: List[Dict] = []
    selected_fingerprints: List[str] = []
    seen = set()
    for row in candidate_rows:
        record = dict(row)
        fingerprint = record.get("graph_fingerprint")
        if not fingerprint or fingerprint in seen:
            continue
        seen.add(fingerprint)
        selected_fingerprints.append(fingerprint)
        grouped.append(record)
        if len(grouped) >= n:
            break
    if not grouped:
        return []

    spread_by_fp = {}
    chunk_size = 900
    for start in range(0, len(selected_fingerprints), chunk_size):
        chunk = selected_fingerprints[start : start + chunk_size]
        placeholders = ", ".join("?" for _ in chunk)
        spread_rows = nb.conn.execute(
            f"""
            SELECT
                graph_fingerprint,
                COUNT(*) AS repeat_count,
                COUNT(DISTINCT experiment_id) AS repeat_experiment_span,
                MIN(timestamp) AS repeat_first_seen_ts,
                MAX(timestamp) AS repeat_last_seen_ts,
                MIN(loss_ratio) AS repeat_loss_min,
                MAX(loss_ratio) AS repeat_loss_max,
                AVG(loss_ratio) AS repeat_loss_mean,
                MIN(novelty_score) AS repeat_novelty_min,
                MAX(novelty_score) AS repeat_novelty_max
            FROM program_results
            WHERE stage1_passed = 1
              AND graph_fingerprint IN ({placeholders})
              {trust_filter}
            GROUP BY graph_fingerprint
            """,
            tuple(chunk),
        ).fetchall()
        spread_by_fp.update(
            {row["graph_fingerprint"]: dict(row) for row in spread_rows}
        )

    for record in grouped:
        spread = spread_by_fp.get(record.get("graph_fingerprint"), {})
        record["repeat_count"] = int(spread.get("repeat_count") or 1)
        record["repeat_experiment_span"] = int(
            spread.get("repeat_experiment_span") or 1
        )
        record["repeat_first_seen_ts"] = spread.get("repeat_first_seen_ts")
        record["repeat_last_seen_ts"] = spread.get("repeat_last_seen_ts")
        record["repeat_loss_min"] = spread.get("repeat_loss_min")
        record["repeat_loss_max"] = spread.get("repeat_loss_max")
        record["repeat_loss_mean"] = spread.get("repeat_loss_mean")
        record["repeat_novelty_min"] = spread.get("repeat_novelty_min")
        record["repeat_novelty_max"] = spread.get("repeat_novelty_max")
    return ensure_architecture_family(nb, grouped)
