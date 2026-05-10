"""Export fingerprints that still need BPE/tiktoken eval backfill."""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from research.tools.db_health import assert_sqlite_health


BPE_VERSION = "bpe_eval_v1"
REQUIRED_EVAL_COLUMNS = (
    "wikitext_perplexity",
    "tinystories_perplexity",
    "hellaswag_acc",
    "blimp_overall_accuracy",
)


def _default_output_path() -> Path:
    ts = time.strftime("%Y%m%dT%H%M%S")
    return Path("research/reports") / f"bpe_backfill_inventory_{ts}.json"


def _candidate_where(scope: str) -> str:
    stale = (
        "COALESCE(pr.screening_wikitext_metric_version, '') <> :metric_version "
        "OR pr.wikitext_perplexity IS NULL "
        "OR pr.tinystories_perplexity IS NULL "
        "OR pr.hellaswag_acc IS NULL "
        "OR pr.blimp_overall_accuracy IS NULL"
    )
    if scope == "off_leaderboard":
        return f"({stale}) AND lb.entry_id IS NULL"
    if scope == "leaderboard":
        return f"({stale}) AND lb.entry_id IS NOT NULL"
    if scope == "all":
        return f"({stale})"
    raise ValueError(f"unsupported scope: {scope}")


def export_inventory(
    db_path: str | Path,
    *,
    output_path: str | Path,
    scope: str,
    limit: int | None,
) -> dict[str, Any]:
    assert_sqlite_health(db_path, label="pre-inventory")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(str(db_path)) as conn:
        table = _program_results_read_table(conn)

    sql = f"""
        SELECT
            pr.graph_fingerprint,
            COUNT(*) AS row_count,
            MAX(pr.timestamp) AS latest_timestamp,
            MAX(COALESCE(lb.composite_score, pr.loss_ratio, 0.0)) AS priority_score,
            MAX(CASE WHEN pr.graph_json IS NOT NULL AND length(pr.graph_json) > 10 THEN 1 ELSE 0 END) AS has_graph_json,
            SUM(CASE WHEN COALESCE(pr.screening_wikitext_metric_version, '') = :metric_version THEN 1 ELSE 0 END) AS current_bpe_rows,
            SUM(CASE WHEN COALESCE(pr.screening_wikitext_metric_version, '') <> :metric_version THEN 1 ELSE 0 END) AS legacy_or_unversioned_rows,
            SUM(CASE WHEN pr.wikitext_perplexity IS NULL THEN 1 ELSE 0 END) AS missing_wikitext_rows,
            SUM(CASE WHEN pr.tinystories_perplexity IS NULL THEN 1 ELSE 0 END) AS missing_tinystories_rows,
            SUM(CASE WHEN pr.hellaswag_acc IS NULL THEN 1 ELSE 0 END) AS missing_hellaswag_rows,
            SUM(CASE WHEN pr.blimp_overall_accuracy IS NULL THEN 1 ELSE 0 END) AS missing_blimp_rows,
            GROUP_CONCAT(DISTINCT COALESCE(lb.tier, 'off_leaderboard')) AS tiers,
            GROUP_CONCAT(DISTINCT COALESCE(pr.trust_label, '')) AS trust_labels,
            GROUP_CONCAT(DISTINCT COALESCE(pr.comparability_label, '')) AS comparability_labels
        FROM {table} pr
        LEFT JOIN leaderboard lb ON lb.result_id = pr.result_id
        WHERE pr.graph_fingerprint IS NOT NULL
          AND {_candidate_where(scope)}
        GROUP BY pr.graph_fingerprint
        HAVING has_graph_json = 1
        ORDER BY priority_score DESC, latest_timestamp DESC
    """
    params: dict[str, Any] = {"metric_version": BPE_VERSION}
    if limit is not None and limit > 0:
        sql += " LIMIT :limit"
        params["limit"] = int(limit)

    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(row) for row in conn.execute(sql, params)]

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "db_path": str(db_path),
        "metric_version": BPE_VERSION,
        "scope": scope,
        "required_eval_columns": list(REQUIRED_EVAL_COLUMNS),
        "fingerprint_count": len(rows),
        "rows": rows,
    }
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"output_path": str(output), "fingerprint_count": len(rows)}


def _program_results_read_table(conn: sqlite3.Connection) -> str:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = 'program_results_compat' LIMIT 1"
    ).fetchone()
    return "program_results_compat" if row else "program_results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="research/runs.db")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--scope",
        choices=("off_leaderboard", "leaderboard", "all"),
        default="off_leaderboard",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    result = export_inventory(
        args.db,
        output_path=args.output or _default_output_path(),
        scope=args.scope,
        limit=args.limit,
    )
    print(f"output={result['output_path']}")
    print(f"fingerprints={result['fingerprint_count']}")


if __name__ == "__main__":
    main()
