#!/usr/bin/env python3
"""Association integrity checks for experiments, fingerprints, and lineage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.scientist.notebook import LabNotebook


def build_report(nb: LabNotebook) -> Dict[str, Any]:
    q = nb.conn

    completed_without_program_rows = q.execute(
        """
        SELECT COUNT(*) AS n
        FROM experiments e
        LEFT JOIN program_results p ON p.experiment_id = e.experiment_id
        WHERE e.status = 'completed' AND p.result_id IS NULL
        """
    ).fetchone()["n"]

    mismatch = q.execute(
        """
        WITH gaps AS (
            SELECT
                e.experiment_id,
                COALESCE(e.n_stage1_passed, 0) AS exp_s1,
                COALESCE(SUM(CASE WHEN p.stage1_passed = 1 THEN 1 ELSE 0 END), 0) AS pr_s1
            FROM experiments e
            LEFT JOIN program_results p ON p.experiment_id = e.experiment_id
            GROUP BY e.experiment_id
        )
        SELECT COUNT(*) AS n FROM gaps WHERE exp_s1 != pr_s1
        """
    ).fetchone()["n"]

    missing_refinement_source = q.execute(
        """
        SELECT COUNT(*) AS n
        FROM program_results child
        WHERE json_extract(child.graph_json, '$.metadata.refinement.source_result_id') IS NOT NULL
          AND TRIM(json_extract(child.graph_json, '$.metadata.refinement.source_result_id')) != ''
          AND NOT EXISTS (
                SELECT 1 FROM program_results src
                WHERE src.result_id = json_extract(child.graph_json, '$.metadata.refinement.source_result_id')
          )
        """
    ).fetchone()["n"]

    missing_lineage_parent = q.execute(
        """
        SELECT COUNT(*) AS n
        FROM program_results child
        WHERE json_extract(child.graph_json, '$.metadata.lineage.parent') IS NOT NULL
          AND TRIM(json_extract(child.graph_json, '$.metadata.lineage.parent')) != ''
          AND NOT EXISTS (
                SELECT 1 FROM program_results p
                WHERE p.graph_fingerprint = json_extract(child.graph_json, '$.metadata.lineage.parent')
          )
        """
    ).fetchone()["n"]

    missing_normalized_lineage = q.execute(
        """
        SELECT COUNT(*) AS n
        FROM program_results pr
        WHERE (
            (json_extract(pr.graph_json, '$.metadata.refinement.source_result_id') IS NOT NULL
             AND TRIM(json_extract(pr.graph_json, '$.metadata.refinement.source_result_id')) != '')
            OR (json_extract(pr.graph_json, '$.metadata.lineage.parent') IS NOT NULL
             AND TRIM(json_extract(pr.graph_json, '$.metadata.lineage.parent')) != '')
            OR (json_extract(pr.graph_json, '$.metadata.refinement.seed_fingerprint') IS NOT NULL
             AND TRIM(json_extract(pr.graph_json, '$.metadata.refinement.seed_fingerprint')) != '')
        )
          AND NOT EXISTS (
                SELECT 1 FROM result_lineage rl WHERE rl.result_id = pr.result_id
          )
        """
    ).fetchone()["n"]

    invalid_experiments = q.execute(
        "SELECT COUNT(*) AS n FROM experiments WHERE status = 'invalid'"
    ).fetchone()["n"]

    return {
        "completed_without_program_results": int(completed_without_program_rows or 0),
        "stage1_counter_mismatches": int(mismatch or 0),
        "missing_refinement_source_links": int(missing_refinement_source or 0),
        "missing_lineage_parent_fingerprints": int(missing_lineage_parent or 0),
        "missing_normalized_lineage_rows": int(missing_normalized_lineage or 0),
        "invalid_experiments": int(invalid_experiments or 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check experiment/fingerprint association integrity.")
    parser.add_argument("--db", default="research/lab_notebook.db", help="Path to lab_notebook.db")
    parser.add_argument(
        "--allow-missing-db",
        action="store_true",
        help="Exit success when DB file does not exist (useful for CI on fresh checkouts).",
    )
    args = parser.parse_args()
    db_path = Path(args.db)
    if not db_path.exists():
        output = {
            "ok": bool(args.allow_missing_db),
            "skipped": True,
            "reason": f"database not found: {db_path}",
        }
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0 if args.allow_missing_db else 2

    nb = LabNotebook(db_path)
    try:
        report = build_report(nb)
    finally:
        nb.close()

    failing = (
        report["completed_without_program_results"] > 0
        or report["stage1_counter_mismatches"] > 0
        or report["missing_refinement_source_links"] > 0
        or report["missing_lineage_parent_fingerprints"] > 0
        or report["missing_normalized_lineage_rows"] > 0
    )
    output = {"ok": not failing, "report": report}
    print(json.dumps(output, indent=2, sort_keys=True))
    return 1 if failing else 0


if __name__ == "__main__":
    raise SystemExit(main())
