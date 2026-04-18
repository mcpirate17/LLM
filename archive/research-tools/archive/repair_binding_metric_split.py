#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

from research.scientist.notebook import LabNotebook

DB_PATH = Path("research/lab_notebook.db")
CURRICULUM_PROTOCOL = "copy_curriculum_v1"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Repair historical rows after curriculum binding was written into the "
            "legacy binding_auc column."
        )
    )
    p.add_argument("--db", default=str(DB_PATH))
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _count_candidates(conn: sqlite3.Connection) -> dict[str, int]:
    row = conn.execute(
        """
        SELECT
            SUM(CASE WHEN binding_auc IS NOT NULL AND binding_auc_curriculum IS NULL THEN 1 ELSE 0 END) AS legacy_only_rows,
            SUM(CASE WHEN stage1_passed = 1 AND binding_auc IS NOT NULL AND binding_auc_curriculum IS NULL THEN 1 ELSE 0 END) AS legacy_only_s1_rows,
            SUM(CASE WHEN binding_auc IS NOT NULL AND binding_auc_curriculum IS NULL AND binding_auc < 0.0005 THEN 1 ELSE 0 END) AS legacy_only_display_zero_rows,
            SUM(CASE WHEN binding_auc IS NOT NULL AND binding_auc_curriculum IS NULL AND binding_auc >= 0.0005 THEN 1 ELSE 0 END) AS legacy_only_display_nonzero_rows
        FROM program_results
        """
    ).fetchone()
    return {
        "legacy_only_rows": int(row[0] or 0),
        "legacy_only_s1_rows": int(row[1] or 0),
        "legacy_only_display_zero_rows": int(row[2] or 0),
        "legacy_only_display_nonzero_rows": int(row[3] or 0),
    }


def _copy_curriculum_fields(nb: LabNotebook) -> int:
    with nb.batch():
        nb.conn.execute(
            """
            UPDATE program_results
            SET
                binding_auc_curriculum = binding_auc,
                binding_distance_accuracies_curriculum_json = binding_distance_accuracies_json,
                binding_probe_curriculum_elapsed_ms = binding_probe_elapsed_ms,
                binding_probe_curriculum_protocol_version = COALESCE(
                    binding_probe_curriculum_protocol_version,
                    ?
                )
            WHERE binding_auc IS NOT NULL
              AND binding_auc_curriculum IS NULL
            """,
            (CURRICULUM_PROTOCOL,),
        )
        updated = int(nb.conn.execute("SELECT changes()").fetchone()[0] or 0)
    return updated


def _template_zero_counts(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute(
        """
        WITH template_rows AS (
            SELECT
                gf.template_name,
                AVG(pr.binding_auc_curriculum) AS avg_binding_curriculum,
                COUNT(*) AS n
            FROM program_graph_features gf
            JOIN program_results pr ON pr.result_id = gf.result_id
            WHERE gf.template_name IS NOT NULL
              AND gf.template_name <> ''
              AND pr.binding_auc_curriculum IS NOT NULL
            GROUP BY gf.template_name
        )
        SELECT
            SUM(CASE WHEN avg_binding_curriculum > 0 AND avg_binding_curriculum < 0.0005 THEN 1 ELSE 0 END) AS displayed_zero_but_nonzero_templates,
            SUM(CASE WHEN avg_binding_curriculum >= 0.0005 THEN 1 ELSE 0 END) AS displayed_nonzero_templates
        FROM template_rows
        """
    ).fetchone()
    return {
        "displayed_zero_but_nonzero_templates": int(rows[0] or 0),
        "displayed_nonzero_templates": int(rows[1] or 0),
    }


def main() -> None:
    args = _parse_args()
    db_path = str(Path(args.db))
    nb = LabNotebook(db_path)
    try:
        before = _count_candidates(nb.conn)
        if args.dry_run:
            print(json.dumps({"db": db_path, "before": before}, indent=2, sort_keys=True))
            return

        t0 = time.time()
        updated = _copy_curriculum_fields(nb)
        after = _count_candidates(nb.conn)
        template_counts = _template_zero_counts(nb.conn)
        print(
            json.dumps(
                {
                    "db": db_path,
                    "updated_rows": updated,
                    "before": before,
                    "after": after,
                    "template_counts": template_counts,
                    "elapsed_s": round(time.time() - t0, 2),
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        nb.close()


if __name__ == "__main__":
    main()
