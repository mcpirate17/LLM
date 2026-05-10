from __future__ import annotations

"""Classify graph rows by retention tier before graph payload archival."""

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from research.defaults import RUNS_DB
from research.tools._db_maintenance import connect_readonly


ACTIVE_LEADERBOARD_TIERS = {
    "screening",
    "validation",
    "investigation",
    "investigation_failed",
    "breakthrough",
}
PROMOTABLE_TRUST = {"candidate_grade", "reference"}
PROMOTABLE_COMPARABILITY = {"candidate_comparable", "reference_comparable"}
TRUSTED_TRAINING_TRUST = {"candidate_screening", "candidate_grade", "reference"}
TRUSTED_TRAINING_COMPARABILITY = {
    "screening_only",
    "candidate_comparable",
    "reference_comparable",
}


def _scalar(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> int:
    row = conn.execute(sql, params).fetchone()
    return int((row[0] if row else 0) or 0)


def _graph_where(alias: str = "pr") -> str:
    return (
        f"{alias}.graph_json IS NOT NULL "
        f"AND TRIM(CAST({alias}.graph_json AS TEXT)) NOT IN ('', '{{}}')"
    )


def build_report(db_path: Path = Path(RUNS_DB)) -> dict[str, Any]:
    with connect_readonly(db_path) as conn:
        active_tiers = tuple(sorted(ACTIVE_LEADERBOARD_TIERS))
        placeholders = ",".join("?" for _ in active_tiers)
        hot_active_sql = f"""
            SELECT COUNT(DISTINCT pr.result_id)
            FROM program_results_compat pr
            JOIN leaderboard l ON l.result_id = pr.result_id
            WHERE {_graph_where("pr")}
              AND COALESCE(l.tier, '') IN ({placeholders})
        """
        hot_active = _scalar(conn, hot_active_sql, active_tiers)

        active_followups = _scalar(
            conn,
            f"""
            SELECT COUNT(DISTINCT pr.result_id)
            FROM program_results_compat pr
            JOIN followup_tasks ft
              ON EXISTS (
                  SELECT 1 FROM json_each(ft.result_ids_json) refs
                  WHERE refs.value = pr.result_id
              )
            WHERE {_graph_where("pr")}
              AND COALESCE(ft.status, '') IN ('queued', 'running')
            """,
        )
        ml_training = _scalar(
            conn,
            f"""
            SELECT COUNT(DISTINCT pr.result_id)
            FROM program_results_compat pr
            WHERE {_graph_where("pr")}
              AND (
                json_extract(COALESCE(pr.data_provenance_json, '{{}}'),
                             '$.eligible_for_screening_model_training') = 1
                OR (
                  COALESCE(pr.trust_label, '') IN ({",".join("?" for _ in TRUSTED_TRAINING_TRUST)})
                  AND COALESCE(pr.comparability_label, '') IN ({",".join("?" for _ in TRUSTED_TRAINING_COMPARABILITY)})
                )
              )
            """,
            tuple(sorted(TRUSTED_TRAINING_TRUST))
            + tuple(sorted(TRUSTED_TRAINING_COMPARABILITY)),
        )
        promotable = _scalar(
            conn,
            f"""
            SELECT COUNT(DISTINCT pr.result_id)
            FROM program_results_compat pr
            WHERE {_graph_where("pr")}
              AND COALESCE(pr.trust_label, '') IN ({",".join("?" for _ in PROMOTABLE_TRUST)})
              AND COALESCE(pr.comparability_label, '') IN ({",".join("?" for _ in PROMOTABLE_COMPARABILITY)})
            """,
            tuple(sorted(PROMOTABLE_TRUST)) + tuple(sorted(PROMOTABLE_COMPARABILITY)),
        )
        meta_analysis = _scalar(
            conn,
            f"SELECT COUNT(*) FROM program_results_compat pr WHERE {_graph_where('pr')}",
        )
        intermediate_validation = _scalar(
            conn,
            f"""
            SELECT COUNT(DISTINCT pr.result_id)
            FROM program_results_compat pr
            WHERE {_graph_where("pr")}
              AND (
                pr.induction_intermediate_auc IS NOT NULL
                OR pr.binding_intermediate_auc IS NOT NULL
                OR pr.ar_intermediate_auc IS NOT NULL
                OR pr.induction_validation_auc IS NOT NULL
                OR pr.ar_validation_rank_score IS NOT NULL
                OR pr.binding_multislot_auc IS NOT NULL
              )
            """,
        )
        causal_replay = _scalar(
            conn,
            f"""
            SELECT COUNT(DISTINCT pr.result_id)
            FROM program_results_compat pr
            WHERE {_graph_where("pr")}
              AND (
                EXISTS (
                  SELECT 1 FROM causal_rule_evidence ev
                  WHERE ev.parent_result_id = pr.result_id
                )
                OR EXISTS (
                  SELECT 1 FROM causal_ablation_child_observations obs
                  WHERE obs.parent_result_id = pr.result_id
                     OR obs.child_result_id = pr.result_id
                )
              )
            """,
        )
        total_graph_rows = _scalar(
            conn,
            f"SELECT COUNT(*) FROM program_results_compat pr WHERE {_graph_where('pr')}",
        )
        inline_graph_bytes = _scalar(
            conn,
            f"""
            SELECT COALESCE(SUM(LENGTH(pr.graph_json)), 0)
            FROM program_results_compat pr
            WHERE {_graph_where("pr")}
              AND pr.graph_json NOT LIKE '%"_notebook_artifact"%'
            """,
        )
        graph_artifact_rows = _scalar(
            conn,
            """
            SELECT COUNT(*)
            FROM notebook_artifacts
            WHERE table_name = 'program_results'
              AND column_name = 'graph_json'
            """,
        )
        cold_archive_candidates = _scalar(
            conn,
            f"""
            SELECT COUNT(DISTINCT pr.result_id)
            FROM program_results_compat pr
            WHERE {_graph_where("pr")}
              AND NOT EXISTS (
                SELECT 1 FROM leaderboard l
                WHERE l.result_id = pr.result_id
                  AND COALESCE(l.tier, '') IN ({placeholders})
              )
              AND NOT (
                COALESCE(pr.trust_label, '') IN ({",".join("?" for _ in PROMOTABLE_TRUST)})
                AND COALESCE(pr.comparability_label, '') IN ({",".join("?" for _ in PROMOTABLE_COMPARABILITY)})
              )
              AND NOT (
                pr.induction_intermediate_auc IS NOT NULL
                OR pr.binding_intermediate_auc IS NOT NULL
                OR pr.ar_intermediate_auc IS NOT NULL
                OR pr.induction_validation_auc IS NOT NULL
                OR pr.ar_validation_rank_score IS NOT NULL
                OR pr.binding_multislot_auc IS NOT NULL
              )
              AND NOT EXISTS (
                SELECT 1 FROM causal_rule_evidence ev
                WHERE ev.parent_result_id = pr.result_id
              )
              AND NOT EXISTS (
                SELECT 1 FROM causal_ablation_child_observations obs
                WHERE obs.parent_result_id = pr.result_id
                   OR obs.child_result_id = pr.result_id
              )
            """,
            active_tiers
            + tuple(sorted(PROMOTABLE_TRUST))
            + tuple(sorted(PROMOTABLE_COMPARABILITY)),
        )
        dbstat = [
            {"name": str(row["name"]), "bytes": int(row["bytes"] or 0)}
            for row in conn.execute(
                """
                SELECT name, SUM(pgsize) AS bytes
                FROM dbstat
                GROUP BY name
                ORDER BY bytes DESC
                LIMIT 15
                """
            )
        ]
    return {
        "db_path": str(db_path),
        "total_graph_rows": total_graph_rows,
        "retention_classes": {
            "hot_active_leaderboard": hot_active,
            "active_followup_refs": active_followups,
            "ml_training_rows": ml_training,
            "promotable_rows": promotable,
            "meta_analysis_rows": meta_analysis,
            "intermediate_or_validation_rows": intermediate_validation,
            "causal_replay_rows": causal_replay,
            "cold_archive_candidates": cold_archive_candidates,
        },
        "graph_payloads": {
            "inline_graph_bytes": inline_graph_bytes,
            "graph_artifact_rows": graph_artifact_rows,
        },
        "largest_sqlite_objects": dbstat,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path(RUNS_DB))
    args = parser.parse_args()
    print(json.dumps(build_report(args.db), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
