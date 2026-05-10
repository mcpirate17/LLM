from __future__ import annotations

"""Move bulky notebook payloads from SQLite rows into zstd artifacts."""

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from research.scientist.notebook.artifact_store import (
    NotebookArtifactStore,
    artifact_pointer_json,
    parse_artifact_pointer,
)
from research.defaults import RUNS_DB
from research.tools._db_maintenance import check_writer_lock, table_columns
from research.tools.db_health import assert_sqlite_health
from research.tools.graph_retention_report import (
    ACTIVE_LEADERBOARD_TIERS,
    PROMOTABLE_COMPARABILITY,
    PROMOTABLE_TRUST,
)


DEFAULT_DB = Path(RUNS_DB)
DEFAULT_PROGRAM_COLUMNS = (
    "rapid_screening_metrics_json",
    "external_benchmarks_json",
    "failure_details_json",
    "blimp_subtask_accuracies_json",
    "diagnostic_tasks_json",
    "language_control_s10_checkpoints_json",
    "language_control_investigation_checkpoints_json",
    "ar_validation_learning_curve_json",
)
GRAPH_JSON_COLUMNS = ("graph_json",)
# Keep experiments.config_json/results_json inline for now: multiple runner,
# dashboard, and analysis paths still parse those columns directly.
EXPERIMENT_COLUMNS: tuple[str, ...] = ()
REPORT_COLUMNS = (
    ("report_snapshots", "snapshot_key", "payload_json"),
    ("attribution_reports", "report_id", "report_json"),
    ("construction_prior_snapshots", "version", "payload_json"),
)
OPERATIONAL_COLUMNS = (
    ("healer_tasks", "task_id", "trigger_payload_json"),
    ("healer_tasks", "task_id", "result_json"),
    ("entries", "entry_id", "metadata_json"),
)


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=15000")
    return conn


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _ensure_artifacts_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS notebook_artifacts (
            artifact_id TEXT PRIMARY KEY,
            table_name TEXT NOT NULL,
            row_pk TEXT NOT NULL,
            column_name TEXT NOT NULL,
            path TEXT NOT NULL,
            compression TEXT NOT NULL,
            content_type TEXT NOT NULL,
            sha256_uncompressed TEXT NOT NULL,
            sha256_compressed TEXT NOT NULL,
            uncompressed_bytes INTEGER NOT NULL,
            compressed_bytes INTEGER NOT NULL,
            created_at REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_notebook_artifacts_lookup
        ON notebook_artifacts(table_name, row_pk, column_name)
        """
    )


def _insert_metadata(conn: sqlite3.Connection, metadata: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO notebook_artifacts
        (artifact_id, table_name, row_pk, column_name, path, compression,
         content_type, sha256_uncompressed, sha256_compressed,
         uncompressed_bytes, compressed_bytes, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            metadata["artifact_id"],
            metadata["table_name"],
            metadata["row_pk"],
            metadata["column_name"],
            metadata["path"],
            metadata["compression"],
            metadata["content_type"],
            metadata["sha256_uncompressed"],
            metadata["sha256_compressed"],
            metadata["uncompressed_bytes"],
            metadata["compressed_bytes"],
            metadata["created_at"],
        ),
    )


def _payload_size(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bytes):
        return len(value)
    return len(str(value).encode("utf-8"))


def _candidate_stats(
    conn: sqlite3.Connection,
    *,
    table: str,
    pk: str,
    columns: Iterable[str],
    min_bytes: int,
    extra_where: str = "",
    extra_params: tuple[Any, ...] = (),
) -> list[dict[str, Any]]:
    existing = set(table_columns(conn, table))
    stats = []
    for column in columns:
        if column not in existing:
            continue
        row = conn.execute(
            f"""
            SELECT COUNT(*) AS rows, SUM(LENGTH({column})) AS bytes
            FROM {table}
            WHERE {column} IS NOT NULL
              AND LENGTH({column}) >= ?
              AND CAST({column} AS TEXT) NOT LIKE '%"_notebook_artifact"%'
              {extra_where}
            """,
            (min_bytes, *extra_params),
        ).fetchone()
        stats.append(
            {
                "table": table,
                "pk": pk,
                "column": column,
                "rows": int(row["rows"] or 0),
                "bytes": int(row["bytes"] or 0),
                "_extra_where": extra_where,
                "_extra_params": extra_params,
            }
        )
    return stats


def _externalize_column(
    conn: sqlite3.Connection,
    store: NotebookArtifactStore,
    *,
    table: str,
    pk: str,
    column: str,
    min_bytes: int,
    limit: int | None,
    extra_where: str = "",
    extra_params: tuple[Any, ...] = (),
) -> dict[str, int]:
    processed = 0
    raw_bytes = 0
    compressed_bytes = 0
    batch_limit = 500
    while limit is None or processed < limit:
        take = batch_limit if limit is None else min(batch_limit, limit - processed)
        if take <= 0:
            break
        rows = conn.execute(
            f"""
            SELECT {pk} AS row_pk, {column} AS payload
            FROM {table}
            WHERE {column} IS NOT NULL
              AND LENGTH({column}) >= ?
              AND CAST({column} AS TEXT) NOT LIKE '%"_notebook_artifact"%'
              {extra_where}
            LIMIT ?
            """,
            (min_bytes, *extra_params, take),
        ).fetchall()
        if not rows:
            break
        changed = 0
        for row in rows:
            value = row["payload"]
            if parse_artifact_pointer(value):
                continue
            row_pk = str(row["row_pk"])
            content_type = (
                "application/octet-stream"
                if isinstance(value, bytes)
                else "application/json"
            )
            metadata = store.write(
                table_name=table,
                row_pk=row_pk,
                column_name=column,
                payload=value,
                content_type=content_type,
            )
            _insert_metadata(conn, metadata)
            pointer = artifact_pointer_json(
                metadata["artifact_id"],
                path=metadata["path"],
            )
            stored_pointer: str | bytes = pointer
            if isinstance(value, bytes):
                stored_pointer = pointer.encode("utf-8")
            conn.execute(
                f"UPDATE {table} SET {column} = ? WHERE {pk} = ?",
                (stored_pointer, row_pk),
            )
            processed += 1
            changed += 1
            raw_bytes += int(metadata["uncompressed_bytes"])
            compressed_bytes += int(metadata["compressed_bytes"])
        conn.commit()
        if changed == 0:
            break
    return {
        "rows": processed,
        "raw_bytes": raw_bytes,
        "compressed_bytes": compressed_bytes,
    }


def _cold_graph_where(table: str) -> tuple[str, tuple[Any, ...]]:
    active_tiers = tuple(sorted(ACTIVE_LEADERBOARD_TIERS))
    promotable_trust = tuple(sorted(PROMOTABLE_TRUST))
    promotable_comparability = tuple(sorted(PROMOTABLE_COMPARABILITY))
    if table != "graphs":
        return (
            f"""
                  AND NOT EXISTS (
                    SELECT 1 FROM leaderboard l
                    WHERE l.result_id = graph_runs.result_id
                      AND COALESCE(l.tier, '') IN ({",".join("?" for _ in active_tiers)})
                  )
                  AND NOT (
                    COALESCE(graph_runs.trust_label, '') IN ({",".join("?" for _ in promotable_trust)})
                    AND COALESCE(graph_runs.comparability_label, '') IN ({",".join("?" for _ in promotable_comparability)})
                  )
                  AND NOT (
                    graph_runs.induction_intermediate_auc IS NOT NULL
                    OR graph_runs.binding_intermediate_auc IS NOT NULL
                    OR graph_runs.ar_intermediate_auc IS NOT NULL
                    OR graph_runs.induction_validation_auc IS NOT NULL
                    OR graph_runs.ar_validation_rank_score IS NOT NULL
                    OR graph_runs.binding_multislot_auc IS NOT NULL
                  )
                  AND NOT EXISTS (
                    SELECT 1 FROM causal_rule_evidence ev
                    WHERE ev.parent_result_id = graph_runs.result_id
                  )
                  AND NOT EXISTS (
                    SELECT 1 FROM causal_ablation_child_observations obs
                    WHERE obs.parent_result_id = graph_runs.result_id
                       OR obs.child_result_id = graph_runs.result_id
                  )
            """,
            active_tiers + promotable_trust + promotable_comparability,
        )

    graph_run_match = "graph_runs.graph_fingerprint = graphs.graph_fingerprint"
    return (
        f"""
              AND NOT EXISTS (
                SELECT 1 FROM graph_runs
                JOIN leaderboard l ON l.result_id = graph_runs.result_id
                WHERE {graph_run_match}
                  AND COALESCE(l.tier, '') IN ({",".join("?" for _ in active_tiers)})
              )
              AND NOT EXISTS (
                SELECT 1 FROM graph_runs
                WHERE {graph_run_match}
                  AND COALESCE(graph_runs.trust_label, '') IN ({",".join("?" for _ in promotable_trust)})
                  AND COALESCE(graph_runs.comparability_label, '') IN ({",".join("?" for _ in promotable_comparability)})
              )
              AND NOT EXISTS (
                SELECT 1 FROM graph_runs
                WHERE {graph_run_match}
                  AND (
                    graph_runs.induction_intermediate_auc IS NOT NULL
                    OR graph_runs.binding_intermediate_auc IS NOT NULL
                    OR graph_runs.ar_intermediate_auc IS NOT NULL
                    OR graph_runs.induction_validation_auc IS NOT NULL
                    OR graph_runs.ar_validation_rank_score IS NOT NULL
                    OR graph_runs.binding_multislot_auc IS NOT NULL
                  )
              )
              AND NOT EXISTS (
                SELECT 1 FROM graph_runs
                JOIN causal_rule_evidence ev ON ev.parent_result_id = graph_runs.result_id
                WHERE {graph_run_match}
              )
              AND NOT EXISTS (
                SELECT 1 FROM graph_runs
                JOIN causal_ablation_child_observations obs
                  ON obs.parent_result_id = graph_runs.result_id
                  OR obs.child_result_id = graph_runs.result_id
                WHERE {graph_run_match}
              )
        """,
        active_tiers + promotable_trust + promotable_comparability,
    )


def _public_stat(item: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if not key.startswith("_")}


def _externalize_training_curves(
    conn: sqlite3.Connection,
    store: NotebookArtifactStore,
    *,
    limit: int | None,
) -> dict[str, int]:
    if not _table_exists(conn, "training_curves"):
        return {"rows": 0, "raw_bytes": 0, "compressed_bytes": 0}
    result_rows = conn.execute(
        """
        SELECT result_id, COUNT(*) AS n
        FROM training_curves
        GROUP BY result_id
        ORDER BY n DESC
        """
    ).fetchall()
    processed = 0
    raw_bytes = 0
    compressed_bytes = 0
    for row in result_rows:
        if limit is not None and processed >= limit:
            break
        result_id = str(row["result_id"])
        existing = conn.execute(
            """SELECT 1 FROM notebook_artifacts
               WHERE table_name = 'training_curves'
                 AND row_pk = ?
                 AND column_name = 'curve_json'
               LIMIT 1""",
            (result_id,),
        ).fetchone()
        if existing:
            conn.execute(
                "DELETE FROM training_curves WHERE result_id = ?", (result_id,)
            )
            conn.commit()
            continue
        curve = [
            dict(curve_row)
            for curve_row in conn.execute(
                """SELECT step, loss, grad_norm, step_time_ms
                   FROM training_curves
                   WHERE result_id = ?
                   ORDER BY step""",
                (result_id,),
            )
        ]
        metadata = store.write(
            table_name="training_curves",
            row_pk=result_id,
            column_name="curve_json",
            payload=curve,
            content_type="application/json",
        )
        _insert_metadata(conn, metadata)
        conn.execute("DELETE FROM training_curves WHERE result_id = ?", (result_id,))
        conn.commit()
        processed += 1
        raw_bytes += int(metadata["uncompressed_bytes"])
        compressed_bytes += int(metadata["compressed_bytes"])
    return {
        "rows": processed,
        "raw_bytes": raw_bytes,
        "compressed_bytes": compressed_bytes,
    }


def run(
    *,
    db_path: Path,
    min_bytes: int,
    apply: bool,
    limit: int | None,
    vacuum: bool,
    include_graph_json: bool = False,
    graph_json_cold_only: bool = True,
) -> dict[str, Any]:
    if apply:
        check_writer_lock(Path(f"{db_path.resolve()}.writer-lock"))
    assert_sqlite_health(db_path, label="pre-artifact-migration")
    conn = _connect(db_path)
    try:
        _ensure_artifacts_table(conn)
        conn.commit()
        stats = []
        stats.extend(
            _candidate_stats(
                conn,
                table="graph_runs",
                pk="result_id",
                columns=DEFAULT_PROGRAM_COLUMNS,
                min_bytes=min_bytes,
            )
        )
        if include_graph_json:
            graph_table = "graphs" if _table_exists(conn, "graphs") else "graph_runs"
            graph_pk = "graph_fingerprint" if graph_table == "graphs" else "result_id"
            graph_extra_where = ""
            graph_extra_params: tuple[Any, ...] = ()
            if graph_json_cold_only:
                graph_extra_where, graph_extra_params = _cold_graph_where(graph_table)
            stats.extend(
                _candidate_stats(
                    conn,
                    table=graph_table,
                    pk=graph_pk,
                    columns=GRAPH_JSON_COLUMNS,
                    min_bytes=min_bytes,
                    extra_where=graph_extra_where,
                    extra_params=graph_extra_params,
                )
            )
        stats.extend(
            _candidate_stats(
                conn,
                table="experiments",
                pk="experiment_id",
                columns=EXPERIMENT_COLUMNS,
                min_bytes=min_bytes,
            )
        )
        for table, pk, column in (*REPORT_COLUMNS, *OPERATIONAL_COLUMNS):
            if _table_exists(conn, table):
                stats.extend(
                    _candidate_stats(
                        conn,
                        table=table,
                        pk=pk,
                        columns=(column,),
                        min_bytes=min_bytes,
                    )
                )
        curve_groups = (
            conn.execute(
                "SELECT COUNT(DISTINCT result_id) FROM training_curves"
            ).fetchone()[0]
            if _table_exists(conn, "training_curves")
            else 0
        )
        report: dict[str, Any] = {
            "dry_run": not apply,
            "candidate_stats": [_public_stat(item) for item in stats],
            "training_curve_groups": int(curve_groups or 0),
            "applied": [],
        }
        if not apply:
            return report

        store = NotebookArtifactStore(db_path)
        for item in stats:
            if int(item["rows"]) <= 0:
                continue
            applied = _externalize_column(
                conn,
                store,
                table=item["table"],
                pk=item["pk"],
                column=item["column"],
                min_bytes=min_bytes,
                limit=limit,
                extra_where=str(item.get("_extra_where") or ""),
                extra_params=tuple(item.get("_extra_params") or ()),
            )
            report["applied"].append({**_public_stat(item), **applied})
        curve_applied = _externalize_training_curves(conn, store, limit=limit)
        report["training_curves_applied"] = curve_applied
    finally:
        conn.close()
    assert_sqlite_health(db_path, label="post-artifact-migration")
    if apply and vacuum:
        with sqlite3.connect(str(db_path), timeout=30.0) as vacuum_conn:
            vacuum_conn.execute("VACUUM")
        assert_sqlite_health(db_path, label="post-vacuum")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--min-bytes", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--vacuum", action="store_true")
    parser.add_argument("--include-graph-json", action="store_true")
    parser.add_argument(
        "--all-graph-json",
        action="store_true",
        help="When used with --include-graph-json, include hot graph_json rows too.",
    )
    args = parser.parse_args(argv)
    report = run(
        db_path=args.db,
        min_bytes=args.min_bytes,
        apply=args.apply,
        limit=args.limit,
        vacuum=args.vacuum,
        include_graph_json=args.include_graph_json,
        graph_json_cold_only=not args.all_graph_json,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
