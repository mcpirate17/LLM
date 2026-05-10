"""Report remaining readiness work for retiring the legacy lab notebook DB.

This tool is intentionally read-only.  It audits the split layout, artifact
state, local backup bundles, large inline graph payloads, and remaining direct
``lab_notebook.db`` references so operators can decide when the legacy DB is
ready to become an archival-only compatibility source.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from research.defaults import (
    LAB_NOTEBOOK_DB,
    NOTEBOOK_ARTIFACTS_DIR,
    PROJECT_ROOT,
    RUNS_DB,
    RUNTIME_EVENTS_DIR,
)
from research.tools._db_maintenance import connect_readonly


APPROVED_LEGACY_TOOL_PATHS = {
    "research/tools/backup_and_prune_db_files.py",
    "research/tools/create_runs_db.py",
    "research/tools/delete_corrupted_experiment_6b1f55ca.py",
    "research/tools/restore_lab_notebook.py",
    "research/tools/restore_split_bundle_drill.py",
}
LEGACY_REFERENCE_CATEGORIES = {
    "research/tools/backup_and_prune_db_files.py": "split_backup_compatibility",
    "research/tools/create_runs_db.py": "runs_db_rebuild_source",
    "research/tools/delete_corrupted_experiment_6b1f55ca.py": "historical_corruption_recovery",
    "research/tools/restore_lab_notebook.py": "legacy_restore",
    "research/tools/restore_split_bundle_drill.py": "split_restore_drill",
}
TEXT_SUFFIXES = {".py", ".md", ".sh", ".sql"}
SENSITIVE_POINTER_COLUMNS = (
    ("experiments", "config_json"),
    ("experiments", "results_json"),
    ("program_results", "data_provenance_json"),
)
EXPECTED_POINTER_COLUMNS = (
    ("healer_tasks", "trigger_payload_json"),
    ("healer_tasks", "result_json"),
    ("entries", "metadata_json"),
)


def _rel(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _file_state(path: Path, project_root: Path) -> dict[str, Any]:
    state: dict[str, Any] = {
        "path": _rel(path, project_root),
        "exists": path.exists(),
    }
    if path.exists() and path.is_file():
        state["bytes"] = path.stat().st_size
    return state


def _count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file())


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        is not None
    )


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    if not _table_exists(conn, table):
        return False
    return any(
        str(row[1]) == column for row in conn.execute(f"PRAGMA table_info({table})")
    )


def _pointer_count(conn: sqlite3.Connection, table: str, column: str) -> int | None:
    if not _column_exists(conn, table, column):
        return None
    row = conn.execute(
        f"""
        SELECT COUNT(*)
        FROM {table}
        WHERE {column} LIKE '%"_notebook_artifact"%'
        """
    ).fetchone()
    return int(row[0] or 0)


def graph_json_stats(db_path: Path) -> dict[str, Any]:
    if not db_path.exists():
        return {"available": False, "reason": "missing_db"}
    with connect_readonly(db_path) as conn:
        if not _column_exists(conn, "program_results", "graph_json"):
            return {"available": False, "reason": "missing_program_results_graph_json"}
        row = conn.execute(
            """
            SELECT COUNT(*) AS rows,
                   COALESCE(SUM(LENGTH(graph_json)), 0) AS total_bytes,
                   COALESCE(MAX(LENGTH(graph_json)), 0) AS max_bytes
            FROM program_results_compat
            WHERE graph_json IS NOT NULL
              AND TRIM(CAST(graph_json AS TEXT)) <> ''
            """
        ).fetchone()
    return {
        "available": True,
        "non_empty_rows": int(row["rows"] or 0),
        "total_bytes": int(row["total_bytes"] or 0),
        "max_bytes": int(row["max_bytes"] or 0),
    }


def db_pointer_summary(db_path: Path) -> dict[str, Any]:
    if not db_path.exists():
        return {"available": False, "reason": "missing_db"}
    with connect_readonly(db_path) as conn:
        sensitive = {
            f"{table}.{column}": _pointer_count(conn, table, column)
            for table, column in SENSITIVE_POINTER_COLUMNS
        }
        expected = {
            f"{table}.{column}": _pointer_count(conn, table, column)
            for table, column in EXPECTED_POINTER_COLUMNS
        }
        artifact_rows = (
            int(conn.execute("SELECT COUNT(*) FROM notebook_artifacts").fetchone()[0])
            if _table_exists(conn, "notebook_artifacts")
            else None
        )
    return {
        "available": True,
        "notebook_artifact_rows": artifact_rows,
        "sensitive_pointer_counts": sensitive,
        "expected_pointer_counts": expected,
    }


def local_split_bundles(
    research_root: Path, project_root: Path
) -> list[dict[str, Any]]:
    roots = (
        research_root / "tmp" / "db-backup-upload",
        research_root / "tmp" / "db-backup-download",
    )
    bundles: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for item in sorted(root.rglob("*")):
            if item.is_file() and item.name in {"db-backups.tar.zst", "manifest.json"}:
                bundles.append(
                    {
                        "path": _rel(item, project_root),
                        "bytes": item.stat().st_size,
                    }
                )
    return bundles


def legacy_db_references(project_root: Path) -> dict[str, Any]:
    tools_root = project_root / "research" / "tools"
    refs: list[dict[str, Any]] = []
    if not tools_root.exists():
        return {"total": 0, "unapproved_total": 0, "references": refs}
    for path in sorted(tools_root.rglob("*")):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="replace")
        rel_path = _rel(path, project_root)
        approved = rel_path in APPROVED_LEGACY_TOOL_PATHS
        for lineno, line in enumerate(text.splitlines(), start=1):
            if "lab_notebook.db" not in line:
                continue
            refs.append(
                {
                    "path": rel_path,
                    "line": lineno,
                    "approved_compatibility_reference": approved,
                    "category": LEGACY_REFERENCE_CATEGORIES.get(
                        rel_path,
                        "active_or_unclassified",
                    ),
                    "text": line.strip()[:180],
                }
            )
    return {
        "total": len(refs),
        "unapproved_total": sum(
            1 for ref in refs if not ref["approved_compatibility_reference"]
        ),
        "references": refs,
    }


def build_report(
    *,
    project_root: Path = PROJECT_ROOT,
    runs_db: Path | None = None,
    lab_db: Path | None = None,
    artifact_dir: Path | None = None,
    runtime_events_dir: Path | None = None,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    research_root = project_root / "research"
    runs_db = runs_db or project_root / RUNS_DB
    lab_db = lab_db or project_root / LAB_NOTEBOOK_DB
    artifact_dir = artifact_dir or project_root / NOTEBOOK_ARTIFACTS_DIR
    runtime_events_dir = runtime_events_dir or project_root / RUNTIME_EVENTS_DIR

    pointer_summary = db_pointer_summary(runs_db)
    sensitive_counts = pointer_summary.get("sensitive_pointer_counts") or {}
    blockers = [
        key
        for key, count in sensitive_counts.items()
        if count is not None and int(count) > 0
    ]
    legacy_refs = legacy_db_references(project_root)
    if legacy_refs["unapproved_total"]:
        blockers.append("unapproved_lab_notebook_db_tool_references")
    if not runs_db.exists():
        blockers.append("missing_runs_db")
    if not lab_db.exists():
        blockers.append("missing_legacy_lab_db")

    return {
        "project_root": str(project_root),
        "databases": {
            "runs": _file_state(runs_db, project_root),
            "legacy_lab": _file_state(lab_db, project_root),
        },
        "artifacts": {
            "path": _rel(artifact_dir, project_root),
            "exists": artifact_dir.exists(),
            "files": _count_files(artifact_dir),
        },
        "runtime_events": {
            "path": _rel(runtime_events_dir, project_root),
            "exists": runtime_events_dir.exists(),
            "files": _count_files(runtime_events_dir),
        },
        "local_split_backup_bundles": local_split_bundles(research_root, project_root),
        "runs_db_pointer_summary": pointer_summary,
        "program_results_graph_json": graph_json_stats(runs_db),
        "legacy_lab_notebook_references": legacy_refs,
        "retirement_blockers": blockers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--runs-db", type=Path, default=None)
    parser.add_argument("--lab-db", type=Path, default=None)
    parser.add_argument(
        "--fail-on-blockers",
        action="store_true",
        help="Exit non-zero when retirement blockers remain.",
    )
    args = parser.parse_args()

    report = build_report(
        project_root=args.project_root,
        runs_db=args.runs_db,
        lab_db=args.lab_db,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if args.fail_on_blockers and report["retirement_blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
