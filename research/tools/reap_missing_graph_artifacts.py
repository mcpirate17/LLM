#!/usr/bin/env python3
"""Reap rows whose ``graph_json`` points to a graph artifact file that no
longer exists on disk.

Background
----------
``graph_json`` columns store either inline JSON or a small pointer of the form
``{"_notebook_artifact": "<id>", "path": "<rel/path>.zst", ...}`` resolved via
``research/scientist/notebook/graph_artifacts.py``. When the on-disk artifact
is missing — typically because it was auto-pruned — every downstream consumer
(observability, exp-coverage, mathspace impact, predictors, replays) hits a
``FileNotFoundError`` and either skips the row or floods the log.

This script scans tables with a ``graph_json`` column, classifies each row,
and (under ``--apply``) heals missing-artifact rows by either replacing the
pointer with ``'{}'`` (default) or deleting the row outright
(``--strategy=delete``). A timestamped JSON report of every affected row is
written under ``research/reports/`` so the action is reversible from the audit
trail even after the bad pointers are gone.

Safety
------
- Default mode is dry-run; ``--apply`` is required to mutate the DB.
- ``--apply`` refuses to run while another writer holds
  ``<db>.writer-lock`` (covers both the aria-db Rust writer and the Python
  ``_db_maintenance`` PID-file convention).
- A row-level backup is taken into ``program_results_artifact_reap_backup``
  (or table-specific equivalent) before any UPDATE/DELETE, so each reaped
  row's prior state can be inspected or replayed.

Usage
-----
    # Just scan & report (no DB changes)
    python -m research.tools.reap_missing_graph_artifacts

    # Heal pointers (set graph_json='{}')
    python -m research.tools.reap_missing_graph_artifacts --apply

    # Or delete the rows entirely
    python -m research.tools.reap_missing_graph_artifacts --apply \\
        --strategy=delete

    # Different DB / artifacts root
    python -m research.tools.reap_missing_graph_artifacts \\
        --db research/lab_notebook.db \\
        --artifacts-root research/artifacts/notebook
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from research.defaults import NOTEBOOK_ARTIFACTS_DIR, RUNS_DB
from research.scientist.notebook.artifact_store import (
    parse_artifact_pointer,
)
from research.tools._db_maintenance import (
    check_writer_lock,
    connect_readonly,
    connect_writer,
    ensure_backup_table,
    quoted_columns,
    table_columns,
)

# Tables with a ``graph_json`` column. ``pk`` is used for backup keying and
# for DELETE statements.
_TARGET_TABLES: tuple[tuple[str, str], ...] = (
    ("program_results", "result_id"),
    ("workflow_definitions", "workflow_id"),
    ("scaffold_profile_results", "profile_result_id"),
)

_BACKUP_SUFFIX = "_artifact_reap_backup"
_EMPTY_GRAPH_JSON = "{}"


@dataclass
class _TableReport:
    table: str
    pk: str
    scanned: int = 0
    inline: int = 0
    pointer_ok: int = 0
    pointer_missing: int = 0
    pointer_unresolvable: int = 0
    missing_examples: list[dict[str, str]] = field(default_factory=list)
    missing_pks: list[str] = field(default_factory=list)


def _resolve_artifact_path(
    pointer: dict[str, object], artifacts_root: Path
) -> Path | None:
    rel = pointer.get("path")
    if not isinstance(rel, str) or not rel:
        return None
    return artifacts_root / rel


def _classify_row(
    graph_json: object,
    artifacts_root: Path,
) -> tuple[str, Path | None]:
    """Return ``(category, expected_path)``.

    Categories: ``inline``, ``pointer_ok``, ``pointer_missing``,
    ``pointer_unresolvable``.
    """
    pointer = parse_artifact_pointer(graph_json)
    if pointer is None:
        return "inline", None
    path = _resolve_artifact_path(pointer, artifacts_root)
    if path is None:
        return "pointer_unresolvable", None
    if path.is_file():
        return "pointer_ok", path
    return "pointer_missing", path


def _scan_table(
    conn: sqlite3.Connection,
    table: str,
    pk: str,
    artifacts_root: Path,
    *,
    limit: int | None,
    examples_cap: int,
) -> _TableReport:
    rep = _TableReport(table=table, pk=pk)
    if table not in {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }:
        return rep
    cols = set(table_columns(conn, table))
    if "graph_json" not in cols or pk not in cols:
        return rep
    sql = f'SELECT "{pk}" AS pk, graph_json FROM {table}'
    if limit:
        sql += f" LIMIT {int(limit)}"
    for row in conn.execute(sql):
        rep.scanned += 1
        pk_value = str(row["pk"])
        category, path = _classify_row(row["graph_json"], artifacts_root)
        if category == "inline":
            rep.inline += 1
        elif category == "pointer_ok":
            rep.pointer_ok += 1
        elif category == "pointer_unresolvable":
            rep.pointer_unresolvable += 1
            rep.missing_pks.append(pk_value)
            if len(rep.missing_examples) < examples_cap:
                rep.missing_examples.append(
                    {"pk": pk_value, "reason": "unresolvable_pointer"}
                )
        else:  # pointer_missing
            rep.pointer_missing += 1
            rep.missing_pks.append(pk_value)
            if len(rep.missing_examples) < examples_cap:
                rep.missing_examples.append(
                    {"pk": pk_value, "expected_path": str(path)}
                )
    return rep


def _backup_rows(
    conn: sqlite3.Connection, table: str, pk: str, pks: Iterable[str]
) -> None:
    backup = f"{table}{_BACKUP_SUFFIX}"
    ensure_backup_table(
        conn,
        backup_table=backup,
        source_table=table,
        extra_columns=(("reaped_at", "REAL"),),
        indexes=((f"idx_{backup}_pk", (pk,)),),
    )
    cols = table_columns(conn, table)
    select_cols = quoted_columns(cols)
    placeholders = ",".join(["?"] * len(cols))
    insert_sql = (
        f"INSERT INTO {backup} ({select_cols}, reaped_at) VALUES ({placeholders}, ?)"
    )
    now = time.time()
    batch: list[tuple] = []
    pk_list = list(pks)
    if not pk_list:
        return
    chunk = 900
    for start in range(0, len(pk_list), chunk):
        slice_pks = pk_list[start : start + chunk]
        marks = ",".join(["?"] * len(slice_pks))
        rows = conn.execute(
            f"SELECT {select_cols} FROM {table} WHERE {pk} IN ({marks})",
            slice_pks,
        ).fetchall()
        for row in rows:
            batch.append(tuple(row) + (now,))
        if batch:
            conn.executemany(insert_sql, batch)
            batch.clear()


def _apply_clear(conn: sqlite3.Connection, table: str, pk: str, pks: list[str]) -> int:
    n = 0
    chunk = 900
    for start in range(0, len(pks), chunk):
        slice_pks = pks[start : start + chunk]
        marks = ",".join(["?"] * len(slice_pks))
        cur = conn.execute(
            f"UPDATE {table} SET graph_json = ? WHERE {pk} IN ({marks})",
            (_EMPTY_GRAPH_JSON, *slice_pks),
        )
        n += cur.rowcount or 0
    return n


def _apply_delete(conn: sqlite3.Connection, table: str, pk: str, pks: list[str]) -> int:
    n = 0
    chunk = 900
    for start in range(0, len(pks), chunk):
        slice_pks = pks[start : start + chunk]
        marks = ",".join(["?"] * len(slice_pks))
        cur = conn.execute(f"DELETE FROM {table} WHERE {pk} IN ({marks})", slice_pks)
        n += cur.rowcount or 0
    return n


def _print_report(reports: list[_TableReport]) -> None:
    width = max(len(r.table) for r in reports) if reports else 20
    header = (
        f"{'table'.ljust(width)}  scanned  inline  pointer_ok  missing  unresolvable"
    )
    print(header)
    print("-" * len(header))
    totals = _TableReport(table="TOTAL", pk="")
    for r in reports:
        print(
            f"{r.table.ljust(width)}  {r.scanned:>7}  {r.inline:>6}  "
            f"{r.pointer_ok:>10}  {r.pointer_missing:>7}  {r.pointer_unresolvable:>12}"
        )
        totals.scanned += r.scanned
        totals.inline += r.inline
        totals.pointer_ok += r.pointer_ok
        totals.pointer_missing += r.pointer_missing
        totals.pointer_unresolvable += r.pointer_unresolvable
    print("-" * len(header))
    print(
        f"{totals.table.ljust(width)}  {totals.scanned:>7}  {totals.inline:>6}  "
        f"{totals.pointer_ok:>10}  {totals.pointer_missing:>7}  "
        f"{totals.pointer_unresolvable:>12}"
    )
    for r in reports:
        if not r.missing_examples:
            continue
        print(f"\n{r.table} sample missing rows (first {len(r.missing_examples)}):")
        for ex in r.missing_examples:
            print(f"  {ex}")


def _write_report_json(
    reports: list[_TableReport],
    *,
    report_dir: Path,
    db_path: Path,
    artifacts_root: Path,
    applied: bool,
    strategy: str,
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    path = report_dir / f"{stamp}_artifact_reaper.json"
    payload = {
        "timestamp": time.time(),
        "db": str(db_path),
        "artifacts_root": str(artifacts_root),
        "applied": applied,
        "strategy": strategy if applied else None,
        "tables": [
            {
                "table": r.table,
                "pk": r.pk,
                "scanned": r.scanned,
                "inline": r.inline,
                "pointer_ok": r.pointer_ok,
                "pointer_missing": r.pointer_missing,
                "pointer_unresolvable": r.pointer_unresolvable,
                "missing_pks": r.missing_pks,
            }
            for r in reports
        ],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


_SHORT_DESC = "Reap rows whose graph_json points to a missing artifact file."


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=_SHORT_DESC)
    p.add_argument(
        "--db",
        type=Path,
        default=Path(RUNS_DB),
        help="SQLite DB to scan (default: %(default)s).",
    )
    p.add_argument(
        "--artifacts-root",
        type=Path,
        default=None,
        help=(
            "Root directory for graph artifacts. Defaults to "
            f"<db_dir>/artifacts/notebook (i.e. {NOTEBOOK_ARTIFACTS_DIR} for "
            "the project default DB)."
        ),
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Actually mutate the DB. Without this flag the script is a "
        "read-only scan and reports counts only.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Required with --apply to acknowledge that the writer (dashboard "
        "server, training run) must be stopped first.",
    )
    p.add_argument(
        "--strategy",
        choices=("clear", "delete"),
        default="clear",
        help="With --apply: 'clear' replaces graph_json with '{}' (keeps "
        "metrics, makes the row safely ignorable by graph consumers). "
        "'delete' removes the row entirely.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, scan only the first N rows per table (debug aid).",
    )
    p.add_argument(
        "--examples-cap",
        type=int,
        default=10,
        help="Max example PKs to keep per table in the printed report "
        "(default: %(default)s). The JSON report always carries the full "
        "list of affected PKs.",
    )
    p.add_argument(
        "--report-dir",
        type=Path,
        default=Path("research/reports"),
        help="Directory for the JSON report (default: %(default)s).",
    )
    return p


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path] | int:
    db_path: Path = args.db
    if not db_path.exists():
        print(f"error: db not found: {db_path}", file=sys.stderr)
        return 2
    artifacts_root = (
        args.artifacts_root
        if args.artifacts_root is not None
        else db_path.parent / "artifacts" / "notebook"
    )
    if not artifacts_root.exists():
        print(
            f"warning: artifacts root does not exist: {artifacts_root} "
            "(every pointer will be reported missing)",
            file=sys.stderr,
        )
    return db_path, artifacts_root


def _preflight_apply(args: argparse.Namespace, db_path: Path) -> int:
    if not args.apply:
        return 0
    if not args.force:
        print(
            "error: --apply requires --force to acknowledge the writer "
            "(dashboard / training process) must be stopped first. "
            "See feedback_aria_db_wal_hygiene memory.",
            file=sys.stderr,
        )
        return 2
    try:
        check_writer_lock(Path(f"{db_path}.writer-lock"))
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    return 0


def _run_scan(
    args: argparse.Namespace, db_path: Path, artifacts_root: Path
) -> list[_TableReport]:
    limit = args.limit if args.limit > 0 else None
    conn = connect_readonly(db_path)
    try:
        return [
            _scan_table(
                conn,
                table,
                pk,
                artifacts_root,
                limit=limit,
                examples_cap=args.examples_cap,
            )
            for table, pk in _TARGET_TABLES
        ]
    finally:
        conn.close()


def _run_apply(
    args: argparse.Namespace, db_path: Path, reports: list[_TableReport]
) -> dict[str, int]:
    write_conn = connect_writer(db_path)
    per_table: dict[str, int] = {}
    try:
        write_conn.execute("BEGIN")
        for r in reports:
            if not r.missing_pks:
                continue
            _backup_rows(write_conn, r.table, r.pk, r.missing_pks)
            if args.strategy == "clear":
                per_table[r.table] = _apply_clear(
                    write_conn, r.table, r.pk, r.missing_pks
                )
            else:
                per_table[r.table] = _apply_delete(
                    write_conn, r.table, r.pk, r.missing_pks
                )
        write_conn.commit()
    except Exception:
        write_conn.rollback()
        raise
    finally:
        write_conn.close()
    return per_table


def _emit_report(
    args: argparse.Namespace,
    reports: list[_TableReport],
    *,
    db_path: Path,
    artifacts_root: Path,
    applied: bool,
) -> None:
    report_path = _write_report_json(
        reports,
        report_dir=args.report_dir,
        db_path=db_path,
        artifacts_root=artifacts_root,
        applied=applied,
        strategy=args.strategy,
    )
    print(f"Report: {report_path}")


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    resolved = _resolve_paths(args)
    if isinstance(resolved, int):
        return resolved
    db_path, artifacts_root = resolved

    err = _preflight_apply(args, db_path)
    if err:
        return err

    reports = _run_scan(args, db_path, artifacts_root)
    _print_report(reports)

    total_missing = sum(r.pointer_missing + r.pointer_unresolvable for r in reports)
    if total_missing == 0:
        print("\nNo missing-artifact rows found. Nothing to do.")
        _emit_report(
            args, reports, db_path=db_path, artifacts_root=artifacts_root, applied=False
        )
        return 0

    if not args.apply:
        print(
            f"\nDry run: would {args.strategy} {total_missing} row(s). "
            "Re-run with --apply --force to execute.",
        )
        _emit_report(
            args, reports, db_path=db_path, artifacts_root=artifacts_root, applied=False
        )
        return 0

    per_table = _run_apply(args, db_path, reports)
    print(f"\nApplied strategy={args.strategy}:")
    for table, n in per_table.items():
        print(f"  {table}: {n} row(s) affected")
    _emit_report(
        args, reports, db_path=db_path, artifacts_root=artifacts_root, applied=True
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
