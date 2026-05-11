#!/usr/bin/env python
"""Import AR Validation/easy25 fingerprint sweep CSVs into ``program_results``.

Default mode is a dry-run. Use ``--write`` to persist updates. Matching prefers
an exact ``result_id`` and only falls back to ``graph_fingerprint`` when that
fingerprint maps to exactly one DB row. Existing AR Validation fields are never
overwritten unless ``--overwrite`` is passed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, TextIO

from research.tools.check_backup_freshness import main as check_backup_freshness_main


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = PROJECT_ROOT / "research/runs.db"
DEFAULT_CSV_ROOT = PROJECT_ROOT / "research/runtime/ar_validation_fingerprint_sweep"

AR_VALIDATION_COLUMNS: dict[str, str] = {
    "ar_validation_metric_version": "TEXT",
    "ar_validation_final_acc": "REAL",
    "ar_validation_held_pair_acc": "REAL",
    "ar_validation_held_class_acc": "REAL",
    "ar_validation_learning_curve_json": "TEXT",
    "ar_validation_steps_to_floor": "INTEGER",
    "ar_validation_rank_score": "REAL",
    "ar_validation_status": "TEXT",
    "ar_validation_elapsed_ms": "REAL",
}

_FLOAT_COLUMNS = {
    "ar_validation_final_acc",
    "ar_validation_held_pair_acc",
    "ar_validation_held_class_acc",
    "ar_validation_rank_score",
    "ar_validation_elapsed_ms",
}
_INT_COLUMNS = {"ar_validation_steps_to_floor"}


@dataclass(frozen=True)
class CsvMetricRow:
    source_csv: Path
    source_line: int
    run_id: str
    created_unix: float
    result_id: str
    graph_fingerprint: str
    values: dict[str, Any]


@dataclass(frozen=True)
class MatchResult:
    result_id: str | None
    match_mode: str
    reason: str | None = None


@dataclass(frozen=True)
class ImportDecision:
    csv_row: CsvMetricRow
    result_id: str | None
    action: str
    reason: str
    match_mode: str
    values: dict[str, Any]


def _table_columns(conn: sqlite3.Connection, table: str) -> dict[str, str]:
    return {
        str(row[1]): str(row[2]) for row in conn.execute(f"PRAGMA table_info({table})")
    }


def ensure_ar_validation_columns(conn: sqlite3.Connection) -> list[str]:
    """Idempotently add missing AR Validation columns and return added names."""
    existing = _table_columns(conn, "graph_runs")
    added: list[str] = []
    for name, col_type in AR_VALIDATION_COLUMNS.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE graph_runs ADD COLUMN {name} {col_type}")
            added.append(name)
    return added


def discover_csv_paths(paths: Iterable[Path]) -> list[Path]:
    discovered: list[Path] = []
    for path in paths:
        if path.is_dir():
            discovered.extend(sorted(path.glob("**/*.csv")))
        elif path.exists():
            discovered.append(path)
    return sorted(dict.fromkeys(discovered))


def _blank_to_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _parse_float(value: Any) -> float | None:
    text = _blank_to_none(value)
    if text is None:
        return None
    return float(text)


def _parse_int(value: Any) -> int | None:
    text = _blank_to_none(value)
    if text is None:
        return None
    return int(float(text))


def _normalize_learning_curve(row: dict[str, Any]) -> str | None:
    raw = _blank_to_none(row.get("ar_validation_learning_curve_json"))
    if raw is None:
        raw = _blank_to_none(row.get("learning_curve_json"))
    if raw is None:
        return None
    parsed = json.loads(raw)
    return json.dumps(parsed, sort_keys=True, separators=(",", ":"))


def parse_csv_metric_row(
    row: dict[str, Any], *, source_csv: Path, source_line: int
) -> CsvMetricRow:
    values: dict[str, Any] = {}
    for name in AR_VALIDATION_COLUMNS:
        if name == "ar_validation_learning_curve_json":
            values[name] = _normalize_learning_curve(row)
        elif name in _FLOAT_COLUMNS:
            values[name] = _parse_float(row.get(name))
        elif name in _INT_COLUMNS:
            values[name] = _parse_int(row.get(name))
        else:
            values[name] = _blank_to_none(row.get(name))
    values = {key: value for key, value in values.items() if value is not None}
    return CsvMetricRow(
        source_csv=source_csv,
        source_line=int(source_line),
        run_id=_blank_to_none(row.get("run_id")) or "",
        created_unix=_parse_float(row.get("created_unix")) or 0.0,
        result_id=_blank_to_none(row.get("result_id")) or "",
        graph_fingerprint=_blank_to_none(row.get("graph_fingerprint")) or "",
        values=values,
    )


def load_csv_metric_rows(paths: Iterable[Path]) -> list[CsvMetricRow]:
    rows: list[CsvMetricRow] = []
    for path in discover_csv_paths(paths):
        with path.open(newline="") as handle:
            for line_no, row in enumerate(csv.DictReader(handle), start=2):
                rows.append(
                    parse_csv_metric_row(row, source_csv=path, source_line=line_no)
                )
    return rows


def _csv_row_priority(row: CsvMetricRow) -> tuple[int, int, float]:
    status = str(row.values.get("ar_validation_status") or "").strip().lower()
    return (
        1 if status == "ok" else 0,
        1 if row.values.get("ar_validation_rank_score") is not None else 0,
        float(row.created_unix),
    )


def _load_db_indexes(
    conn: sqlite3.Connection,
) -> tuple[dict[str, sqlite3.Row], dict[str, list[sqlite3.Row]]]:
    conn.row_factory = sqlite3.Row
    select_cols = ["result_id", "graph_fingerprint", *AR_VALIDATION_COLUMNS.keys()]
    table = _program_results_read_table(conn)
    rows = conn.execute(
        f"SELECT {', '.join(select_cols)} FROM {table}",
    ).fetchall()
    by_result_id: dict[str, sqlite3.Row] = {}
    by_fingerprint: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        rid = str(row["result_id"] or "")
        fp = str(row["graph_fingerprint"] or "")
        if rid:
            by_result_id[rid] = row
        if fp:
            by_fingerprint.setdefault(fp, []).append(row)
    return by_result_id, by_fingerprint


def _program_results_read_table(conn: sqlite3.Connection) -> str:
    """Canonical read source. Prefers graph_runs (post-Phase-5b)."""
    for candidate in ("graph_runs", "program_results_compat", "program_results"):
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = ? LIMIT 1", (candidate,)
        ).fetchone()
        if row:
            return candidate
    raise RuntimeError("no program_results-compatible table found")


def match_csv_row(
    csv_row: CsvMetricRow,
    *,
    by_result_id: dict[str, sqlite3.Row],
    by_fingerprint: dict[str, list[sqlite3.Row]],
) -> MatchResult:
    if csv_row.result_id and csv_row.result_id in by_result_id:
        db_row = by_result_id[csv_row.result_id]
        db_fp = str(db_row["graph_fingerprint"] or "")
        if csv_row.graph_fingerprint and db_fp and db_fp != csv_row.graph_fingerprint:
            return MatchResult(None, "result_id", "fingerprint_mismatch")
        return MatchResult(csv_row.result_id, "result_id")
    if not csv_row.graph_fingerprint:
        return MatchResult(None, "none", "missing_result_id_and_fingerprint")
    candidates = by_fingerprint.get(csv_row.graph_fingerprint, [])
    if len(candidates) == 1:
        return MatchResult(str(candidates[0]["result_id"]), "fingerprint")
    if len(candidates) > 1:
        return MatchResult(None, "fingerprint", "ambiguous_fingerprint")
    return MatchResult(None, "none", "no_matching_db_row")


def _has_existing_ar_validation(row: sqlite3.Row) -> bool:
    return any(row[name] is not None for name in AR_VALIDATION_COLUMNS)


def plan_import(
    rows: Iterable[CsvMetricRow],
    *,
    by_result_id: dict[str, sqlite3.Row],
    by_fingerprint: dict[str, list[sqlite3.Row]],
    overwrite: bool,
) -> list[ImportDecision]:
    decisions: list[ImportDecision] = []
    planned_targets: set[str] = set()
    for csv_row in rows:
        if not csv_row.values:
            decisions.append(
                ImportDecision(
                    csv_row, None, "skip", "empty_metric_payload", "none", {}
                ),
            )
            continue
        match = match_csv_row(
            csv_row,
            by_result_id=by_result_id,
            by_fingerprint=by_fingerprint,
        )
        if match.result_id is None:
            decisions.append(
                ImportDecision(
                    csv_row,
                    None,
                    "skip",
                    match.reason or "no_match",
                    match.match_mode,
                    {},
                ),
            )
            continue
        if match.result_id in planned_targets:
            decisions.append(
                ImportDecision(
                    csv_row,
                    match.result_id,
                    "skip",
                    "duplicate_target_in_sources",
                    match.match_mode,
                    {},
                ),
            )
            continue
        db_row = by_result_id[match.result_id]
        if _has_existing_ar_validation(db_row) and not overwrite:
            decisions.append(
                ImportDecision(
                    csv_row,
                    match.result_id,
                    "skip",
                    "existing_ar_validation_values",
                    match.match_mode,
                    {},
                ),
            )
            continue
        planned_targets.add(match.result_id)
        decisions.append(
            ImportDecision(
                csv_row,
                match.result_id,
                "update",
                "overwrite"
                if overwrite and _has_existing_ar_validation(db_row)
                else "missing_ar_validation",
                match.match_mode,
                dict(csv_row.values),
            ),
        )
    return decisions


def _merge_provenance(raw_payload: Any, entry: dict[str, Any]) -> str:
    try:
        payload = json.loads(raw_payload) if raw_payload else {}
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        print(
            f"WARNING: dropping malformed data_provenance_json for "
            f"result_id={entry.get('source_csv', '?')}:{entry.get('source_line', '?')}: {exc}",
            file=sys.stderr,
        )
        payload = {}
    if not isinstance(payload, dict):
        print(
            f"WARNING: data_provenance_json was not a dict; replacing "
            f"(source={entry.get('source_csv', '?')}:{entry.get('source_line', '?')})",
            file=sys.stderr,
        )
        payload = {}
    history = payload.get("metric_backfills")
    if not isinstance(history, list):
        history = []
    history = [item for item in history if isinstance(item, dict)]
    history.append(entry)
    payload["metric_backfills"] = history[-5:]
    payload["last_metric_backfill"] = entry
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def apply_import_decisions(
    conn: sqlite3.Connection, decisions: Iterable[ImportDecision], *, overwrite: bool
) -> int:
    columns = _table_columns(conn, "graph_runs")
    now = time.time()
    updated = 0
    for decision in decisions:
        if decision.action != "update" or decision.result_id is None:
            continue
        items = [
            (key, value) for key, value in decision.values.items() if key in columns
        ]
        if not items:
            continue
        provenance_entry = {
            "source": "ar_validation_fingerprint_sweep_csv_import",
            "source_csv": str(decision.csv_row.source_csv),
            "source_line": decision.csv_row.source_line,
            "run_id": decision.csv_row.run_id,
            "match_mode": decision.match_mode,
            "overwrite": bool(overwrite),
            "imported_at_unix": round(now, 3),
        }
        if "data_provenance_json" in columns:
            table = _program_results_read_table(conn)
            row = conn.execute(
                f"SELECT data_provenance_json FROM {table} WHERE result_id = ?",
                (decision.result_id,),
            ).fetchone()
            raw = row["data_provenance_json"] if row else None
            items.append(
                ("data_provenance_json", _merge_provenance(raw, provenance_entry))
            )
        set_clause = ", ".join(f"{key} = ?" for key, _value in items)
        params = [value for _key, value in items]
        params.append(decision.result_id)
        conn.execute(f"UPDATE graph_runs SET {set_clause} WHERE result_id = ?", params)
        updated += 1
    conn.commit()
    return updated


def summarize_decisions(decisions: Iterable[ImportDecision]) -> dict[str, Any]:
    summary: dict[str, Any] = {"actions": {}, "reasons": {}, "match_modes": {}}
    for decision in decisions:
        summary["actions"][decision.action] = (
            summary["actions"].get(decision.action, 0) + 1
        )
        summary["reasons"][decision.reason] = (
            summary["reasons"].get(decision.reason, 0) + 1
        )
        summary["match_modes"][decision.match_mode] = (
            summary["match_modes"].get(decision.match_mode, 0) + 1
        )
    return summary


def _print_report(
    *,
    out: TextIO,
    csv_paths: list[Path],
    decisions: list[ImportDecision],
    mode: str,
    updated: int = 0,
) -> None:
    summary = summarize_decisions(decisions)
    print(
        json.dumps(
            {
                "mode": mode,
                "csv_file_count": len(csv_paths),
                "csv_row_count": len(decisions),
                "updated": int(updated),
                **summary,
            },
            sort_keys=True,
        ),
        file=out,
    )
    for decision in decisions[:25]:
        print(
            json.dumps(
                {
                    "action": decision.action,
                    "reason": decision.reason,
                    "match_mode": decision.match_mode,
                    "result_id": decision.result_id,
                    "csv_result_id": decision.csv_row.result_id,
                    "graph_fingerprint": decision.csv_row.graph_fingerprint,
                    "source_csv": str(decision.csv_row.source_csv),
                    "source_line": decision.csv_row.source_line,
                    "fields": sorted(decision.values),
                },
                sort_keys=True,
            ),
            file=out,
        )
    if len(decisions) > 25:
        print(f"omitted_decisions={len(decisions) - 25}", file=out)


def run(args: argparse.Namespace, out: TextIO = sys.stdout) -> int:
    csv_paths = discover_csv_paths(args.csv_paths or [DEFAULT_CSV_ROOT])
    if not csv_paths:
        print("no_csv_files_found", file=out)
        return 1
    if args.write:
        rc = check_backup_freshness_main([])
        if rc != 0:
            return rc
        conn = sqlite3.connect(str(args.db), timeout=30.0)
    else:
        conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        if args.write:
            ensure_ar_validation_columns(conn)
        else:
            missing = [
                name
                for name in AR_VALIDATION_COLUMNS
                if name not in _table_columns(conn, "graph_runs")
            ]
            if missing:
                print(f"missing_ar_validation_columns={','.join(missing)}", file=out)
                return 1
        rows = sorted(
            load_csv_metric_rows(csv_paths), key=_csv_row_priority, reverse=True
        )
        by_result_id, by_fingerprint = _load_db_indexes(conn)
        decisions = plan_import(
            rows,
            by_result_id=by_result_id,
            by_fingerprint=by_fingerprint,
            overwrite=bool(args.overwrite),
        )
        updated = 0
        if args.write:
            updated = apply_import_decisions(
                conn, decisions, overwrite=bool(args.overwrite)
            )
        _print_report(
            out=out,
            csv_paths=csv_paths,
            decisions=decisions,
            mode="WRITE" if args.write else "DRY-RUN",
            updated=updated,
        )
        return 0
    finally:
        conn.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_paths", nargs="*", type=Path, help="CSV files or directories to import."
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--write", action="store_true", help="Persist updates. Default is dry-run."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing AR Validation values.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
