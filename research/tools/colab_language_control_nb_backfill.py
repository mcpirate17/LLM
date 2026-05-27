"""Colab-friendly NB0.5/NB1.0 language-control backfill.

Runs one language-control nano-bind tier at a time against explicit candidate
TSVs from the Google Drive bundle. This avoids the broader ladder runner's
top-N selection and keeps NB0.5 and NB1.0 independently resumable.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import torch

from research.tools.language_control_backfill import (
    LANGUAGE_CONTROL_METRIC_VERSION,
    _ensure_backfill_columns,
    _run_one,
    _write_row,
)
from research.scientist.notebook.graph_artifacts import resolve_graph_json_value


_TIER_FOR_PROBE = {"nb05": "s05", "nb10": "s10"}


def _candidate_result_ids(path: Path | None) -> list[str]:
    if path is None:
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []
    header = lines[0].split("\t")
    try:
        idx = header.index("result_id")
    except ValueError:
        idx = 0
    out: list[str] = []
    seen: set[str] = set()
    for line in lines[1:]:
        parts = line.split("\t")
        if idx < len(parts):
            rid = parts[idx].strip()
            if rid and rid not in seen:
                seen.add(rid)
                out.append(rid)
    return out


def _completed_result_ids(report_path: Path, probe: str) -> set[str]:
    """Result IDs already scored in a prior report run (restart resume).

    A row counts as done only when its report record carries a non-null binding
    score for the probe's tier. ``error`` / ``None`` / ``no_updates`` rows stay
    in the work set so a restart re-runs them. The report is append-only across
    restarts, so the latest non-null score for any id wins.
    """
    if not report_path.exists():
        return set()
    score_key = (
        "language_control_s05_binding_score"
        if probe == "nb05"
        else "language_control_s10_binding_score"
    )
    done: set[str] = set()
    with report_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = record.get("result_id")
            updates = record.get("updates")
            if rid and isinstance(updates, dict) and updates.get(score_key) is not None:
                done.add(str(rid))
    return done


def _load_candidate_rows(path: Path, *, limit: int | None) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"candidate JSONL missing or empty: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            # JSONL candidates carry only result_id; the DB path aliases it to
            # entry_id. Normalize here so every downstream consumer (logging,
            # _run_one) sees a uniform row contract and no missing-entry_id
            # KeyError can ever surface again.
            row.setdefault("entry_id", row.get("result_id"))
            rows.append(row)
            if limit is not None and limit > 0 and len(rows) >= limit:
                break
    return rows


def _open_existing_db(path: Path) -> sqlite3.Connection:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(
            f"DB snapshot is missing or empty: {path}. "
            "Re-check the Drive path and rerun the notebook setup cell."
        )
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def _program_source(conn: sqlite3.Connection) -> tuple[str, str]:
    names = {
        str(row["name"])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        )
    }
    if "program_results_compat" in names:
        return "program_results_compat pr", "pr.graph_json"
    if "graph_runs" in names and "graphs" in names:
        return "graph_runs pr LEFT JOIN graphs g USING (graph_fingerprint)", (
            "COALESCE(g.graph_json, '{}')"
        )
    raise sqlite3.OperationalError(
        "DB snapshot has no program_results_compat view and no graph_runs+graphs "
        f"tables; available objects={sorted(names)[:40]}"
    )


def _fetch_rows(
    conn: sqlite3.Connection,
    db_path: Path,
    *,
    probe: str,
    result_ids: list[str],
    limit: int | None,
    force: bool,
) -> list[dict[str, Any]]:
    primary = (
        "language_control_s05_binding_score"
        if probe == "nb05"
        else "language_control_s10_binding_score"
    )
    source, graph_json = _program_source(conn)
    sql = f"""
        SELECT pr.result_id,
               pr.result_id AS entry_id,
               pr.graph_fingerprint,
               {graph_json} AS graph_json,
               COALESCE(lb.tier, 'off_leaderboard') AS tier,
               COALESCE(lb.composite_score, pr.loss_ratio, 0.0) AS composite_score,
               pr.fp_jacobian_erf_density AS erf_density,
               pr.fp_jacobian_erf_decay_slope AS erf_decay_slope,
               pr.graph_category_histogram AS graph_category_histogram
        FROM {source}
        LEFT JOIN leaderboard lb ON lb.result_id = pr.result_id
        WHERE pr.graph_fingerprint IS NOT NULL
          AND TRIM(COALESCE({graph_json}, '')) <> ''
          AND {graph_json} <> '{{}}'
          AND length({graph_json}) > 10
          AND pr.stage0_passed = 1
          AND pr.stage05_passed = 1
    """
    params: list[Any] = []
    if result_ids:
        marks = ",".join("?" for _ in result_ids)
        sql += f" AND pr.result_id IN ({marks})"
        params.extend(result_ids)
    if not force:
        sql += f" AND pr.{primary} IS NULL"
    sql += """
        ORDER BY
            CASE WHEN tier IN ('breakthrough', 'validation', 'investigation') THEN 0 ELSE 1 END,
            composite_score DESC
    """
    if limit is not None and limit > 0:
        sql += f" LIMIT {int(limit)}"
    rows: list[dict[str, Any]] = []
    for row in conn.execute(sql, params):
        d = dict(row)
        d["graph_json"] = resolve_graph_json_value(conn, db_path, d.get("graph_json"))
        rows.append(d)
    return rows


def _filter_updates(probe: str, updates: dict[str, Any]) -> dict[str, Any]:
    if probe == "nb05":
        allowed = {
            "language_control_metric_version",
            "language_control_s05_sentence_assoc_score",
            "language_control_s05_binding_order_acc",
            "language_control_s05_binding_score",
        }
    else:
        allowed = {
            "language_control_metric_version",
            "language_control_s10_sentence_assoc_score",
            "language_control_s10_binding_order_acc",
            "language_control_s10_binding_score",
            "language_control_s10_checkpoints_json",
        }
    return {k: v for k, v in updates.items() if k in allowed and v is not None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("research/runs.db"))
    parser.add_argument("--candidates-jsonl", type=Path)
    parser.add_argument("--probe", choices=("nb05", "nb10"), required=True)
    parser.add_argument("--from-report", type=Path)
    parser.add_argument("--report-jsonl", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "re-run every candidate even if it already has a scored row in "
            "--report-jsonl (default: skip already-scored rows on restart)"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.report_jsonl.parent.mkdir(parents=True, exist_ok=True)
    resume = not args.force and not args.no_resume
    completed = (
        _completed_result_ids(args.report_jsonl, args.probe) if resume else set()
    )
    conn = None
    if args.candidates_jsonl:
        # When resuming, load the full candidate set so --limit applies to the
        # not-yet-scored remainder rather than rows we are about to skip.
        rows = _load_candidate_rows(
            args.candidates_jsonl, limit=None if completed else args.limit
        )
    else:
        result_ids = _candidate_result_ids(args.from_report)
        conn = _open_existing_db(args.db)
        conn.execute("PRAGMA journal_mode=WAL")
        _ensure_backfill_columns(conn)
        rows = _fetch_rows(
            conn,
            args.db,
            probe=args.probe,
            result_ids=result_ids,
            limit=None if completed else args.limit,
            force=args.force,
        )
    if completed:
        before = len(rows)
        rows = [r for r in rows if str(r.get("result_id")) not in completed]
        if args.limit and args.limit > 0:
            rows = rows[: args.limit]
        print(
            f"resume: skipped {before - len(rows)} already-scored rows; "
            f"{len(rows)} remaining",
            flush=True,
        )
    tier = _TIER_FOR_PROBE[args.probe]
    print(
        f"{args.probe} rows={len(rows)} tier={tier} version={LANGUAGE_CONTROL_METRIC_VERSION}",
        flush=True,
    )
    if args.dry_run:
        return
    ok = 0
    t0 = time.perf_counter()
    with args.report_jsonl.open("a", encoding="utf-8") as report:
        for idx, row in enumerate(rows, start=1):
            try:
                updates = _run_one(row, device=args.device, tier_names=(tier,)) or {}
                updates = _filter_updates(args.probe, updates)
                if updates:
                    if conn is not None:
                        _write_row(conn, str(row["result_id"]), updates)
                        conn.commit()
                    ok += 1
                status = "updated" if updates else "no_updates"
                score = updates.get(
                    "language_control_s05_binding_score"
                    if args.probe == "nb05"
                    else "language_control_s10_binding_score"
                )
                record = {
                    "idx": idx,
                    "total": len(rows),
                    "status": status,
                    "result_id": row["result_id"],
                    "graph_fingerprint": row["graph_fingerprint"],
                    "tier": row["tier"],
                    "updates": updates,
                    "elapsed_s": round(time.perf_counter() - t0, 3),
                }
                print(
                    f"[{idx}/{len(rows)}] {row['result_id'][:12]} {args.probe}={score} status={status}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                record = {
                    "idx": idx,
                    "total": len(rows),
                    "status": "error",
                    "result_id": row["result_id"],
                    "graph_fingerprint": row["graph_fingerprint"],
                    "error": f"{type(exc).__name__}: {str(exc)[:240]}",
                    "elapsed_s": round(time.perf_counter() - t0, 3),
                }
                print(
                    f"[{idx}/{len(rows)}] {row['result_id'][:12]} error={record['error']}",
                    flush=True,
                )
                if args.device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            report.write(json.dumps(record, sort_keys=True) + "\n")
            report.flush()
    print(
        f"done ok={ok} total={len(rows)} elapsed_s={time.perf_counter() - t0:.1f}",
        flush=True,
    )
    if conn is not None:
        conn.close()


if __name__ == "__main__":
    main()
