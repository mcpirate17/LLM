"""Colab-friendly NanoBind no-go backfill.

Runs ``research.eval.nano_bind.nano_bind`` against explicit candidate rows and
updates only no-go failures on the snapshot DB. Detailed probe outputs are
written to JSONL so they remain useful even though the production DB only has
failure columns for this gate.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import torch

from research.eval.nano_bind import NANO_BIND_METRIC_VERSION, nano_bind
from research.scientist.notebook.graph_artifacts import resolve_graph_json_value


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


def _load_candidate_rows(path: Path, *, limit: int | None) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"candidate JSONL missing or empty: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
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
    result_ids: list[str],
    limit: int | None,
    force: bool,
) -> list[dict[str, Any]]:
    source, graph_json = _program_source(conn)
    sql = """
        SELECT pr.result_id,
               pr.graph_fingerprint,
               {graph_json} AS graph_json,
               pr.failure_op,
               COALESCE(lb.tier, 'off_leaderboard') AS tier,
               COALESCE(lb.composite_score, pr.loss_ratio, 0.0) AS priority_score
        FROM {source}
        LEFT JOIN leaderboard lb ON lb.result_id = pr.result_id
        WHERE pr.graph_fingerprint IS NOT NULL
          AND TRIM(COALESCE({graph_json}, '')) <> ''
          AND {graph_json} <> '{{}}'
          AND length({graph_json}) > 10
          AND pr.stage0_passed = 1
          AND pr.stage05_passed = 1
    """.format(source=source, graph_json=graph_json)
    params: list[Any] = []
    if result_ids:
        marks = ",".join("?" for _ in result_ids)
        sql += f" AND pr.result_id IN ({marks})"
        params.extend(result_ids)
    if not force:
        sql += " AND COALESCE(pr.failure_op, '') <> 'nano_bind'"
    sql += """
        ORDER BY
            CASE WHEN tier IN ('breakthrough', 'validation', 'investigation') THEN 0 ELSE 1 END,
            priority_score DESC
    """
    if limit is not None and limit > 0:
        sql += f" LIMIT {int(limit)}"
    rows: list[dict[str, Any]] = []
    for row in conn.execute(sql, params):
        d = dict(row)
        d["graph_json"] = resolve_graph_json_value(conn, db_path, d.get("graph_json"))
        rows.append(d)
    return rows


def _failure_details(result) -> str:
    return json.dumps(
        {
            "reason": "nano_bind_persistent_zero",
            "scores": list(result.scores),
            "metric_version": NANO_BIND_METRIC_VERSION,
            "checkpoints": list((result.sweep_metadata or {}).get("checkpoints", [])),
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("research/runs.db"))
    parser.add_argument("--candidates-jsonl", type=Path)
    parser.add_argument("--from-report", type=Path)
    parser.add_argument("--report-jsonl", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.report_jsonl.parent.mkdir(parents=True, exist_ok=True)
    conn = None
    if args.candidates_jsonl:
        rows = _load_candidate_rows(args.candidates_jsonl, limit=args.limit)
    else:
        result_ids = _candidate_result_ids(args.from_report)
        conn = _open_existing_db(args.db)
        conn.execute("PRAGMA journal_mode=WAL")
        rows = _fetch_rows(
            conn,
            args.db,
            result_ids=result_ids,
            limit=args.limit,
            force=args.force,
        )
    print(f"nano_bind rows={len(rows)} seed={args.seed}", flush=True)
    if args.dry_run:
        return

    no_go = 0
    ok = 0
    t0 = time.perf_counter()
    with args.report_jsonl.open("a", encoding="utf-8") as report:
        for idx, row in enumerate(rows, start=1):
            try:
                result = nano_bind(
                    row["graph_json"], device=args.device, seed=args.seed
                )
                decision = (
                    "no_go"
                    if result.status == "ok" and bool(result.is_no_go)
                    else "pass"
                    if result.status == "ok"
                    else "error"
                )
                if decision == "no_go":
                    no_go += 1
                    if conn is not None:
                        conn.execute(
                            "UPDATE graph_runs SET failure_op = 'nano_bind', failure_details_json = ? WHERE result_id = ?",
                            (_failure_details(result), row["result_id"]),
                        )
                        conn.commit()
                elif decision == "pass":
                    ok += 1
                record = {
                    "idx": idx,
                    "total": len(rows),
                    "status": decision,
                    "result_id": row["result_id"],
                    "graph_fingerprint": row["graph_fingerprint"],
                    "tier": row["tier"],
                    "nano_bind": result.to_dict(),
                    "elapsed_s": round(time.perf_counter() - t0, 3),
                }
                print(
                    f"[{idx}/{len(rows)}] {row['result_id'][:12]} decision={decision} scores={list(result.scores)}",
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
        f"done pass={ok} no_go={no_go} total={len(rows)} elapsed_s={time.perf_counter() - t0:.1f}",
        flush=True,
    )
    if conn is not None:
        conn.close()


if __name__ == "__main__":
    main()
