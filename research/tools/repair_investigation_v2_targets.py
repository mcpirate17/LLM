"""Build a targeted v2-probe repair list for historical investigation rows.

This does not change generation.  It finds completed investigation/leaderboard
graphs that are still missing investigation-tier v2 induction or binding
metrics, writes a prioritized JSONL target file, and optionally runs the
existing v2 backfill command on that file.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPORT_DIR = Path("research/reports")


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _rank_key(row: dict[str, Any]) -> tuple[int, float, float]:
    loss = _float_or_none(row.get("loss_ratio"))
    timestamp = _float_or_none(row.get("timestamp")) or 0.0
    return (
        1 if row.get("stage1_passed") else 0,
        -(loss if loss is not None else float("inf")),
        timestamp,
    )


def find_repair_targets(
    db_path: str,
    *,
    include_leaderboard: bool = True,
    failed_only: bool = False,
    limit: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    db_uri = f"file:{Path(db_path).resolve()}?mode=ro"
    conn = sqlite3.connect(db_uri, uri=True, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT pr.result_id, pr.graph_fingerprint, pr.experiment_id, pr.timestamp,
                   pr.stage1_passed, pr.loss_ratio, pr.model_source, pr.result_cohort,
                   pr.induction_intermediate_auc,
                   pr.binding_intermediate_auc,
                   pr.induction_intermediate_status,
                   pr.binding_intermediate_status,
                   e.experiment_type, e.status AS experiment_status,
                   l.tier
            FROM program_results_compat pr
            LEFT JOIN experiments e ON e.experiment_id = pr.experiment_id
            LEFT JOIN leaderboard l ON l.result_id = pr.result_id
            WHERE TRIM(COALESCE(pr.graph_json, '')) <> ''
              AND pr.graph_json <> '{}'
              AND TRIM(COALESCE(pr.graph_fingerprint, '')) <> ''
              AND (
                    e.experiment_type = 'investigation'
                    OR (? AND l.tier IN ('investigation', 'validation', 'breakthrough'))
                  )
              AND (
                    pr.induction_intermediate_auc IS NULL
                    OR pr.binding_intermediate_auc IS NULL
                    OR COALESCE(pr.induction_intermediate_status, '') NOT IN ('', 'ok')
                    OR COALESCE(pr.binding_intermediate_status, '') NOT IN ('', 'ok')
                  )
              AND (
                    NOT ?
                    OR COALESCE(pr.induction_intermediate_status, '') NOT IN ('', 'ok')
                    OR COALESCE(pr.binding_intermediate_status, '') NOT IN ('', 'ok')
                  )
            """,
            (1 if include_leaderboard else 0, 1 if failed_only else 0),
        ).fetchall()
    finally:
        conn.close()

    by_fp: dict[str, dict[str, Any]] = {}
    duplicate_rows = 0
    for raw in rows:
        row = dict(raw)
        fp = str(row.get("graph_fingerprint") or "").strip()
        if not fp:
            continue
        current = by_fp.get(fp)
        if current is not None:
            duplicate_rows += 1
        if current is None or _rank_key(row) > _rank_key(current):
            missing = []
            if row.get("induction_intermediate_auc") is None:
                missing.append("induction_intermediate_auc")
            if row.get("binding_intermediate_auc") is None:
                missing.append("binding_intermediate_auc")
            if (
                row.get("induction_intermediate_status")
                and row.get("induction_intermediate_status") != "ok"
            ):
                missing.append("induction_intermediate_status")
            if (
                row.get("binding_intermediate_status")
                and row.get("binding_intermediate_status") != "ok"
            ):
                missing.append("binding_intermediate_status")
            by_fp[fp] = {
                "result_id": str(row.get("result_id") or ""),
                "fp": fp,
                "graph_fingerprint": fp,
                "experiment_id": row.get("experiment_id"),
                "experiment_type": row.get("experiment_type"),
                "experiment_status": row.get("experiment_status"),
                "tier": row.get("tier"),
                "stage1_passed": bool(row.get("stage1_passed")),
                "loss_ratio": _float_or_none(row.get("loss_ratio")),
                "model_source": row.get("model_source"),
                "result_cohort": row.get("result_cohort"),
                "missing_signals": missing,
            }

    targets = sorted(
        by_fp.values(),
        key=lambda item: (
            1 if item.get("stage1_passed") else 0,
            -(item.get("loss_ratio") if item.get("loss_ratio") is not None else 999.0),
        ),
        reverse=True,
    )
    if limit > 0:
        targets = targets[:limit]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "db_path": db_path,
        "include_leaderboard": bool(include_leaderboard),
        "failed_only": bool(failed_only),
        "n_raw_rows": len(rows),
        "n_duplicate_fingerprint_rows": duplicate_rows,
        "n_targets": len(targets),
        "limit": int(limit),
    }
    return targets, summary


def write_targets(
    targets: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    output_prefix: Path,
) -> tuple[Path, Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    jsonl_path = output_prefix.with_suffix(".jsonl")
    json_path.write_text(
        json.dumps({"summary": summary, "targets": targets}, indent=2),
        encoding="utf-8",
    )
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in targets:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return json_path, jsonl_path


def run_backfill(jsonl_path: Path, *, top: int, device: str, force: bool) -> None:
    cmd = [
        sys.executable,
        "-m",
        "research.tools.backfill",
        "--probe",
        "induction_intermediate,binding_intermediate",
        "--fingerprint-file",
        str(jsonl_path),
        "--top",
        str(max(0, int(top))),
        "--device",
        device,
    ]
    if force:
        cmd.append("--force")
    completed = subprocess.run(cmd, check=False)
    if completed.returncode:
        raise RuntimeError(
            "v2 backfill command failed with exit code "
            f"{completed.returncode}: {' '.join(cmd)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="research/runs.db")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--include-leaderboard", action="store_true", default=True)
    parser.add_argument("--investigation-only", action="store_true")
    parser.add_argument(
        "--failed-only",
        action="store_true",
        help="Only target rows where a prior v2 probe status is non-ok.",
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Output path without suffix. Defaults to research/reports/investigation_v2_repair_targets_YYYY-MM-DD.",
    )
    args = parser.parse_args()

    targets, summary = find_repair_targets(
        args.db,
        include_leaderboard=not bool(args.investigation_only),
        failed_only=bool(args.failed_only),
        limit=max(0, int(args.limit)),
    )
    prefix = (
        Path(args.output_prefix)
        if args.output_prefix
        else REPORT_DIR
        / f"investigation_v2_repair_targets_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
    )
    json_path, jsonl_path = write_targets(targets, summary, output_prefix=prefix)
    print(f"Wrote {len(targets)} v2 repair targets")
    print(f"JSON:  {json_path}")
    print(f"JSONL: {jsonl_path}")
    for row in targets[:10]:
        print(
            f"{row['result_id']} fp={row['fp'][:12]} "
            f"tier={row.get('tier') or '-'} loss={row.get('loss_ratio')} "
            f"missing={','.join(row.get('missing_signals') or [])}"
        )
    if args.apply and targets:
        try:
            run_backfill(
                jsonl_path,
                top=len(targets),
                device=str(args.device or "cuda"),
                force=bool(args.force),
            )
        except RuntimeError as exc:
            print(f"Apply failed: {exc}", file=sys.stderr)
            print(
                "If the dashboard or continuous runner is active, run the repair "
                "from that process or stop it before launching this writer.",
                file=sys.stderr,
            )
            raise SystemExit(1)


if __name__ == "__main__":
    main()
