"""Queue targeted investigation follow-ups from the offline investigation queue.

Default mode is dry-run.  Use ``--apply`` to insert follow-up tasks into the
notebook's existing ``followup_tasks`` table for the runner to claim.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research.defaults import RUNS_DB
from research.scientist.notebook import LabNotebook
from research.scientist.runner import RunConfig
from research.tools.build_investigation_queue import build_queue, write_reports

REPORT_DIR = Path("research/reports")


def _active_investigation_ids(nb: LabNotebook) -> set[str]:
    active: set[str] = set()
    for status in ("queued", "running"):
        for task in nb.get_followup_tasks(
            stage="investigation",
            status=status,
            limit=500,
        ):
            for result_id in task.get("result_ids_json") or []:
                rid = str(result_id or "").strip()
                if rid:
                    active.add(rid)
    return active


def _candidate_result_id(candidate: dict[str, Any]) -> str:
    return str(candidate.get("result_id") or "").strip()


def _chunks(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    size = max(1, int(size))
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


def queue_followups(
    db_path: str,
    *,
    limit: int = 6,
    batch_size: int = 3,
    apply: bool = False,
    include_investigated: bool = False,
    device: str = "cuda",
    investigation_steps: int | None = None,
) -> dict[str, Any]:
    candidates, summary = build_queue(
        db_path,
        limit=max(limit * 4, limit),
        include_investigated=include_investigated,
    )
    nb = LabNotebook(db_path)
    try:
        active_ids = _active_investigation_ids(nb)
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        suppressed_active = 0
        missing_id = 0
        for candidate in candidates:
            result_id = _candidate_result_id(candidate)
            if not result_id:
                missing_id += 1
                continue
            canonical = str(nb.resolve_canonical_result_id(result_id) or result_id)
            if canonical in active_ids:
                suppressed_active += 1
                continue
            if canonical in seen:
                continue
            selected_candidate = dict(candidate)
            selected_candidate["result_id"] = canonical
            selected.append(selected_candidate)
            seen.add(canonical)
            if len(selected) >= limit:
                break

        config = RunConfig()
        config.device = device
        if investigation_steps is not None:
            config.investigation_steps = max(1, int(investigation_steps))
        config.gbm_prescreener_enabled = False
        config.allow_unproven_ml_influence = False

        queued_tasks: list[dict[str, Any]] = []
        for batch in _chunks(selected, batch_size):
            result_ids = [_candidate_result_id(item) for item in batch]
            priority_score = max(float(item.get("rank_score") or 0.0) for item in batch)
            priority_reasons = {
                "policy": "offline_investigation_queue",
                "rank_scores": {
                    _candidate_result_id(item): float(item.get("rank_score") or 0.0)
                    for item in batch
                },
                "missing_signals": {
                    _candidate_result_id(item): item.get("missing_signals") or []
                    for item in batch
                },
            }
            metadata = {
                "source_tool": "queue_investigation_followups",
                "queue_generated_at": summary.get("generated_at"),
                "uses_screening_ensemble_gate": False,
                "uses_learned_generation_influence": False,
                "candidate_ranks": {
                    _candidate_result_id(item): int(item.get("rank") or 0)
                    for item in batch
                },
            }
            task_id = None
            if apply:
                task_id = nb.enqueue_followup_task(
                    stage="investigation",
                    result_ids=result_ids,
                    hypothesis=(
                        "Targeted investigation queue: run v2 induction/binding "
                        "and validation follow-up for empirically strong S1 candidates "
                        "with missing investigation-tier evidence."
                    ),
                    config=config.to_dict(),
                    evidence_pack={
                        "queue_summary": summary,
                        "candidates": batch,
                    },
                    source_context="offline_investigation_queue",
                    priority_score=priority_score,
                    priority_reasons=priority_reasons,
                    metadata=metadata,
                )
            queued_tasks.append(
                {
                    "task_id": task_id,
                    "result_ids": result_ids,
                    "priority_score": priority_score,
                    "dry_run": not apply,
                    "candidate_ranks": metadata["candidate_ranks"],
                    "missing_signals": priority_reasons["missing_signals"],
                }
            )
    finally:
        nb.close()

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "db_path": db_path,
        "apply": bool(apply),
        "limit": int(limit),
        "batch_size": int(batch_size),
        "selected_count": len(selected),
        "queued_task_count": len(queued_tasks) if apply else 0,
        "dry_run_task_count": len(queued_tasks) if not apply else 0,
        "suppressed_active": suppressed_active,
        "missing_result_id": missing_id,
        "queue_summary": summary,
        "tasks": queued_tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=RUNS_DB)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--include-investigated", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--investigation-steps", type=int, default=None)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Output path without suffix. Defaults to research/reports/investigation_followups_YYYY-MM-DD.",
    )
    args = parser.parse_args()

    report = queue_followups(
        args.db,
        limit=max(1, int(args.limit)),
        batch_size=max(1, int(args.batch_size)),
        apply=bool(args.apply),
        include_investigated=bool(args.include_investigated),
        device=str(args.device or "cuda"),
        investigation_steps=args.investigation_steps,
    )
    prefix = (
        Path(args.output_prefix)
        if args.output_prefix
        else REPORT_DIR
        / f"investigation_followups_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
    )
    task_rows = list(report.get("tasks") or [])
    report["output_kind"] = "followup_task_queue"
    json_path, jsonl_path = write_reports(
        task_rows,
        {k: v for k, v in report.items() if k != "tasks"},
        output_prefix=prefix,
    )
    print(
        f"{'Queued' if args.apply else 'Dry-run'} {len(task_rows)} "
        f"investigation follow-up task batches"
    )
    print(f"JSON:  {json_path}")
    print(f"JSONL: {jsonl_path}")
    for row in task_rows:
        print(
            f"task={row.get('task_id') or '-'} "
            f"priority={float(row.get('priority_score') or 0.0):.3f} "
            f"results={','.join(row.get('result_ids') or [])}"
        )


if __name__ == "__main__":
    main()
