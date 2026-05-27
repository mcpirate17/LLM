"""Colab-friendly AR-gate backfill with multi-seed support.

This is separate from the legacy ``ar`` probe. It writes the active
``ar_gate_*`` columns on ``graph_runs`` and keeps an append-only JSONL report
next to the Colab bundle so progress survives notebook disconnects.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import time
from pathlib import Path
from typing import Any

from research.eval.ar_gate import ARGateConfig, ar_gate, ar_gate_is_no_go


def _parse_seeds(raw: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def _load_candidate_rows(path: Path, *, limit: int | None) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"candidate JSONL missing or empty: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            row["_candidate_idx"] = len(rows) + 1
            rows.append(row)
            if limit is not None and limit > 0 and len(rows) >= limit:
                break
    return rows


def _completed_result_ids(paths: list[Path]) -> set[str]:
    completed: set[str] = set()
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("status") != "computed":
                    continue
                result_id = str(record.get("result_id") or "")
                if result_id:
                    completed.add(result_id)
    return completed


def _mean(values: list[float]) -> float:
    return round(float(st.mean(values)), 6) if values else 0.0


def _run_one(
    row: dict[str, Any],
    *,
    device: str,
    seeds: tuple[int, ...],
    warmup_steps: int,
    finetune_steps: int,
    timeout_s: float,
    n_pairs_per_noun: int,
    reps: int,
    n_distractors: int,
    n_adjectives: int,
    n_objects: int,
) -> dict[str, Any]:
    graph_json = str(row["graph_json"])
    seed_results = []
    for seed in seeds:
        cfg = ARGateConfig(
            seed=int(seed),
            wikitext_warmup_steps=int(warmup_steps),
            finetune_steps=int(finetune_steps),
            n_pairs_per_noun=int(n_pairs_per_noun),
            reps=int(reps),
            n_distractors=int(n_distractors),
            n_adjectives=int(n_adjectives),
            n_objects=int(n_objects),
            timeout_s=float(timeout_s),
            from_s1=False,
        )
        result = ar_gate(graph_json=graph_json, device=device, cfg=cfg)
        seed_results.append(
            {
                "seed": seed,
                "metric_version": result.metric_version,
                "in_dist_pair_acc": result.in_dist_pair_acc,
                "in_dist_class_acc": result.in_dist_class_acc,
                "held_pair_acc": result.held_pair_acc,
                "held_class_acc": result.held_class_acc,
                "score": round(
                    0.6 * result.in_dist_pair_acc + 0.4 * result.held_class_acc,
                    6,
                ),
                "status": result.status,
                "error": result.error,
                "elapsed_ms": result.elapsed_ms,
                "finetune_steps_done": result.finetune_steps_done,
                "no_go": int(ar_gate_is_no_go(result)),
            }
        )

    ok = [r for r in seed_results if r["status"] == "ok"]
    used = ok if ok else seed_results
    payload = {
        "ar_gate_metric_version": (
            f"ar_gate_v0_colab_{len(seeds)}seed_w{int(warmup_steps)}_ft{int(finetune_steps)}"
        ),
        "ar_gate_in_dist_pair_acc": _mean([r["in_dist_pair_acc"] for r in used]),
        "ar_gate_in_dist_class_acc": _mean([r["in_dist_class_acc"] for r in used]),
        "ar_gate_held_pair_acc": _mean([r["held_pair_acc"] for r in used]),
        "ar_gate_held_class_acc": _mean([r["held_class_acc"] for r in used]),
        "ar_gate_score": _mean([r["score"] for r in used]),
        "ar_gate_status": "ok" if ok else "all_failed",
        "ar_gate_elapsed_ms": round(
            sum(float(r["elapsed_ms"] or 0.0) for r in seed_results), 3
        ),
        "ar_gate_train_steps_done": int(
            max(r["finetune_steps_done"] or 0 for r in seed_results)
        ),
        "ar_gate_no_go": int(all(int(r["no_go"]) for r in ok)) if ok else None,
    }
    return {
        "result_id": row["result_id"],
        "graph_fingerprint": row["graph_fingerprint"],
        "tier": row["tier"],
        "priority_score": row["priority_score"],
        "seed_results": seed_results,
        "payload": payload,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-jsonl", type=Path, required=True)
    parser.add_argument("--report-jsonl", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seeds", type=_parse_seeds, default=(0, 1, 2))
    parser.add_argument("--wikitext-warmup-steps", type=int, default=0)
    parser.add_argument("--finetune-steps", type=int, default=400)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--n-pairs-per-noun", type=int, default=1)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--n-distractors", type=int, default=480)
    parser.add_argument("--n-adjectives", type=int, default=20)
    parser.add_argument("--n-objects", type=int, default=25)
    parser.add_argument(
        "--load-processed-from-report",
        type=Path,
        action="append",
        default=[],
        help="Additional JSONL report to use when skipping completed result IDs.",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.report_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows = _load_candidate_rows(args.candidates_jsonl, limit=args.limit)
    skipped = 0
    if not args.no_resume:
        resume_reports = [args.report_jsonl, *args.load_processed_from_report]
        completed = _completed_result_ids(resume_reports)
        if completed:
            before = len(rows)
            rows = [
                row for row in rows if str(row.get("result_id") or "") not in completed
            ]
            skipped = before - len(rows)
    print(
        "ar_gate rows=%d skipped_completed=%d seeds=%s warmup=%d ft=%d"
        % (
            len(rows),
            skipped,
            args.seeds,
            args.wikitext_warmup_steps,
            args.finetune_steps,
        ),
        flush=True,
    )
    if args.dry_run:
        return

    t0 = time.perf_counter()
    ok = 0
    with args.report_jsonl.open("a", encoding="utf-8") as report:
        for idx, row in enumerate(rows, start=1):
            try:
                record = _run_one(
                    row,
                    device=args.device,
                    seeds=args.seeds,
                    warmup_steps=args.wikitext_warmup_steps,
                    finetune_steps=args.finetune_steps,
                    timeout_s=args.timeout_s,
                    n_pairs_per_noun=args.n_pairs_per_noun,
                    reps=args.reps,
                    n_distractors=args.n_distractors,
                    n_adjectives=args.n_adjectives,
                    n_objects=args.n_objects,
                )
                ok += 1
                record["status"] = "computed"
                score = record["payload"].get("ar_gate_score")
                print(
                    "[remaining %d/%d source %s] %s ar_gate=%s status=computed"
                    % (
                        idx,
                        len(rows),
                        row.get("_candidate_idx", "?"),
                        row["result_id"][:12],
                        score,
                    ),
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                record = {
                    "result_id": row["result_id"],
                    "graph_fingerprint": row["graph_fingerprint"],
                    "status": "error",
                    "error": f"{type(exc).__name__}: {str(exc)[:240]}",
                }
                print(
                    "[remaining %d/%d source %s] %s error=%s"
                    % (
                        idx,
                        len(rows),
                        row.get("_candidate_idx", "?"),
                        row["result_id"][:12],
                        record["error"],
                    ),
                    flush=True,
                )
            record["idx"] = idx
            record["candidate_idx"] = row.get("_candidate_idx")
            record["total"] = len(rows)
            record["elapsed_s"] = round(time.perf_counter() - t0, 3)
            report.write(json.dumps(record, sort_keys=True) + "\n")
            report.flush()
    print(
        f"done ok={ok} total={len(rows)} elapsed_s={time.perf_counter() - t0:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
