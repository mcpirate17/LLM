#!/usr/bin/env python
"""Grade matched model registries on hardened corrected binding tasks."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from component_fab.harness.binding_validity import (
    BINDING_VALIDITY_VERSION,
    HARD_BINDING_VALIDITY_TASKS,
    BindingValidityTask,
    binding_validity_load_ladder,
    run_binding_validity_task,
)
from research.tools.grade_matched_corrected import DIM, _build_registry

_REPORT_DIR = Path(__file__).resolve().parents[2] / "research" / "reports"


def _tasks(task_set: str) -> tuple[BindingValidityTask, ...]:
    if task_set == "hard":
        return HARD_BINDING_VALIDITY_TASKS
    if task_set == "load":
        return binding_validity_load_ladder()
    raise ValueError(f"unknown task set: {task_set}")


def run_grade(args: argparse.Namespace) -> dict[str, Any]:
    tasks = _tasks(args.task_set)
    seeds = tuple(range(args.seed, args.seed + args.seed_count))
    registry = _build_registry(args.mode)
    selected = tuple(
        name.strip() for name in args.models.split(",") if name.strip()
    )
    unknown = sorted(set(selected) - registry.keys())
    if unknown:
        raise ValueError(f"unknown models: {unknown}")
    registry = {name: registry[name] for name in selected}
    if args.dry_run:
        return {
            "dry_run": True,
            "task_semantics_version": BINDING_VALIDITY_VERSION,
            "mode": args.mode,
            "tasks": [asdict(task) for task in tasks],
            "models": {
                name: {"params": params, "mflops_L64": flops / 1e6}
                for name, (_, params, flops) in registry.items()
            },
        }

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    started = time.monotonic()
    rows: dict[str, Any] = {}
    for model_index, (name, (factory, params, flops)) in enumerate(
        registry.items(), 1
    ):
        by_task: dict[str, Any] = {}
        for task in tasks:
            accuracies: list[float] = []
            for seed in seeds:
                result = run_binding_validity_task(
                    factory,
                    task,
                    mixer_label=name,
                    dim=DIM,
                    n_train_steps=args.steps,
                    batch_size=args.batch_size,
                    n_eval_batches=args.eval_batches,
                    seed=seed,
                    device=device,
                )
                accuracies.append(result.eval_accuracy)
            by_task[task.name] = {
                "accuracy_mean": statistics.fmean(accuracies),
                "accuracy_stdev": (
                    statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
                ),
                "chance": task.chance_accuracy,
                "seed_accuracies": accuracies,
            }
            print(
                f"[{model_index}/{len(registry)}] {name:16s} "
                f"{task.name:46s} acc={statistics.fmean(accuracies):.3f}",
                flush=True,
            )
        rows[name] = {
            "params": params,
            "mflops_L64": round(flops / 1e6, 3),
            "task_mean": statistics.fmean(
                row["accuracy_mean"] for row in by_task.values()
            ),
            "per_task": by_task,
        }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_semantics_version": BINDING_VALIDITY_VERSION,
        "task_set": args.task_set,
        "resource_match_policy": args.mode,
        "device": device,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "eval_batches": args.eval_batches,
        "seeds": list(seeds),
        "tasks": [asdict(task) for task in tasks],
        "rows": rows,
        "elapsed_s": round(time.monotonic() - started, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-set", choices=("hard", "load"), default="hard")
    parser.add_argument("--mode", choices=("params", "flops"), default="params")
    parser.add_argument(
        "--models", default="softmax_4h,legendre_ssm,mamba2,softmax_1h"
    )
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    report = run_grade(args)
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = args.out or (
        _REPORT_DIR
        / f"hardened_corrected_{args.task_set}_{args.mode}_{stamp}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
