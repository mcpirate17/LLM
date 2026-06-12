#!/usr/bin/env python
"""Grade existing lanes on adversarial episodic retention conditions."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from component_fab.generator.memory_primitives import (
    CausalFastWeightMemoryLane,
    DataDependentDecayMemoryLane,
    HierarchicalResidualCompressorLane,
)
from component_fab.harness.adversarial_retention import (
    DEFAULT_RETENTION_TASKS,
    RetentionTask,
    run_retention_task,
)
from component_fab.harness.tiny_lm import lane_factory_for_baseline

_REPO = Path(__file__).resolve().parents[2]
_REPORT_DIR = _REPO / "research" / "reports"

MODEL_FACTORIES: dict[str, Callable[[int], nn.Module]] = {
    "ddecay": DataDependentDecayMemoryLane,
    "fast_weight": CausalFastWeightMemoryLane,
    "hier_compress": HierarchicalResidualCompressorLane,
    "gpt2": lane_factory_for_baseline("gpt2"),
}


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _task_map() -> dict[str, RetentionTask]:
    return {task.name: task for task in DEFAULT_RETENTION_TASKS}


def run_grade(args: argparse.Namespace) -> dict[str, object]:
    model_names = _parse_csv(args.models)
    task_names = _parse_csv(args.tasks)
    seeds = tuple(int(seed) for seed in _parse_csv(args.seeds))
    unknown_models = sorted(set(model_names) - MODEL_FACTORIES.keys())
    unknown_tasks = sorted(set(task_names) - _task_map().keys())
    if unknown_models:
        raise ValueError(f"unknown models: {unknown_models}")
    if unknown_tasks:
        raise ValueError(f"unknown tasks: {unknown_tasks}")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    raw: dict[str, dict[str, list[dict[str, object]]]] = {}
    started = time.monotonic()
    for model_name in model_names:
        raw[model_name] = {}
        for task_name in task_names:
            task = _task_map()[task_name]
            rows: list[dict[str, object]] = []
            for seed in seeds:
                result = run_retention_task(
                    MODEL_FACTORIES[model_name],
                    task,
                    mixer_label=model_name,
                    dim=args.dim,
                    n_blocks=args.blocks,
                    n_train_steps=args.steps,
                    batch_size=args.batch_size,
                    learning_rate=args.lr,
                    n_eval_batches=args.eval_batches,
                    seed=seed,
                    device=device,
                )
                row = asdict(result)
                row["seed"] = seed
                rows.append(row)
            raw[model_name][task_name] = rows
            mean_acc = statistics.fmean(float(row["eval_accuracy"]) for row in rows)
            print(
                f"{model_name:14s} {task_name:30s} acc={mean_acc:.3f} "
                f"({time.monotonic() - started:.0f}s)",
                flush=True,
            )

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for model_name, by_task in raw.items():
        summary[model_name] = {}
        for task_name, rows in by_task.items():
            accuracies = [float(row["eval_accuracy"]) for row in rows]
            lifts = [float(row["chance_normalized_lift"]) for row in rows]
            summary[model_name][task_name] = {
                "accuracy_mean": statistics.fmean(accuracies),
                "accuracy_stdev": statistics.stdev(accuracies)
                if len(accuracies) > 1
                else 0.0,
                "lift_mean": statistics.fmean(lifts),
                "lift_stdev": statistics.stdev(lifts) if len(lifts) > 1 else 0.0,
                "chance": float(rows[0]["chance_accuracy"]),
            }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "dim": args.dim,
        "n_blocks": args.blocks,
        "train_steps": args.steps,
        "batch_size": args.batch_size,
        "eval_batches": args.eval_batches,
        "seeds": list(seeds),
        "models": list(model_names),
        "tasks": [asdict(_task_map()[name]) for name in task_names],
        "summary": summary,
        "raw": raw,
        "elapsed_s": round(time.monotonic() - started, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="ddecay,fast_weight,hier_compress,gpt2")
    parser.add_argument(
        "--tasks", default=",".join(task.name for task in DEFAULT_RETENTION_TASKS)
    )
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    report = run_grade(args)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = args.out or (_REPORT_DIR / f"adversarial_retention_{stamp}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
