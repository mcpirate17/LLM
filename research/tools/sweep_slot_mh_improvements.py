"""Cumulative ablation of multi-head slot-memory write and read improvements."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from torch import nn

from component_fab.generator.memory_primitives import MultiHeadSlotTableMemoryLane
from component_fab.harness.binding_validity import (
    BINDING_VALIDITY_VERSION,
    HARD_BINDING_VALIDITY_TASKS,
    run_binding_validity_task,
)
from component_fab.harness.tiny_lm import MultiHeadCausalAttention

MEMORY_DIM = 56
DIM = 64


def _slot_factory(
    *, memory_dim: int = MEMORY_DIM, **options: bool
) -> Callable[[int], nn.Module]:
    def factory(dim: int) -> nn.Module:
        return MultiHeadSlotTableMemoryLane(
            dim,
            memory_dim=memory_dim,
            n_slots=8,
            n_heads=4,
            **options,
        )

    return factory


CONFIGS: tuple[tuple[str, Callable[[int], nn.Module]], ...] = (
    (
        "baseline",
        _slot_factory(
            use_null_write=False,
            use_composer=False,
            use_delta_update=False,
            normalize_read=False,
        ),
    ),
    (
        "+null_write",
        _slot_factory(
            use_null_write=True,
            use_composer=False,
            use_delta_update=False,
            normalize_read=False,
        ),
    ),
    (
        "+composer",
        _slot_factory(
            use_null_write=True,
            use_composer=True,
            use_delta_update=False,
            normalize_read=False,
        ),
    ),
    (
        "+delta",
        _slot_factory(
            use_null_write=True,
            use_composer=True,
            use_delta_update=True,
            normalize_read=False,
        ),
    ),
    ("+normalized_read", _slot_factory()),
    (
        "+normalized_read_grouped_router_fixed_width",
        _slot_factory(grouped_router=True),
    ),
    (
        "+normalized_read_grouped_router_matched",
        _slot_factory(memory_dim=60, grouped_router=True),
    ),
    ("softmax_4h", lambda dim: MultiHeadCausalAttention(dim, n_heads=4)),
)


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated config labels to run; default runs every config.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("research/reports/sweep_slot_mh_improvements.json"),
    )
    args = parser.parse_args()
    seeds = _parse_seeds(args.seeds)

    started = time.monotonic()
    selected = {part.strip() for part in args.only.split(",") if part.strip()}
    unknown = selected - {label for label, _ in CONFIGS}
    if unknown:
        raise ValueError(f"unknown config labels: {sorted(unknown)}")
    configs = tuple(
        (label, factory)
        for label, factory in CONFIGS
        if not selected or label in selected
    )
    rows: dict[str, dict[str, object]] = {}
    for label, factory in configs:
        per_task: dict[str, dict[str, object]] = {}
        for task in HARD_BINDING_VALIDITY_TASKS:
            accuracies = [
                run_binding_validity_task(
                    factory,
                    task,
                    mixer_label=label,
                    dim=DIM,
                    n_train_steps=args.steps,
                    seed=seed,
                    device=args.device,
                ).eval_accuracy
                for seed in seeds
            ]
            per_task[task.name] = {
                "mean": statistics.fmean(accuracies),
                "stdev": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
                "seed_accuracies": accuracies,
            }
        task_mean = statistics.fmean(
            float(task_row["mean"]) for task_row in per_task.values()
        )
        params = sum(parameter.numel() for parameter in factory(DIM).parameters())
        rows[label] = {
            "params": params,
            "task_mean": task_mean,
            "per_task": per_task,
        }
        print(
            f"{label:18s} params={params:5d} task_mean={task_mean:.3f} "
            f"elapsed={time.monotonic() - started:.0f}s",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "task_semantics_version": BINDING_VALIDITY_VERSION,
                "memory_dim": MEMORY_DIM,
                "dim": DIM,
                "steps": args.steps,
                "seeds": list(seeds),
                "tasks": [asdict(task) for task in HARD_BINDING_VALIDITY_TASKS],
                "rows": rows,
                "elapsed_s": round(time.monotonic() - started, 1),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
