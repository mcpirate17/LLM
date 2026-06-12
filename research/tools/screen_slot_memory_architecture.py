"""Screen composer-centered slot-memory architecture variants on HARD binding."""

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

DIM = 64
MEMORY_DIM = 56


def _factory(
    *,
    normalize_read: bool,
    use_query_lift: bool,
    route_from_input: bool,
    bilinear_read: bool = False,
    refine_write_route: bool = False,
    consolidate_slots: bool = False,
) -> Callable[[int], nn.Module]:
    def factory(dim: int) -> nn.Module:
        return MultiHeadSlotTableMemoryLane(
            dim,
            memory_dim=MEMORY_DIM,
            n_slots=8,
            n_heads=4,
            use_null_write=True,
            use_composer=True,
            use_delta_update=False,
            normalize_read=normalize_read,
            grouped_router=False,
            use_query_lift=use_query_lift,
            route_from_input=route_from_input,
            bilinear_read=bilinear_read,
            refine_write_route=refine_write_route,
            consolidate_slots=consolidate_slots,
        )

    return factory


def _factorial_configs() -> dict[str, Callable[[int], nn.Module]]:
    configs: dict[str, Callable[[int], nn.Module]] = {}
    for normalize_read in (False, True):
        for use_query_lift in (False, True):
            for route_from_input in (False, True):
                label = (
                    f"norm{int(normalize_read)}_"
                    f"qlift{int(use_query_lift)}_"
                    f"route{'input' if route_from_input else 'k'}"
                )
                configs[label] = _factory(
                    normalize_read=normalize_read,
                    use_query_lift=use_query_lift,
                    route_from_input=route_from_input,
                )
    configs["norm1_qlift0_routeinput_bilinear"] = _factory(
        normalize_read=True,
        use_query_lift=False,
        route_from_input=True,
        bilinear_read=True,
    )
    configs["norm1_routeinput_content_route"] = _factory(
        normalize_read=True,
        use_query_lift=False,
        route_from_input=True,
        refine_write_route=True,
    )
    configs["norm1_routeinput_consolidate"] = _factory(
        normalize_read=True,
        use_query_lift=False,
        route_from_input=True,
        consolidate_slots=True,
    )
    configs["norm1_routeinput_content_route_consolidate"] = _factory(
        normalize_read=True,
        use_query_lift=False,
        route_from_input=True,
        refine_write_route=True,
        consolidate_slots=True,
    )
    return configs


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--only", default="")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("research/reports/screen_slot_memory_architecture.json"),
    )
    args = parser.parse_args()
    seeds = tuple(int(seed) for seed in _parse_csv(args.seeds))
    if not seeds:
        raise ValueError("at least one seed is required")

    configs = _factorial_configs()
    selected = set(_parse_csv(args.only))
    unknown = selected - configs.keys()
    if unknown:
        raise ValueError(f"unknown configs: {sorted(unknown)}")
    if selected:
        configs = {
            name: factory for name, factory in configs.items() if name in selected
        }

    started = time.monotonic()
    rows: dict[str, dict[str, object]] = {}
    for label, factory in configs.items():
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
            float(task_result["mean"]) for task_result in per_task.values()
        )
        params = sum(parameter.numel() for parameter in factory(DIM).parameters())
        rows[label] = {
            "params": params,
            "task_mean": task_mean,
            "per_task": per_task,
        }
        print(
            f"{label:28s} params={params:5d} mean={task_mean:.3f} "
            f"elapsed={time.monotonic() - started:.0f}s",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "task_semantics_version": BINDING_VALIDITY_VERSION,
                "steps": args.steps,
                "seeds": list(seeds),
                "dim": DIM,
                "memory_dim": MEMORY_DIM,
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
