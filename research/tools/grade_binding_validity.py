#!/usr/bin/env python
"""Audit legacy binding ambiguity and grade corrected episodic tasks."""

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
from component_fab.harness.binding_validity import (
    DEFAULT_BINDING_VALIDITY_TASKS,
    BindingValidityTask,
    audit_flat_writes,
    run_binding_validity_task,
)
from component_fab.harness.harder_binding_tasks import (
    _BATCH_GENERATORS,
    default_hard_binding_tasks,
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


def _legacy_audit(seed: int, examples: int) -> dict[str, object]:
    tasks = {task.name: task for task in default_hard_binding_tasks(seed=seed)}
    output: dict[str, object] = {}
    for name in (
        "multi_query_kv_recall",
        "distractor_kv_recall",
        "variable_layout_recall",
        "heldout_pair_recall",
    ):
        task = tasks[name]
        ids, _, _ = _BATCH_GENERATORS[name](
            task, examples, False, torch.Generator().manual_seed(seed)
        )
        write_pairs = task.n_pairs_in_seq * (1 + task.distractors_per_key)
        audit = audit_flat_writes(
            ids, n_write_pairs=write_pairs, key_upper_bound=task.n_keys
        )
        output[name] = asdict(audit) | {
            "duplicate_key_rate": audit.duplicate_key_rate,
            "conflicting_value_rate": audit.conflicting_value_rate,
        }
    output["semantic_findings"] = {
        "distractor_uses_exact_same_key": True,
        "compositional_value_rule": "(entity + 7 * attribute) % n_values",
        "compositional_values_resampled_per_episode": False,
    }
    return output


def run_grade(args: argparse.Namespace) -> dict[str, object]:
    model_names = _parse_csv(args.models)
    task_names = _parse_csv(args.tasks)
    seeds = tuple(int(seed) for seed in _parse_csv(args.seeds))
    task_map = {task.name: task for task in DEFAULT_BINDING_VALIDITY_TASKS}
    unknown_models = sorted(set(model_names) - MODEL_FACTORIES.keys())
    unknown_tasks = sorted(set(task_names) - task_map.keys())
    if unknown_models:
        raise ValueError(f"unknown models: {unknown_models}")
    if unknown_tasks:
        raise ValueError(f"unknown tasks: {unknown_tasks}")
    if not seeds:
        raise ValueError("at least one seed is required")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    started = time.monotonic()
    raw: dict[str, dict[str, list[dict[str, object]]]] = {}
    for model_name in model_names:
        raw[model_name] = {}
        for task_name in task_names:
            rows: list[dict[str, object]] = []
            task: BindingValidityTask = task_map[task_name]
            for seed in seeds:
                result = run_binding_validity_task(
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
                rows.append(asdict(result) | {"seed": seed})
            raw[model_name][task_name] = rows
            mean_accuracy = statistics.fmean(
                float(row["eval_accuracy"]) for row in rows
            )
            print(
                f"{model_name:14s} {task_name:36s} acc={mean_accuracy:.3f}",
                flush=True,
            )
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for model_name, by_task in raw.items():
        summary[model_name] = {}
        for task_name, rows in by_task.items():
            accuracies = [float(row["eval_accuracy"]) for row in rows]
            summary[model_name][task_name] = {
                "accuracy_mean": statistics.fmean(accuracies),
                "accuracy_stdev": (
                    statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
                ),
                "chance": float(rows[0]["chance_accuracy"]),
            }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "config": {
            "dim": args.dim,
            "blocks": args.blocks,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "eval_batches": args.eval_batches,
            "seeds": list(seeds),
        },
        "legacy_audit": _legacy_audit(seeds[0], args.audit_examples),
        "tasks": [asdict(task_map[name]) for name in task_names],
        "summary": summary,
        "raw": raw,
        "elapsed_s": round(time.monotonic() - started, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="ddecay,fast_weight,hier_compress,gpt2")
    parser.add_argument(
        "--tasks",
        default=",".join(task.name for task in DEFAULT_BINDING_VALIDITY_TASKS),
    )
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--audit-examples", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    report = run_grade(args)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = args.out or (_REPORT_DIR / f"binding_validity_{stamp}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
