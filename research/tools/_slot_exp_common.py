"""Shared runner for slot-lever experiment scripts (exp_slot_levers, exp_static_key_fixes).

Both scripts share an identical main-loop body:
  for model × task: collect per-seed accuracies → stats → print → write JSON → print deltas.

Only the MODELS dict and base_name differ between callers.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Callable

from torch import nn

from component_fab.harness.binding_validity import (
    HARD_BINDING_VALIDITY_TASKS,
    run_binding_validity_task,
)


def run_binding_validity_sweep(
    models: dict[str, Callable[[int], nn.Module]],
    *,
    base_name: str,
    seed_count: int,
    steps: int,
    device: str,
    out: Path,
) -> dict[str, dict]:
    """Evaluate every model in *models* over all HARD_BINDING_VALIDITY_TASKS.

    Parameters
    ----------
    models:
        Ordered mapping of ``label → factory(dim)``.
    base_name:
        Key in *models* used as the baseline for delta reporting.
    seed_count:
        Number of random seeds to average over.
    steps:
        Training steps per seed.
    device:
        PyTorch device string.
    out:
        Destination JSON path.  Parent directories are created automatically.

    Returns
    -------
    ``rows`` dict: label → {"task_mean": float, "per_task": {task_name: {"mean", "stdev"}}}.
    """
    if base_name not in models:
        raise ValueError(f"base_name {base_name!r} not found in models")

    rows: dict[str, dict] = {}
    for name, fac in models.items():
        per_task: dict[str, dict] = {}
        for task in HARD_BINDING_VALIDITY_TASKS:
            accs = [
                run_binding_validity_task(
                    fac,
                    task,
                    mixer_label=name,
                    dim=64,
                    n_train_steps=steps,
                    seed=seed,
                    device=device,
                ).eval_accuracy
                for seed in range(seed_count)
            ]
            per_task[task.name] = {
                "mean": statistics.fmean(accs),
                "stdev": statistics.pstdev(accs),
            }
        mean = statistics.fmean(v["mean"] for v in per_task.values())
        rows[name] = {"task_mean": mean, "per_task": per_task}
        short = {
            k.replace("hard_", "").split("_")[0]: round(v["mean"], 3)
            for k, v in per_task.items()
        }
        print(f"  {name:12s} mean={mean:.3f}  {short}", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"seeds": seed_count, "rows": rows}, indent=2))

    base_mean = rows[base_name]["task_mean"]
    for name in models:
        if name != base_name:
            print(
                f"  Δ {name} vs {base_name}: {rows[name]['task_mean'] - base_mean:+.3f}"
            )

    return rows
