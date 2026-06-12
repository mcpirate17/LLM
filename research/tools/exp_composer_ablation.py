"""Is the causal composer THE mechanism that cracks content-binding?

The locked stack binds atomic [key,value] / [entity,attr,value] token groups via a causal
width-3 composer before routing. Hypothesis: that composer is the causal lever behind the
0.99 binding result. Ablate it (use_composer=False) from the otherwise-identical production
stack (RMSNorm + input-route + null-write) and grade on HARD tasks at the convergence budget.
If binding collapses without the composer, the mechanism is isolated for the writeup.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from component_fab.generator.memory_primitives import MultiHeadSlotTableMemoryLane
from component_fab.harness.binding_validity import (
    HARD_BINDING_VALIDITY_TASKS,
    run_binding_validity_task,
)


def _factory(*, use_composer: bool):
    def make(d: int):
        memory_dim = max(4, ((7 * d) // 32) * 4)
        return MultiHeadSlotTableMemoryLane(
            d,
            memory_dim=memory_dim,
            use_delta_update=False,
            route_from_input=True,
            normalize_slot_values=True,
            use_composer=use_composer,
        )

    return make


MODELS = {
    "locked_composer_on": _factory(use_composer=True),
    "locked_composer_off": _factory(use_composer=False),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed-count", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--out", type=Path, default=Path("research/reports/exp_composer_ablation.json")
    )
    args = ap.parse_args()

    rows: dict[str, dict] = {}
    for name, fac in MODELS.items():
        per_task: dict[str, dict] = {}
        for task in HARD_BINDING_VALIDITY_TASKS:
            accs = [
                run_binding_validity_task(
                    fac,
                    task,
                    mixer_label=name,
                    dim=64,
                    n_train_steps=args.steps,
                    seed=seed,
                    device=args.device,
                ).eval_accuracy
                for seed in range(args.seed_count)
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
        print(f"  {name:20s} mean={mean:.3f}  {short}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"seeds": args.seed_count, "steps": args.steps, "rows": rows}, indent=2
        )
    )
    on = rows["locked_composer_on"]["task_mean"]
    off = rows["locked_composer_off"]["task_mean"]
    print(f"\n  composer contribution: on={on:.3f} off={off:.3f} (Δ={on - off:+.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
