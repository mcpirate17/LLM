"""Router-prior (codex #2): the one static-key mechanism orthogonal to RMSNorm.

Read-side static keys are subsumed by RMSNorm (exp_static_key_fixes: all variants net-negative,
none beats base on interference). Router-prior is different — it biases WRITE routing toward
learned per-slot prototypes (stable slot specialization) while leaving the dynamic content READ
untouched. RMSNorm is a read-side value normalization; this is a write-side routing prior, so it
could add what RMSNorm doesn't.

Originally tested via a `_refine_route` override with null-write OFF, which showed the prior is a
real interference lever but could not answer whether it stacks with null-write. This runner now
uses the lane-native pre-gate logit bias and can run only the current stacked variant.
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


def _factory(*, use_router_prior: bool):
    def make(d: int):
        memory_dim = max(4, ((7 * d) // 32) * 4)
        return MultiHeadSlotTableMemoryLane(
            d,
            memory_dim=memory_dim,
            use_delta_update=False,
            route_from_input=True,
            normalize_slot_values=True,
            use_null_write=True,
            use_router_prior=use_router_prior,
        )

    return make


MODELS = {
    "prod_base": _factory(use_router_prior=False),
    "prod_router_prior": _factory(use_router_prior=True),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed-count", type=int, default=5)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--variant",
        choices=tuple(MODELS),
        action="append",
        help="Run only the selected current-lane variant; repeat to select multiple.",
    )
    ap.add_argument(
        "--out", type=Path, default=Path("research/reports/exp_router_prior.json")
    )
    args = ap.parse_args()

    rows: dict[str, dict] = {}
    selected = args.variant or list(MODELS)
    for name in selected:
        fac = MODELS[name]
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
        print(f"  {name:12s} mean={mean:.3f}  {short}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"seeds": args.seed_count, "rows": rows}, indent=2))
    if {"prod_base", "prod_router_prior"} <= rows.keys():
        d = rows["prod_router_prior"]["task_mean"] - rows["prod_base"]["task_mean"]
        print(f"\n  Δ prod_router_prior vs prod_base: {d:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
