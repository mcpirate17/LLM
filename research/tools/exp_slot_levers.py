"""Two more untested levers from codex's list, as standalone subclasses (no lane edits).

* Output gating: a per-head sigmoid gate on the read, so the lane can suppress the slot read on
  positions with no relevant write (PAD/NOISE/QUERY). Targets contamination / interference.
* Static learned slot keys with content-only values: address fixed learned per-head/per-slot key
  prototypes instead of the running-mean key; values stay the content running-mean. A learned
  codebook rather than content-derived addressing.

Graded vs the locked base on HARD tasks, 10 seeds. (Empty-slot masking is N/A under soft routing;
value-aware routing, heterogeneous slot counts, and global/local hierarchy need forward changes /
a lane flag and are deferred to codex.)
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from torch import nn

from component_fab.generator.memory_primitives import MultiHeadSlotTableMemoryLane
from component_fab.harness.binding_validity import (
    HARD_BINDING_VALIDITY_TASKS,
    run_binding_validity_task,
)


class OutputGateLane(MultiHeadSlotTableMemoryLane):
    def __init__(self, *a, **k) -> None:
        super().__init__(*a, **k)
        self.read_gate = nn.Linear(self.head_dim, 1)

    def _read_slots(self, q, slot_key, slot_val):
        read = super()._read_slots(q, slot_key, slot_val)
        return read * torch.sigmoid(self.read_gate(q))


class StaticKeyLane(MultiHeadSlotTableMemoryLane):
    def __init__(self, *a, **k) -> None:
        super().__init__(*a, **k)
        self.static_key = nn.Parameter(
            torch.randn(self.n_heads, self.n_slots, self.head_dim) * 0.02
        )

    def _read_slots(self, q, slot_key, slot_val):
        b, seq = q.shape[0], q.shape[1]
        sk = self.static_key.view(1, 1, self.n_heads, self.n_slots, self.head_dim)
        sk = sk.expand(b, seq, -1, -1, -1)
        return super()._read_slots(q, sk, slot_val)


def _factory(cls):
    def make(d: int):
        memory_dim = max(4, ((7 * d) // 32) * 4)
        return cls(
            d, memory_dim=memory_dim, use_delta_update=False, route_from_input=True
        )

    return make


MODELS = {
    "locked_base": _factory(MultiHeadSlotTableMemoryLane),
    "output_gate": _factory(OutputGateLane),
    "static_key": _factory(StaticKeyLane),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed-count", type=int, default=10)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--out", type=Path, default=Path("research/reports/exp_slot_levers.json")
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
        print(f"  {name:12s} mean={mean:.3f}  {short}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"seeds": args.seed_count, "rows": rows}, indent=2))
    base = rows["locked_base"]["task_mean"]
    for name in MODELS:
        if name != "locked_base":
            print(f"  Δ {name} vs base: {rows[name]['task_mean'] - base:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
