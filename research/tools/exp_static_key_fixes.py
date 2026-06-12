"""Gemini's static-key fixes, screened ON the RMSNorm production base (do they stack?).

static_key alone was net-negative (-0.080) but uniquely raised the interference holdout
(0.273->0.324). Gemini proposes keeping that interference benefit while restoring unique recall.
The three cleanly-subclassable fixes (others need forward/aux-loss changes):

  * anchor      : slot_key = content_key + static_key (elastic prototype, gemini's top pick)
  * gated       : slot_key = (1-g)*static + g*content, g = sigmoid(per-head param)
  * multiscale  : half the heads use static keys, half use content keys (global/local within heads)

All run with the production flags (normalize_slot_values + normalize_read + route_from_input,
delta off) and override only `_read_slots`, so super() still applies RMSNorm + normalized read.
Screened vs the production base, HARD tasks. Promotion bar: beat prod_base mean AND not regress
unique below ~0.95.
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


class _StaticBase(MultiHeadSlotTableMemoryLane):
    def __init__(self, *a, **k) -> None:
        super().__init__(*a, **k)
        self.static_key = nn.Parameter(
            torch.randn(self.n_heads, self.n_slots, self.head_dim) * 0.02
        )

    def _static(self, b: int, seq: int) -> torch.Tensor:
        return self.static_key.view(
            1, 1, self.n_heads, self.n_slots, self.head_dim
        ).expand(b, seq, -1, -1, -1)


class PureStaticLane(_StaticBase):
    def _read_slots(self, q, slot_key, slot_val):
        b, seq = q.shape[0], q.shape[1]
        return super()._read_slots(q, self._static(b, seq), slot_val)


class AnchorLane(_StaticBase):
    def _read_slots(self, q, slot_key, slot_val):
        b, seq = q.shape[0], q.shape[1]
        return super()._read_slots(q, slot_key + self._static(b, seq), slot_val)


class GatedLane(_StaticBase):
    def __init__(self, *a, **k) -> None:
        super().__init__(*a, **k)
        self.mix_gate = nn.Parameter(torch.zeros(self.n_heads))

    def _read_slots(self, q, slot_key, slot_val):
        b, seq = q.shape[0], q.shape[1]
        g = torch.sigmoid(self.mix_gate).view(1, 1, self.n_heads, 1, 1)
        sk = (1.0 - g) * self._static(b, seq) + g * slot_key
        return super()._read_slots(q, sk, slot_val)


class MultiScaleLane(_StaticBase):
    def _read_slots(self, q, slot_key, slot_val):
        b, seq = q.shape[0], q.shape[1]
        half = self.n_heads // 2
        static = self._static(b, seq)
        sk = torch.cat([static[:, :, :half], slot_key[:, :, half:]], dim=2)
        return super()._read_slots(q, sk, slot_val)


def _factory(cls):
    def make(d: int):
        memory_dim = max(4, ((7 * d) // 32) * 4)
        return cls(
            d,
            memory_dim=memory_dim,
            use_delta_update=False,
            route_from_input=True,
            normalize_slot_values=True,
        )

    return make


MODELS = {
    "prod_base": _factory(MultiHeadSlotTableMemoryLane),
    "pure_static": _factory(PureStaticLane),
    "anchor": _factory(AnchorLane),
    "gated": _factory(GatedLane),
    "multiscale": _factory(MultiScaleLane),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed-count", type=int, default=5)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--out", type=Path, default=Path("research/reports/exp_static_key_fixes.json")
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
    base = rows["prod_base"]["task_mean"]
    for name in MODELS:
        if name != "prod_base":
            print(f"  Δ {name} vs prod_base: {rows[name]['task_mean'] - base:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
