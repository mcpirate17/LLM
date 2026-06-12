"""#7 defensible re-grade: matched-param lanes on codex's CORRECTED tasks.

Fixes BOTH confounds the matrix had at once:
- benchmark: grades on `binding_validity` corrected episodic tasks (codex) —
  unique_multi_query / distinct_key_interference (the REAL distractor) /
  same_key_overwrite / episodic_compositional — not the ambiguous legacy 6.
- model: sizes every tunable lane to a common ~16K mixer-param budget (the
  attention level) at fixed dim64, so accuracy is read at matched capacity.

Frontier attention (gpt2, 4-head) and SSM (mamba2) are graded at their natural
size with params disclosed (their structure isn't freely resizable). FLOPs can't
be matched across O(L^2) attention vs O(L*m) memory — params are matched, FLOPs
disclosed (see fairness_audit.py). Budget matched: 1500 steps, Adam 3e-3, 3 seeds.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from component_fab.generator.memory_primitives import (
    CausalFastWeightMemoryLane,
    DataDependentDecayMemoryLane,
    HierarchicalResidualCompressorLane,
    LegendreSSMLane,
    PowerSemiringMemoryLane,
)
from component_fab.harness.binding_validity import (
    BINDING_VALIDITY_VERSION,
    DEFAULT_BINDING_VALIDITY_TASKS,
    run_binding_validity_task,
)
from component_fab.harness.tiny_lm import lane_factory_for_baseline

_REPORT = Path(__file__).resolve().parents[2] / "research" / "reports"
TARGET_PARAMS = 16384
TARGET_FLOPS = 3_150_000  # softmax_4h matmul-FLOPs @ dim64, L64 (the task seq_len)
FLOP_REF_L = 64
DIM = 64


def _params(lane: nn.Module) -> int:
    return sum(p.numel() for p in lane.parameters())


def _flops(lane: nn.Module, seq_len: int = FLOP_REF_L) -> int:
    from torch.utils.flop_counter import FlopCounterMode

    lane.eval()
    x = torch.randn(1, seq_len, DIM)
    try:
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            lane(x)
        return int(fc.get_total_flops())
    except Exception:  # noqa: BLE001
        return -1


def _size(
    make: Callable[[int, int], nn.Module], knobs: list[int], mode: str
) -> tuple[int, int]:
    """Pick the knob closest to the param (mode='params') or FLOP target."""
    target = TARGET_PARAMS if mode == "params" else TARGET_FLOPS
    measure = _params if mode == "params" else _flops
    best: tuple[int, int] | None = None
    for kv in knobs:
        try:
            val = measure(make(DIM, kv))
        except Exception:  # noqa: BLE001
            continue
        if val < 0:
            continue
        if best is None or abs(val - target) < abs(best[1] - target):
            best = (kv, val)
    assert best is not None
    return best


def _build_registry(
    mode: str,
) -> dict[str, tuple[Callable[[int], nn.Module], int, int]]:
    mem = list(range(16, 257, 8))
    reg: dict[str, tuple[Callable[[int], nn.Module], int, int]] = {}

    # attention head axis (1h vs 4h) + frontier SSM — fixed structure, disclosed
    from component_fab.harness.tiny_lm import MultiHeadCausalAttention

    def _ref(
        factory: Callable[[int], nn.Module],
    ) -> tuple[Callable[[int], nn.Module], int, int]:
        m = factory(DIM)
        return factory, _params(m), _flops(m)

    reg["softmax_1h"] = _ref(lambda d: MultiHeadCausalAttention(d, n_heads=1))
    reg["softmax_4h"] = _ref(lambda d: MultiHeadCausalAttention(d, n_heads=4))
    reg["mamba2"] = _ref(lambda d: lane_factory_for_baseline("mamba2")(d))

    from research.tools.gemini_slot_snapshot import GeminiSlotMemoryLane
    from research.tools.gemini_master_snapshot import UniversalMasterLane
    from component_fab.generator.memory_primitives import (
        MultiHeadSlotTableMemoryLane,
        SlotTableMemoryLane,
    )

    for name, cls, knob_name, knobs in [
        ("ddecay", DataDependentDecayMemoryLane, "memory_dim", mem),
        ("fast_weight", CausalFastWeightMemoryLane, "memory_dim", mem),
        ("power_semiring", PowerSemiringMemoryLane, "memory_dim", mem),
        ("legendre_ssm", LegendreSSMLane, "state_dim", list(range(16, 513, 16))),
        ("hier_compress", HierarchicalResidualCompressorLane, "n_levels", [1, 2, 3, 4]),
        ("gemini_slot", GeminiSlotMemoryLane, "memory_dim", mem),
        ("gemini_master", UniversalMasterLane, "memory_dim", mem),
        ("slot_table", SlotTableMemoryLane, "memory_dim", mem),
        ("slot_table_mh", MultiHeadSlotTableMemoryLane, "memory_dim", mem),
    ]:

        def make(
            d: int,
            kv: int,
            c: type[nn.Module] = cls,
            kn: str = knob_name,
        ) -> nn.Module:
            return c(d, **{kn: kv})

        knob, _ = _size(make, knobs, mode)

        def fac(
            d: int,
            c: type[nn.Module] = cls,
            kn: str = knob_name,
            kv: int = knob,
        ) -> nn.Module:
            return c(d, **{kn: kv})

        built = fac(DIM)
        reg[name] = (fac, _params(built), _flops(built))
        print(
            f"  sized {name:16s} {knob_name}={knob:<4d} -> {_params(built)} params, "
            f"{_flops(built) / 1e6:.2f} MFLOP@L{FLOP_REF_L}",
            flush=True,
        )
    return reg


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["params", "flops"], default="params")
    args = ap.parse_args()
    seeds = (0, 1, 2)
    steps = 800  # corrected tasks separate fast (codex: gpt2=1.0 by ~200 steps)
    target = TARGET_PARAMS if args.mode == "params" else TARGET_FLOPS
    print(f"sizing lanes to matched {args.mode} (~{target}):")
    registry = _build_registry(args.mode)
    tasks = DEFAULT_BINDING_VALIDITY_TASKS
    task_names = [t.name for t in tasks]

    started = time.monotonic()
    rows: dict[str, Any] = {}
    n = len(registry)
    for i, (name, (factory, params, flops)) in enumerate(registry.items(), 1):
        per_task = {}
        for task in tasks:
            accs = []
            for s in seeds:
                r = run_binding_validity_task(
                    factory,
                    task,
                    mixer_label=name,
                    dim=DIM,
                    n_train_steps=steps,
                    seed=s,
                    device="cuda",
                )
                accs.append(r.eval_accuracy)
            per_task[task.name] = sum(accs) / len(accs)
        avg = sum(per_task.values()) / len(per_task)
        rows[name] = {
            "params": params,
            "mflops_L64": round(flops / 1e6, 2),
            "avg": avg,
            "per_task": per_task,
        }
        print(
            f"[{i}/{n}] {name:16s} p={params:>6d} f={flops / 1e6:>5.1f}M avg={avg:.3f}  "
            + " ".join(f"{t.split('_')[-1][:6]}={a:.2f}" for t, a in per_task.items())
            + f"  ({time.monotonic() - started:.0f}s)",
            flush=True,
        )

    out = _REPORT / f"matched_corrected_regrade_{args.mode}.json"
    out.write_text(
        json.dumps(
            {
                "mode": args.mode,
                "target": target,
                "task_semantics_version": BINDING_VALIDITY_VERSION,
                "steps": steps,
                "seeds": list(seeds),
                "task_names": task_names,
                "rows": rows,
            },
            indent=1,
        )
    )
    print(f"\n[report -> {out}]  ({time.monotonic() - started:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
