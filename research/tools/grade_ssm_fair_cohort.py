#!/usr/bin/env python
"""SSM-fair cohort: rank non-QKV mechanisms on the state-tracking axis.

The Tier-2 binding suite is all key-value recall (attention-favoured by
construction), so it cannot say whether component_fab's non-QKV mechanisms —
routing, compression, state-space / memory — are the win. This cohort grades
them on the complementary axis: the ``probe_tasks`` MSE state-tracking / copy /
compression battery (``score_state_tracking``), with attention (softmax / gpt2)
included only as the contrast that *should* lose on state-tracking. Output is a
2-D Pareto profile (state_tracking vs copy_compression vs recall_induction),
ranked by continuous MSE loss-reduction — not a single "beats frontier" bit.

Tiny probes (dim32, seq32, ~100 steps) → CPU, fast, no GPU contention.

Usage:
    python -m research.tools.grade_ssm_fair_cohort --seeds 0,1,2 --steps 150
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import time
from pathlib import Path
from typing import Any, Callable

from torch import nn

from component_fab.generator.memory_primitives import (
    CausalFastWeightMemoryLane,
    DataDependentDecayMemoryLane,
    HierarchicalResidualCompressorLane,
    LegendreSSMLane,
    PowerSemiringMemoryLane,
)
from component_fab.generator.routing_primitives import SparseMoRLane
from component_fab.harness.state_tracking_suite import AXES, score_state_tracking
from component_fab.harness.tiny_lm import CausalConv1dLane, lane_factory_for_baseline

_REPO = Path(__file__).resolve().parents[2]
_REPORT_DIR = _REPO / "research" / "reports"

# kind: "candidate" = non-QKV mechanism fab drives; "contrast" = attention bar.
MODELS: dict[str, tuple[str, Callable[[int], nn.Module]]] = {
    # state-space / memory / alternative structures
    "power_semiring": ("candidate", lambda d: PowerSemiringMemoryLane(d)),
    "ddecay_memory": ("candidate", lambda d: DataDependentDecayMemoryLane(d)),
    "legendre_ssm": ("candidate", lambda d: LegendreSSMLane(d)),
    "fast_weight": ("candidate", lambda d: CausalFastWeightMemoryLane(d)),
    "gated_delta": ("candidate", lane_factory_for_baseline("mamba2")),
    "selective_scan": ("candidate", lane_factory_for_baseline("mamba")),
    # compression
    "hier_compress": ("candidate", lambda d: HierarchicalResidualCompressorLane(d)),
    # routing (MoR over a local-conv base — routing is the tested variable)
    "sparse_mor_conv": (
        "candidate",
        lambda d: SparseMoRLane(lambda dd: CausalConv1dLane(dd), d),
    ),
    # attention contrast (should LOSE on state-tracking)
    "softmax_attn": ("contrast", lane_factory_for_baseline("softmax_attention")),
    "gpt2": ("contrast", lane_factory_for_baseline("gpt2")),
}


def _run(args: argparse.Namespace) -> dict[str, Any]:
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    rows: dict[str, Any] = {}
    started = time.monotonic()
    for i, (name, (kind, factory)) in enumerate(MODELS.items(), 1):
        t0 = time.monotonic()
        score = score_state_tracking(
            factory,
            dim=args.dim,
            seq_len=args.seq_len,
            n_steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            seeds=seeds,
            device=args.device,
        )
        rows[name] = {"kind": kind, **score}
        print(
            f"[{i}/{len(MODELS)}] {name:16s} ({kind}) "
            + " ".join(f"{a}={score['per_axis'].get(a, 0):.2f}" for a in AXES)
            + f"  ({time.monotonic() - t0:.0f}s)",
            flush=True,
        )
    return {
        "dim": args.dim,
        "seq_len": args.seq_len,
        "n_steps": args.steps,
        "seeds": list(seeds),
        "device": args.device,
        "axes": {a: list(t) for a, t in AXES.items()},
        "models": rows,
        "elapsed_s": round(time.monotonic() - started, 1),
    }


def _print_leaderboard(report: dict[str, Any]) -> None:
    rows = report["models"]
    axis_names = list(report["axes"])
    print("\n=== SSM-fair leaderboard (MSE loss-reduction ratio, higher=better) ===")
    print(
        f"{'model':16s} {'kind':9s} " + " ".join(f"{a[:12]:>13s}" for a in axis_names)
    )
    for name, r in sorted(
        rows.items(),
        key=lambda kv: kv[1]["per_axis"].get("state_tracking", 0.0),
        reverse=True,
    ):
        cells = " ".join(f"{r['per_axis'].get(a, 0.0):13.2f}" for a in axis_names)
        print(f"{name:16s} {r['kind']:9s} {cells}")
    print("\nper-task ratio:")
    task_order = [t for a in axis_names for t in report["axes"][a]]
    print(f"{'model':16s} " + " ".join(f"{t[:8]:>9s}" for t in task_order))
    for name, r in rows.items():
        cells = " ".join(
            f"{r['per_task'].get(t, {}).get('ratio', 0):9.2f}" for t in task_order
        )
        print(f"{name:16s} {cells}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)

    report = _run(args)
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = args.output or (_REPORT_DIR / f"ssm_fair_cohort_{stamp}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    _print_leaderboard(report)
    print(f"\n[report -> {out}]  ({report['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
