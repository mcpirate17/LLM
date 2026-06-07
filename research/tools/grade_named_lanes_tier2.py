#!/usr/bin/env python
"""Tier-2 6-task binding scoring for *named* lanes vs the frontier preset.

The fab Tier-2 engine (``run_tier2_binding_cohort``/``run_deep_probe``) only
accepts ledger ``proposal_id``s. This tool runs four hand-written mechanisms —
the branch's binder leads plus the MoR memory lane — directly through the same
6-task ``harder_binding_tasks`` suite against the frontier baselines
(softmax / gpt2 / mamba / mamba2):

    semiring            -> learnable_semiring_attention   (compiled mixer)
    reciprocal          -> reciprocal_rank_attention
    semiring_reciprocal -> reciprocal_semiring_attention
    native_semiring_mor -> MoRSemiringSurpriseMemoryLane

Binding only separates after thousands of steps, so each (model, task, seed) is
trained ONE trajectory and evaluated at every ``--eval-at`` checkpoint (default
2000 + 3000) — checkpoint at 2K, continue to 3K, no retraining. All 4 candidates
and 4 baselines are trained once per (task, seed); the candidate Δ is measured
against the best frontier baseline, per the official niche-survival rule.

Usage:
    python -m research.tools.grade_named_lanes_tier2 \
        --dim 64 --seeds 0,1,2 --eval-at 2000,3000 --device cuda
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from component_fab.generator.mor_surprise_memory import MoRSemiringSurpriseMemoryLane
from component_fab.harness.harder_binding_tasks import (
    default_hard_binding_tasks,
    run_one_task_checkpoints,
)
from component_fab.harness.tiny_lm import (
    FRONTIER_BASELINE_NAMES,
    lane_factory_for_baseline,
)
from research.tools.run_tier2_binding_cohort import _niche_survival

_REPO = Path(__file__).resolve().parents[2]
_REPORT_DIR = _REPO / "research" / "reports"


def _attention_mixer_factory(op: str) -> Callable[[int], nn.Module]:
    """A pure ``[B,L,D]`` mixer lane compiled from a single NAS attention op.

    Same construction as ``tiny_lm._compiled_reference_mixer_factory`` for the
    mamba/mamba2 baselines: build a one-op graph and take ``layers[0]`` (TinyLM
    wraps the norm/residual around it).
    """

    def factory(dim: int) -> nn.Module:
        from research.synthesis.compiler import compile_model
        from research.synthesis.graph import ComputationGraph

        g = ComputationGraph(dim)
        inp = g.add_input()
        g.set_output(g.add_op(op, [inp]))
        return compile_model([g], use_ir=False).layers[0]

    return factory


def _candidate_factories() -> dict[str, Callable[[int], nn.Module]]:
    return {
        "semiring": _attention_mixer_factory("learnable_semiring_attention"),
        "reciprocal": _attention_mixer_factory("reciprocal_rank_attention"),
        "semiring_reciprocal": _attention_mixer_factory(
            "reciprocal_semiring_attention"
        ),
        "native_semiring_mor": lambda dim: MoRSemiringSurpriseMemoryLane(dim),
    }


def _aggregate(
    # acc[step][model][task] -> list of eval_acc across seeds
    acc: dict[int, dict[str, dict[str, list[float]]]],
    candidate_names: list[str],
    baseline_names: tuple[str, ...],
    task_names: list[str],
) -> dict[int, dict[str, Any]]:
    """Per checkpoint, per candidate: Δ-vs-best-frontier on each task + survival."""
    out: dict[int, dict[str, Any]] = {}
    for step, by_model in acc.items():
        # best frontier baseline accuracy per task (mean over seeds), per seed too
        out[step] = {}
        for cand in candidate_names:
            per_task: dict[str, dict[str, Any]] = {}
            for task in task_names:
                cand_seeds = by_model.get(cand, {}).get(task, [])
                base_seeds = [by_model.get(b, {}).get(task, []) for b in baseline_names]
                n_seeds = len(cand_seeds)
                seed_deltas = []
                for i in range(n_seeds):
                    base_max = max(
                        (bs[i] for bs in base_seeds if i < len(bs)), default=0.0
                    )
                    seed_deltas.append(cand_seeds[i] - base_max)
                cand_mean = sum(cand_seeds) / n_seeds if cand_seeds else 0.0
                base_mean_max = (
                    sum(
                        max((bs[i] for bs in base_seeds if i < len(bs)), default=0.0)
                        for i in range(n_seeds)
                    )
                    / n_seeds
                    if n_seeds
                    else 0.0
                )
                mean_delta = sum(seed_deltas) / len(seed_deltas) if seed_deltas else 0.0
                per_task[task] = {
                    "candidate_eval_acc": cand_mean,
                    "baseline_max": base_mean_max,
                    "delta": mean_delta,
                    "beats": mean_delta > 0.0,
                    "seed_deltas": seed_deltas,
                }
            pass_count = sum(1 for v in per_task.values() if v["beats"])
            out[step][cand] = {
                "per_task": per_task,
                "pass_count": pass_count,
                "n_tasks": len(per_task),
                "tier2_passed_niche": _niche_survival(per_task),
                "mean_delta_vs_frontier": (
                    sum(v["delta"] for v in per_task.values()) / len(per_task)
                    if per_task
                    else 0.0
                ),
            }
    return out


def _print_summary(report: dict[str, Any]) -> None:
    baselines = ", ".join(report["baseline_names"])
    print(f"\nfrontier baselines: {baselines}")
    print(f"seeds={report['seeds']} dim={report['dim']} device={report['device']}")
    for step in report["eval_at_steps"]:
        agg = report["by_step"][str(step)]
        print(f"\n=== checkpoint @ {step} steps ===")
        print(
            f"{'candidate':22s} {'niche':6s} {'pass':5s} "
            f"{'meanΔ':>8s}  per-task Δ vs best-frontier"
        )
        for cand, row in agg.items():
            deltas = " ".join(
                f"{t[:6]}={v['delta']:+.3f}" for t, v in row["per_task"].items()
            )
            flag = "BEATS" if row["tier2_passed_niche"] else "  -  "
            print(
                f"{cand:22s} {flag:6s} {row['pass_count']}/{row['n_tasks']}  "
                f"{row['mean_delta_vs_frontier']:+.4f}  {deltas}"
            )


def _run_sweep(
    model_factories: dict[str, Callable[[int], nn.Module]],
    seeds: list[int],
    eval_at: tuple[int, ...],
    args: argparse.Namespace,
    started: float,
) -> dict[int, dict[str, dict[str, list[float]]]]:
    """Train every model on every (task, seed) once; collect eval acc per checkpoint.

    Returns ``acc[step][model][task] -> [eval_acc per seed]``.
    """
    acc: dict[int, dict[str, dict[str, list[float]]]] = {
        step: defaultdict(lambda: defaultdict(list)) for step in eval_at
    }
    n_runs = len(seeds) * 6 * len(model_factories)
    done = 0
    for seed in seeds:
        for task in default_hard_binding_tasks(seed=seed):
            for model_name, factory in model_factories.items():
                rows = run_one_task_checkpoints(
                    factory,
                    task,
                    eval_at_steps=eval_at,
                    mixer_label=model_name,
                    dim=args.dim,
                    n_blocks=args.n_blocks,
                    batch_size=args.batch_size,
                    learning_rate=args.lr,
                    seed=seed,
                    device=args.device,
                )
                for step, res in rows.items():
                    acc[step][model_name][task.name].append(res.eval_accuracy)
                done += 1
            print(
                f"[{done}/{n_runs}] seed={seed} task={task.name} done "
                f"({time.monotonic() - started:.0f}s)",
                flush=True,
            )
    return {
        step: {m: dict(tasks_) for m, tasks_ in by_model.items()}
        for step, by_model in acc.items()
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--n-blocks", type=int, default=2)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--eval-at", type=str, default="2000,3000")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument(
        "--candidates",
        type=str,
        default=None,
        help="comma-separated subset of candidate names to run (default: all 4). "
        "Frontier baselines are always retrained alongside whatever is selected, "
        "so each run is internally fair. Useful to split the fast attention leads "
        "from the slow MoR memory lane.",
    )
    args = ap.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    eval_at = tuple(int(s) for s in args.eval_at.split(",") if s.strip())
    candidates = _candidate_factories()
    if args.candidates:
        want = {c.strip() for c in args.candidates.split(",") if c.strip()}
        unknown = want - set(candidates)
        if unknown:
            raise SystemExit(f"unknown candidates: {sorted(unknown)}")
        candidates = {k: v for k, v in candidates.items() if k in want}
    baseline_names = FRONTIER_BASELINE_NAMES
    model_factories: dict[str, Callable[[int], nn.Module]] = {
        **candidates,
        **{name: lane_factory_for_baseline(name) for name in baseline_names},
    }
    task_names = [t.name for t in default_hard_binding_tasks(seed=0)]
    started = time.monotonic()
    plain_acc = _run_sweep(model_factories, seeds, eval_at, args, started)
    by_step = _aggregate(plain_acc, list(candidates), baseline_names, task_names)
    report = {
        "dim": args.dim,
        "n_blocks": args.n_blocks,
        "seeds": seeds,
        "eval_at_steps": list(eval_at),
        "device": args.device,
        "baseline_names": list(baseline_names),
        "candidate_names": list(candidates),
        "task_names": task_names,
        "raw_eval_acc": plain_acc,
        "by_step": {str(k): v for k, v in by_step.items()},
        "elapsed_s": round(time.monotonic() - started, 1),
    }

    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = args.output or (_REPORT_DIR / f"named_lanes_tier2_{stamp}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    _print_summary(report)
    print(f"\n[report -> {out_path}]  ({report['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
