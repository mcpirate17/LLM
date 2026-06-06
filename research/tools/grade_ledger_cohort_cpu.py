"""Batch-grade the top-N gradeable ledger candidates vs frontier on CPU.

The deep-probe bake-off (``run_tier2_binding_cohort.run_cohort``) re-trains the 4
frontier baselines for every candidate. The baselines are identical across
candidates, so that is ~4/5 wasted compute. This driver trains each baseline
ONCE per task, caches ``baseline_max`` per task, and then trains only the
candidate per task — turning ``N × 5`` trainings into ``N + 4``. That makes a
broad read (how many of the top-K actually beat frontier) cheap enough to run on
CPU without touching a live GPU job.

Selection = top-N ledger entries by recent-mean composite that are present in
``proposals.jsonl`` (re-gradeable). Survival = the cohort's niche rule.

Usage::

    CUDA_VISIBLE_DEVICES="" python -m research.tools.grade_ledger_cohort_cpu \
        --top-n 40 --dim 32 --steps 600
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import time
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from component_fab.generator.code_generator import generate_module_from_spec
from component_fab.harness.harder_binding_tasks import (
    default_hard_binding_tasks,
    run_one_task,
)
from component_fab.harness.tiny_lm import (
    FRONTIER_BASELINE_NAMES,
    lane_factory_for_baseline,
    striped_lane_factory,
)
from component_fab.state.ledger import Ledger
from research.tools.run_tier2_binding_cohort import (
    _load_proposals_by_id,
    _niche_survival,
)

_REPO = Path(__file__).resolve().parents[2]
_REPORTS = _REPO / "research" / "reports"


def _recent_mean(history: list[float], window: int = 2) -> float:
    if not history:
        return 0.0
    recent = history[-window:]
    return sum(recent) / len(recent)


def _select_candidates(
    top_n: int, *, dedup: bool = True
) -> list[tuple[str, str, float]]:
    """Top-N (proposal_id, name, composite) gradeable ledger entries."""
    specs_by_id = _load_proposals_by_id()
    ledger = Ledger(include_rotated=True)
    rows = [
        (e.proposal_id, e.name, _recent_mean(e.composite_history))
        for e in ledger.all_entries()
        if e.composite_history and e.proposal_id in specs_by_id
    ]
    rows.sort(key=lambda r: r[2], reverse=True)
    if dedup:
        # The high-composite region is mode-collapsed (n=50 read: 31/50 were clones
        # of one math_axes family). Keep only the highest-composite spec per distinct
        # math_axes signature so grading compute is spent on distinct mechanisms.
        seen: set[str] = set()
        deduped: list[tuple[str, str, float]] = []
        for pid, name, comp in rows:
            spec = specs_by_id.get(pid)
            sig = (
                repr(sorted((spec.math_axes or {}).items()))
                if spec is not None
                else pid
            )
            if sig in seen:
                continue
            seen.add(sig)
            deduped.append((pid, name, comp))
        rows = deduped
    return rows[: max(0, top_n)]


def _baseline_max_per_task(
    tasks: tuple, *, dim: int, n_blocks: int, steps: int, seed: int
) -> dict[str, float]:
    """Train each frontier baseline once per task; cache the best per task."""
    best: dict[str, float] = {}
    for name in FRONTIER_BASELINE_NAMES:
        factory = lane_factory_for_baseline(name)
        for task in tasks:
            res = run_one_task(
                factory,
                task,
                mixer_label=name,
                dim=dim,
                n_blocks=n_blocks,
                n_train_steps=steps,
                seed=seed,
            )
            best[task.name] = max(best.get(task.name, 0.0), float(res.eval_accuracy))
    return best


def _grade_candidate(
    make_factory: Callable[[], Callable[[int], nn.Module]],
    label: str,
    tasks: tuple,
    baseline_max: dict[str, float],
    *,
    dim: int,
    n_blocks: int,
    steps: int,
    seed: int,
) -> dict[str, Any]:
    per_task: dict[str, dict[str, Any]] = {}
    for task in tasks:
        # Fresh factory per task — the striped hybrid factory carries block-position
        # state, so reusing one across tasks would mis-stripe after the first.
        res = run_one_task(
            make_factory(),
            task,
            mixer_label=label,
            dim=dim,
            n_blocks=n_blocks,
            n_train_steps=steps,
            seed=seed,
        )
        base = baseline_max.get(task.name, 0.0)
        acc = float(res.eval_accuracy)
        per_task[task.name] = {
            "candidate_eval_acc": acc,
            "baseline_max": base,
            "delta": acc - base,
            "beats": acc > base,
        }
    n_beats = sum(1 for v in per_task.values() if v["beats"])
    mean_delta = sum(v["delta"] for v in per_task.values()) / max(1, len(per_task))
    return {
        "per_task": per_task,
        "n_beats": n_beats,
        "mean_delta": mean_delta,
        "tier2_passed": _niche_survival(per_task),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--top-n", default=40, type=int)
    p.add_argument("--dim", default=32, type=int)
    p.add_argument("--n-blocks", default=2, type=int)
    p.add_argument("--steps", default=600, type=int)
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--threads", default=4, type=int)
    p.add_argument(
        "--hybrid",
        action="store_true",
        help="grade each candidate STRIPED with attention (Lahoti/Poli: standalone "
        "quality doesn't predict hybrid quality) instead of standalone",
    )
    p.add_argument(
        "--attn-every",
        default=2,
        type=int,
        help="in --hybrid: 1 full-attention block every N blocks (2=1:1, 4=1:3)",
    )
    p.add_argument("--no-dedup", action="store_true", help="grade math_axes clones too")
    return p


def _make_factory_builder(
    spec: Any, *, hybrid: bool, attn_every: int
) -> Callable[[], Callable[[int], nn.Module]]:
    """Zero-arg builder yielding a fresh (possibly striped) lane factory per task."""

    def base_factory() -> Callable[[int], nn.Module]:
        return lambda d: generate_module_from_spec(spec, dim=d)

    if not hybrid:
        return base_factory
    return lambda: striped_lane_factory(base_factory(), attn_every=attn_every)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    torch.set_num_threads(max(1, args.threads))

    candidates = _select_candidates(args.top_n, dedup=not args.no_dedup)
    specs_by_id = _load_proposals_by_id()
    tasks = default_hard_binding_tasks(seed=args.seed)
    mode = f"HYBRID (attn every {args.attn_every})" if args.hybrid else "standalone"
    print(
        f"grading {len(candidates)} candidates [{mode}] (dim={args.dim}, "
        f"{args.steps} steps) vs {', '.join(FRONTIER_BASELINE_NAMES)} on CPU"
    )

    t0 = time.monotonic()
    baseline_max = _baseline_max_per_task(
        tasks, dim=args.dim, n_blocks=args.n_blocks, steps=args.steps, seed=args.seed
    )
    print(f"[baselines trained in {time.monotonic() - t0:.0f}s] {baseline_max}")

    graded: list[dict[str, Any]] = []
    for i, (pid, name, composite) in enumerate(candidates):
        spec = specs_by_id.get(pid)
        if spec is None:
            continue
        tc = time.monotonic()
        try:
            res = _grade_candidate(
                _make_factory_builder(
                    spec, hybrid=args.hybrid, attn_every=args.attn_every
                ),
                name,
                tasks,
                baseline_max,
                dim=args.dim,
                n_blocks=args.n_blocks,
                steps=args.steps,
                seed=args.seed,
            )
        except Exception as exc:  # noqa: BLE001 — one bad candidate must not abort the sweep
            print(f"[{i + 1}/{len(candidates)}] {name[:48]} FAILED: {exc}")
            continue
        graded.append({"proposal_id": pid, "name": name, "composite": composite, **res})
        print(
            f"[{i + 1}/{len(candidates)}] {name[:46]:46} "
            f"comp={composite:.3f} beats={res['n_beats']}/6 "
            f"Δ={res['mean_delta']:+.3f} tier2={'PASS' if res['tier2_passed'] else ''} "
            f"({time.monotonic() - tc:.0f}s)"
        )

    graded.sort(
        key=lambda g: (g["tier2_passed"], g["n_beats"], g["mean_delta"]), reverse=True
    )
    n_pass = sum(1 for g in graded if g["tier2_passed"])
    n_any = sum(1 for g in graded if g["n_beats"] > 0)
    elapsed = time.monotonic() - t0

    print(f"\n=== SUMMARY ({len(graded)} graded, {elapsed:.0f}s) ===")
    print(f"tier2 survivors (beat frontier on niche rule): {n_pass}/{len(graded)}")
    print(f"beat frontier on >=1 task:                      {n_any}/{len(graded)}")
    print("\ntop 15 by (tier2, n_beats, mean Δ):")
    for g in graded[:15]:
        print(
            f"  {'PASS' if g['tier2_passed'] else '    '} beats={g['n_beats']}/6 "
            f"Δ={g['mean_delta']:+.3f} comp={g['composite']:.3f}  {g['name'][:58]}"
        )

    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = _REPORTS / f"ledger_cohort_grade_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "dim": args.dim,
                "n_blocks": args.n_blocks,
                "steps": args.steps,
                "hybrid": args.hybrid,
                "attn_every": args.attn_every if args.hybrid else None,
                "deduped": not args.no_dedup,
                "baselines": list(FRONTIER_BASELINE_NAMES),
                "baseline_max": baseline_max,
                "n_graded": len(graded),
                "n_tier2_survivors": n_pass,
                "n_beat_any": n_any,
                "elapsed_s": round(elapsed, 1),
                "graded": graded,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"\n[report → {out}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
