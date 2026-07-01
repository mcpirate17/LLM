"""Induction step-ladder for a loss-monster family champion (ACTION 1, cheap-first).

Context: ``recursive_depth_router`` was flagged induction-capable-but-mislabelled — its
induction AUC jumps 0.008 (1000 steps, historical nano screening) -> 0.420 (3000 steps).
Before spending GPU on a full scale run we want to know:
  - emergence: at how many steps does induction actually form?
  - ceiling: does AUC keep climbing past 3k (5k, 10k), or has it saturated?

``capability_probe_monsters.py`` can't answer this cleanly: it runs induction with the
default ``timeout_s=120`` (so the 5k/10k rungs silently truncate) and only seed 0, and it
recomputes ar_gate + nano_bind on every call (wasted GPU — neither depends on step count).

This tool varies ONLY the induction training-step budget, lifts the timeout so every rung
runs to completion, and runs >=3 seeds per rung (model init + data both reseeded) so the
emergence/ceiling claim survives the induction seed-variance rule. Model config matches
``capability_probe_monsters.py`` (n_layers=6, vocab 512, seq 256) for direct comparability
with the historical 0.420@3k number.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from research.scientist.native_runner import compile_model_native_first
from research.synthesis.serializer import graph_from_json
from research.eval.induction_probe import induction_score
from research.tools.loss_monster_screen import (
    _OUT_DIR,
    _RUNS_DB,
    select_family_champions,
)


def _run_rung(graph_json: str, n_layers: int, steps: int, seed: int, args) -> dict:
    """One (steps, seed) cell: fresh model init under `seed`, induction over `steps`."""
    torch.manual_seed(seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = compile_model_native_first(
        [graph_from_json(graph_json)] * n_layers,
        vocab_size=512,
        max_seq_len=256,
    ).to(args.device)
    res = induction_score(
        model,
        n_train_steps=steps,
        device=args.device,
        seed=seed,
        timeout_s=args.timeout_s,
    )
    return {
        "auc": round(float(res.auc), 4),
        "status": res.status,
        "steps_trained": int(res.steps_trained),
        "gap_accuracies": res.gap_accuracies,
        "elapsed_ms": round(float(res.elapsed_ms), 1),
    }


def _run_ladder_rung(champ, steps: int, args) -> dict:
    """All seeds for one step budget; aggregate to median/spread, flag truncation."""
    cells = []
    for seed in args.seeds:
        cell = _run_rung(champ.graph_json, args.n_layers, steps, seed, args)
        cell["seed"] = seed
        cells.append(cell)
        print(
            f"  steps={steps:6d} seed={seed} auc={cell['auc']} "
            f"status={cell['status']} trained={cell['steps_trained']} "
            f"({cell['elapsed_ms'] / 1000:.1f}s)",
            flush=True,
        )
    aucs = [c["auc"] for c in cells]
    truncated = [c for c in cells if c["steps_trained"] < steps]
    rung = {
        "steps": steps,
        "median_auc": round(statistics.median(aucs), 4),
        "mean_auc": round(statistics.fmean(aucs), 4),
        "min_auc": round(min(aucs), 4),
        "max_auc": round(max(aucs), 4),
        "n_truncated": len(truncated),
        "cells": cells,
    }
    warn = f"  ⚠ {len(truncated)} cell(s) truncated by timeout" if truncated else ""
    print(
        f"  -> steps={steps:6d} median_auc={rung['median_auc']} "
        f"[{rung['min_auc']}, {rung['max_auc']}]{warn}\n",
        flush=True,
    )
    return rung


def _print_summary(rungs: list[dict]) -> None:
    """Emergence/ceiling read-out: steps -> median, and last-rung gain direction."""
    medians = [(r["steps"], r["median_auc"]) for r in rungs]
    print("Summary (steps -> median_auc):")
    for s, m in medians:
        print(f"  {s:6d}  {m}")
    if len(medians) >= 2:
        last_gain = medians[-1][1] - medians[-2][1]
        print(
            f"\nLast rung gain ({medians[-2][0]}->{medians[-1][0]}): {last_gain:+.4f} "
            f"({'still climbing' if last_gain > 0.02 else 'plateauing'})"
        )


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", default="recursive_depth_router")
    ap.add_argument("--steps", type=int, nargs="*", default=[1000, 3000, 5000, 10000])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument(
        "--timeout-s",
        type=float,
        default=1800.0,
        help="per-cell induction timeout; default high so 10k rung completes",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(_OUT_DIR / "induction_step_ladder.json"))
    return ap


def main() -> int:
    args = _build_parser().parse_args()

    champ = next(
        (c for c in select_family_champions(_RUNS_DB) if c.family == args.family), None
    )
    if champ is None:
        print(f"No family champion for {args.family!r}")
        return 1

    print(
        f"Induction step-ladder for {args.family} "
        f"(loss_ratio={champ.screening_loss_ratio:.3f}, n_layers={args.n_layers})\n"
        f"steps={args.steps} seeds={args.seeds} timeout_s={args.timeout_s} "
        f"device={args.device}\n",
        flush=True,
    )

    rungs = [_run_ladder_rung(champ, steps, args) for steps in args.steps]
    _print_summary(rungs)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(
            {
                "config": vars(args),
                "family": args.family,
                "loss_ratio": champ.screening_loss_ratio,
                "rungs": rungs,
            },
            indent=2,
        )
    )
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
