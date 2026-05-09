#!/usr/bin/env python
"""AR Curriculum control comparison.

Runs the AR curriculum probe in 3 modes per architecture:
  - cumulative: standard run (train each stage in sequence)
  - frozen_s0:  compute-matched control (train ONLY on stage 0 for n_stages*steps total)
  - untrained:  empirical chance baseline (no training)

The compute-matched frozen_s0 control answers: "is the cumulative-training acc on
stage 0 lower than what we'd see if we trained the same total steps on S0 only?"
That's the strict test of catastrophic forgetting. The untrained baseline gives
the empirical chance floor per stage.

Output:
  research/runtime/ar_curriculum_experiment/controls_<run_id>.json
  research/runtime/ar_curriculum_experiment/controls_<run_id>.md
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics as st
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from research.synthesis.reference_architectures import REFERENCE_ARCHITECTURES
from research.tools.ar_curriculum_experiment import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EVAL_BATCHES,
    DEFAULT_LR,
    RUNTIME_ROOT,
    STAGE_SETS,
    TRAINING_MODES,
    VOCAB_SIZE_BY_SET,
    CurriculumResult,
    run_arch,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def aggregate(results: list[CurriculumResult]) -> dict[str, Any]:
    if not results:
        return {}
    n_stages = len(results[0].per_stage_final)
    per_stage_acc_mean: list[float] = []
    per_stage_acc_std: list[float] = []
    per_stage_lift_mean: list[float] = []
    per_stage_z_mean: list[float] = []
    for i in range(n_stages):
        accs = [
            r.per_stage_final[i]["held_pair_acc"]
            for r in results
            if i < len(r.per_stage_final)
        ]
        lifts = [
            r.per_stage_final[i]["lift_pair"]
            for r in results
            if i < len(r.per_stage_final)
        ]
        zs = [
            r.per_stage_final[i]["z_score_pair"]
            for r in results
            if i < len(r.per_stage_final)
        ]
        per_stage_acc_mean.append(round(st.mean(accs), 3) if accs else 0.0)
        per_stage_acc_std.append(round(st.stdev(accs), 3) if len(accs) > 1 else 0.0)
        per_stage_lift_mean.append(round(st.mean(lifts), 3) if lifts else 0.0)
        per_stage_z_mean.append(round(st.mean(zs), 2) if zs else 0.0)
    finals = [r.auc_pair_final for r in results]
    return {
        "n_seeds": len(results),
        "auc_pair_final_mean": round(st.mean(finals), 3),
        "auc_pair_final_std": round(st.stdev(finals), 3) if len(finals) > 1 else 0.0,
        "per_stage_acc_mean": per_stage_acc_mean,
        "per_stage_acc_std": per_stage_acc_std,
        "per_stage_lift_mean": per_stage_lift_mean,
        "per_stage_z_mean": per_stage_z_mean,
        "wall_mean_s": round(st.mean(r.elapsed_s for r in results), 1),
    }


def write_report(
    cells: dict[tuple[str, str], dict[str, Any]],
    out_dir: Path,
    run_id: str,
    *,
    arch_keys: tuple[str, ...],
    seeds: tuple[int, ...],
    stage_set: str,
    steps_per_stage: int,
    n_eval_examples: int,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"controls_{run_id}.json"
    md_path = out_dir / f"controls_{run_id}.md"

    chance_per_stage = []
    n_stages = 0
    for cell in cells.values():
        if cell.get("per_stage_acc_mean"):
            n_stages = len(cell["per_stage_acc_mean"])
            break
    stage_configs = STAGE_SETS[stage_set]
    chance_per_stage = [round(1.0 / cfg["n_values"], 4) for cfg in stage_configs]

    payload = {
        "run_id": run_id,
        "arch_keys": list(arch_keys),
        "seeds": list(seeds),
        "stage_set": stage_set,
        "steps_per_stage": steps_per_stage,
        "n_eval_examples": n_eval_examples,
        "chance_per_stage": chance_per_stage,
        "stage_configs": list(stage_configs),
        "cells": {f"{a}|{m}": v for (a, m), v in cells.items()},
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    lines: list[str] = [
        f"# AR Curriculum controls — {run_id}",
        "",
        f"stage_set={stage_set} steps_per_stage={steps_per_stage} "
        f"eval_examples={n_eval_examples} archs={list(arch_keys)} seeds={list(seeds)}",
        "",
        "**Chance per stage** (1/n_values): "
        + ", ".join(f"S{i}={c:.3f}" for i, c in enumerate(chance_per_stage)),
        "",
        "## AUC pair final (mean ± std across seeds)",
        "",
        "| arch | cumulative | frozen_s0 (matched compute) | untrained (empirical chance) | Δ cum-vs-frozen |",
        "|---|---:|---:|---:|---:|",
    ]
    random_auc = (
        sum(chance_per_stage) / len(chance_per_stage) if chance_per_stage else 0.0
    )
    for arch_key in arch_keys:
        cum = cells.get((arch_key, "cumulative"), {})
        froz = cells.get((arch_key, "frozen_s0"), {})
        unt = cells.get((arch_key, "untrained"), {})
        arch_name = REFERENCE_ARCHITECTURES[arch_key]["name"]
        cum_v = cum.get("auc_pair_final_mean", 0.0)
        froz_v = froz.get("auc_pair_final_mean", 0.0)
        unt_v = unt.get("auc_pair_final_mean", 0.0)
        delta = cum_v - froz_v
        lines.append(
            f"| {arch_name} | "
            f"{cum_v:.3f}±{cum.get('auc_pair_final_std', 0):.3f} | "
            f"{froz_v:.3f}±{froz.get('auc_pair_final_std', 0):.3f} | "
            f"{unt_v:.3f}±{unt.get('auc_pair_final_std', 0):.3f} | "
            f"{delta:+.3f} |"
        )
    lines.append("")
    lines.append(f"_Theoretical random AUC (mean of 1/n_values): {random_auc:.3f}_")

    lines += [
        "",
        "## Stage-0 forgetting test — strict",
        "",
        "If cumulative S0 acc < frozen_s0 S0 acc, the model FORGOT what it could "
        "have learned with the same compute. If cumulative ≥ frozen_s0, no forgetting "
        "(joint training generalizes as well or better).",
        "",
        "| arch | cumulative S0 acc | frozen_s0 S0 acc | Δ (forgetting) |",
        "|---|---:|---:|---:|",
    ]
    for arch_key in arch_keys:
        cum = cells.get((arch_key, "cumulative"), {})
        froz = cells.get((arch_key, "frozen_s0"), {})
        arch_name = REFERENCE_ARCHITECTURES[arch_key]["name"]
        cum_s0 = (
            cum.get("per_stage_acc_mean", [0.0])[0]
            if cum.get("per_stage_acc_mean")
            else 0.0
        )
        froz_s0 = (
            froz.get("per_stage_acc_mean", [0.0])[0]
            if froz.get("per_stage_acc_mean")
            else 0.0
        )
        delta = cum_s0 - froz_s0
        forget_marker = " (forgetting)" if delta < -0.05 else ""
        lines.append(
            f"| {arch_name} | {cum_s0:.3f} | {froz_s0:.3f} | {delta:+.3f}{forget_marker} |"
        )

    lines += [
        "",
        "## Per-stage acc — cumulative vs frozen_s0 vs untrained",
        "",
    ]
    for arch_key in arch_keys:
        arch_name = REFERENCE_ARCHITECTURES[arch_key]["name"]
        lines.append(f"### {arch_name}")
        lines.append("")
        header = "| mode | " + " | ".join(f"S{i}" for i in range(n_stages)) + " | AUC |"
        sep = "|---|" + "---:|" * (n_stages + 1)
        lines.append(header)
        lines.append(sep)
        chance_row = (
            ["chance"] + [f"{c:.3f}" for c in chance_per_stage] + [f"{random_auc:.3f}"]
        )
        lines.append("| " + " | ".join(chance_row) + " |")
        for mode in ("cumulative", "frozen_s0", "untrained"):
            cell = cells.get((arch_key, mode), {})
            if not cell:
                continue
            row = (
                [mode]
                + [f"{v:.3f}" for v in cell.get("per_stage_acc_mean", [])]
                + [f"{cell.get('auc_pair_final_mean', 0):.3f}"]
            )
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        lines.append("Z-scores (≥2 = significantly above chance):")
        lines.append("")
        lines.append(header.replace(" AUC ", " mean Z "))
        lines.append(sep)
        for mode in ("cumulative", "frozen_s0", "untrained"):
            cell = cells.get((arch_key, mode), {})
            if not cell:
                continue
            zs = cell.get("per_stage_z_mean", [])
            mean_z = sum(zs) / len(zs) if zs else 0.0
            row = [mode] + [f"{z:+.1f}" for z in zs] + [f"{mean_z:+.1f}"]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archs", default="gpt2,mamba,rwkv,retrieval_augmented")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--stage-set", default="fine", choices=tuple(STAGE_SETS.keys()))
    p.add_argument("--steps-per-stage", type=int, default=1000)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--eval-batches", type=int, default=DEFAULT_EVAL_BATCHES)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument(
        "--modes",
        default=",".join(TRAINING_MODES),
        help=f"Comma-separated modes from {TRAINING_MODES}",
    )
    p.add_argument("--run-id", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    arch_keys = tuple(a.strip() for a in str(args.archs).split(",") if a.strip())
    seeds = tuple(int(s.strip()) for s in str(args.seeds).split(",") if s.strip())
    modes = tuple(m.strip() for m in str(args.modes).split(",") if m.strip())
    for m in modes:
        if m not in TRAINING_MODES:
            raise SystemExit(f"unknown mode: {m}")
    device = torch.device(args.device)
    stage_configs = STAGE_SETS[args.stage_set]
    vocab_size = VOCAB_SIZE_BY_SET[args.stage_set]
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    logger.info(
        "controls run %s archs=%s seeds=%s modes=%s stage_set=%s steps=%d eval_batches=%d device=%s",
        run_id,
        arch_keys,
        seeds,
        modes,
        args.stage_set,
        args.steps_per_stage,
        args.eval_batches,
        device,
    )

    cells: dict[tuple[str, str], dict[str, Any]] = {}
    t_start = time.perf_counter()
    for arch_key in arch_keys:
        for mode in modes:
            mode_results: list[CurriculumResult] = []
            for seed in seeds:
                logger.info("=== %s mode=%s seed=%d ===", arch_key, mode, seed)
                r = run_arch(
                    arch_key,
                    device=device,
                    seed=seed,
                    d_model=int(args.d_model),
                    n_layers=int(args.n_layers),
                    steps_per_stage=int(args.steps_per_stage),
                    batch_size=int(args.batch_size),
                    lr=float(args.lr),
                    eval_batches=int(args.eval_batches),
                    stage_configs=stage_configs,
                    stage_set_name=args.stage_set,
                    vocab_size=vocab_size,
                    mode=mode,
                )
                mode_results.append(r)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            cells[(arch_key, mode)] = aggregate(mode_results)
            logger.info(
                "[%s %s] AUC final=%.3f wall_mean=%.1fs",
                arch_key,
                mode,
                cells[(arch_key, mode)]["auc_pair_final_mean"],
                cells[(arch_key, mode)]["wall_mean_s"],
            )

    n_eval = int(args.eval_batches) * int(args.batch_size)
    json_path, md_path = write_report(
        cells,
        RUNTIME_ROOT,
        run_id,
        arch_keys=arch_keys,
        seeds=seeds,
        stage_set=args.stage_set,
        steps_per_stage=int(args.steps_per_stage),
        n_eval_examples=n_eval,
    )
    logger.info("Total wall: %.1fs", time.perf_counter() - t_start)
    logger.info("Wrote %s", json_path)
    logger.info("Wrote %s", md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
