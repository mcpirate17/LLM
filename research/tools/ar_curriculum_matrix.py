#!/usr/bin/env python
"""AR Curriculum factor-matrix experiment.

Runs the AR curriculum probe across 6 conditions to identify which factor
(stage granularity, model capacity, step budget) drives the AUC spread among
the 4 reference architectures.

Conditions:
  - baseline: 6 stages, d_model=256/n_layers=6, 250 steps
  - granular: 9 stages (incl. vocab=8), d_model=256/n_layers=6, 250 steps
  - bump:     6 stages, d_model=384/n_layers=8, 250 steps
  - s500:     6 stages, d_model=256/n_layers=6, 500 steps
  - s750:     6 stages, d_model=256/n_layers=6, 750 steps
  - s1000:    6 stages, d_model=256/n_layers=6, 1000 steps

Each condition: 4 archs (gpt2, mamba, rwkv, retrieval_augmented) x 3 seeds.

Output:
  research/runtime/ar_curriculum_experiment/matrix_<run_id>.json
  research/runtime/ar_curriculum_experiment/matrix_<run_id>.md
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


CONDITIONS: tuple[dict[str, Any], ...] = (
    {
        "name": "baseline",
        "stage_set": "default",
        "d_model": 256,
        "n_layers": 6,
        "steps": 250,
    },
    {
        "name": "granular",
        "stage_set": "fine",
        "d_model": 256,
        "n_layers": 6,
        "steps": 250,
    },
    {
        "name": "bump",
        "stage_set": "default",
        "d_model": 384,
        "n_layers": 8,
        "steps": 250,
    },
    {
        "name": "s500",
        "stage_set": "default",
        "d_model": 256,
        "n_layers": 6,
        "steps": 500,
    },
    {
        "name": "s750",
        "stage_set": "default",
        "d_model": 256,
        "n_layers": 6,
        "steps": 750,
    },
    {
        "name": "s1000",
        "stage_set": "default",
        "d_model": 256,
        "n_layers": 6,
        "steps": 1000,
    },
    {
        "name": "granular_s1000",
        "stage_set": "fine",
        "d_model": 256,
        "n_layers": 6,
        "steps": 1000,
    },
)

DEFAULT_ARCHS = ("gpt2", "mamba", "rwkv", "retrieval_augmented")
DEFAULT_SEEDS = (0, 1, 2)


def aggregate_per_arch(results: list[CurriculumResult]) -> dict[str, dict[str, Any]]:
    """Aggregate seed results per arch_key."""
    by_arch: dict[str, list[CurriculumResult]] = {}
    for r in results:
        by_arch.setdefault(r.arch_key, []).append(r)
    out: dict[str, dict[str, Any]] = {}
    for arch_key, rs in by_arch.items():
        finals = [r.auc_pair_final for r in rs]
        peaks = [r.auc_pair_peak for r in rs]
        retentions = [r.retention_pair for r in rs]
        wallts = [r.elapsed_s for r in rs]
        n_stages = max((len(r.per_stage_final) for r in rs), default=0)
        per_stage_means: list[float] = []
        for i in range(n_stages):
            col = [
                r.per_stage_final[i]["held_pair_acc"]
                for r in rs
                if i < len(r.per_stage_final)
            ]
            per_stage_means.append(round(st.mean(col), 3) if col else 0.0)
        out[arch_key] = {
            "arch_name": rs[0].arch_name,
            "paradigm": rs[0].paradigm,
            "n_seeds": len(rs),
            "auc_final_mean": round(st.mean(finals), 3),
            "auc_final_std": round(st.stdev(finals), 3) if len(finals) > 1 else 0.0,
            "auc_peak_mean": round(st.mean(peaks), 3),
            "auc_peak_std": round(st.stdev(peaks), 3) if len(peaks) > 1 else 0.0,
            "retention_mean": round(st.mean(retentions), 3),
            "wall_mean_s": round(st.mean(wallts), 1),
            "per_stage_mean": per_stage_means,
        }
    return out


def run_condition(
    cond: dict[str, Any],
    arch_keys: tuple[str, ...],
    seeds: tuple[int, ...],
    *,
    device: torch.device,
    batch_size: int,
    eval_batches: int,
    lr: float,
) -> tuple[list[CurriculumResult], dict[str, dict[str, Any]]]:
    stage_configs = STAGE_SETS[cond["stage_set"]]
    vocab_size = VOCAB_SIZE_BY_SET[cond["stage_set"]]
    cond_results: list[CurriculumResult] = []
    for arch_key in arch_keys:
        for seed in seeds:
            t0 = time.perf_counter()
            r = run_arch(
                arch_key,
                device=device,
                seed=seed,
                d_model=int(cond["d_model"]),
                n_layers=int(cond["n_layers"]),
                steps_per_stage=int(cond["steps"]),
                batch_size=batch_size,
                lr=lr,
                eval_batches=eval_batches,
                stage_configs=stage_configs,
                stage_set_name=cond["stage_set"],
                vocab_size=vocab_size,
            )
            cond_results.append(r)
            logger.info(
                "[%s] %s seed=%d AUC final=%.3f peak=%.3f ret=%.2f wall=%.1fs",
                cond["name"],
                arch_key,
                seed,
                r.auc_pair_final,
                r.auc_pair_peak,
                r.retention_pair,
                time.perf_counter() - t0,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return cond_results, aggregate_per_arch(cond_results)


def write_matrix_report(
    matrix: dict[str, dict[str, Any]],
    out_dir: Path,
    run_id: str,
    arch_keys: tuple[str, ...],
    seeds: tuple[int, ...],
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"matrix_{run_id}.json"
    md_path = out_dir / f"matrix_{run_id}.md"

    serializable = {
        "run_id": run_id,
        "arch_keys": list(arch_keys),
        "seeds": list(seeds),
        "conditions": list(CONDITIONS),
        "matrix": matrix,
    }
    json_path.write_text(
        json.dumps(serializable, indent=2, default=str), encoding="utf-8"
    )

    lines: list[str] = [
        f"# AR Curriculum factor-matrix — {run_id}",
        "",
        f"archs={list(arch_keys)} seeds={list(seeds)} ({len(seeds)} per cell)",
        "",
        "## Conditions",
        "",
        "| name | stages | d_model | n_layers | steps |",
        "|---|---|---:|---:|---:|",
    ]
    for cond in CONDITIONS:
        n_stages = len(STAGE_SETS[cond["stage_set"]])
        lines.append(
            f"| {cond['name']} | {cond['stage_set']} ({n_stages}) | "
            f"{cond['d_model']} | {cond['n_layers']} | {cond['steps']} |"
        )

    lines += [
        "",
        "## AUC pair (final, mean ± std across seeds) — by condition × arch",
        "",
    ]
    arch_names = [REFERENCE_ARCHITECTURES[k]["name"] for k in arch_keys]
    header = "| condition | " + " | ".join(arch_names) + " | wall(s) |"
    sep = "|---|" + "---:|" * (len(arch_names) + 1)
    lines.append(header)
    lines.append(sep)
    for cond in CONDITIONS:
        cells = [cond["name"]]
        agg = matrix.get(cond["name"], {})
        wall_total = 0.0
        for arch_key in arch_keys:
            a = agg.get(arch_key, {})
            if not a:
                cells.append("—")
                continue
            cells.append(f"{a['auc_final_mean']:.3f}±{a['auc_final_std']:.3f}")
            wall_total += a["wall_mean_s"] * a["n_seeds"]
        cells.append(f"{wall_total:.0f}")
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "## AUC pair (peak, mean across seeds)",
        "",
        header.replace(" wall(s) ", " — "),
        sep,
    ]
    for cond in CONDITIONS:
        cells = [cond["name"]]
        agg = matrix.get(cond["name"], {})
        for arch_key in arch_keys:
            a = agg.get(arch_key, {})
            cells.append(f"{a.get('auc_peak_mean', 0):.3f}" if a else "—")
        cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Retention (final / peak) — closer to 1.0 = no forgetting",
        "",
        header.replace(" wall(s) ", " — "),
        sep,
    ]
    for cond in CONDITIONS:
        cells = [cond["name"]]
        agg = matrix.get(cond["name"], {})
        for arch_key in arch_keys:
            a = agg.get(arch_key, {})
            cells.append(f"{a.get('retention_mean', 0):.2f}" if a else "—")
        cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Spread (max - min AUC final across archs, per condition)",
        "",
        "Higher spread = better discrimination among the 4 archs.",
        "",
        "| condition | spread | best arch | worst arch |",
        "|---|---:|---|---|",
    ]
    for cond in CONDITIONS:
        agg = matrix.get(cond["name"], {})
        if not agg:
            continue
        scored = [(k, v["auc_final_mean"]) for k, v in agg.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        if not scored:
            continue
        best_k, best_v = scored[0]
        worst_k, worst_v = scored[-1]
        spread = best_v - worst_v
        lines.append(
            f"| {cond['name']} | {spread:.3f} | "
            f"{REFERENCE_ARCHITECTURES[best_k]['name']} ({best_v:.3f}) | "
            f"{REFERENCE_ARCHITECTURES[worst_k]['name']} ({worst_v:.3f}) |"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archs", default=",".join(DEFAULT_ARCHS))
    p.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--eval-batches", type=int, default=DEFAULT_EVAL_BATCHES)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument(
        "--conditions",
        default=None,
        help="Comma-separated condition names (default: all 6)",
    )
    p.add_argument("--run-id", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    arch_keys = tuple(a.strip() for a in str(args.archs).split(",") if a.strip())
    seeds = tuple(int(s.strip()) for s in str(args.seeds).split(",") if s.strip())
    device = torch.device(args.device)
    selected = (
        tuple(c.strip() for c in str(args.conditions).split(",") if c.strip())
        if args.conditions
        else tuple(c["name"] for c in CONDITIONS)
    )
    conditions_to_run = [c for c in CONDITIONS if c["name"] in selected]
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    logger.info(
        "Matrix run %s on conditions=%s archs=%s seeds=%s device=%s",
        run_id,
        [c["name"] for c in conditions_to_run],
        arch_keys,
        seeds,
        device,
    )

    matrix: dict[str, dict[str, Any]] = {}
    t0 = time.perf_counter()
    for cond in conditions_to_run:
        logger.info("=== condition %s ===", cond["name"])
        _cond_results, agg = run_condition(
            cond,
            arch_keys,
            seeds,
            device=device,
            batch_size=int(args.batch_size),
            eval_batches=int(args.eval_batches),
            lr=float(args.lr),
        )
        matrix[cond["name"]] = agg

    json_path, md_path = write_matrix_report(
        matrix, RUNTIME_ROOT, run_id, arch_keys, seeds
    )
    logger.info("Total wall: %.1fs", time.perf_counter() - t0)
    logger.info("Wrote %s", json_path)
    logger.info("Wrote %s", md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
