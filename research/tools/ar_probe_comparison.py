#!/usr/bin/env python
"""Head-to-head comparison of non-gate AR probes.

Runs ar_intermediate_probe, run_ar_validation, and the new ar_curriculum probe
on the same 4 reference architectures and reports:

  - Headline score per (arch, probe)
  - Wall time per (arch, probe)
  - Spearman/Pearson correlation between probes' rankings (duplication signal)

Excludes ar_gate (screening tier). Same seed across probes produces identical
random-init weights so probe scores are directly comparable per arch.

Output:
  research/runtime/ar_curriculum_experiment/probe_compare_<run_id>.json
  research/runtime/ar_curriculum_experiment/probe_compare_<run_id>.md
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import statistics as st
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import torch

from research.eval.ar_intermediate_probe import (
    ARIntermediateConfig,
    ar_intermediate_probe,
)
from research.eval.ar_validation import ARValidationConfig, run_ar_validation
from research.synthesis.compiler import compile_model
from research.synthesis.reference_architectures import (
    REFERENCE_ARCHITECTURES,
    build_reference,
)
from research.tools.ar_curriculum_experiment import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EVAL_BATCHES,
    DEFAULT_LR,
    RUNTIME_ROOT,
    STAGE_SETS,
    VOCAB_SIZE_BY_SET,
    run_arch,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Vocab must satisfy all 3 probes:
#   ar_intermediate default: vocab_lo=512 + n_keys=256 + n_values=48 + 2 special = 818
#   ar_validation default:   vocab_lo=1000 + n_keys=1024 + n_values=96 + 2 = 2122
#   ar_curriculum fine:      4096
COMPARE_VOCAB_SIZE = 4096
DEFAULT_D_MODEL = 256
DEFAULT_N_LAYERS = 6
CURRICULUM_STEPS_PER_STAGE = 1000
CURRICULUM_STAGE_SET = "fine"


@dataclass(slots=True)
class ProbeOutcome:
    arch_key: str
    arch_name: str
    seed: int
    probe: str
    headline: float
    headline_label: str
    wall_s: float
    status: str
    extras: dict[str, Any] = field(default_factory=dict)


def _build_model(arch_key: str, seed: int, device: torch.device) -> torch.nn.Module:
    torch.manual_seed(int(seed))
    layer_graphs = [
        build_reference(arch_key, d_model=DEFAULT_D_MODEL)
        for _ in range(DEFAULT_N_LAYERS)
    ]
    return compile_model(layer_graphs, vocab_size=COMPARE_VOCAB_SIZE).to(device)


def _run_intermediate(
    model: torch.nn.Module, *, device: torch.device, seed: int
) -> tuple[float, float, dict[str, Any]]:
    cfg = ARIntermediateConfig(seed=seed)
    t0 = time.perf_counter()
    res = ar_intermediate_probe(model, cfg=cfg, device=str(device))
    wall = time.perf_counter() - t0
    extras = {
        "held_pair_acc": res.held_pair_acc,
        "held_class_acc": res.held_class_acc,
        "held_pair_lift": res.held_pair_lift,
        "auc": res.auc,
        "auc_lift": res.auc_lift,
        "score": res.score,
        "status": res.status,
    }
    return res.held_pair_lift, wall, extras


def _run_validation(
    model: torch.nn.Module, *, device: torch.device, seed: int
) -> tuple[float, float, dict[str, Any]]:
    cfg = ARValidationConfig(seed=seed)
    t0 = time.perf_counter()
    res = run_ar_validation(model, cfg=cfg, device=str(device))
    wall = time.perf_counter() - t0
    extras = {
        "final_acc": res.final_acc,
        "held_pair_acc": res.held_pair_acc,
        "held_class_acc": res.held_class_acc,
        "score": res.score,
        "steps_to_floor": res.steps_to_floor,
        "status": res.status,
    }
    return res.held_pair_acc, wall, extras


def _run_curriculum(
    arch_key: str, *, device: torch.device, seed: int, eval_batches: int
) -> tuple[float, float, dict[str, Any]]:
    stage_configs = STAGE_SETS[CURRICULUM_STAGE_SET]
    vocab_size = VOCAB_SIZE_BY_SET[CURRICULUM_STAGE_SET]
    t0 = time.perf_counter()
    cum = run_arch(
        arch_key,
        device=device,
        seed=seed,
        d_model=DEFAULT_D_MODEL,
        n_layers=DEFAULT_N_LAYERS,
        steps_per_stage=CURRICULUM_STEPS_PER_STAGE,
        batch_size=DEFAULT_BATCH_SIZE,
        lr=DEFAULT_LR,
        eval_batches=eval_batches,
        stage_configs=stage_configs,
        stage_set_name=CURRICULUM_STAGE_SET,
        vocab_size=vocab_size,
        mode="cumulative",
    )
    froz = run_arch(
        arch_key,
        device=device,
        seed=seed,
        d_model=DEFAULT_D_MODEL,
        n_layers=DEFAULT_N_LAYERS,
        steps_per_stage=CURRICULUM_STEPS_PER_STAGE,
        batch_size=DEFAULT_BATCH_SIZE,
        lr=DEFAULT_LR,
        eval_batches=eval_batches,
        stage_configs=stage_configs,
        stage_set_name=CURRICULUM_STAGE_SET,
        vocab_size=vocab_size,
        mode="frozen_s0",
    )
    wall = time.perf_counter() - t0
    headline = round(cum.auc_pair_final - froz.auc_pair_final, 4)
    extras = {
        "cumulative_auc": cum.auc_pair_final,
        "frozen_s0_auc": froz.auc_pair_final,
        "curriculum_learning": headline,
        "s0_retention": round(
            cum.per_stage_final[0]["held_pair_acc"]
            / max(froz.per_stage_final[0]["held_pair_acc"], 1e-9),
            3,
        ),
        "cumulative_status": cum.status,
        "frozen_status": froz.status,
        "cumulative_wall_s": cum.elapsed_s,
        "frozen_wall_s": froz.elapsed_s,
    }
    return headline, wall, extras


def _spearman(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    x_rank = _ranks(xs)
    y_rank = _ranks(ys)
    return _pearson(x_rank, y_rank)


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx > 0 and dy > 0 else 0.0


def _ranks(xs: list[float]) -> list[float]:
    paired = sorted(enumerate(xs), key=lambda p: p[1])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(paired):
        j = i
        while j + 1 < len(paired) and paired[j + 1][1] == paired[i][1]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[paired[k][0]] = avg_rank
        i = j + 1
    return ranks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archs", default="gpt2,mamba,rwkv,retrieval_augmented")
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--eval-batches", type=int, default=DEFAULT_EVAL_BATCHES)
    p.add_argument(
        "--probes",
        default="intermediate,validation,curriculum",
        help="Subset of probes to run",
    )
    p.add_argument("--run-id", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    arch_keys = tuple(a.strip() for a in str(args.archs).split(",") if a.strip())
    seeds = tuple(int(s.strip()) for s in str(args.seeds).split(",") if s.strip())
    probes = tuple(p.strip() for p in str(args.probes).split(",") if p.strip())
    device = torch.device(args.device)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if "validation" in probes and device.type != "cuda":
        raise SystemExit("ar_validation requires CUDA")

    logger.info(
        "probe-comparison run %s archs=%s seeds=%s probes=%s device=%s",
        run_id,
        arch_keys,
        seeds,
        probes,
        device,
    )

    outcomes: list[ProbeOutcome] = []
    t_start = time.perf_counter()
    for arch_key in arch_keys:
        arch_name = REFERENCE_ARCHITECTURES[arch_key]["name"]
        for seed in seeds:
            logger.info("=== %s seed=%d ===", arch_key, seed)
            base_model = _build_model(arch_key, seed, device)

            if "intermediate" in probes:
                model_for_int = copy.deepcopy(base_model).to(device)
                hl, w, ex = _run_intermediate(model_for_int, device=device, seed=seed)
                outcomes.append(
                    ProbeOutcome(
                        arch_key=arch_key,
                        arch_name=arch_name,
                        seed=seed,
                        probe="ar_intermediate",
                        headline=hl,
                        headline_label="held_pair_lift",
                        wall_s=round(w, 1),
                        status=ex.get("status", "ok"),
                        extras=ex,
                    )
                )
                logger.info(
                    "  ar_intermediate seed=%d held_pair_lift=%.3f wall=%.1fs",
                    seed,
                    hl,
                    w,
                )
                del model_for_int
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            if "validation" in probes:
                model_for_val = copy.deepcopy(base_model).to(device)
                hl, w, ex = _run_validation(model_for_val, device=device, seed=seed)
                outcomes.append(
                    ProbeOutcome(
                        arch_key=arch_key,
                        arch_name=arch_name,
                        seed=seed,
                        probe="ar_validation",
                        headline=hl,
                        headline_label="held_pair_acc",
                        wall_s=round(w, 1),
                        status=ex.get("status", "ok"),
                        extras=ex,
                    )
                )
                logger.info(
                    "  ar_validation   seed=%d held_pair_acc=%.3f wall=%.1fs",
                    seed,
                    hl,
                    w,
                )
                del model_for_val
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            del base_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

            if "curriculum" in probes:
                hl, w, ex = _run_curriculum(
                    arch_key,
                    device=device,
                    seed=seed,
                    eval_batches=int(args.eval_batches),
                )
                outcomes.append(
                    ProbeOutcome(
                        arch_key=arch_key,
                        arch_name=arch_name,
                        seed=seed,
                        probe="ar_curriculum",
                        headline=hl,
                        headline_label="curriculum_learning (cum_AUC - frozen_S0_AUC)",
                        wall_s=round(w, 1),
                        status="ok",
                        extras=ex,
                    )
                )
                logger.info(
                    "  ar_curriculum   seed=%d curriculum_learning=%.3f wall=%.1fs",
                    seed,
                    hl,
                    w,
                )

    by_arch_probe: dict[tuple[str, str], list[ProbeOutcome]] = {}
    for o in outcomes:
        by_arch_probe.setdefault((o.arch_key, o.probe), []).append(o)

    aggregated: dict[str, dict[str, dict[str, Any]]] = {}
    for (arch_key, probe), os in by_arch_probe.items():
        hls = [o.headline for o in os]
        walls = [o.wall_s for o in os]
        aggregated.setdefault(arch_key, {})[probe] = {
            "headline_mean": round(st.mean(hls), 4),
            "headline_std": round(st.stdev(hls), 4) if len(hls) > 1 else 0.0,
            "wall_mean_s": round(st.mean(walls), 1),
            "headline_label": os[0].headline_label,
        }

    correlations: dict[str, dict[str, float]] = {}
    for p1 in probes:
        for p2 in probes:
            if p1 >= p2:
                continue
            x = []
            y = []
            for arch_key in arch_keys:
                if (
                    arch_key in aggregated
                    and ("ar_" + p1) in aggregated[arch_key]
                    and ("ar_" + p2) in aggregated[arch_key]
                ):
                    x.append(aggregated[arch_key]["ar_" + p1]["headline_mean"])
                    y.append(aggregated[arch_key]["ar_" + p2]["headline_mean"])
            correlations.setdefault(f"ar_{p1}", {})[f"ar_{p2}"] = round(
                _spearman(x, y), 3
            )
            correlations.setdefault(f"ar_{p1}", {})[f"ar_{p2}_pearson"] = round(
                _pearson(x, y), 3
            )

    out_dir = RUNTIME_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"probe_compare_{run_id}.json"
    md_path = out_dir / f"probe_compare_{run_id}.md"

    payload = {
        "run_id": run_id,
        "arch_keys": list(arch_keys),
        "seeds": list(seeds),
        "probes": list(probes),
        "outcomes": [
            {
                "arch_key": o.arch_key,
                "arch_name": o.arch_name,
                "seed": o.seed,
                "probe": o.probe,
                "headline": o.headline,
                "headline_label": o.headline_label,
                "wall_s": o.wall_s,
                "status": o.status,
                "extras": o.extras,
            }
            for o in outcomes
        ],
        "aggregated": aggregated,
        "correlations": correlations,
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    lines: list[str] = [
        f"# AR probe comparison — {run_id}",
        "",
        f"archs={list(arch_keys)} seeds={list(seeds)} probes={list(probes)}",
        "",
        "## Headline (mean ± std across seeds)",
        "",
        "| arch | ar_intermediate (held_pair_lift) | ar_validation (held_pair_acc) | ar_curriculum (curriculum_learning) |",
        "|---|---:|---:|---:|",
    ]
    for arch_key in arch_keys:
        arch_name = REFERENCE_ARCHITECTURES[arch_key]["name"]
        agg = aggregated.get(arch_key, {})
        cells = [arch_name]
        for probe in ("ar_intermediate", "ar_validation", "ar_curriculum"):
            row = agg.get(probe)
            if row is None:
                cells.append("—")
            else:
                cells.append(f"{row['headline_mean']:.3f}±{row['headline_std']:.3f}")
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Wall time (mean across seeds, seconds)",
        "",
        "| arch | ar_intermediate | ar_validation | ar_curriculum (cum + frozen) | total |",
        "|---|---:|---:|---:|---:|",
    ]
    for arch_key in arch_keys:
        arch_name = REFERENCE_ARCHITECTURES[arch_key]["name"]
        agg = aggregated.get(arch_key, {})
        wall_int = agg.get("ar_intermediate", {}).get("wall_mean_s", 0.0)
        wall_val = agg.get("ar_validation", {}).get("wall_mean_s", 0.0)
        wall_cur = agg.get("ar_curriculum", {}).get("wall_mean_s", 0.0)
        total = wall_int + wall_val + wall_cur
        lines.append(
            f"| {arch_name} | {wall_int:.1f} | {wall_val:.1f} | {wall_cur:.1f} | {total:.1f} |"
        )

    lines += [
        "",
        "## Probe rank-correlation (Spearman) — does the probe rank archs the same way?",
        "",
        "Spearman ρ ≈ 1: probes are duplicates (same ranking). ρ ≈ 0: orthogonal "
        "(complementary signals). ρ ≈ -1: anti-correlated (one probe's winner is "
        "the other's loser).",
        "",
        "| pair | spearman | pearson |",
        "|---|---:|---:|",
    ]
    for p1, others in correlations.items():
        for p2_key, val in others.items():
            if p2_key.endswith("_pearson"):
                continue
            pearson = others.get(f"{p2_key}_pearson", 0.0)
            lines.append(f"| {p1} ↔ {p2_key} | {val:+.3f} | {pearson:+.3f} |")

    lines += [
        "",
        "## Per-arch / per-seed details",
        "",
    ]
    for o in outcomes:
        lines.append(
            f"- **{o.arch_name}** seed={o.seed} **{o.probe}**: "
            f"{o.headline_label}={o.headline:.3f} wall={o.wall_s:.1f}s status={o.status}"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    logger.info("Total wall: %.1fs", time.perf_counter() - t_start)
    logger.info("Wrote %s", json_path)
    logger.info("Wrote %s", md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
