#!/usr/bin/env python
"""AR Curriculum experimental probe.

Continuous associative-recall test that grows the corpus across 6 stages with
disjoint token ranges. The same model is trained cumulatively (single optimizer
state) for 250 steps per stage; after each stage transition we evaluate all
stages seen so far so we can spot catastrophic forgetting alongside the
breaking-point.

This is an *experiment* — not yet wired into the leaderboard. Run it on the
4 reference architectures (gpt2, mamba, rwkv, retrieval_augmented) and inspect
the per-stage curves before deciding whether to promote it to a real probe.

Output:
  research/runtime/ar_curriculum_experiment/<run_id>.json
  research/runtime/ar_curriculum_experiment/<run_id>.md
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from research.eval._ar_curriculum_common import (
    STAGE_CONFIGS_DEFAULT,
    STAGE_CONFIGS_FINE,
    VOCAB_LO,
    StageSpec,
    build_stage_specs,
    chance_class_acc,
    chance_pair_acc,
    evaluate_stage,
    make_stage_batch,
    stage_to_device,
    train_stage_one_batch,
)
from research.eval.associative_recall import _get_special_tokens
from research.eval.utils import make_adamw
from research.scientist.native_runner import compile_model_native_first as compile_model
from research.synthesis.reference_architectures import (
    REFERENCE_ARCHITECTURES,
    build_reference,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = REPO / "research/runtime/ar_curriculum_experiment"

VOCAB_SIZE = 3200
DEFAULT_D_MODEL = 256
DEFAULT_N_LAYERS = 6
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 1e-3
DEFAULT_STEPS_PER_STAGE = 250
DEFAULT_EVAL_BATCHES = 32
TRAINING_MODES = ("cumulative", "frozen_s0", "untrained")

STAGE_SETS = {"default": STAGE_CONFIGS_DEFAULT, "fine": STAGE_CONFIGS_FINE}

# Vocab needs to fit the largest stage set. Fine = 8+4+16+6+32+8+64+12+128+16+256+24+512+32+768+40+1024+48 = 2998 + VOCAB_LO 1000 + 2 special = 4000.
VOCAB_SIZE_BY_SET: dict[str, int] = {"default": 3200, "fine": 4096}


@dataclass(slots=True)
class CurriculumResult:
    arch_key: str
    arch_name: str
    paradigm: str
    seed: int
    d_model: int
    n_layers: int
    vocab_size: int
    steps_per_stage: int
    total_steps: int
    stage_set: str
    elapsed_s: float
    status: str = "ok"
    error: str | None = None
    auc_pair_final: float = 0.0
    auc_pair_peak: float = 0.0
    auc_class_final: float = 0.0
    auc_class_peak: float = 0.0
    retention_pair: float = 0.0
    max_passing_stage: int = -1
    mode: str = "cumulative"
    n_eval_examples: int = 0
    per_stage_final: list[dict[str, Any]] = field(default_factory=list)
    per_stage_peak_pair: list[float] = field(default_factory=list)
    forgetting_matrix_pair: list[list[float]] = field(default_factory=list)
    forgetting_matrix_class: list[list[float]] = field(default_factory=list)
    per_step_loss: list[float] = field(default_factory=list)


def run_arch(
    arch_key: str,
    *,
    device: torch.device,
    seed: int,
    d_model: int,
    n_layers: int,
    steps_per_stage: int,
    batch_size: int,
    lr: float,
    eval_batches: int,
    stage_configs: tuple[dict[str, int], ...] = STAGE_CONFIGS_DEFAULT,
    stage_set_name: str = "default",
    vocab_size: int = VOCAB_SIZE,
    mode: str = "cumulative",
) -> CurriculumResult:
    if mode not in TRAINING_MODES:
        raise ValueError(f"mode must be one of {TRAINING_MODES} (got {mode!r})")
    arch_meta = REFERENCE_ARCHITECTURES[arch_key]
    arch_name = str(arch_meta.get("name", arch_key))
    paradigm = str(arch_meta.get("paradigm", ""))

    t0 = time.perf_counter()
    result = CurriculumResult(
        arch_key=arch_key,
        arch_name=arch_name,
        paradigm=paradigm,
        seed=seed,
        d_model=d_model,
        n_layers=n_layers,
        vocab_size=vocab_size,
        steps_per_stage=steps_per_stage,
        total_steps=steps_per_stage * len(stage_configs),
        stage_set=stage_set_name,
        elapsed_s=0.0,
    )

    try:
        torch.manual_seed(int(seed))
        layer_graphs = [
            build_reference(arch_key, d_model=d_model) for _ in range(n_layers)
        ]
        model = compile_model(layer_graphs, vocab_size=vocab_size).to(device)
    except Exception as exc:  # noqa: BLE001
        result.status = "compile_failed"
        result.error = str(exc)
        result.elapsed_s = round(time.perf_counter() - t0, 2)
        return result

    sep_token, ans_token = _get_special_tokens(model)
    stages = [
        stage_to_device(s, device) for s in build_stage_specs(seed, stage_configs)
    ]
    n_stages = len(stages)
    n_eval_examples = int(eval_batches) * int(batch_size)
    result.mode = mode
    result.n_eval_examples = n_eval_examples

    forgetting_pair: list[list[float]] = []
    forgetting_class: list[list[float]] = []
    final_per_stage: list[dict[str, Any]] = []
    train_gen = torch.Generator(device=device)
    train_gen.manual_seed(int(seed))

    def _train_steps_on_stage(
        stage: StageSpec, n_steps: int, opt: torch.optim.Optimizer
    ) -> str | None:
        ans_pos = 3 * int(stage.pairs_per_example) + 3
        model.train()
        for step in range(n_steps):
            ids, targets = make_stage_batch(
                stage,
                split="train",
                batch_size=batch_size,
                sep_token=sep_token,
                ans_token=ans_token,
                device=device,
                generator=train_gen,
                episodic_values=True,
            )
            loss = train_stage_one_batch(
                model, ids, targets, opt=opt, stage=stage, ans_pos=ans_pos
            )
            if loss is None:
                return f"non_finite_loss at step={step}"
            if step % 100 == 0 or step == n_steps - 1:
                result.per_step_loss.append(round(loss, 4))
        return None

    def _eval_all_stages_into_row() -> tuple[list[float], list[float]]:
        row_pair: list[float] = []
        row_class: list[float] = []
        for s in stages:
            pa, ca = evaluate_stage(
                model,
                s,
                sep_token=sep_token,
                ans_token=ans_token,
                device=device,
                seed=seed + 1000,
                eval_batches=eval_batches,
                batch_size=batch_size,
            )
            row_pair.append(round(pa, 4))
            row_class.append(round(ca, 4))
        return row_pair, row_class

    if mode == "cumulative":
        opt = make_adamw(model.parameters(), lr=lr)
        for stage_idx, stage in enumerate(stages):
            err = _train_steps_on_stage(stage, steps_per_stage, opt)
            if err is not None:
                result.status = "non_finite_loss"
                result.error = f"stage={stage_idx} {err}"
                result.elapsed_s = round(time.perf_counter() - t0, 2)
                return result
            row_pair: list[float] = []
            row_class: list[float] = []
            for prev_stage in stages[: stage_idx + 1]:
                pa, ca = evaluate_stage(
                    model,
                    prev_stage,
                    sep_token=sep_token,
                    ans_token=ans_token,
                    device=device,
                    seed=seed + 1000,
                    eval_batches=eval_batches,
                    batch_size=batch_size,
                )
                row_pair.append(round(pa, 4))
                row_class.append(round(ca, 4))
            pad_pair = row_pair + [float("nan")] * (n_stages - len(row_pair))
            pad_class = row_class + [float("nan")] * (n_stages - len(row_class))
            forgetting_pair.append(pad_pair)
            forgetting_class.append(pad_class)
            logger.info(
                "[%s s=%d %s] stage=%d eval_curve=%s",
                arch_key,
                seed,
                mode,
                stage_idx,
                [f"{x:.2f}" for x in row_pair],
            )
    elif mode == "frozen_s0":
        opt = make_adamw(model.parameters(), lr=lr)
        total_steps = steps_per_stage * n_stages
        err = _train_steps_on_stage(stages[0], total_steps, opt)
        if err is not None:
            result.status = "non_finite_loss"
            result.error = f"frozen_s0 {err}"
            result.elapsed_s = round(time.perf_counter() - t0, 2)
            return result
        row_pair, row_class = _eval_all_stages_into_row()
        forgetting_pair.append(row_pair)
        forgetting_class.append(row_class)
        logger.info(
            "[%s s=%d %s] trained %d steps on S0; eval=%s",
            arch_key,
            seed,
            mode,
            total_steps,
            [f"{x:.2f}" for x in row_pair],
        )
    elif mode == "untrained":
        row_pair, row_class = _eval_all_stages_into_row()
        forgetting_pair.append(row_pair)
        forgetting_class.append(row_class)
        logger.info(
            "[%s s=%d %s] eval-only (no training): %s",
            arch_key,
            seed,
            mode,
            [f"{x:.2f}" for x in row_pair],
        )

    final_pair_row = forgetting_pair[-1]
    final_class_row = forgetting_class[-1]
    for s, pa, ca in zip(stages, final_pair_row, final_class_row):
        chance_pa = chance_pair_acc(s)
        chance_ca = chance_class_acc(s)
        lift_pa = (pa - chance_pa) / (1.0 - chance_pa) if (1.0 - chance_pa) > 0 else 0.0
        se_pa = (chance_pa * (1.0 - chance_pa) / max(n_eval_examples, 1)) ** 0.5
        z_pa = (pa - chance_pa) / se_pa if se_pa > 0 else 0.0
        final_per_stage.append(
            {
                "stage": s.stage_idx,
                "n_key_tokens": s.n_key_tokens,
                "n_value_tokens": s.n_value_tokens,
                "pairs_per_example": s.pairs_per_example,
                "held_pair_acc": round(float(pa), 4),
                "held_class_acc": round(float(ca), 4),
                "chance_pair_acc": round(chance_pa, 4),
                "chance_class_acc": round(chance_ca, 4),
                "lift_pair": round(lift_pa, 4),
                "z_score_pair": round(z_pa, 2),
            }
        )

    result.per_stage_final = final_per_stage
    result.forgetting_matrix_pair = forgetting_pair
    result.forgetting_matrix_class = forgetting_class

    peak_pair: list[float] = []
    peak_class: list[float] = []
    for stage_idx in range(n_stages):
        col_pair = [
            row[stage_idx]
            for row in forgetting_pair
            if not (
                isinstance(row[stage_idx], float) and row[stage_idx] != row[stage_idx]
            )
        ]
        col_class = [
            row[stage_idx]
            for row in forgetting_class
            if not (
                isinstance(row[stage_idx], float) and row[stage_idx] != row[stage_idx]
            )
        ]
        peak_pair.append(round(max(col_pair), 4) if col_pair else 0.0)
        peak_class.append(round(max(col_class), 4) if col_class else 0.0)
    result.per_stage_peak_pair = peak_pair

    final_pair = [r["held_pair_acc"] for r in final_per_stage]
    final_class = [r["held_class_acc"] for r in final_per_stage]
    result.auc_pair_final = round(sum(final_pair) / max(len(final_pair), 1), 4)
    result.auc_class_final = round(sum(final_class) / max(len(final_class), 1), 4)
    result.auc_pair_peak = round(sum(peak_pair) / max(len(peak_pair), 1), 4)
    result.auc_class_peak = round(sum(peak_class) / max(len(peak_class), 1), 4)
    result.retention_pair = round(
        result.auc_pair_final / result.auc_pair_peak
        if result.auc_pair_peak > 0
        else 0.0,
        4,
    )

    last_passing = -1
    for r in final_per_stage:
        if r["held_pair_acc"] >= max(0.4, r["chance_pair_acc"] * 4.0):
            last_passing = r["stage"]
    result.max_passing_stage = last_passing
    result.elapsed_s = round(time.perf_counter() - t0, 2)
    return result


def _result_to_dict(r: CurriculumResult) -> dict[str, Any]:
    return {
        "arch_key": r.arch_key,
        "arch_name": r.arch_name,
        "paradigm": r.paradigm,
        "seed": r.seed,
        "d_model": r.d_model,
        "n_layers": r.n_layers,
        "steps_per_stage": r.steps_per_stage,
        "total_steps": r.total_steps,
        "stage_set": r.stage_set,
        "vocab_size": r.vocab_size,
        "elapsed_s": r.elapsed_s,
        "status": r.status,
        "error": r.error,
        "auc_pair_final": r.auc_pair_final,
        "auc_pair_peak": r.auc_pair_peak,
        "auc_class_final": r.auc_class_final,
        "auc_class_peak": r.auc_class_peak,
        "retention_pair": r.retention_pair,
        "max_passing_stage": r.max_passing_stage,
        "mode": r.mode,
        "n_eval_examples": r.n_eval_examples,
        "per_stage_final": r.per_stage_final,
        "per_stage_peak_pair": r.per_stage_peak_pair,
        "forgetting_matrix_pair": r.forgetting_matrix_pair,
        "forgetting_matrix_class": r.forgetting_matrix_class,
        "per_step_loss": r.per_step_loss,
    }


def _aggregate_seeds(seed_results: list[CurriculumResult]) -> dict[str, Any]:
    """Mean/std across seeds for one arch+condition."""
    import statistics as st

    if not seed_results:
        return {}
    arch_name = seed_results[0].arch_name
    paradigm = seed_results[0].paradigm
    pair_finals = [r.auc_pair_final for r in seed_results]
    pair_peaks = [r.auc_pair_peak for r in seed_results]
    retentions = [r.retention_pair for r in seed_results]
    n_stages = len(seed_results[0].per_stage_final)
    per_stage_pair_mean = []
    per_stage_pair_std = []
    for i in range(n_stages):
        col = [
            r.per_stage_final[i]["held_pair_acc"]
            for r in seed_results
            if i < len(r.per_stage_final)
        ]
        per_stage_pair_mean.append(round(st.mean(col), 4) if col else 0.0)
        per_stage_pair_std.append(round(st.stdev(col), 4) if len(col) > 1 else 0.0)
    return {
        "arch_name": arch_name,
        "paradigm": paradigm,
        "n_seeds": len(seed_results),
        "auc_pair_final_mean": round(st.mean(pair_finals), 4),
        "auc_pair_final_std": round(st.stdev(pair_finals), 4)
        if len(pair_finals) > 1
        else 0.0,
        "auc_pair_peak_mean": round(st.mean(pair_peaks), 4),
        "auc_pair_peak_std": round(st.stdev(pair_peaks), 4)
        if len(pair_peaks) > 1
        else 0.0,
        "retention_pair_mean": round(st.mean(retentions), 4),
        "per_stage_pair_mean": per_stage_pair_mean,
        "per_stage_pair_std": per_stage_pair_std,
    }


def write_report(
    results: list[CurriculumResult],
    out_dir: Path,
    run_id: str,
    *,
    stage_configs: tuple[dict[str, int], ...],
    vocab_size: int,
    stage_set_name: str,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{run_id}.json"
    md_path = out_dir / f"{run_id}.md"

    by_arch: dict[str, list[CurriculumResult]] = {}
    for r in results:
        by_arch.setdefault(r.arch_key, []).append(r)
    aggregated = {arch_key: _aggregate_seeds(rs) for arch_key, rs in by_arch.items()}

    payload = {
        "run_id": run_id,
        "vocab_size": vocab_size,
        "vocab_lo": VOCAB_LO,
        "stage_set": stage_set_name,
        "stage_configs": list(stage_configs),
        "archs": [_result_to_dict(r) for r in results],
        "aggregated": aggregated,
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    n_stages = len(stage_configs)
    lines: list[str] = [
        f"# AR Curriculum experiment — {run_id}",
        "",
        f"stage_set={stage_set_name} vocab_lo={VOCAB_LO} vocab_size={vocab_size} "
        f"stages={n_stages} steps_per_stage={results[0].steps_per_stage if results else '?'} "
        f"d_model={results[0].d_model if results else '?'} "
        f"n_layers={results[0].n_layers if results else '?'} "
        f"seeds={len(by_arch.get(next(iter(by_arch), ''), []))}",
        "",
        "## Stage configuration",
        "",
        "| stage | n_keys | n_values | pairs/ex | n_train | n_held |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for i, cfg in enumerate(stage_configs):
        lines.append(
            f"| {i} | {cfg['n_keys']} | {cfg['n_values']} | "
            f"{cfg['pairs_per_example']} | {cfg['n_train_pairs']} | {cfg['n_held_pairs']} |"
        )

    lines += [
        "",
        "## Headline (mean ± std across seeds)",
        "",
        "| arch | paradigm | seeds | AUC final | AUC peak | retention |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for arch_key, agg in aggregated.items():
        lines.append(
            f"| {agg['arch_name']} | {agg['paradigm']} | {agg['n_seeds']} | "
            f"{agg['auc_pair_final_mean']:.3f}±{agg['auc_pair_final_std']:.3f} | "
            f"{agg['auc_pair_peak_mean']:.3f}±{agg['auc_pair_peak_std']:.3f} | "
            f"{agg['retention_pair_mean']:.2f} |"
        )

    lines += ["", "## Per-stage held_pair_acc (mean across seeds)", ""]
    header = "| arch | " + " | ".join(f"S{i}" for i in range(n_stages)) + " |"
    sep_line = "|---|" + "---:|" * n_stages
    lines.append(header)
    lines.append(sep_line)
    for arch_key, agg in aggregated.items():
        cells = [agg["arch_name"]]
        for i in range(n_stages):
            if i < len(agg["per_stage_pair_mean"]):
                cells.append(f"{agg['per_stage_pair_mean'][i]:.2f}")
            else:
                cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--archs",
        default="gpt2,mamba,rwkv,retrieval_augmented",
        help="Comma-separated reference arch keys",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--seeds",
        default="0",
        help="Comma-separated seeds (e.g., '0,1,2' for 3-seed mean)",
    )
    p.add_argument("--d-model", type=int, default=DEFAULT_D_MODEL)
    p.add_argument("--n-layers", type=int, default=DEFAULT_N_LAYERS)
    p.add_argument("--steps-per-stage", type=int, default=DEFAULT_STEPS_PER_STAGE)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--eval-batches", type=int, default=DEFAULT_EVAL_BATCHES)
    p.add_argument(
        "--stage-set",
        choices=tuple(STAGE_SETS.keys()),
        default="default",
        help="Stage curriculum to use (default=6 stages, fine=9 stages incl. vocab=8)",
    )
    p.add_argument(
        "--mode",
        choices=TRAINING_MODES,
        default="cumulative",
        help="Training mode (cumulative=normal curriculum, frozen_s0=train only S0, untrained=no training)",
    )
    p.add_argument("--run-id", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    arch_keys = [a.strip() for a in str(args.archs).split(",") if a.strip()]
    unknown = [a for a in arch_keys if a not in REFERENCE_ARCHITECTURES]
    if unknown:
        raise SystemExit(f"Unknown reference archs: {unknown}")
    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]
    stage_configs = STAGE_SETS[args.stage_set]
    vocab_size = VOCAB_SIZE_BY_SET[args.stage_set]

    device = torch.device(args.device)
    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        "ar_curriculum_%Y%m%d_%H%M%S"
    )
    logger.info(
        "Run %s on archs=%s seeds=%s stage_set=%s device=%s d_model=%d "
        "n_layers=%d steps/stage=%d",
        run_id,
        arch_keys,
        seeds,
        args.stage_set,
        device,
        args.d_model,
        args.n_layers,
        args.steps_per_stage,
    )

    results: list[CurriculumResult] = []
    for arch_key in arch_keys:
        for seed in seeds:
            logger.info("=== %s seed=%d ===", arch_key, seed)
            result = run_arch(
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
                mode=args.mode,
            )
            logger.info(
                "%s seed=%d: AUC final=%.3f peak=%.3f retention=%.2f wall=%.1fs status=%s",
                arch_key,
                seed,
                result.auc_pair_final,
                result.auc_pair_peak,
                result.retention_pair,
                result.elapsed_s,
                result.status,
            )
            results.append(result)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    json_path, md_path = write_report(
        results,
        RUNTIME_ROOT,
        run_id,
        stage_configs=stage_configs,
        vocab_size=vocab_size,
        stage_set_name=args.stage_set,
    )
    logger.info("Wrote %s", json_path)
    logger.info("Wrote %s", md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
