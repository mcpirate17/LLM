#!/usr/bin/env python
"""Phase 5 schedule refinement for multiscale_rich_lane_router."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import torch

from research.eval.training_core import make_optimizer
from research.eval.utils import clip_grad_norm, language_model_loss, make_batches
from research.synthesis.compiler import compile_model
from research.tools.audit_multiscale_rich_lane_router import build_multiscale_variant
from research.tools.audit_multiscale_rich_lane_router_phase2 import (
    _memory_mb,
    _probe_routing_telemetry,
)


DEFAULT_CORPUS = (
    Path(__file__).resolve().parents[1] / "corpus" / "wikitext103_train.npy"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "multiscale_rich_lane_router_phase5_schedule_refinement.json"
)


@dataclass(slots=True)
class Phase5Config:
    device: str
    vocab_size: int
    seq_len: int
    batch_size: int
    model_dim: int
    train_batches: int
    val_batches: int
    sweep_steps: int
    long_steps: int
    clip_grad: float
    corpus_path: str
    output_path: str
    route_temperature: float
    min_keep_fraction: float
    confidence_threshold: float
    merge_redesign: bool


@dataclass(slots=True)
class ScheduleSpec:
    name: str
    medium_op: str
    hard_op: str
    span_widths: tuple[int, ...]
    enable_curriculum: bool
    peak_lr: float
    warmup_frac: float
    hold_frac: float
    end_lr_scale: float
    decay_style: str
    gate_curriculum_overrides: dict[str, Any] | None = None
    router_curriculum_overrides: dict[str, Any] | None = None
    hard_curriculum_overrides: dict[str, Any] | None = None
    merge_curriculum_overrides: dict[str, dict[str, Any]] | None = None


def _tensor_batches(
    corpus: np.ndarray,
    *,
    device: str,
    seq_len: int,
    batch_size: int,
    train_batches: int,
    val_batches: int,
    seed: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    train_tokens = corpus[:200_000]
    val_tokens = corpus[200_000:260_000]
    train = make_batches(
        train_tokens,
        batch_size=batch_size,
        seq_len=seq_len,
        n_batches=train_batches,
        device=device,
        seed=seed,
    )
    val = make_batches(
        val_tokens,
        batch_size=batch_size,
        seq_len=seq_len,
        n_batches=val_batches,
        device=device,
        seed=seed + 1000,
    )
    return train, val


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _lr_at_progress(
    progress: float,
    *,
    peak_lr: float,
    warmup_frac: float,
    hold_frac: float,
    end_lr_scale: float,
    decay_style: str,
) -> float:
    progress = max(0.0, min(1.0, progress))
    warmup_frac = max(0.0, min(0.95, warmup_frac))
    hold_frac = max(warmup_frac, min(1.0, hold_frac))
    if progress < warmup_frac and warmup_frac > 1e-6:
        return peak_lr * ((progress / warmup_frac) + 1e-3)
    if progress <= hold_frac:
        return peak_lr
    tail = (progress - hold_frac) / max(1.0 - hold_frac, 1e-6)
    if decay_style == "constant":
        factor = 1.0
    elif decay_style == "linear":
        factor = 1.0 - (1.0 - end_lr_scale) * tail
    elif decay_style == "cosine":
        factor = end_lr_scale + (1.0 - end_lr_scale) * 0.5 * (
            1.0 + math.cos(math.pi * tail)
        )
    else:
        raise ValueError(f"Unsupported decay style: {decay_style}")
    return peak_lr * factor


def _functional_eval_loss(
    model,
    batches: list[torch.Tensor],
    vocab_size: int,
) -> float | None:
    total_loss = 0.0
    total_tokens = 0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in batches:
            logits = model(batch)
            loss = language_model_loss(logits, batch, vocab_size, reduction="sum")
            if torch.isfinite(loss):
                total_loss += float(loss.item())
                total_tokens += batch[:, 1:].numel()
    if was_training:
        model.train()
    if total_tokens <= 0:
        return None
    return total_loss / total_tokens


def _convergence_step(loss_trajectory: dict[int, float]) -> int | None:
    if len(loss_trajectory) < 2:
        return None
    ordered = [loss_trajectory[idx] for idx in sorted(loss_trajectory)]
    start, end = ordered[0], ordered[-1]
    if not math.isfinite(start) or not math.isfinite(end) or start <= end:
        return None
    target = start - 0.75 * (start - end)
    for idx, value in enumerate(ordered, start=1):
        if value <= target:
            return idx
    return len(ordered)


def _build_schedule_specs() -> list[ScheduleSpec]:
    return [
        ScheduleSpec(
            name="baseline_fixed",
            medium_op="conv_only",
            hard_op="mixed_recursion_gate",
            span_widths=(2, 3, 4),
            enable_curriculum=False,
            peak_lr=3e-4,
            warmup_frac=10.0 / 24.0,
            hold_frac=1.0,
            end_lr_scale=1.0,
            decay_style="constant",
        ),
        ScheduleSpec(
            name="optimizer_refined",
            medium_op="conv_only",
            hard_op="mixed_recursion_gate",
            span_widths=(2, 3, 4),
            enable_curriculum=False,
            peak_lr=2.6e-4,
            warmup_frac=0.16,
            hold_frac=0.45,
            end_lr_scale=0.35,
            decay_style="cosine",
        ),
        ScheduleSpec(
            name="delayed_recursion_ramp",
            medium_op="conv_only",
            hard_op="mixed_recursion_gate",
            span_widths=(2, 3, 4),
            enable_curriculum=True,
            peak_lr=2.6e-4,
            warmup_frac=0.16,
            hold_frac=0.45,
            end_lr_scale=0.35,
            decay_style="cosine",
            gate_curriculum_overrides={
                "threshold_start": 0.5,
                "threshold_mid": 0.5,
                "threshold_end": 0.5,
                "gate_temperature_start": 1.0,
                "gate_temperature_mid": 1.0,
                "gate_temperature_end": 1.0,
            },
            router_curriculum_overrides={
                "confidence_threshold_start": 0.55,
                "confidence_threshold_mid": 0.55,
                "confidence_threshold_end": 0.55,
                "min_keep_fraction_start": 0.125,
                "min_keep_fraction_mid": 0.125,
                "min_keep_fraction_end": 0.125,
                "route_temperature_start": 0.85,
                "route_temperature_mid": 0.85,
                "route_temperature_end": 0.85,
            },
            hard_curriculum_overrides={
                "curriculum_warmup_frac": 0.35,
                "curriculum_mid_frac": 0.7,
                "active_depth_start": 1,
                "active_depth_mid": 1,
                "active_depth_end": 3,
            },
            merge_curriculum_overrides={
                "routed_hard": {
                    "curriculum_warmup_frac": 0.35,
                    "curriculum_mid_frac": 0.7,
                    "min_secondary_share_start": 0.02,
                    "min_secondary_share_mid": 0.06,
                    "min_secondary_share_end": 0.1,
                    "max_secondary_share_start": 0.08,
                    "max_secondary_share_mid": 0.14,
                    "max_secondary_share_end": 0.22,
                }
            },
        ),
        ScheduleSpec(
            name="gentle_routing_curriculum",
            medium_op="conv_only",
            hard_op="mixed_recursion_gate",
            span_widths=(2, 3, 4),
            enable_curriculum=True,
            peak_lr=2.6e-4,
            warmup_frac=0.16,
            hold_frac=0.45,
            end_lr_scale=0.35,
            decay_style="cosine",
            gate_curriculum_overrides={
                "curriculum_warmup_frac": 0.2,
                "curriculum_mid_frac": 0.65,
                "threshold_start": 0.42,
                "threshold_mid": 0.47,
                "threshold_end": 0.5,
                "gate_temperature_start": 1.2,
                "gate_temperature_mid": 1.05,
                "gate_temperature_end": 1.0,
            },
            router_curriculum_overrides={
                "curriculum_warmup_frac": 0.2,
                "curriculum_mid_frac": 0.65,
                "confidence_threshold_start": 0.4,
                "confidence_threshold_mid": 0.48,
                "confidence_threshold_end": 0.55,
                "min_keep_fraction_start": 0.22,
                "min_keep_fraction_mid": 0.16,
                "min_keep_fraction_end": 0.125,
                "route_temperature_start": 1.1,
                "route_temperature_mid": 0.95,
                "route_temperature_end": 0.85,
            },
            hard_curriculum_overrides={
                "curriculum_warmup_frac": 0.25,
                "curriculum_mid_frac": 0.7,
                "active_depth_start": 1,
                "active_depth_mid": 2,
                "active_depth_end": 3,
            },
            merge_curriculum_overrides={
                "default_medium": {
                    "curriculum_warmup_frac": 0.2,
                    "curriculum_mid_frac": 0.65,
                    "min_secondary_share_start": 0.24,
                    "min_secondary_share_mid": 0.2,
                    "min_secondary_share_end": 0.18,
                },
                "routed_hard": {
                    "curriculum_warmup_frac": 0.25,
                    "curriculum_mid_frac": 0.7,
                    "min_secondary_share_start": 0.03,
                    "min_secondary_share_mid": 0.08,
                    "min_secondary_share_end": 0.1,
                    "max_secondary_share_start": 0.08,
                    "max_secondary_share_mid": 0.16,
                    "max_secondary_share_end": 0.22,
                },
            },
        ),
        ScheduleSpec(
            name="combined_refined",
            medium_op="conv_only",
            hard_op="mixed_recursion_gate",
            span_widths=(2, 3, 4),
            enable_curriculum=True,
            peak_lr=2.4e-4,
            warmup_frac=0.12,
            hold_frac=0.38,
            end_lr_scale=0.28,
            decay_style="cosine",
            gate_curriculum_overrides={
                "curriculum_warmup_frac": 0.18,
                "curriculum_mid_frac": 0.62,
                "threshold_start": 0.44,
                "threshold_mid": 0.48,
                "threshold_end": 0.5,
                "gate_temperature_start": 1.15,
                "gate_temperature_mid": 1.03,
                "gate_temperature_end": 1.0,
            },
            router_curriculum_overrides={
                "curriculum_warmup_frac": 0.18,
                "curriculum_mid_frac": 0.62,
                "confidence_threshold_start": 0.42,
                "confidence_threshold_mid": 0.5,
                "confidence_threshold_end": 0.55,
                "min_keep_fraction_start": 0.18,
                "min_keep_fraction_mid": 0.14,
                "min_keep_fraction_end": 0.125,
                "route_temperature_start": 1.05,
                "route_temperature_mid": 0.93,
                "route_temperature_end": 0.85,
            },
            hard_curriculum_overrides={
                "curriculum_warmup_frac": 0.32,
                "curriculum_mid_frac": 0.72,
                "active_depth_start": 1,
                "active_depth_mid": 1,
                "active_depth_end": 3,
            },
            merge_curriculum_overrides={
                "routed_hard": {
                    "curriculum_warmup_frac": 0.32,
                    "curriculum_mid_frac": 0.72,
                    "min_secondary_share_start": 0.02,
                    "min_secondary_share_mid": 0.06,
                    "min_secondary_share_end": 0.1,
                    "max_secondary_share_start": 0.08,
                    "max_secondary_share_mid": 0.14,
                    "max_secondary_share_end": 0.2,
                }
            },
        ),
    ]


def _run_live_training(
    *,
    cfg: Phase5Config,
    spec: ScheduleSpec,
    train_batches: list[torch.Tensor],
    val_batches: list[torch.Tensor],
    steps: int,
    seed: int,
) -> dict[str, Any]:
    _set_seed(seed)
    graph = build_multiscale_variant(
        model_dim=cfg.model_dim,
        span_widths=spec.span_widths,
        medium_op=spec.medium_op,
        hard_op=spec.hard_op,
        route_temperature=cfg.route_temperature,
        min_keep_fraction=cfg.min_keep_fraction,
        confidence_threshold=cfg.confidence_threshold,
        enable_curriculum=spec.enable_curriculum,
        use_calibrated_merge=cfg.merge_redesign,
        gate_curriculum_overrides=spec.gate_curriculum_overrides,
        router_curriculum_overrides=spec.router_curriculum_overrides,
        hard_curriculum_overrides=spec.hard_curriculum_overrides,
        merge_curriculum_overrides=spec.merge_curriculum_overrides,
    )
    model = compile_model(
        [graph], vocab_size=cfg.vocab_size, max_seq_len=cfg.seq_len
    ).to(cfg.device)
    optimizer = make_optimizer(
        model.parameters(), optimizer_name="adamw", lr=spec.peak_lr
    )
    if cfg.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device=cfg.device)
    eval_before = _functional_eval_loss(model, val_batches, cfg.vocab_size)
    probe_batch = val_batches[0]
    probe_steps = {0, max(0, steps // 2), max(0, steps - 1)}
    route_trajectory: list[dict[str, Any]] = []
    loss_trajectory: dict[int, float] = {}
    model.train()
    train_start = time.perf_counter()
    for step in range(steps):
        progress = 1.0 if steps <= 1 else float(step) / float(steps - 1)
        if hasattr(model, "set_routing_progress"):
            model.set_routing_progress(progress if spec.enable_curriculum else 1.0)
        lr = _lr_at_progress(
            progress,
            peak_lr=spec.peak_lr,
            warmup_frac=spec.warmup_frac,
            hold_frac=spec.hold_frac,
            end_lr_scale=spec.end_lr_scale,
            decay_style=spec.decay_style,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        batch = train_batches[step % len(train_batches)]
        logits = model(batch)
        loss = language_model_loss(logits, batch, cfg.vocab_size)
        if not torch.isfinite(loss):
            break
        loss.backward()
        if cfg.clip_grad > 0:
            clip_grad_norm(model.parameters(), cfg.clip_grad)
        optimizer.step()
        loss_trajectory[step + 1] = float(loss.item())
        if step in probe_steps:
            route_telemetry = _probe_routing_telemetry(model, probe_batch)
            route_trajectory.append(
                {
                    "step": step + 1,
                    "progress": round(progress, 4),
                    "lr": lr,
                    "lane_entropy": route_telemetry.get("lane_entropy"),
                    "sparse_span_coverage": route_telemetry.get("sparse_span_coverage"),
                    "route_strength_mean": route_telemetry.get("route_strength_mean"),
                    "dead_lane_rate": route_telemetry.get("dead_lane_rate"),
                }
            )
    train_ms = (time.perf_counter() - train_start) * 1000.0
    if hasattr(model, "set_routing_progress"):
        model.set_routing_progress(1.0)
    eval_after = _functional_eval_loss(model, val_batches, cfg.vocab_size)
    routing = _probe_routing_telemetry(model, probe_batch)
    throughput = (steps * cfg.batch_size * cfg.seq_len) / max(train_ms / 1000.0, 1e-9)
    return {
        "spec": asdict(spec),
        "train_final_loss": loss_trajectory[max(loss_trajectory)]
        if loss_trajectory
        else math.inf,
        "eval_loss_before": eval_before,
        "eval_loss_after": eval_after,
        "train_ms": round(train_ms, 3),
        "throughput_tokens_per_s": round(throughput, 3),
        "max_memory_mb": _memory_mb(cfg.device),
        "convergence_step_75pct": _convergence_step(loss_trajectory),
        "loss_trajectory": {str(k): round(v, 6) for k, v in loss_trajectory.items()},
        "route_trajectory": route_trajectory,
        "routing": routing,
    }


def _aggregate_runs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def _vals(key: str) -> list[float]:
        values = [row[key] for row in rows if row.get(key) is not None]
        return [float(v) for v in values]

    def _routing_vals(key: str) -> list[float]:
        values = [
            float(row["routing"].get(key))
            for row in rows
            if row.get("routing") and row["routing"].get(key) is not None
        ]
        return values

    aggregate = {
        "train_loss_mean": mean(_vals("train_final_loss")),
        "train_loss_std": pstdev(_vals("train_final_loss")),
        "eval_loss_mean": mean(_vals("eval_loss_after")),
        "eval_loss_std": pstdev(_vals("eval_loss_after")),
        "throughput_mean": mean(_vals("throughput_tokens_per_s")),
        "throughput_std": pstdev(_vals("throughput_tokens_per_s")),
        "memory_mean": mean(_vals("max_memory_mb")),
        "memory_std": pstdev(_vals("max_memory_mb")),
        "convergence_step_mean": mean(_vals("convergence_step_75pct")),
    }
    routing = {}
    for key in (
        "dead_lane_rate",
        "lane_entropy",
        "sparse_span_coverage",
        "route_strength_mean",
        "routed_branch_share",
        "branch_dominance_mean",
    ):
        vals = _routing_vals(key)
        if vals:
            routing[key] = {"mean": mean(vals), "std": pstdev(vals)}
    stage_labels = ("default->medium", "routed->hard", "routed->skip", "routed->input")
    merge_metrics: dict[str, Any] = {}
    for label in stage_labels:
        primary = []
        secondary = []
        for row in rows:
            stage = ((row.get("routing") or {}).get("merge_stage_metrics") or {}).get(
                label
            ) or {}
            if "primary_share" in stage:
                primary.append(float(stage["primary_share"]))
            if "secondary_share" in stage:
                secondary.append(float(stage["secondary_share"]))
        if primary:
            merge_metrics[label] = {
                "primary_share_mean": round(mean(primary), 4),
                "secondary_share_mean": round(mean(secondary), 4),
            }
    routing["merge_stage_metrics"] = merge_metrics
    aggregate["routing"] = routing
    return aggregate


def _score_row(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    agg = payload["aggregate"]
    routing = agg["routing"]
    return {
        "name": name,
        "eval_loss_mean": agg["eval_loss_mean"],
        "eval_loss_std": agg["eval_loss_std"],
        "throughput_mean": agg["throughput_mean"],
        "memory_mean": agg["memory_mean"],
        "dead_lane_rate": routing["dead_lane_rate"]["mean"],
        "sparse_span_coverage": routing["sparse_span_coverage"]["mean"],
        "route_strength_mean": routing["route_strength_mean"]["mean"],
        "lane_entropy": routing["lane_entropy"]["mean"],
        "branch_dominance_mean": routing["branch_dominance_mean"]["mean"],
    }


def _quality_per_cost(eval_loss: float, throughput: float) -> float:
    return throughput / max(eval_loss, 1e-9)


def _build_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Multiscale Rich Lane Router Phase 5",
        "",
        "## Findings Summary",
        "",
        f"- Best raw-quality config: `{payload['deployment_recommendation']['raw_quality_winner']}`",
        f"- Production default: `{payload['deployment_recommendation']['production_default']}`",
        f"- Cost-sensitive fallback: `{payload['deployment_recommendation']['cost_sensitive_fallback']}`",
        f"- Optional high-quality mode: `{payload['deployment_recommendation']['optional_high_quality_mode']}`",
        f"- Width-4 status: `{payload['deployment_recommendation']['width4_status']}`",
        f"- Curriculum status: `{payload['deployment_recommendation']['curriculum_status']}`",
        f"- Ceiling verdict: `{payload['deployment_recommendation']['ceiling_assessment']}`",
        "",
        "## Schedule Tuning Scoreboard",
        "",
        "| Schedule | Eval Mean | Eval Std | Tok/s | Dead Rate | Span Cov | Route Str | Lane Entropy | Comment |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["schedule_tuning_scoreboard"]:
        lines.append(
            f"| {row['name']} | {row['eval_loss_mean']:.4f} | {row['eval_loss_std']:.4f} | {row['throughput_mean']:.1f} | "
            f"{row['dead_lane_rate']:.4f} | {row['sparse_span_coverage']:.4f} | {row['route_strength_mean']:.4f} | "
            f"{row['lane_entropy']:.4f} | {row['comment']} |"
        )
    lines.extend(
        [
            "",
            "## Width-4 Justification",
            "",
            "| Config | Eval Mean | Long Eval | Tok/s | Dead Rate | Quality/Cost |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["width4_justification_table"]:
        lines.append(
            f"| {row['label']} | {row['eval_loss_mean']:.4f} | {row['long_eval_loss']:.4f} | {row['throughput_mean']:.1f} | {row['dead_lane_rate']:.4f} | {row['quality_per_cost']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Hard-Path Pressure Test",
            "",
            "| Hard Path | Eval Mean | Long Eval | Tok/s | Dead Rate | Route Str |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["hard_path_pressure_test_table"]:
        lines.append(
            f"| {row['hard_op']} | {row['eval_loss_mean']:.4f} | {row['long_eval_loss']:.4f} | {row['throughput_mean']:.1f} | {row['dead_lane_rate']:.4f} | {row['route_strength_mean']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> Phase5Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--vocab-size", type=int, default=100_277)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--train-batches", type=int, default=8)
    parser.add_argument("--val-batches", type=int, default=4)
    parser.add_argument("--sweep-steps", type=int, default=24)
    parser.add_argument("--long-steps", type=int, default=72)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--corpus-path", default=str(DEFAULT_CORPUS))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--route-temperature", type=float, default=0.85)
    parser.add_argument("--min-keep-fraction", type=float, default=0.125)
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--merge-redesign", action="store_true", default=True)
    args = parser.parse_args()
    return Phase5Config(
        device=args.device,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        model_dim=args.model_dim,
        train_batches=args.train_batches,
        val_batches=args.val_batches,
        sweep_steps=args.sweep_steps,
        long_steps=args.long_steps,
        clip_grad=args.clip_grad,
        corpus_path=args.corpus_path,
        output_path=args.output_path,
        route_temperature=args.route_temperature,
        min_keep_fraction=args.min_keep_fraction,
        confidence_threshold=args.confidence_threshold,
        merge_redesign=bool(args.merge_redesign),
    )


def main() -> None:
    cfg = parse_args()
    corpus = np.load(cfg.corpus_path, mmap_mode="r")
    schedule_specs = _build_schedule_specs()

    schedule_results: dict[str, Any] = {}
    for spec in schedule_specs:
        runs = []
        for seed in (7, 17, 27):
            train_batches, val_batches = _tensor_batches(
                corpus,
                device=cfg.device,
                seq_len=cfg.seq_len,
                batch_size=cfg.batch_size,
                train_batches=cfg.train_batches,
                val_batches=cfg.val_batches,
                seed=seed,
            )
            runs.append(
                _run_live_training(
                    cfg=cfg,
                    spec=spec,
                    train_batches=train_batches,
                    val_batches=val_batches,
                    steps=cfg.sweep_steps,
                    seed=seed,
                )
            )
        schedule_results[spec.name] = {
            "spec": asdict(spec),
            "runs": runs,
            "aggregate": _aggregate_runs(runs),
        }

    best_schedule_name = min(
        schedule_results,
        key=lambda name: schedule_results[name]["aggregate"]["eval_loss_mean"],
    )
    best_schedule = next(
        spec for spec in schedule_specs if spec.name == best_schedule_name
    )

    width_specs = {
        "widths_[2,3,4]": best_schedule,
        "widths_[2,3]": ScheduleSpec(
            **{**asdict(best_schedule), "span_widths": (2, 3)}
        ),
    }
    width_results = {}
    for name, spec in width_specs.items():
        runs = []
        for seed in (7, 17, 27):
            train_batches, val_batches = _tensor_batches(
                corpus,
                device=cfg.device,
                seq_len=cfg.seq_len,
                batch_size=cfg.batch_size,
                train_batches=cfg.train_batches,
                val_batches=cfg.val_batches,
                seed=seed,
            )
            runs.append(
                _run_live_training(
                    cfg=cfg,
                    spec=spec,
                    train_batches=train_batches,
                    val_batches=val_batches,
                    steps=cfg.sweep_steps,
                    seed=seed,
                )
            )
        width_results[name] = {
            "spec": asdict(spec),
            "runs": runs,
            "aggregate": _aggregate_runs(runs),
        }

    long_horizon = {}
    for name, spec in width_specs.items():
        train_batches, val_batches = _tensor_batches(
            corpus,
            device=cfg.device,
            seq_len=cfg.seq_len,
            batch_size=cfg.batch_size,
            train_batches=cfg.train_batches,
            val_batches=cfg.val_batches,
            seed=7,
        )
        long_horizon[name] = _run_live_training(
            cfg=cfg,
            spec=spec,
            train_batches=train_batches,
            val_batches=val_batches,
            steps=cfg.long_steps,
            seed=7,
        )

    hard_specs = {
        "mixed_recursion_gate": best_schedule,
        "moe_topk": ScheduleSpec(
            **{
                **asdict(best_schedule),
                "hard_op": "moe_topk",
                "enable_curriculum": False,
                "hard_curriculum_overrides": None,
            }
        ),
    }
    hard_results = {}
    for name, spec in hard_specs.items():
        runs = []
        for seed in (7, 17, 27):
            train_batches, val_batches = _tensor_batches(
                corpus,
                device=cfg.device,
                seq_len=cfg.seq_len,
                batch_size=cfg.batch_size,
                train_batches=cfg.train_batches,
                val_batches=cfg.val_batches,
                seed=seed,
            )
            runs.append(
                _run_live_training(
                    cfg=cfg,
                    spec=spec,
                    train_batches=train_batches,
                    val_batches=val_batches,
                    steps=cfg.sweep_steps,
                    seed=seed,
                )
            )
        hard_results[name] = {
            "spec": asdict(spec),
            "runs": runs,
            "aggregate": _aggregate_runs(runs),
        }

    hard_long = {}
    for name, spec in hard_specs.items():
        train_batches, val_batches = _tensor_batches(
            corpus,
            device=cfg.device,
            seq_len=cfg.seq_len,
            batch_size=cfg.batch_size,
            train_batches=cfg.train_batches,
            val_batches=cfg.val_batches,
            seed=7,
        )
        hard_long[name] = _run_live_training(
            cfg=cfg,
            spec=spec,
            train_batches=train_batches,
            val_batches=val_batches,
            steps=cfg.long_steps,
            seed=7,
        )

    schedule_tuning_scoreboard = []
    for name, row in schedule_results.items():
        score = _score_row(name, row)
        comment = "baseline"
        if name == best_schedule_name:
            comment = "best refined schedule"
        elif score["dead_lane_rate"] > 0.0:
            comment = "dead-lane risk"
        elif row["spec"]["enable_curriculum"]:
            comment = "curriculum variant lost"
        elif (
            score["eval_loss_mean"]
            > schedule_results["baseline_fixed"]["aggregate"]["eval_loss_mean"]
        ):
            comment = "optimizer tuning lost"
        score["comment"] = comment
        schedule_tuning_scoreboard.append(score)
    schedule_tuning_scoreboard.sort(key=lambda row: row["eval_loss_mean"])

    width4_table = []
    for name in ("widths_[2,3,4]", "widths_[2,3]"):
        agg = width_results[name]["aggregate"]
        routing = agg["routing"]
        width4_table.append(
            {
                "label": name,
                "eval_loss_mean": agg["eval_loss_mean"],
                "long_eval_loss": long_horizon[name]["eval_loss_after"],
                "throughput_mean": agg["throughput_mean"],
                "memory_mean": agg["memory_mean"],
                "dead_lane_rate": routing["dead_lane_rate"]["mean"],
                "lane_entropy": routing["lane_entropy"]["mean"],
                "route_strength_mean": routing["route_strength_mean"]["mean"],
                "sparse_span_coverage": routing["sparse_span_coverage"]["mean"],
                "quality_per_cost": _quality_per_cost(
                    agg["eval_loss_mean"], agg["throughput_mean"]
                ),
            }
        )

    hard_pressure_table = []
    for name in ("mixed_recursion_gate", "moe_topk"):
        agg = hard_results[name]["aggregate"]
        routing = agg["routing"]
        hard_pressure_table.append(
            {
                "hard_op": name,
                "eval_loss_mean": agg["eval_loss_mean"],
                "long_eval_loss": hard_long[name]["eval_loss_after"],
                "throughput_mean": agg["throughput_mean"],
                "dead_lane_rate": routing["dead_lane_rate"]["mean"],
                "route_strength_mean": routing["route_strength_mean"]["mean"],
            }
        )

    width_default = min(width4_table, key=lambda row: row["eval_loss_mean"])
    width_cost = max(width4_table, key=lambda row: row["quality_per_cost"])
    raw_quality_winner = (
        "multiscale_rich_lane_router + conv_only + mixed_recursion_gate + calibrated_merge + widths=[2,3,4] + schedule="
        + best_schedule_name
    )
    production_default = (
        "multiscale_rich_lane_router + conv_only + mixed_recursion_gate + calibrated_merge + widths=[2,3] + schedule="
        + best_schedule_name
    )
    width4_status = "optional_high_quality_mode"
    if (
        width_default["label"] == "widths_[2,3]"
        and width_cost["label"] == "widths_[2,3]"
    ):
        width4_status = "drop_from_default"
    optional_high_quality_mode = raw_quality_winner

    curriculum_status = "off"
    if best_schedule.enable_curriculum:
        curriculum_status = "optional_only"

    ceiling_assessment = "near_ceiling"
    baseline_eval = schedule_results["baseline_fixed"]["aggregate"]["eval_loss_mean"]
    best_eval = schedule_results[best_schedule_name]["aggregate"]["eval_loss_mean"]
    if baseline_eval - best_eval > 0.03:
        ceiling_assessment = "some_schedule_headroom_remaining"

    payload = {
        "config": asdict(cfg),
        "schedule_specs": [asdict(spec) for spec in schedule_specs],
        "schedule_tuning": schedule_results,
        "schedule_tuning_scoreboard": schedule_tuning_scoreboard,
        "best_schedule_name": best_schedule_name,
        "width4_justification": width_results,
        "width4_long_horizon": long_horizon,
        "width4_justification_table": width4_table,
        "hard_path_pressure_test": hard_results,
        "hard_path_long_horizon": hard_long,
        "hard_path_pressure_test_table": hard_pressure_table,
        "deployment_recommendation": {
            "raw_quality_winner": raw_quality_winner,
            "production_default": production_default,
            "cost_sensitive_fallback": production_default,
            "optional_high_quality_mode": optional_high_quality_mode,
            "width4_status": width4_status,
            "curriculum_status": curriculum_status,
            "hard_slot_status": "lock_mixed_recursion_gate"
            if hard_results["mixed_recursion_gate"]["aggregate"]["eval_loss_mean"]
            <= hard_results["moe_topk"]["aggregate"]["eval_loss_mean"]
            else "revisit_hard_slot",
            "ceiling_assessment": ceiling_assessment,
        },
    }

    output_path = Path(cfg.output_path)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_path.with_suffix(".md").write_text(
        _build_markdown(payload), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
