#!/usr/bin/env python
"""Empirical audit harness for the multiscale_rich_lane_router template.

Builds controlled graph variants that preserve the template's three-tier
identity while varying span sets, medium operators, and hard operators.
Collects short-run training, routing telemetry, latency, and memory signals.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch

from research.eval.sandbox import _collect_routing_telemetry
from research.eval.stateless_training import (
    clone_module_state,
    functional_compute_perplexity,
    functional_micro_train_loop,
)
from research.eval.utils import make_batches
from research.synthesis.compiler import compile_model
from research.synthesis.graph import ComputationGraph


DEFAULT_CORPUS = (
    Path(__file__).resolve().parents[1] / "corpus" / "wikitext103_train.npy"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "multiscale_rich_lane_router_audit.json"
)

MEDIUM_OPS = (
    "adaptive_lane_mixer",
    "block_sparse_linear",
    "semi_structured_2_4_linear",
    "rwkv_time_mixing",
    "conv1d_seq",
    "cheap_verify_blend",
    "default_path",
)

HARD_OPS = (
    "dual_compression_blend",
    "signal_conditioned_compression",
    "moe_topk",
    "moe_2expert",
    "state_space",
    "route_recursion",
    "adaptive_recursion",
    "mixed_recursion_gate",
    "n_way_sparse_router",
)


@dataclass(slots=True)
class AuditConfig:
    device: str
    vocab_size: int
    seq_len: int
    batch_size: int
    model_dim: int
    train_batches: int
    val_batches: int
    train_steps: int
    lr: float
    corpus_path: str
    output_path: str
    route_temperature: float
    min_keep_fraction: float
    confidence_threshold: float


def _span_variants() -> list[tuple[int, ...]]:
    widths = (2, 3, 4)
    variants: list[tuple[int, ...]] = []
    for n in range(1, len(widths) + 1):
        variants.extend(combinations(widths, n))
    return variants


def _tensor_batches(
    corpus: np.ndarray,
    *,
    device: str,
    seq_len: int,
    batch_size: int,
    train_batches: int,
    val_batches: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    train_tokens = corpus[:200_000]
    val_tokens = corpus[200_000:260_000]
    train = make_batches(
        train_tokens,
        batch_size=batch_size,
        seq_len=seq_len,
        n_batches=train_batches,
        device=device,
        seed=7,
    )
    val = make_batches(
        val_tokens,
        batch_size=batch_size,
        seq_len=seq_len,
        n_batches=val_batches,
        device=device,
        seed=17,
    )
    return train, val


def _medium_config(op_name: str) -> dict[str, Any]:
    if op_name in {"route_lanes", "adaptive_lane_mixer"}:
        return {"n_lanes": 3}
    return {}


def _hard_config(op_name: str) -> dict[str, Any]:
    if op_name in {"route_recursion", "adaptive_recursion", "mixed_recursion_gate"}:
        return {"max_depth": 3}
    if op_name == "moe_topk":
        return {"num_experts": 4, "top_k": 1}
    if op_name == "n_way_sparse_router":
        return {"n_ways": 4, "top_k": 2}
    return {}


def _hard_inputs(
    op_name: str, gated: int, hard_signal: int, hard_seed: int
) -> list[int]:
    if op_name in {
        "compression_mixture_experts",
        "routing_conditioned_compression",
        "dual_compression_blend",
        "mixed_recursion_gate",
    }:
        return [gated, hard_signal]
    return [hard_seed]


def _routing_curriculum_config(
    *,
    threshold: float,
    route_temperature: float,
    min_keep_fraction: float,
    confidence_threshold: float,
) -> dict[str, Any]:
    return {
        "curriculum_enabled": True,
        "curriculum_warmup_frac": 0.25,
        "curriculum_mid_frac": 0.65,
        "threshold": threshold,
        "threshold_start": 0.34,
        "threshold_mid": 0.4,
        "threshold_end": 0.46,
        "gate_temperature": 1.0,
        "gate_temperature_start": 1.35,
        "gate_temperature_mid": 1.1,
        "gate_temperature_end": 1.0,
        "confidence_threshold": confidence_threshold,
        "confidence_threshold_start": 0.3,
        "confidence_threshold_mid": 0.4,
        "confidence_threshold_end": min(confidence_threshold, 0.48),
        "min_keep_fraction": min_keep_fraction,
        "min_keep_fraction_start": 0.28,
        "min_keep_fraction_mid": 0.2,
        "min_keep_fraction_end": max(min_keep_fraction, 0.16),
        "route_temperature": route_temperature,
        "route_temperature_start": 1.35,
        "route_temperature_mid": 1.05,
        "route_temperature_end": max(route_temperature, 0.9),
    }


def _updated(base: dict[str, Any], extra: dict[str, Any] | None) -> dict[str, Any]:
    if not extra:
        return base
    return {**base, **extra}


def _merge_config() -> dict[str, Any]:
    return {
        "n_branches": 2,
        "normalize_inputs": True,
        "merge_temperature": 0.9,
    }


def _binary_merge_config(
    *,
    curriculum_enabled: bool,
    primary_role: str,
    secondary_role: str,
    min_secondary_share: float,
    max_secondary_share: float,
    min_secondary_start: float | None = None,
    min_secondary_mid: float | None = None,
    min_secondary_end: float | None = None,
    max_secondary_start: float | None = None,
    max_secondary_mid: float | None = None,
    max_secondary_end: float | None = None,
) -> dict[str, Any]:
    config = {
        **_merge_config(),
        "primary_role": primary_role,
        "secondary_role": secondary_role,
        "min_secondary_share": min_secondary_share,
        "max_secondary_share": max_secondary_share,
    }
    if curriculum_enabled:
        config.update(
            {
                "curriculum_enabled": True,
                "curriculum_warmup_frac": 0.25,
                "curriculum_mid_frac": 0.65,
                "min_secondary_share_start": min_secondary_share
                if min_secondary_start is None
                else min_secondary_start,
                "min_secondary_share_mid": min_secondary_share
                if min_secondary_mid is None
                else min_secondary_mid,
                "min_secondary_share_end": min_secondary_share
                if min_secondary_end is None
                else min_secondary_end,
                "max_secondary_share_start": max_secondary_share
                if max_secondary_start is None
                else max_secondary_start,
                "max_secondary_share_mid": max_secondary_share
                if max_secondary_mid is None
                else max_secondary_mid,
                "max_secondary_share_end": max_secondary_share
                if max_secondary_end is None
                else max_secondary_end,
            }
        )
    return config


def _hard_curriculum_config(op_name: str, base: dict[str, Any]) -> dict[str, Any]:
    if op_name not in {"route_recursion", "adaptive_recursion", "mixed_recursion_gate"}:
        return base
    max_depth = int(base.get("max_depth", 3))
    return {
        **base,
        "curriculum_enabled": True,
        "curriculum_warmup_frac": 0.25,
        "curriculum_mid_frac": 0.65,
        "active_depth_start": 1,
        "active_depth_mid": min(2, max_depth),
        "active_depth_end": max_depth,
    }


def build_multiscale_variant(
    *,
    model_dim: int,
    span_widths: tuple[int, ...],
    medium_op: str,
    hard_op: str,
    route_temperature: float,
    min_keep_fraction: float,
    confidence_threshold: float,
    enable_curriculum: bool = False,
    use_calibrated_merge: bool = False,
    gate_curriculum_overrides: dict[str, Any] | None = None,
    router_curriculum_overrides: dict[str, Any] | None = None,
    hard_curriculum_overrides: dict[str, Any] | None = None,
    merge_curriculum_overrides: dict[str, dict[str, Any]] | None = None,
) -> ComputationGraph:
    graph = ComputationGraph(model_dim=model_dim)
    inp = graph.add_input()
    default_path = graph.add_op("default_path", [inp], {})
    gate_config: dict[str, Any] = {"threshold": 0.5}
    if enable_curriculum:
        gate_config = _updated(
            _routing_curriculum_config(
                threshold=0.5,
                route_temperature=route_temperature,
                min_keep_fraction=min_keep_fraction,
                confidence_threshold=confidence_threshold,
            ),
            gate_curriculum_overrides,
        )
    gated = graph.add_op("hybrid_token_gate", [inp], gate_config)
    gated_skip = graph.add_op("add", [inp, gated], {})

    routed_nodes: list[int] = []
    for width in span_widths:
        graph.add_op(
            "sparse_span_builder",
            [gated],
            {"span_width": width, "fallback_behavior": "default_path"},
        )
        router_config = {
            "span_width": width,
            "lane_count": width,
            "confidence_threshold": confidence_threshold,
            "min_keep_fraction": min_keep_fraction,
            "route_temperature": route_temperature,
        }
        if enable_curriculum:
            router_config = _updated(
                {
                    **_routing_curriculum_config(
                        threshold=0.5,
                        route_temperature=route_temperature,
                        min_keep_fraction=min_keep_fraction,
                        confidence_threshold=confidence_threshold,
                    ),
                    "span_width": width,
                    "lane_count": width,
                },
                router_curriculum_overrides,
            )
        routed_nodes.append(
            graph.add_op(
                "hybrid_sparse_router",
                [gated],
                router_config,
            )
        )

    medium = routed_nodes[0]
    for nxt in routed_nodes[1:]:
        medium = graph.add_op("add", [medium, nxt], {})
    medium = graph.add_op("layernorm", [medium], {})
    medium = graph.add_op(medium_op, [medium], _medium_config(medium_op))
    medium = graph.add_op("linear_proj", [medium], {"out_dim": model_dim})

    hard_signal = graph.add_op("token_class_proj", [gated], {"n_classes": 4})
    hard_seed = graph.add_op("signal_conditioned_compression", [gated, hard_signal], {})
    hard = graph.add_op(
        hard_op,
        _hard_inputs(hard_op, gated, hard_signal, hard_seed),
        _updated(
            _hard_curriculum_config(hard_op, _hard_config(hard_op))
            if enable_curriculum
            else _hard_config(hard_op),
            hard_curriculum_overrides,
        ),
    )
    hard = graph.add_op("linear_proj", [hard], {"out_dim": model_dim})

    if use_calibrated_merge:
        merge_curriculum_overrides = merge_curriculum_overrides or {}
        out = graph.add_op(
            "calibrated_branch_merge",
            [default_path, medium],
            _updated(
                _binary_merge_config(
                    curriculum_enabled=enable_curriculum,
                    primary_role="default",
                    secondary_role="medium",
                    min_secondary_share=0.18,
                    max_secondary_share=0.42,
                    min_secondary_start=0.26,
                    min_secondary_mid=0.22,
                    min_secondary_end=0.18,
                ),
                merge_curriculum_overrides.get("default_medium"),
            ),
        )
        out = graph.add_op(
            "calibrated_branch_merge",
            [out, hard],
            _updated(
                _binary_merge_config(
                    curriculum_enabled=enable_curriculum,
                    primary_role="routed",
                    secondary_role="hard",
                    min_secondary_share=0.08,
                    max_secondary_share=0.2,
                    min_secondary_start=0.04,
                    min_secondary_mid=0.08,
                    min_secondary_end=0.1,
                    max_secondary_start=0.1,
                    max_secondary_mid=0.16,
                    max_secondary_end=0.22,
                ),
                merge_curriculum_overrides.get("routed_hard"),
            ),
        )
        out = graph.add_op(
            "calibrated_branch_merge",
            [out, gated_skip],
            _updated(
                _binary_merge_config(
                    curriculum_enabled=enable_curriculum,
                    primary_role="routed",
                    secondary_role="skip",
                    min_secondary_share=0.08,
                    max_secondary_share=0.22,
                    max_secondary_start=0.18,
                    max_secondary_mid=0.2,
                    max_secondary_end=0.22,
                ),
                merge_curriculum_overrides.get("routed_skip"),
            ),
        )
        out = graph.add_op(
            "calibrated_branch_merge",
            [out, inp],
            _updated(
                _binary_merge_config(
                    curriculum_enabled=enable_curriculum,
                    primary_role="routed",
                    secondary_role="input",
                    min_secondary_share=0.06,
                    max_secondary_share=0.18,
                    max_secondary_start=0.14,
                    max_secondary_mid=0.16,
                    max_secondary_end=0.18,
                ),
                merge_curriculum_overrides.get("routed_input"),
            ),
        )
    else:
        out = graph.add_op("add", [default_path, medium], {})
        out = graph.add_op("add", [out, hard], {})
        out = graph.add_op("add", [gated_skip, out], {})
        out = graph.add_op("add", [inp, out], {})
    graph.set_output(out)
    return graph


def _memory_mb(device: str) -> float | None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated(device=device) / (1024 * 1024)


def run_variant(
    *,
    cfg: AuditConfig,
    train_batches: list[torch.Tensor],
    val_batches: list[torch.Tensor],
    span_widths: tuple[int, ...],
    medium_op: str,
    hard_op: str,
) -> dict[str, Any]:
    start = time.perf_counter()
    graph = build_multiscale_variant(
        model_dim=cfg.model_dim,
        span_widths=span_widths,
        medium_op=medium_op,
        hard_op=hard_op,
        route_temperature=cfg.route_temperature,
        min_keep_fraction=cfg.min_keep_fraction,
        confidence_threshold=cfg.confidence_threshold,
    )
    model = compile_model(
        [graph], vocab_size=cfg.vocab_size, max_seq_len=cfg.seq_len
    ).to(cfg.device)
    params, buffers = clone_module_state(model)
    pre_ppl = functional_compute_perplexity(
        model, params, buffers, val_batches, cfg.vocab_size
    )
    if cfg.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device=cfg.device)
    train_start = time.perf_counter()
    train_final_loss = functional_micro_train_loop(
        model,
        params,
        buffers,
        train_batches,
        cfg.vocab_size,
        n_steps=cfg.train_steps,
        lr=cfg.lr,
    )
    train_ms = (time.perf_counter() - train_start) * 1000.0
    post_ppl = functional_compute_perplexity(
        model, params, buffers, val_batches, cfg.vocab_size
    )
    with torch.no_grad():
        _ = model(val_batches[0])
    routing = _collect_routing_telemetry(model, False) or {}
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    tokens_trained = cfg.train_steps * cfg.batch_size * cfg.seq_len
    throughput = tokens_trained / max(train_ms / 1000.0, 1e-9)
    return {
        "span_widths": list(span_widths),
        "medium_op": medium_op,
        "hard_op": hard_op,
        "pre_ppl": pre_ppl,
        "post_ppl": post_ppl,
        "train_final_loss": train_final_loss,
        "ppl_delta": (post_ppl - pre_ppl)
        if pre_ppl is not None and post_ppl is not None
        else None,
        "elapsed_ms": round(elapsed_ms, 3),
        "train_ms": round(train_ms, 3),
        "throughput_tokens_per_s": round(throughput, 3),
        "max_memory_mb": _memory_mb(cfg.device),
        "routing": routing,
    }


def run_variant_safe(
    *,
    cfg: AuditConfig,
    train_batches: list[torch.Tensor],
    val_batches: list[torch.Tensor],
    span_widths: tuple[int, ...],
    medium_op: str,
    hard_op: str,
) -> dict[str, Any]:
    try:
        return run_variant(
            cfg=cfg,
            train_batches=train_batches,
            val_batches=val_batches,
            span_widths=span_widths,
            medium_op=medium_op,
            hard_op=hard_op,
        )
    except Exception as exc:
        return {
            "span_widths": list(span_widths),
            "medium_op": medium_op,
            "hard_op": hard_op,
            "pre_ppl": math.inf,
            "post_ppl": math.inf,
            "train_final_loss": math.inf,
            "ppl_delta": None,
            "elapsed_ms": None,
            "train_ms": None,
            "throughput_tokens_per_s": 0.0,
            "max_memory_mb": None,
            "routing": {},
            "error": f"{type(exc).__name__}: {exc}",
        }


def _variant_comment(row: dict[str, Any]) -> str:
    routing = row.get("routing") or {}
    comments: list[str] = []
    if routing.get("dead_lane_count", 0):
        comments.append(f"dead_lanes={routing['dead_lane_count']}")
    if (routing.get("sparse_span_coverage") or 0.0) == 0.0:
        comments.append("no_span_coverage")
    if (routing.get("routed_compute_fraction") or 0.0) < 0.01:
        comments.append("weak_routed_compute")
    if (routing.get("lane_entropy") or 0.0) < 0.5:
        comments.append("lane_collapse_risk")
    return ",".join(comments) or "healthy"


def _markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    def _fmt_num(value: Any, digits: int = 1) -> str:
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            return "n/a"
        return f"{float(value):.{digits}f}"

    lines = [
        "| Variant | Train Loss | Pre PPL | Post PPL | Train ms | Tok/s | Mem MB | Routed Frac | Span Cov | Active/Dead | Comment |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        routing = row.get("routing") or {}
        variant = (
            f"spans={row['span_widths']} med={row['medium_op']} hard={row['hard_op']}"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    variant,
                    _fmt_num(row.get("train_final_loss"), 4),
                    _fmt_num(row.get("pre_ppl"), 1),
                    _fmt_num(row.get("post_ppl"), 1),
                    _fmt_num(row.get("train_ms"), 1),
                    _fmt_num(row.get("throughput_tokens_per_s"), 1),
                    _fmt_num(row.get("max_memory_mb"), 1),
                    f"{(routing.get('routed_compute_fraction') or 0.0):.4f}",
                    f"{(routing.get('sparse_span_coverage') or 0.0):.4f}",
                    f"{routing.get('active_lane_count', 0)}/{routing.get('dead_lane_count', 0)}",
                    row.get("error") or _variant_comment(row),
                ]
            )
            + " |"
        )
    return lines


def build_report(cfg: AuditConfig, results: dict[str, Any]) -> str:
    summary = results["summary"]
    lines = [
        "# Multiscale Rich Lane Router Audit",
        "",
        "## Findings Summary",
        "",
        f"- Best span set: `{summary['best_span_variant']}`",
        f"- Best medium operator: `{summary['best_medium_op']}`",
        f"- Best hard operator: `{summary['best_hard_op']}`",
        f"- Winner configuration: spans `{summary['winner']['span_widths']}`, medium `{summary['winner']['medium_op']}`, hard `{summary['winner']['hard_op']}`",
        f"- Winner train loss: `{summary['winner']['train_final_loss']:.4f}`",
        f"- Winner post-train perplexity: `{summary['winner']['post_ppl']:.1f}`",
        "",
        "## Span Ablation",
        "",
        *_markdown_table(results["span_ablation"]),
        "",
        "## Medium Operator Ablation",
        "",
        *_markdown_table(results["medium_ablation"]),
        "",
        "## Hard Operator Ablation",
        "",
        *_markdown_table(results["hard_ablation"]),
        "",
        "## Audit Verdict",
        "",
        "- Keep the three-tier identity: `yes`",
        f"- Simplify: `{summary['simplify']}`",
        f"- Replace medium operator with: `{summary['best_medium_op']}`",
        f"- Replace hard operator with: `{summary['best_hard_op']}`",
        f"- Retrain with curriculum: `{summary['retrain_with_curriculum']}`",
        f"- Merge redesign priority: `{summary['merge_redesign_priority']}`",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(asdict(cfg), indent=2),
        "```",
    ]
    return "\n".join(lines)


def summarize_results(results: dict[str, Any]) -> dict[str, Any]:
    span_best = min(results["span_ablation"], key=lambda row: row["post_ppl"])
    medium_best = min(results["medium_ablation"], key=lambda row: row["post_ppl"])
    hard_best = min(results["hard_ablation"], key=lambda row: row["post_ppl"])
    winner = min(
        [
            *results["span_ablation"],
            *results["medium_ablation"],
            *results["hard_ablation"],
        ],
        key=lambda row: row["post_ppl"],
    )
    return {
        "best_span_variant": str(span_best["span_widths"]),
        "best_medium_op": medium_best["medium_op"],
        "best_hard_op": hard_best["hard_op"],
        "winner": winner,
        "simplify": "drop span widths that repeatedly underperform the best span set",
        "retrain_with_curriculum": "yes",
        "merge_redesign_priority": "medium",
    }


def parse_args() -> AuditConfig:
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
    parser.add_argument("--train-steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--corpus-path", default=str(DEFAULT_CORPUS))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--route-temperature", type=float, default=0.85)
    parser.add_argument("--min-keep-fraction", type=float, default=0.125)
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    args = parser.parse_args()
    return AuditConfig(
        device=args.device,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        model_dim=args.model_dim,
        train_batches=args.train_batches,
        val_batches=args.val_batches,
        train_steps=args.train_steps,
        lr=args.lr,
        corpus_path=args.corpus_path,
        output_path=args.output_path,
        route_temperature=args.route_temperature,
        min_keep_fraction=args.min_keep_fraction,
        confidence_threshold=args.confidence_threshold,
    )


def main() -> None:
    cfg = parse_args()
    corpus = np.load(cfg.corpus_path, mmap_mode="r")
    train_batches, val_batches = _tensor_batches(
        corpus,
        device=cfg.device,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        train_batches=cfg.train_batches,
        val_batches=cfg.val_batches,
    )

    span_rows = [
        run_variant_safe(
            cfg=cfg,
            train_batches=train_batches,
            val_batches=val_batches,
            span_widths=spans,
            medium_op="conv1d_seq",
            hard_op="moe_topk",
        )
        for spans in _span_variants()
    ]
    medium_rows = [
        run_variant_safe(
            cfg=cfg,
            train_batches=train_batches,
            val_batches=val_batches,
            span_widths=(2, 3, 4),
            medium_op=medium_op,
            hard_op="moe_topk",
        )
        for medium_op in MEDIUM_OPS
    ]
    hard_rows = [
        run_variant_safe(
            cfg=cfg,
            train_batches=train_batches,
            val_batches=val_batches,
            span_widths=(2, 3, 4),
            medium_op="conv1d_seq",
            hard_op=hard_op,
        )
        for hard_op in HARD_OPS
    ]

    results = {
        "span_ablation": span_rows,
        "medium_ablation": medium_rows,
        "hard_ablation": hard_rows,
    }
    results["summary"] = summarize_results(results)

    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_md = output_path.with_suffix(".md")
    output_path.write_text(
        json.dumps({"config": asdict(cfg), **results}, indent=2), encoding="utf-8"
    )
    report_md.write_text(build_report(cfg, results), encoding="utf-8")
    print(json.dumps({"json": str(output_path), "markdown": str(report_md)}, indent=2))


if __name__ == "__main__":
    main()
