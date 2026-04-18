#!/usr/bin/env python
"""Phase 2 experiments for multiscale_rich_lane_router.

Compares the current winner baseline against routing curriculum and calibrated
merge variants, then runs a targeted span-width marginal-value check under the
best improved setup.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from research.eval.sandbox import _collect_routing_telemetry
from research.eval.stateless_training import (
    clone_module_state,
    functional_micro_train_loop,
)
from research.eval.utils import language_model_loss, make_batches
from research.synthesis.compiler import compile_model
from research.tools.audit_multiscale_rich_lane_router import (
    DEFAULT_CORPUS,
    build_multiscale_variant,
)


DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "multiscale_rich_lane_router_phase2.json"
)
BRANCH_NAMES = ["default", "medium", "hard", "skip", "input"]


@dataclass(slots=True)
class Phase2Config:
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
    medium_op: str
    hard_op: str


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


def _memory_mb(device: str) -> float | None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated(device=device) / (1024 * 1024)


def _functional_eval_loss(
    model,
    params,
    buffers,
    batches: list[torch.Tensor],
    vocab_size: int,
) -> float | None:
    from torch.func import functional_call

    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in batches:
            logits = functional_call(model, (params, buffers), (batch,))
            loss = language_model_loss(logits, batch, vocab_size, reduction="sum")
            if torch.isfinite(loss):
                total_loss += float(loss.item())
                total_tokens += batch[:, 1:].numel()
    if total_tokens <= 0:
        return None
    return total_loss / total_tokens


def _progress_callback(model, enabled: bool):
    if not enabled:
        return None

    def _callback(step: int, total_steps: int) -> None:
        progress = 1.0 if total_steps <= 1 else float(step) / float(total_steps - 1)
        model.set_routing_progress(progress)

    return _callback


def _convergence_steps(loss_trajectory: dict[int, float]) -> int | None:
    if len(loss_trajectory) < 2:
        return None
    ordered = [loss_trajectory[idx] for idx in sorted(loss_trajectory)]
    start, end = ordered[0], ordered[-1]
    if not math.isfinite(start) or not math.isfinite(end) or start <= end:
        return None
    target = start - 0.75 * (start - end)
    for step_idx, loss in enumerate(ordered, start=1):
        if loss <= target:
            return step_idx
    return len(ordered)


def _augment_routing_metrics(routing: dict[str, Any]) -> dict[str, Any]:
    payload = dict(routing or {})
    branch_weights = payload.get("branch_weight_mean") or []
    if branch_weights:
        named = {
            name: round(float(branch_weights[idx]), 4)
            for idx, name in enumerate(BRANCH_NAMES[: len(branch_weights)])
        }
        payload["branch_weight_named"] = named
        payload["dominant_branch"] = max(named.items(), key=lambda item: item[1])[0]
    lane_count = int(payload.get("lane_count", 0) or 0)
    dead_count = int(payload.get("dead_lane_count", 0) or 0)
    if lane_count > 0:
        payload["dead_lane_rate"] = round(dead_count / lane_count, 4)
    return payload


def _disable_subgraph_dispatch(model) -> None:
    for layer in getattr(model, "layers", []):
        if hasattr(layer, "_subgraph_dispatcher"):
            layer._subgraph_dispatcher = None


def _clear_routing_telemetry(model) -> None:
    for module in model.modules():
        if hasattr(module, "routing_telemetry"):
            delattr(module, "routing_telemetry")


def _probe_routing_telemetry(model, batch: torch.Tensor) -> dict[str, Any]:
    _clear_routing_telemetry(model)
    captured_inputs: dict[int, tuple[torch.Tensor, ...]] = {}
    handles = []
    for module in model.modules():
        if getattr(module, "op_name", None) == "calibrated_branch_merge":

            def _capture(mod, args):
                captured_inputs[id(mod)] = tuple(arg.detach() for arg in args)

            handles.append(module.register_forward_pre_hook(_capture))
    with torch.no_grad():
        _ = model(batch)
    for handle in handles:
        handle.remove()
    for module in model.modules():
        if (
            getattr(module, "op_name", None) == "calibrated_branch_merge"
            and id(module) in captured_inputs
        ):
            with torch.no_grad():
                _ = module(*captured_inputs[id(module)])
    routing = _augment_routing_metrics(_collect_routing_telemetry(model, False) or {})
    merge_metrics: dict[str, Any] = {}
    dominance_values: list[float] = []
    for module in model.modules():
        rt = getattr(module, "routing_telemetry", None)
        if not rt or rt.get("routing_mode") != "calibrated_branch_merge":
            continue
        payload = rt.get("trace_payload") or {}
        label = f"{payload.get('primary_role', 'primary')}->{payload.get('secondary_role', 'secondary')}"
        weights = rt.get("branch_weight_sum")
        count = int(rt.get("branch_weight_count", 0) or 0)
        if weights is None or count <= 0:
            continue
        secondary_share = float(weights[1].item()) / count
        primary_share = float(weights[0].item()) / count
        dominance = float(rt.get("branch_dominance_sum", 0.0) or 0.0) / count
        dominance_values.append(dominance)
        merge_metrics[label] = {
            "primary_share": round(primary_share, 4),
            "secondary_share": round(secondary_share, 4),
            "dominance": round(dominance, 4),
        }
    if merge_metrics:
        routing["merge_stage_metrics"] = merge_metrics
        routing["branch_weight_named"] = merge_metrics
        routing["routed_branch_share"] = merge_metrics.get("routed->input", {}).get(
            "primary_share"
        )
        if routing["routed_branch_share"] is None:
            routing["routed_branch_share"] = merge_metrics.get("routed->skip", {}).get(
                "primary_share"
            )
        routing["dominant_branch"] = max(
            merge_metrics.items(), key=lambda item: item[1]["dominance"]
        )[0]
        routing["branch_dominance_mean"] = round(
            sum(dominance_values) / len(dominance_values), 4
        )
    return routing


def run_experiment(
    *,
    cfg: Phase2Config,
    train_batches: list[torch.Tensor],
    val_batches: list[torch.Tensor],
    variant_name: str,
    span_widths: tuple[int, ...],
    enable_curriculum: bool,
    use_calibrated_merge: bool,
) -> dict[str, Any]:
    graph = build_multiscale_variant(
        model_dim=cfg.model_dim,
        span_widths=span_widths,
        medium_op=cfg.medium_op,
        hard_op=cfg.hard_op,
        route_temperature=cfg.route_temperature,
        min_keep_fraction=cfg.min_keep_fraction,
        confidence_threshold=cfg.confidence_threshold,
        enable_curriculum=enable_curriculum,
        use_calibrated_merge=use_calibrated_merge,
    )
    model = compile_model(
        [graph], vocab_size=cfg.vocab_size, max_seq_len=cfg.seq_len
    ).to(cfg.device)
    params, buffers = clone_module_state(model)
    eval_loss_before = _functional_eval_loss(
        model, params, buffers, val_batches, cfg.vocab_size
    )
    if cfg.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device=cfg.device)
    loss_trajectory: dict[int, float] = {}
    train_start = time.perf_counter()
    train_final_loss = functional_micro_train_loop(
        model,
        params,
        buffers,
        train_batches,
        cfg.vocab_size,
        n_steps=cfg.train_steps,
        lr=cfg.lr,
        loss_trajectory=loss_trajectory,
        step_callback=_progress_callback(model, enable_curriculum),
    )
    train_ms = (time.perf_counter() - train_start) * 1000.0
    model.set_routing_progress(1.0)
    eval_loss_after = _functional_eval_loss(
        model, params, buffers, val_batches, cfg.vocab_size
    )
    _disable_subgraph_dispatch(model)
    routing = _probe_routing_telemetry(model, val_batches[0])
    tokens_trained = cfg.train_steps * cfg.batch_size * cfg.seq_len
    throughput = tokens_trained / max(train_ms / 1000.0, 1e-9)
    return {
        "variant": variant_name,
        "span_widths": list(span_widths),
        "curriculum": enable_curriculum,
        "merge_redesign": use_calibrated_merge,
        "train_final_loss": train_final_loss,
        "eval_loss_before": eval_loss_before,
        "eval_loss_after": eval_loss_after,
        "eval_loss_delta": None
        if eval_loss_before is None or eval_loss_after is None
        else eval_loss_after - eval_loss_before,
        "train_ms": round(train_ms, 3),
        "throughput_tokens_per_s": round(throughput, 3),
        "max_memory_mb": _memory_mb(cfg.device),
        "convergence_step_75pct": _convergence_steps(loss_trajectory),
        "loss_trajectory": {str(k): round(v, 6) for k, v in loss_trajectory.items()},
        "routing": routing,
    }


def _experiment_comment(row: dict[str, Any]) -> str:
    routing = row.get("routing") or {}
    comments: list[str] = []
    if (routing.get("dead_lane_rate") or 0.0) > 0.0:
        comments.append(f"dead_lane_rate={routing['dead_lane_rate']:.2f}")
    if (routing.get("routed_branch_share") or 0.0) < 0.2:
        comments.append("routed_merge_weak")
    if (routing.get("sparse_span_coverage") or 0.0) <= 0.0:
        comments.append("no_span_coverage")
    if (routing.get("route_strength_mean") or 0.0) <= 0.0:
        comments.append("no_route_strength")
    return ",".join(comments) or "healthy"


def _table(rows: list[dict[str, Any]]) -> list[str]:
    def _fmt(value: Any, digits: int = 4) -> str:
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            return "n/a"
        return f"{float(value):.{digits}f}"

    lines = [
        "| Variant | Train Loss | Eval Loss | Conv Step | Tok/s | Mem MB | Span Cov | Route Str | Lane Ent | Dead Rate | Routed Share | Branch Dominance | Comment |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        routing = row.get("routing") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    row["variant"],
                    _fmt(row.get("train_final_loss")),
                    _fmt(row.get("eval_loss_after")),
                    str(row.get("convergence_step_75pct") or "n/a"),
                    _fmt(row.get("throughput_tokens_per_s"), 1),
                    _fmt(row.get("max_memory_mb"), 1),
                    _fmt(routing.get("sparse_span_coverage")),
                    _fmt(routing.get("route_strength_mean")),
                    _fmt(routing.get("lane_entropy")),
                    _fmt(routing.get("dead_lane_rate")),
                    _fmt(routing.get("routed_branch_share")),
                    _fmt(routing.get("branch_dominance_mean")),
                    _experiment_comment(row),
                ]
            )
            + " |"
        )
    return lines


def summarize(results: dict[str, Any]) -> dict[str, Any]:
    matrix = results["comparison_matrix"]
    best = min(matrix, key=lambda row: row["eval_loss_after"])
    best_span = min(results["span_sanity"], key=lambda row: row["eval_loss_after"])
    return {
        "best_variant": best["variant"],
        "best_eval_loss": best["eval_loss_after"],
        "best_train_loss": best["train_final_loss"],
        "best_span_variant": best_span["span_widths"],
        "recommended_curriculum": best["curriculum"],
        "recommended_merge_redesign": best["merge_redesign"],
        "best_medium_op": results["config"]["medium_op"],
        "best_hard_op": results["config"]["hard_op"],
        "keep_three_scale_identity": True,
    }


def build_report(results: dict[str, Any]) -> str:
    summary = results["summary"]
    lines = [
        "# Multiscale Rich Lane Router Phase 2",
        "",
        "## Findings Summary",
        "",
        f"- Best variant: `{summary['best_variant']}`",
        f"- Best train loss: `{summary['best_train_loss']:.4f}`",
        f"- Best eval loss: `{summary['best_eval_loss']:.4f}`",
        f"- Recommended medium operator: `{summary['best_medium_op']}`",
        f"- Recommended hard operator: `{summary['best_hard_op']}`",
        f"- Recommended span set under phase 2: `{summary['best_span_variant']}`",
        "",
        "## Controlled Matrix",
        "",
        *_table(results["comparison_matrix"]),
        "",
        "## Span Sanity Check",
        "",
        *_table(results["span_sanity"]),
        "",
        "## Final Recommendation",
        "",
        f"- Keep the three-tier identity: `{summary['keep_three_scale_identity']}`",
        f"- Routing curriculum: `{summary['recommended_curriculum']}`",
        f"- Calibrated merge redesign: `{summary['recommended_merge_redesign']}`",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(results["config"], indent=2),
        "```",
    ]
    return "\n".join(lines)


def parse_args() -> Phase2Config:
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
    parser.add_argument("--train-steps", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--corpus-path", default=str(DEFAULT_CORPUS))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--route-temperature", type=float, default=0.85)
    parser.add_argument("--min-keep-fraction", type=float, default=0.125)
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--medium-op", default="conv1d_seq")
    parser.add_argument("--hard-op", default="mixed_recursion_gate")
    args = parser.parse_args()
    return Phase2Config(
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
        medium_op=args.medium_op,
        hard_op=args.hard_op,
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
    comparison_matrix = [
        run_experiment(
            cfg=cfg,
            train_batches=train_batches,
            val_batches=val_batches,
            variant_name="baseline",
            span_widths=(2, 3, 4),
            enable_curriculum=False,
            use_calibrated_merge=False,
        ),
        run_experiment(
            cfg=cfg,
            train_batches=train_batches,
            val_batches=val_batches,
            variant_name="curriculum_only",
            span_widths=(2, 3, 4),
            enable_curriculum=True,
            use_calibrated_merge=False,
        ),
        run_experiment(
            cfg=cfg,
            train_batches=train_batches,
            val_batches=val_batches,
            variant_name="merge_redesign_only",
            span_widths=(2, 3, 4),
            enable_curriculum=False,
            use_calibrated_merge=True,
        ),
        run_experiment(
            cfg=cfg,
            train_batches=train_batches,
            val_batches=val_batches,
            variant_name="curriculum_plus_merge",
            span_widths=(2, 3, 4),
            enable_curriculum=True,
            use_calibrated_merge=True,
        ),
    ]
    span_sanity = [
        run_experiment(
            cfg=cfg,
            train_batches=train_batches,
            val_batches=val_batches,
            variant_name=f"spans_{'_'.join(map(str, spans))}",
            span_widths=spans,
            enable_curriculum=True,
            use_calibrated_merge=True,
        )
        for spans in ((2,), (2, 3), (2, 4), (2, 3, 4))
    ]
    results = {
        "config": asdict(cfg),
        "comparison_matrix": comparison_matrix,
        "span_sanity": span_sanity,
    }
    results["summary"] = summarize(results)
    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_md = output_path.with_suffix(".md")
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    report_md.write_text(build_report(results), encoding="utf-8")
    print(json.dumps({"json": str(output_path), "markdown": str(report_md)}, indent=2))


if __name__ == "__main__":
    main()
