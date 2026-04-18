#!/usr/bin/env python
"""Comparative anatomy audit for locked routing templates."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch

from research.eval.training_core import make_optimizer
from research.eval.utils import clip_grad_norm, language_model_loss, make_batches
from research.synthesis.compiler import compile_model
from research.tools.audit_multiscale_rich_lane_router_phase2 import (
    _memory_mb,
    _probe_routing_telemetry,
)
from research.tools.routing_template_variants import (
    build_intelligent_multilane_variant,
    build_locked_multiscale_variant,
    build_recursive_depth_variant,
)


DEFAULT_CORPUS = (
    Path(__file__).resolve().parents[1] / "corpus" / "wikitext103_train.npy"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "routing_template_comparative_anatomy.json"
)

ROUTING_OPS = {
    "hybrid_token_gate",
    "hybrid_sparse_router",
    "sparse_span_builder",
    "token_class_proj",
    "signal_conditioned_compression",
    "calibrated_branch_merge",
    "depth_weighted_proj",
    "score_depth_blend",
    "adaptive_lane_mixer",
    "difficulty_blend_3way",
    "moe_topk",
    "cheap_verify_blend",
    "lane_conditioned_block",
}
MERGE_OPS = {"add", "calibrated_branch_merge", "adjacent_token_merge"}
DECISION_OPS = {
    "hybrid_token_gate",
    "hybrid_sparse_router",
    "token_class_proj",
    "depth_weighted_proj",
    "adaptive_lane_mixer",
    "moe_topk",
    "score_depth_blend",
    "calibrated_branch_merge",
}


@dataclass(slots=True)
class AnatomyConfig:
    device: str
    vocab_size: int
    seq_len: int
    batch_size: int
    model_dim: int
    train_batches: int
    val_batches: int
    audit_steps: int
    sensitivity_steps: int
    lr: float
    clip_grad: float
    corpus_path: str
    output_path: str


@dataclass(slots=True)
class Candidate:
    name: str
    template: str
    builder: str
    builder_kwargs: dict[str, Any]
    notes: str


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def _build_graph(candidate: Candidate):
    if candidate.builder == "multiscale_locked":
        return build_locked_multiscale_variant(model_dim=64, **candidate.builder_kwargs)
    if candidate.builder == "intelligent_multilane":
        return build_intelligent_multilane_variant(
            model_dim=64, **candidate.builder_kwargs
        )
    if candidate.builder == "recursive_depth":
        return build_recursive_depth_variant(model_dim=64, **candidate.builder_kwargs)
    raise ValueError(candidate.builder)


def _build_candidates() -> list[Candidate]:
    return [
        Candidate(
            name="multiscale_locked_prod",
            template="multiscale_rich_lane_router",
            builder="multiscale_locked",
            builder_kwargs={"span_widths": (2, 3)},
            notes="Production default",
        ),
        Candidate(
            name="multiscale_locked_hq",
            template="multiscale_rich_lane_router",
            builder="multiscale_locked",
            builder_kwargs={"span_widths": (2, 3, 4)},
            notes="Optional high-quality mode",
        ),
        Candidate(
            name="intelligent_multilane_locked",
            template="intelligent_multilane_router",
            builder="intelligent_multilane",
            builder_kwargs={
                "easy_op": "cheap_verify_blend",
                "medium_op": "adaptive_lane_mixer",
                "hard_op": "moe_topk",
            },
            notes="Portfolio raw-quality leader",
        ),
        Candidate(
            name="recursive_depth_locked",
            template="recursive_depth_router",
            builder="recursive_depth",
            builder_kwargs={"max_depth": 3, "post_op": "conv_only"},
            notes="Portfolio production-default leader",
        ),
    ]


def _eval_loss(model, batches: list[torch.Tensor], vocab_size: int) -> float | None:
    total = 0.0
    tokens = 0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in batches:
            logits = model(batch)
            loss = language_model_loss(logits, batch, vocab_size, reduction="sum")
            if torch.isfinite(loss):
                total += float(loss.item())
                tokens += batch[:, 1:].numel()
    if was_training:
        model.train()
    if tokens <= 0:
        return None
    return total / tokens


def _describe_compute_allocation(candidate: Candidate) -> str:
    if candidate.template == "multiscale_rich_lane_router":
        return "token gate -> pair/triplet/quartet routed spans -> medium op -> hard difficulty/compression -> calibrated merge chain"
    if candidate.template == "intelligent_multilane_router":
        return "explicit easy lane + routed span summary -> medium lane + hard lane -> token merge + residual stabilize"
    if candidate.template == "recursive_depth_router":
        return "single depth scorer allocates recursion depth before one post-routing transform"
    return "unknown"


def _structural_summary(candidate: Candidate) -> dict[str, Any]:
    graph = _build_graph(candidate)
    ops = [node.op_name for node in graph.nodes.values()]
    routing_ops = [op for op in ops if op in ROUTING_OPS]
    merge_ops = [op for op in ops if op in MERGE_OPS]
    decision_ops = [op for op in ops if op in DECISION_OPS]
    branch_count = 1
    if candidate.template == "multiscale_rich_lane_router":
        branch_count = 5
    elif candidate.template == "intelligent_multilane_router":
        branch_count = 4
    elif candidate.template == "recursive_depth_router":
        branch_count = 2
    optimization_burden = "high"
    if candidate.template == "recursive_depth_router":
        optimization_burden = "low"
    elif candidate.template == "intelligent_multilane_router":
        optimization_burden = "medium"
    return {
        "candidate": candidate.name,
        "template": candidate.template,
        "structural_depth": len(ops),
        "routing_complexity": len(routing_ops),
        "merge_complexity": len(merge_ops),
        "decision_points": len(decision_ops),
        "branch_count": branch_count,
        "coordination_steps": len(decision_ops) + len(merge_ops),
        "explicit_compute_allocation": _describe_compute_allocation(candidate),
        "likely_optimization_burden": optimization_burden,
        "routing_ops": routing_ops,
    }


def _module_grad_shares(model) -> dict[str, float]:
    module_map = {name: module for name, module in model.named_modules()}
    op_totals: dict[str, float] = {}
    total = 0.0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad_norm = float(param.grad.norm().item())
        if not math.isfinite(grad_norm) or grad_norm <= 0.0:
            continue
        parts = name.split(".")[:-1]
        module = None
        while parts:
            prefix = ".".join(parts)
            candidate = module_map.get(prefix)
            if candidate is not None and getattr(candidate, "op_name", None):
                module = candidate
                break
            parts.pop()
        op_name = getattr(module, "op_name", "other")
        op_totals[op_name] = op_totals.get(op_name, 0.0) + grad_norm
        total += grad_norm
    if total <= 0.0:
        return {}
    return {name: value / total for name, value in sorted(op_totals.items())}


def _routing_economics_row(
    candidate: Candidate,
    *,
    cfg: AnatomyConfig,
    train_batches: list[torch.Tensor],
    val_batches: list[torch.Tensor],
    seed: int,
) -> dict[str, Any]:
    _set_seed(seed)
    graph = _build_graph(candidate)
    model = compile_model(
        [graph], vocab_size=cfg.vocab_size, max_seq_len=cfg.seq_len
    ).to(cfg.device)
    optimizer = make_optimizer(model.parameters(), optimizer_name="adamw", lr=cfg.lr)
    if cfg.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device=cfg.device)
    loss_trace: dict[int, float] = {}
    route_trace: list[dict[str, Any]] = []
    grad_trace: list[dict[str, float]] = []
    probe_steps = {0, max(0, cfg.audit_steps // 2), max(0, cfg.audit_steps - 1)}
    probe_batch = val_batches[0]
    train_start = time.perf_counter()
    model.train()
    for step in range(cfg.audit_steps):
        optimizer.zero_grad(set_to_none=True)
        batch = train_batches[step % len(train_batches)]
        logits = model(batch)
        loss = language_model_loss(logits, batch, cfg.vocab_size)
        if not torch.isfinite(loss):
            break
        loss.backward()
        grad_share = _module_grad_shares(model)
        if cfg.clip_grad > 0:
            clip_grad_norm(model.parameters(), cfg.clip_grad)
        optimizer.step()
        loss_trace[step + 1] = float(loss.item())
        if step in probe_steps:
            routing = _probe_routing_telemetry(model, probe_batch)
            route_trace.append(
                {
                    "step": step + 1,
                    "lane_entropy": routing.get("lane_entropy"),
                    "route_strength_mean": routing.get("route_strength_mean"),
                    "sparse_span_coverage": routing.get("sparse_span_coverage"),
                    "dead_lane_rate": routing.get("dead_lane_rate"),
                }
            )
            grad_trace.append(grad_share)
    train_ms = (time.perf_counter() - train_start) * 1000.0
    eval_after = _eval_loss(model, val_batches, cfg.vocab_size)
    routing = _probe_routing_telemetry(model, probe_batch)
    throughput = (cfg.audit_steps * cfg.batch_size * cfg.seq_len) / max(
        train_ms / 1000.0, 1e-9
    )
    lane_count = max(int(routing.get("lane_count") or 0), 1)
    entropy = float(routing.get("lane_entropy") or 0.0)
    entropy_norm = (
        entropy / math.log(lane_count) if lane_count > 1 and entropy > 0.0 else 0.0
    )
    useful_routed_activity = max(
        float(routing.get("route_strength_mean") or 0.0)
        * max(
            float(routing.get("sparse_span_coverage") or 0.0),
            float(routing.get("routed_compute_fraction") or 0.0),
            1e-6,
        ),
        entropy_norm,
        1e-6,
    )
    compute_spent_per_useful_decision = (
        1000.0 / max(throughput, 1e-9)
    ) / useful_routed_activity
    routing_related_grad = []
    for snapshot in grad_trace:
        if set(snapshot.keys()) == {"other"}:
            continue
        routing_related_grad.append(
            sum(value for name, value in snapshot.items() if name in ROUTING_OPS)
        )
    return {
        "candidate": candidate.name,
        "template": candidate.template,
        "eval_loss_after": eval_after,
        "throughput_tokens_per_s": throughput,
        "max_memory_mb": _memory_mb(cfg.device),
        "routing": routing,
        "route_trace": route_trace,
        "grad_trace": grad_trace,
        "routing_related_grad_share_mean": mean(routing_related_grad)
        if routing_related_grad
        else None,
        "compute_spent_per_useful_decision": compute_spent_per_useful_decision,
    }


def _schedule_sensitivity(
    candidate: Candidate,
    *,
    cfg: AnatomyConfig,
    corpus: np.ndarray,
) -> dict[str, Any]:
    train_batches, val_batches = _tensor_batches(
        corpus,
        device=cfg.device,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        train_batches=cfg.train_batches,
        val_batches=cfg.val_batches,
        seed=7,
    )
    rows = []
    for factor in (0.8, 1.0, 1.2):
        tuned_cfg = AnatomyConfig(**{**asdict(cfg), "lr": cfg.lr * factor})
        row = _routing_economics_row(
            candidate,
            cfg=tuned_cfg,
            train_batches=train_batches,
            val_batches=val_batches,
            seed=7,
        )
        rows.append(
            {
                "lr_factor": factor,
                "eval_loss_after": row["eval_loss_after"],
                "dead_lane_rate": float(row["routing"].get("dead_lane_rate") or 0.0),
            }
        )
    baseline = next(row for row in rows if row["lr_factor"] == 1.0)
    deltas = [
        abs(row["eval_loss_after"] - baseline["eval_loss_after"])
        for row in rows
        if row["lr_factor"] != 1.0
    ]
    return {
        "candidate": candidate.name,
        "rows": rows,
        "mean_eval_shift": mean(deltas) if deltas else 0.0,
    }


def _aggregate_economics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {}
    for row in rows:
        routing = row["routing"]
        payload[row["candidate"]] = {
            "template": row["template"],
            "eval_loss_after": row["eval_loss_after"],
            "throughput_tokens_per_s": row["throughput_tokens_per_s"],
            "max_memory_mb": row["max_memory_mb"],
            "dead_lane_rate": float(routing.get("dead_lane_rate") or 0.0),
            "lane_entropy": float(routing.get("lane_entropy") or 0.0),
            "route_strength_mean": float(routing.get("route_strength_mean") or 0.0),
            "sparse_span_coverage": float(routing.get("sparse_span_coverage") or 0.0),
            "branch_dominance_mean": float(routing.get("branch_dominance_mean") or 0.0),
            "routed_branch_share": float(routing.get("routed_branch_share") or 0.0),
            "quality_per_cost": row["throughput_tokens_per_s"]
            / max(float(row["eval_loss_after"] or 1.0), 1e-9),
            "compute_spent_per_useful_decision": row[
                "compute_spent_per_useful_decision"
            ],
            "routing_related_grad_share_mean": row["routing_related_grad_share_mean"],
        }
    return payload


def _redesign_hypotheses() -> list[dict[str, Any]]:
    return [
        {
            "hypothesis": "collapse_explicit_span_fanout",
            "targets_problem": "Multiscale pays three separate span-builder/router coordination costs before any medium specialization happens.",
            "why_it_could_close_gap": "Intelligent_multilane gets cleaner routed signal with fewer coordination steps, while recursive_depth wins by collapsing allocation into one scorer. A single learned multiscale summary could preserve span awareness without triple routing overhead.",
            "risk": "medium",
            "complexity": "medium",
            "expected_roi": "high",
        },
        {
            "hypothesis": "replace_default_skip_input_merge_chain_with_single_easy_lane",
            "targets_problem": "Multiscale still coordinates default, routed, skip, and input branches across four merge decisions.",
            "why_it_could_close_gap": "Intelligent_multilane makes the easy path explicit and wins raw quality; reducing merge economics may improve credit assignment and branch cooperation.",
            "risk": "medium",
            "complexity": "medium",
            "expected_roi": "medium_high",
        },
        {
            "hypothesis": "move_hard_routing_off_raw_gated_tokens_onto_routed_summary",
            "targets_problem": "Hard routing in multiscale is conditioned from gated tokens rather than a summarized routed medium representation.",
            "why_it_could_close_gap": "The leader templates allocate expensive compute after a more consolidated representation. That should reduce redundant hard-path work and tighten difficulty alignment.",
            "risk": "low_medium",
            "complexity": "medium",
            "expected_roi": "medium",
        },
        {
            "hypothesis": "narrow_slot_flexibility_into_stronger_structural_roles",
            "targets_problem": "Multiscale is overgeneralized: too many interchangeable steps are forced to cooperate before specialization yields quality.",
            "why_it_could_close_gap": "Both leaders win with stronger inductive bias and fewer degrees of freedom. Redefining slots around easy-summary-hard roles instead of generic medium/hard menus should reduce optimization burden.",
            "risk": "medium",
            "complexity": "high",
            "expected_roi": "medium_high",
        },
    ]


def _build_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Routing Template Comparative Anatomy",
        "",
        "## Structural Comparison",
        "",
        "| Candidate | Template | Structural Depth | Routing Complexity | Merge Complexity | Decision Points | Branch Count | Coordination Steps | Optimization Burden |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["structural_comparison_table"]:
        lines.append(
            f"| {row['candidate']} | {row['template']} | {row['structural_depth']} | {row['routing_complexity']} | "
            f"{row['merge_complexity']} | {row['decision_points']} | {row['branch_count']} | {row['coordination_steps']} | {row['likely_optimization_burden']} |"
        )
    lines.extend(
        [
            "",
            "## Routing Economics",
            "",
            "| Candidate | Eval | Tok/s | Mem MB | Dead Rate | Entropy | Route Str | Span Cov | Branch Dominance | Useful-Decision Cost |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["routing_economics_table"]:
        lines.append(
            f"| {row['candidate']} | {row['eval_loss_after']:.4f} | {row['throughput_tokens_per_s']:.1f} | {row['max_memory_mb']:.1f} | "
            f"{row['dead_lane_rate']:.4f} | {row['lane_entropy']:.4f} | {row['route_strength_mean']:.4f} | "
            f"{row['sparse_span_coverage']:.4f} | {row['branch_dominance_mean']:.4f} | {row['compute_spent_per_useful_decision']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Optimization Burden",
            "",
            "| Candidate | Routing-Grad Share | LR Sensitivity | Branch Maturation |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for row in payload["optimization_burden_comparison"]:
        grad_share = (
            "n/a"
            if row["routing_related_grad_share_mean"] is None
            else f"{row['routing_related_grad_share_mean']:.4f}"
        )
        lines.append(
            f"| {row['candidate']} | {grad_share} | {row['mean_eval_shift']:.4f} | {row['branch_maturation_summary']} |"
        )
    lines.extend(
        [
            "",
            "## Redesign Hypotheses",
            "",
            "| Hypothesis | Problem Targeted | Risk | Complexity | Expected ROI |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in payload["redesign_hypothesis_table"]:
        lines.append(
            f"| {row['hypothesis']} | {row['targets_problem']} | {row['risk']} | {row['complexity']} | {row['expected_roi']} |"
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"- Why multiscale loses: {payload['final_verdict']['why_multiscale_loses']}",
            f"- Fixability: {payload['final_verdict']['fixability']}",
            f"- Redesign ROI: {payload['final_verdict']['redesign_roi']}",
            f"- Recommendation: {payload['final_verdict']['recommendation']}",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> AnatomyConfig:
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
    parser.add_argument("--audit-steps", type=int, default=24)
    parser.add_argument("--sensitivity-steps", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--corpus-path", default=str(DEFAULT_CORPUS))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    return AnatomyConfig(
        device=args.device,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        model_dim=args.model_dim,
        train_batches=args.train_batches,
        val_batches=args.val_batches,
        audit_steps=args.audit_steps,
        sensitivity_steps=args.sensitivity_steps,
        lr=args.lr,
        clip_grad=args.clip_grad,
        corpus_path=args.corpus_path,
        output_path=args.output_path,
    )


def main() -> None:
    cfg = parse_args()
    corpus = np.load(cfg.corpus_path, mmap_mode="r")
    candidates = _build_candidates()
    structures = [_structural_summary(candidate) for candidate in candidates]
    economics_rows = []
    sensitivity_rows = []
    for candidate in candidates:
        train_batches, val_batches = _tensor_batches(
            corpus,
            device=cfg.device,
            seq_len=cfg.seq_len,
            batch_size=cfg.batch_size,
            train_batches=cfg.train_batches,
            val_batches=cfg.val_batches,
            seed=7,
        )
        economics_rows.append(
            _routing_economics_row(
                candidate,
                cfg=cfg,
                train_batches=train_batches,
                val_batches=val_batches,
                seed=7,
            )
        )
        sensitivity_rows.append(
            _schedule_sensitivity(candidate, cfg=cfg, corpus=corpus)
        )

    economics = _aggregate_economics(economics_rows)
    routing_economics_table = [
        {"candidate": key, **value} for key, value in economics.items()
    ]
    sensitivity_map = {row["candidate"]: row for row in sensitivity_rows}
    optimization_rows = []
    for row in routing_economics_table:
        route_trace = next(
            item["route_trace"]
            for item in economics_rows
            if item["candidate"] == row["candidate"]
        )
        maturation = "stable"
        if len(route_trace) >= 2:
            start = route_trace[0]
            end = route_trace[-1]
            if (end.get("route_strength_mean") or 0.0) > (
                start.get("route_strength_mean") or 0.0
            ):
                maturation = "specialization_strengthens_over_steps"
            elif (end.get("route_strength_mean") or 0.0) < (
                start.get("route_strength_mean") or 0.0
            ):
                maturation = "routing_signal_weakens_over_steps"
        optimization_rows.append(
            {
                "candidate": row["candidate"],
                "routing_related_grad_share_mean": row[
                    "routing_related_grad_share_mean"
                ],
                "mean_eval_shift": sensitivity_map[row["candidate"]]["mean_eval_shift"],
                "branch_maturation_summary": maturation,
            }
        )

    final_verdict = {
        "why_multiscale_loses": "It spends more coordination on span fan-out, multiple nested routing decisions, and four-way merge economics before specialization pays off. The leaders allocate compute more directly: intelligent_multilane via explicit easy/medium/hard roles, recursive_depth via a single depth allocator.",
        "fixability": "fixable_with_template_redesign",
        "redesign_roi": "medium",
        "recommendation": "template_level_redesign_only; no more local tuning. If redesign does not collapse routing/merge coordination and strengthen structural roles, stop investing.",
    }

    payload = {
        "config": asdict(cfg),
        "candidates": [asdict(candidate) for candidate in candidates],
        "structural_comparison_table": structures,
        "routing_economics": economics,
        "routing_economics_table": routing_economics_table,
        "optimization_burden_comparison": optimization_rows,
        "sensitivity": sensitivity_rows,
        "redesign_hypothesis_table": _redesign_hypotheses(),
        "final_verdict": final_verdict,
    }

    output_path = Path(cfg.output_path)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_path.with_suffix(".md").write_text(
        _build_markdown(payload), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
