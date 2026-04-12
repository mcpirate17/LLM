#!/usr/bin/env python
"""Cross-template benchmark for locked routing winners."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable

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
    build_hybrid_sparse_triplet_variant,
    build_intelligent_multilane_variant,
    build_locked_multiscale_variant,
    build_multiscale_difficulty_variant,
    build_recursive_depth_variant,
)


DEFAULT_CORPUS = (
    Path(__file__).resolve().parents[1] / "corpus" / "wikitext103_train.npy"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "routing_template_portfolio_benchmark.json"
)


@dataclass(slots=True)
class BenchmarkConfig:
    device: str
    vocab_size: int
    seq_len: int
    batch_size: int
    model_dim: int
    train_batches: int
    val_batches: int
    preselect_steps: int
    benchmark_steps: int
    long_steps: int
    lr: float
    clip_grad: float
    corpus_path: str
    output_path: str


@dataclass(slots=True)
class CandidateSpec:
    name: str
    template: str
    builder_key: str
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


def _builder_registry() -> dict[str, Callable[..., Any]]:
    return {
        "multiscale_locked": build_locked_multiscale_variant,
        "hybrid_sparse_triplet": build_hybrid_sparse_triplet_variant,
        "multiscale_difficulty": build_multiscale_difficulty_variant,
        "intelligent_multilane": build_intelligent_multilane_variant,
        "recursive_depth": build_recursive_depth_variant,
    }


def _functional_eval_loss(
    model, batches: list[torch.Tensor], vocab_size: int
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


def _run_candidate(
    *,
    cfg: BenchmarkConfig,
    candidate: CandidateSpec,
    train_batches: list[torch.Tensor],
    val_batches: list[torch.Tensor],
    steps: int,
    seed: int,
) -> dict[str, Any]:
    _set_seed(seed)
    builder = _builder_registry()[candidate.builder_key]
    graph = builder(model_dim=cfg.model_dim, **candidate.builder_kwargs)
    model = compile_model(
        [graph], vocab_size=cfg.vocab_size, max_seq_len=cfg.seq_len
    ).to(cfg.device)
    optimizer = make_optimizer(model.parameters(), optimizer_name="adamw", lr=cfg.lr)
    if cfg.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device=cfg.device)
    eval_before = _functional_eval_loss(model, val_batches, cfg.vocab_size)
    loss_trajectory: dict[int, float] = {}
    train_start = time.perf_counter()
    model.train()
    for step in range(steps):
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
    train_ms = (time.perf_counter() - train_start) * 1000.0
    eval_after = _functional_eval_loss(model, val_batches, cfg.vocab_size)
    routing = _probe_routing_telemetry(model, val_batches[0])
    throughput = (steps * cfg.batch_size * cfg.seq_len) / max(train_ms / 1000.0, 1e-9)
    return {
        "candidate": candidate.name,
        "template": candidate.template,
        "train_final_loss": loss_trajectory[max(loss_trajectory)]
        if loss_trajectory
        else float("inf"),
        "eval_loss_before": eval_before,
        "eval_loss_after": eval_after,
        "throughput_tokens_per_s": round(throughput, 3),
        "max_memory_mb": _memory_mb(cfg.device),
        "routing": routing,
        "builder_kwargs": candidate.builder_kwargs,
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
    }
    routing = {}
    for key in (
        "dead_lane_rate",
        "lane_entropy",
        "route_strength_mean",
        "sparse_span_coverage",
        "routed_branch_share",
        "branch_dominance_mean",
    ):
        vals = _routing_vals(key)
        if vals:
            routing[key] = {"mean": mean(vals), "std": pstdev(vals)}
    aggregate["routing"] = routing
    aggregate["quality_per_cost"] = aggregate["throughput_mean"] / max(
        aggregate["eval_loss_mean"], 1e-9
    )
    return aggregate


def _select_intelligent_candidate(
    *,
    cfg: BenchmarkConfig,
    corpus: np.ndarray,
) -> dict[str, Any]:
    options: list[CandidateSpec] = []
    for easy_op in ("conv_only", "cheap_verify_blend"):
        for medium_op in (
            "adaptive_lane_mixer",
            "block_sparse_linear",
            "rwkv_time_mixing",
        ):
            for hard_op in ("moe_topk", "adaptive_recursion"):
                options.append(
                    CandidateSpec(
                        name=f"intelligent_multilane[{easy_op},{medium_op},{hard_op}]",
                        template="intelligent_multilane_router",
                        builder_key="intelligent_multilane",
                        builder_kwargs={
                            "easy_op": easy_op,
                            "medium_op": medium_op,
                            "hard_op": hard_op,
                        },
                        notes="Phase-6 preselection sweep",
                    )
                )
    train_batches, val_batches = _tensor_batches(
        corpus,
        device=cfg.device,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        train_batches=cfg.train_batches,
        val_batches=cfg.val_batches,
        seed=7,
    )
    rows = [
        _run_candidate(
            cfg=cfg,
            candidate=candidate,
            train_batches=train_batches,
            val_batches=val_batches,
            steps=cfg.preselect_steps,
            seed=7,
        )
        for candidate in options
    ]
    best = min(rows, key=lambda row: row["eval_loss_after"])
    return {
        "candidates": rows,
        "selected": best,
        "selection_basis": "lowest short-run eval loss under the common protocol",
    }


def _select_recursive_candidate(
    *,
    cfg: BenchmarkConfig,
    corpus: np.ndarray,
) -> dict[str, Any]:
    options = [
        CandidateSpec(
            name=f"recursive_depth[max_depth={max_depth},post={post_op}]",
            template="recursive_depth_router",
            builder_key="recursive_depth",
            builder_kwargs={"max_depth": max_depth, "post_op": post_op},
            notes="Phase-6 preselection sweep",
        )
        for max_depth in (3, 4)
        for post_op in ("conv_only", "conv1d_seq")
    ]
    train_batches, val_batches = _tensor_batches(
        corpus,
        device=cfg.device,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        train_batches=cfg.train_batches,
        val_batches=cfg.val_batches,
        seed=7,
    )
    rows = [
        _run_candidate(
            cfg=cfg,
            candidate=candidate,
            train_batches=train_batches,
            val_batches=val_batches,
            steps=cfg.preselect_steps,
            seed=7,
        )
        for candidate in options
    ]
    best = min(rows, key=lambda row: row["eval_loss_after"])
    return {
        "candidates": rows,
        "selected": best,
        "selection_basis": "lowest short-run eval loss under the common protocol",
    }


def _make_locked_candidates(preselection: dict[str, Any]) -> list[CandidateSpec]:
    intelligent = preselection["intelligent"]["selected"]
    recursive = preselection["recursive"]["selected"]
    return [
        CandidateSpec(
            name="multiscale_locked_prod",
            template="multiscale_rich_lane_router",
            builder_key="multiscale_locked",
            builder_kwargs={"span_widths": (2, 3)},
            notes="Locked production default",
        ),
        CandidateSpec(
            name="multiscale_locked_hq",
            template="multiscale_rich_lane_router",
            builder_key="multiscale_locked",
            builder_kwargs={"span_widths": (2, 3, 4)},
            notes="Locked optional high-quality mode",
        ),
        CandidateSpec(
            name="hybrid_sparse_triplet_locked",
            template="hybrid_sparse_triplet_router",
            builder_key="hybrid_sparse_triplet",
            builder_kwargs={},
            notes="Cheap sparse-routing baseline",
        ),
        CandidateSpec(
            name="multiscale_difficulty_locked",
            template="multiscale_difficulty_router",
            builder_key="multiscale_difficulty",
            builder_kwargs={},
            notes="Simpler multiscale sibling with fixed hard MoE",
        ),
        CandidateSpec(
            name="intelligent_multilane_locked",
            template="intelligent_multilane_router",
            builder_key="intelligent_multilane",
            builder_kwargs=intelligent["builder_kwargs"],
            notes="Best preselected intelligent_multilane config",
        ),
        CandidateSpec(
            name="recursive_depth_locked",
            template="recursive_depth_router",
            builder_key="recursive_depth",
            builder_kwargs=recursive["builder_kwargs"],
            notes="Best preselected depth-routing config",
        ),
    ]


def _rank_rows(
    scoreboard: list[dict[str, Any]], *, key: str, reverse: bool = False
) -> list[dict[str, Any]]:
    return sorted(scoreboard, key=lambda row: row[key], reverse=reverse)


def _build_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Routing Template Portfolio Benchmark",
        "",
        "## Competitor Selection",
        "",
        "| Template | Locked Config | Rationale |",
        "| --- | --- | --- |",
    ]
    for row in payload["competitor_selection_rationale"]:
        lines.append(
            f"| {row['template']} | `{row['locked_config']}` | {row['rationale']} |"
        )
    lines.extend(
        [
            "",
            "## Unified Benchmark Scoreboard",
            "",
            "| Candidate | Template | Eval Mean | Eval Std | Long Eval | Tok/s | Mem MB | Dead Rate | Route Str | Span Cov | Quality/Cost |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["unified_benchmark_scoreboard"]:
        lines.append(
            f"| {row['candidate']} | {row['template']} | {row['eval_loss_mean']:.4f} | {row['eval_loss_std']:.4f} | "
            f"{row['long_eval_loss']:.4f} | {row['throughput_mean']:.1f} | {row['memory_mean']:.1f} | "
            f"{row['dead_lane_rate']:.4f} | {row['route_strength_mean']:.4f} | {row['sparse_span_coverage']:.4f} | {row['quality_per_cost']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Rankings",
            "",
            f"- Best raw-quality template: `{payload['portfolio_ranking']['best_raw_quality_template']}`",
            f"- Best production default template: `{payload['portfolio_ranking']['best_production_default_template']}`",
            f"- Best cost-sensitive template: `{payload['portfolio_ranking']['best_cost_sensitive_template']}`",
            f"- Best high-quality optional mode: `{payload['portfolio_ranking']['best_high_quality_optional_mode']}`",
            f"- Most robust across seeds: `{payload['portfolio_ranking']['most_robust_template']}`",
            f"- Most promising non-winning template: `{payload['portfolio_ranking']['most_promising_non_winner']}`",
            "",
            "## Strategic Recommendation",
            "",
            f"- Multiscale primary-template status: `{payload['strategic_recommendation']['multiscale_primary_status']}`",
            f"- Keep `[2,3]` production default: `{payload['strategic_recommendation']['keep_multiscale_prod_default']}`",
            f"- Keep `[2,3,4]` premium mode: `{payload['strategic_recommendation']['keep_multiscale_hq_mode']}`",
            f"- Next investment target: `{payload['strategic_recommendation']['next_investment_target']}`",
            f"- Next work focus: `{payload['strategic_recommendation']['next_work_focus']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> BenchmarkConfig:
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
    parser.add_argument("--preselect-steps", type=int, default=10)
    parser.add_argument("--benchmark-steps", type=int, default=24)
    parser.add_argument("--long-steps", type=int, default=72)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--corpus-path", default=str(DEFAULT_CORPUS))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    return BenchmarkConfig(
        device=args.device,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        model_dim=args.model_dim,
        train_batches=args.train_batches,
        val_batches=args.val_batches,
        preselect_steps=args.preselect_steps,
        benchmark_steps=args.benchmark_steps,
        long_steps=args.long_steps,
        lr=args.lr,
        clip_grad=args.clip_grad,
        corpus_path=args.corpus_path,
        output_path=args.output_path,
    )


def main() -> None:
    cfg = parse_args()
    corpus = np.load(cfg.corpus_path, mmap_mode="r")
    preselection = {
        "intelligent": _select_intelligent_candidate(cfg=cfg, corpus=corpus),
        "recursive": _select_recursive_candidate(cfg=cfg, corpus=corpus),
    }
    locked_candidates = _make_locked_candidates(preselection)

    benchmark = {}
    for candidate in locked_candidates:
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
                _run_candidate(
                    cfg=cfg,
                    candidate=candidate,
                    train_batches=train_batches,
                    val_batches=val_batches,
                    steps=cfg.benchmark_steps,
                    seed=seed,
                )
            )
        benchmark[candidate.name] = {
            "candidate": asdict(candidate),
            "runs": runs,
            "aggregate": _aggregate_runs(runs),
        }

    long_horizon = {}
    for candidate in locked_candidates:
        train_batches, val_batches = _tensor_batches(
            corpus,
            device=cfg.device,
            seq_len=cfg.seq_len,
            batch_size=cfg.batch_size,
            train_batches=cfg.train_batches,
            val_batches=cfg.val_batches,
            seed=7,
        )
        long_horizon[candidate.name] = _run_candidate(
            cfg=cfg,
            candidate=candidate,
            train_batches=train_batches,
            val_batches=val_batches,
            steps=cfg.long_steps,
            seed=7,
        )

    scoreboard = []
    for name, row in benchmark.items():
        agg = row["aggregate"]
        routing = agg["routing"]
        scoreboard.append(
            {
                "candidate": name,
                "template": row["candidate"]["template"],
                "eval_loss_mean": agg["eval_loss_mean"],
                "eval_loss_std": agg["eval_loss_std"],
                "long_eval_loss": long_horizon[name]["eval_loss_after"],
                "train_loss_mean": agg["train_loss_mean"],
                "throughput_mean": agg["throughput_mean"],
                "memory_mean": agg["memory_mean"],
                "dead_lane_rate": routing.get("dead_lane_rate", {}).get("mean", 0.0),
                "route_strength_mean": routing.get("route_strength_mean", {}).get(
                    "mean", 0.0
                ),
                "lane_entropy": routing.get("lane_entropy", {}).get("mean", 0.0),
                "sparse_span_coverage": routing.get("sparse_span_coverage", {}).get(
                    "mean", 0.0
                ),
                "quality_per_cost": agg["quality_per_cost"],
            }
        )
    raw_quality_ranking = _rank_rows(scoreboard, key="long_eval_loss")
    cost_adjusted_ranking = _rank_rows(scoreboard, key="quality_per_cost", reverse=True)
    robust_ranking = _rank_rows(scoreboard, key="eval_loss_std")

    promising_non_winner = next(
        row
        for row in raw_quality_ranking
        if row["candidate"] != raw_quality_ranking[0]["candidate"]
    )
    strategic = {
        "multiscale_primary_status": "keep_as_primary"
        if any(
            row["candidate"].startswith("multiscale_locked")
            and row == raw_quality_ranking[0]
            for row in raw_quality_ranking[:1]
        )
        or any(
            row["candidate"] == "multiscale_locked_prod"
            for row in cost_adjusted_ranking[:2]
        )
        else "keep_as_strong_option",
        "keep_multiscale_prod_default": any(
            row["candidate"] == "multiscale_locked_prod"
            for row in cost_adjusted_ranking[:2]
        ),
        "keep_multiscale_hq_mode": any(
            row["candidate"] == "multiscale_locked_hq"
            for row in raw_quality_ranking[:2]
        ),
        "next_investment_target": promising_non_winner["candidate"],
        "next_work_focus": "cross_template_consolidation_or_training_system_improvements"
        if raw_quality_ranking[0]["template"] != promising_non_winner["template"]
        else "slot_redesign",
    }

    selection_rationale = [
        {
            "template": "hybrid_sparse_triplet_router",
            "locked_config": "triplet sparse router + lane_conditioned_block",
            "rationale": "Cheapest sparse-routing baseline and the cleanest single-span competitor.",
        },
        {
            "template": "multiscale_difficulty_router",
            "locked_config": "fixed pair/triplet/quartet lane blocks + moe_topk hard path",
            "rationale": "Closest simpler sibling to multiscale_rich_lane_router and the main structural ablation competitor.",
        },
        {
            "template": "intelligent_multilane_router",
            "locked_config": str(
                preselection["intelligent"]["selected"]["builder_kwargs"]
            ),
            "rationale": "Architecturally distinct three-tier competitor with explicit easy/medium/hard lanes and token merge; locked from a short common-protocol preselection sweep.",
        },
        {
            "template": "recursive_depth_router",
            "locked_config": str(
                preselection["recursive"]["selected"]["builder_kwargs"]
            ),
            "rationale": "Depth-adaptive routing competitor with a much lighter structure; included as the low-complexity routing alternative.",
        },
    ]

    payload = {
        "config": asdict(cfg),
        "preselection": preselection,
        "competitor_selection_rationale": selection_rationale,
        "locked_candidates": [asdict(candidate) for candidate in locked_candidates],
        "benchmark": benchmark,
        "long_horizon": long_horizon,
        "unified_benchmark_scoreboard": scoreboard,
        "raw_quality_ranking": raw_quality_ranking,
        "cost_adjusted_ranking": cost_adjusted_ranking,
        "portfolio_ranking": {
            "best_raw_quality_template": raw_quality_ranking[0]["candidate"],
            "best_production_default_template": cost_adjusted_ranking[0]["candidate"],
            "best_cost_sensitive_template": cost_adjusted_ranking[0]["candidate"],
            "best_high_quality_optional_mode": raw_quality_ranking[0]["candidate"],
            "most_robust_template": robust_ranking[0]["candidate"],
            "most_promising_non_winner": promising_non_winner["candidate"],
        },
        "strategic_recommendation": strategic,
    }

    output_path = Path(cfg.output_path)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_path.with_suffix(".md").write_text(
        _build_markdown(payload), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
