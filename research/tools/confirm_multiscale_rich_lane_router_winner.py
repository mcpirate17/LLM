#!/usr/bin/env python
"""Winner confirmation for multiscale_rich_lane_router using the full catalogue."""

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
from torch.func import functional_call

from research.eval.stateless_training import (
    clone_module_state,
    functional_micro_train_loop,
)
from research.eval.utils import language_model_loss, make_batches
from research.synthesis.compiler import compile_model
from research.tools.audit_multiscale_rich_lane_router import build_multiscale_variant
from research.tools.audit_multiscale_rich_lane_router_phase2 import (
    _memory_mb,
    _probe_routing_telemetry,
    _progress_callback,
)
from research.tools.multiscale_catalogue import build_multiscale_registry


DEFAULT_CORPUS = (
    Path(__file__).resolve().parents[1] / "corpus" / "wikitext103_train.npy"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "multiscale_rich_lane_router_winner_confirmation.json"
)


@dataclass(slots=True)
class ConfirmConfig:
    device: str
    vocab_size: int
    seq_len: int
    batch_size: int
    model_dim: int
    train_batches: int
    val_batches: int
    sweep_steps: int
    multiseed_steps: int
    long_steps: int
    lr: float
    corpus_path: str
    output_path: str
    route_temperature: float
    min_keep_fraction: float
    confidence_threshold: float
    merge_redesign: bool


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


def _functional_eval_loss(
    model,
    params,
    buffers,
    batches: list[torch.Tensor],
    vocab_size: int,
) -> float | None:
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


def _evaluate_variant(
    *,
    cfg: ConfirmConfig,
    train_batches: list[torch.Tensor],
    val_batches: list[torch.Tensor],
    medium_op: str,
    hard_op: str,
    span_widths: tuple[int, ...],
    enable_curriculum: bool,
    steps: int,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    graph = build_multiscale_variant(
        model_dim=cfg.model_dim,
        span_widths=span_widths,
        medium_op=medium_op,
        hard_op=hard_op,
        route_temperature=cfg.route_temperature,
        min_keep_fraction=cfg.min_keep_fraction,
        confidence_threshold=cfg.confidence_threshold,
        enable_curriculum=enable_curriculum,
        use_calibrated_merge=cfg.merge_redesign,
    )
    model = compile_model(
        [graph], vocab_size=cfg.vocab_size, max_seq_len=cfg.seq_len
    ).to(cfg.device)
    params, buffers = clone_module_state(model)
    eval_before = _functional_eval_loss(
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
        n_steps=steps,
        lr=cfg.lr,
        loss_trajectory=loss_trajectory,
        step_callback=_progress_callback(model, enable_curriculum),
    )
    train_ms = (time.perf_counter() - train_start) * 1000.0
    model.set_routing_progress(1.0)
    eval_after = _functional_eval_loss(
        model, params, buffers, val_batches, cfg.vocab_size
    )
    routing = _probe_routing_telemetry(model, val_batches[0])
    throughput = (steps * cfg.batch_size * cfg.seq_len) / max(train_ms / 1000.0, 1e-9)
    return {
        "medium_op": medium_op,
        "hard_op": hard_op,
        "span_widths": list(span_widths),
        "curriculum": enable_curriculum,
        "train_final_loss": train_final_loss,
        "eval_loss_before": eval_before,
        "eval_loss_after": eval_after,
        "train_ms": round(train_ms, 3),
        "throughput_tokens_per_s": round(throughput, 3),
        "max_memory_mb": _memory_mb(cfg.device),
        "convergence_step_75pct": _convergence_step(loss_trajectory),
        "routing": routing,
    }


def _smoke_candidate(
    *,
    cfg: ConfirmConfig,
    corpus: np.ndarray,
    medium_op: str,
    hard_op: str,
) -> tuple[bool, str | None]:
    try:
        train_batches, val_batches = _tensor_batches(
            corpus,
            device=cfg.device,
            seq_len=cfg.seq_len,
            batch_size=cfg.batch_size,
            train_batches=2,
            val_batches=1,
            seed=11,
        )
        _evaluate_variant(
            cfg=cfg,
            train_batches=train_batches,
            val_batches=val_batches,
            medium_op=medium_op,
            hard_op=hard_op,
            span_widths=(2, 3, 4),
            enable_curriculum=False,
            steps=2,
            seed=11,
        )
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _dedupe_candidates(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    canonical_groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        canonical_groups.setdefault(row["canonical_name"], []).append(row)
    canonical_rows = []
    duplicates = []
    for canonical_name, group in sorted(canonical_groups.items()):
        preferred = next(
            (
                row
                for row in group
                if row["slot_ref"].split("/")[-1] == row["manifest_id"]
            ),
            group[0],
        )
        canonical_rows.append(preferred)
        if len(group) > 1:
            duplicates.append(
                {
                    "canonical_name": canonical_name,
                    "slot_refs": [row["slot_ref"] for row in group],
                }
            )
    return canonical_rows, duplicates


def _aggregate_seed_runs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def _vals(key: str) -> list[float]:
        values = [row[key] for row in rows if row.get(key) is not None]
        return [float(v) for v in values]

    routing_keys = [
        "dead_lane_rate",
        "lane_entropy",
        "sparse_span_coverage",
        "route_strength_mean",
        "routed_branch_share",
        "branch_dominance_mean",
    ]
    aggregate = {
        "train_loss_mean": mean(_vals("train_final_loss")),
        "train_loss_std": pstdev(_vals("train_final_loss")),
        "eval_loss_mean": mean(_vals("eval_loss_after")),
        "eval_loss_std": pstdev(_vals("eval_loss_after")),
        "throughput_mean": mean(_vals("throughput_tokens_per_s")),
        "memory_mean": mean(_vals("max_memory_mb")),
    }
    routing = {}
    for key in routing_keys:
        vals = [
            float(row["routing"].get(key))
            for row in rows
            if row.get("routing") and row["routing"].get(key) is not None
        ]
        if vals:
            routing[key] = {"mean": mean(vals), "std": pstdev(vals)}
    aggregate["routing"] = routing
    return aggregate


def _markdown_table(rows: list[dict[str, Any]], label_key: str) -> list[str]:
    lines = [
        "| Config | Train Loss | Eval Loss | Tok/s | Mem MB | Span Cov | Route Str | Dead Rate | Comment |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        routing = row.get("routing") or {}
        comment = "healthy"
        if (routing.get("dead_lane_rate") or 0.0) > 0.0:
            comment = f"dead_lane_rate={routing['dead_lane_rate']:.2f}"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row[label_key]),
                    f"{float(row['train_final_loss']):.4f}",
                    f"{float(row['eval_loss_after']):.4f}",
                    f"{float(row['throughput_tokens_per_s']):.1f}",
                    f"{float(row['max_memory_mb']):.1f}",
                    f"{float(routing.get('sparse_span_coverage') or 0.0):.4f}",
                    f"{float(routing.get('route_strength_mean') or 0.0):.4f}",
                    f"{float(routing.get('dead_lane_rate') or 0.0):.4f}",
                    comment,
                ]
            )
            + " |"
        )
    return lines


def parse_args() -> ConfirmConfig:
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
    parser.add_argument("--sweep-steps", type=int, default=20)
    parser.add_argument("--multiseed-steps", type=int, default=24)
    parser.add_argument("--long-steps", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--corpus-path", default=str(DEFAULT_CORPUS))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--route-temperature", type=float, default=0.85)
    parser.add_argument("--min-keep-fraction", type=float, default=0.125)
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--merge-redesign", action="store_true", default=True)
    args = parser.parse_args()
    return ConfirmConfig(
        device=args.device,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        model_dim=args.model_dim,
        train_batches=args.train_batches,
        val_batches=args.val_batches,
        sweep_steps=args.sweep_steps,
        multiseed_steps=args.multiseed_steps,
        long_steps=args.long_steps,
        lr=args.lr,
        corpus_path=args.corpus_path,
        output_path=args.output_path,
        route_temperature=args.route_temperature,
        min_keep_fraction=args.min_keep_fraction,
        confidence_threshold=args.confidence_threshold,
        merge_redesign=bool(args.merge_redesign),
    )


def main() -> None:
    cfg = parse_args()
    registry = build_multiscale_registry()
    medium_rows, medium_dupes = _dedupe_candidates(registry["medium_candidates"])
    hard_rows, hard_dupes = _dedupe_candidates(registry["hard_candidates"])

    corpus = np.load(cfg.corpus_path, mmap_mode="r")
    smoke_rows = []
    executable_medium = []
    executable_hard = []
    for row in medium_rows:
        ok, error = _smoke_candidate(
            cfg=cfg,
            corpus=corpus,
            medium_op=row["dispatch_name"],
            hard_op="mixed_recursion_gate",
        )
        smoke_rows.append(
            {"slot": "medium", "name": row["slot_ref"], "ok": ok, "error": error}
        )
        if ok:
            executable_medium.append(row)
    for row in hard_rows:
        ok, error = _smoke_candidate(
            cfg=cfg,
            corpus=corpus,
            medium_op="conv1d_seq",
            hard_op=row["dispatch_name"],
        )
        smoke_rows.append(
            {"slot": "hard", "name": row["slot_ref"], "ok": ok, "error": error}
        )
        if ok:
            executable_hard.append(row)

    train_batches, val_batches = _tensor_batches(
        corpus,
        device=cfg.device,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        train_batches=cfg.train_batches,
        val_batches=cfg.val_batches,
        seed=7,
    )

    winner = _evaluate_variant(
        cfg=cfg,
        train_batches=train_batches,
        val_batches=val_batches,
        medium_op="conv1d_seq",
        hard_op="mixed_recursion_gate",
        span_widths=(2, 3, 4),
        enable_curriculum=False,
        steps=cfg.sweep_steps,
        seed=7,
    )
    fallback = _evaluate_variant(
        cfg=cfg,
        train_batches=train_batches,
        val_batches=val_batches,
        medium_op="conv1d_seq",
        hard_op="mixed_recursion_gate",
        span_widths=(2, 3),
        enable_curriculum=False,
        steps=cfg.sweep_steps,
        seed=7,
    )

    medium_sweep = []
    for row in executable_medium:
        if row["dispatch_name"] == "conv1d_seq":
            continue
        result = _evaluate_variant(
            cfg=cfg,
            train_batches=train_batches,
            val_batches=val_batches,
            medium_op=row["dispatch_name"],
            hard_op="mixed_recursion_gate",
            span_widths=(2, 3, 4),
            enable_curriculum=False,
            steps=cfg.sweep_steps,
            seed=7,
        )
        result["label"] = row["slot_ref"]
        result["canonical_name"] = row["canonical_name"]
        medium_sweep.append(result)

    hard_sweep = []
    for row in executable_hard:
        if row["dispatch_name"] == "mixed_recursion_gate":
            continue
        result = _evaluate_variant(
            cfg=cfg,
            train_batches=train_batches,
            val_batches=val_batches,
            medium_op="conv1d_seq",
            hard_op=row["dispatch_name"],
            span_widths=(2, 3, 4),
            enable_curriculum=False,
            steps=cfg.sweep_steps,
            seed=7,
        )
        result["label"] = row["slot_ref"]
        result["canonical_name"] = row["canonical_name"]
        hard_sweep.append(result)

    best_medium = min(medium_sweep, key=lambda row: row["eval_loss_after"])
    best_hard = min(hard_sweep, key=lambda row: row["eval_loss_after"])
    best_challenger = min(
        [
            {
                "kind": "medium",
                "medium_op": best_medium["medium_op"],
                "hard_op": "mixed_recursion_gate",
                "span_widths": (2, 3, 4),
                "label": best_medium["label"],
                "eval_loss_after": best_medium["eval_loss_after"],
            },
            {
                "kind": "hard",
                "medium_op": "conv1d_seq",
                "hard_op": best_hard["hard_op"],
                "span_widths": (2, 3, 4),
                "label": best_hard["label"],
                "eval_loss_after": best_hard["eval_loss_after"],
            },
        ],
        key=lambda row: row["eval_loss_after"],
    )

    challenger_fallback = _evaluate_variant(
        cfg=cfg,
        train_batches=train_batches,
        val_batches=val_batches,
        medium_op=best_challenger["medium_op"],
        hard_op=best_challenger["hard_op"],
        span_widths=(2, 3),
        enable_curriculum=False,
        steps=cfg.sweep_steps,
        seed=7,
    )

    seed_rows = {}
    for label, spec in {
        "winner": ("conv1d_seq", "mixed_recursion_gate", (2, 3, 4), False),
        "fallback": ("conv1d_seq", "mixed_recursion_gate", (2, 3), False),
        "challenger": (
            best_challenger["medium_op"],
            best_challenger["hard_op"],
            tuple(best_challenger["span_widths"]),
            False,
        ),
        "challenger_fallback": (
            best_challenger["medium_op"],
            best_challenger["hard_op"],
            (2, 3),
            False,
        ),
    }.items():
        rows = []
        for seed in (7, 17, 27):
            train_b, val_b = _tensor_batches(
                corpus,
                device=cfg.device,
                seq_len=cfg.seq_len,
                batch_size=cfg.batch_size,
                train_batches=cfg.train_batches,
                val_batches=cfg.val_batches,
                seed=seed,
            )
            rows.append(
                _evaluate_variant(
                    cfg=cfg,
                    train_batches=train_b,
                    val_batches=val_b,
                    medium_op=spec[0],
                    hard_op=spec[1],
                    span_widths=spec[2],
                    enable_curriculum=spec[3],
                    steps=cfg.multiseed_steps,
                    seed=seed,
                )
            )
        seed_rows[label] = {"runs": rows, "aggregate": _aggregate_seed_runs(rows)}

    long_horizon = {
        "winner": _evaluate_variant(
            cfg=cfg,
            train_batches=train_batches,
            val_batches=val_batches,
            medium_op="conv1d_seq",
            hard_op="mixed_recursion_gate",
            span_widths=(2, 3, 4),
            enable_curriculum=False,
            steps=cfg.long_steps,
            seed=7,
        ),
        "curriculum_variant": _evaluate_variant(
            cfg=cfg,
            train_batches=train_batches,
            val_batches=val_batches,
            medium_op="conv1d_seq",
            hard_op="mixed_recursion_gate",
            span_widths=(2, 3, 4),
            enable_curriculum=True,
            steps=cfg.long_steps,
            seed=7,
        ),
        "challenger": _evaluate_variant(
            cfg=cfg,
            train_batches=train_batches,
            val_batches=val_batches,
            medium_op=best_challenger["medium_op"],
            hard_op=best_challenger["hard_op"],
            span_widths=tuple(best_challenger["span_widths"]),
            enable_curriculum=False,
            steps=cfg.long_steps,
            seed=7,
        ),
    }

    default_choice = min(
        [
            ("winner", seed_rows["winner"]["aggregate"]["eval_loss_mean"]),
            ("challenger", seed_rows["challenger"]["aggregate"]["eval_loss_mean"]),
        ],
        key=lambda item: item[1],
    )[0]
    simplification_choice = (
        "challenger_fallback" if default_choice == "challenger" else "fallback"
    )

    results = {
        "config": asdict(cfg),
        "catalogue": registry,
        "candidate_pool_duplicates": {
            "medium": medium_dupes,
            "hard": hard_dupes,
        },
        "smoke": smoke_rows,
        "winner_baseline": winner,
        "fallback_baseline": fallback,
        "challenger_fallback_baseline": challenger_fallback,
        "medium_sweep": sorted(medium_sweep, key=lambda row: row["eval_loss_after"]),
        "hard_sweep": sorted(hard_sweep, key=lambda row: row["eval_loss_after"]),
        "best_medium_substitute": best_medium,
        "best_hard_substitute": best_hard,
        "best_challenger": best_challenger,
        "multi_seed": seed_rows,
        "long_horizon": long_horizon,
        "deployment_recommendation": {
            "default_config": default_choice,
            "first_simplification_fallback": simplification_choice,
            "best_challenger": best_challenger["label"],
            "curriculum_default": False,
            "width4_justified": seed_rows["challenger"]["aggregate"]["eval_loss_mean"]
            <= seed_rows["challenger_fallback"]["aggregate"]["eval_loss_mean"],
        },
    }

    report_lines = [
        "# Multiscale Rich Lane Router Winner Confirmation",
        "",
        "## Catalogue Hygiene",
        "",
        f"- Total catalogue size: `{registry['summary']['total_catalogue_size']}`",
        f"- Canonical component count: `{registry['summary']['canonical_component_count']}`",
        f"- Routing component count: `{registry['summary']['routing_component_count']}`",
        f"- Reachable-for-template count: `{registry['summary']['reachable_for_template_count']}`",
        f"- Medium candidate count: `{registry['summary']['medium_candidate_count']}`",
        f"- Hard candidate count: `{registry['summary']['hard_candidate_count']}`",
        "",
        "## Candidate Pool Deduplication",
        "",
        f"- Medium logical duplicates removed: `{len(medium_dupes)}`",
        f"- Hard logical duplicates removed: `{len(hard_dupes)}`",
        "",
        "## Medium Sweep",
        "",
        *_markdown_table(results["medium_sweep"][:6], "label"),
        "",
        "## Hard Sweep",
        "",
        *_markdown_table(results["hard_sweep"][:6], "label"),
        "",
        "## Final Scoreboard",
        "",
        *_markdown_table(
            [
                {**winner, "label": "winner"},
                {**fallback, "label": "fallback"},
                {**challenger_fallback, "label": "challenger_fallback"},
                {
                    **results["best_medium_substitute"],
                    "label": f"medium:{results['best_medium_substitute']['label']}",
                },
                {
                    **results["best_hard_substitute"],
                    "label": f"hard:{results['best_hard_substitute']['label']}",
                },
            ],
            "label",
        ),
        "",
        "## Deployment Recommendation",
        "",
        f"- Default config: `{results['deployment_recommendation']['default_config']}`",
        f"- First simplification fallback: `{results['deployment_recommendation']['first_simplification_fallback']}`",
        f"- Best challenger: `{results['deployment_recommendation']['best_challenger']}`",
        f"- Curriculum default: `{results['deployment_recommendation']['curriculum_default']}`",
        f"- Width 4 justified: `{results['deployment_recommendation']['width4_justified']}`",
    ]

    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    output_path.with_suffix(".md").write_text("\n".join(report_lines), encoding="utf-8")
    print(
        json.dumps(
            {"json": str(output_path), "markdown": str(output_path.with_suffix(".md"))},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
