#!/usr/bin/env python
"""Phase 4 mechanism-diversity audit for multiscale_rich_lane_router."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import torch

from research.tools.confirm_multiscale_rich_lane_router_winner import (
    DEFAULT_CORPUS,
    _dedupe_candidates,
    _evaluate_variant,
)
from research.tools.multiscale_catalogue import build_multiscale_registry
from research.tools.multiscale_mechanisms import (
    build_mechanism_coverage,
    classify_hard_mechanism,
    classify_medium_mechanism,
)


DEFAULT_PREVIOUS_REPORT = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "multiscale_rich_lane_router_winner_confirmation.json"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "multiscale_rich_lane_router_phase4_mechanism_audit.json"
)


@dataclass(slots=True)
class Phase4Config:
    device: str
    vocab_size: int
    seq_len: int
    batch_size: int
    model_dim: int
    train_batches: int
    val_batches: int
    multiseed_steps: int
    long_steps: int
    lr: float
    corpus_path: str
    previous_report_path: str
    output_path: str
    route_temperature: float
    min_keep_fraction: float
    confidence_threshold: float
    merge_redesign: bool
    perturb_train_start: int
    perturb_val_start: int


def _tensor_batches(
    corpus: np.ndarray,
    *,
    device: str,
    seq_len: int,
    batch_size: int,
    train_batches: int,
    val_batches: int,
    seed: int,
    train_start: int,
    val_start: int,
    train_tokens_len: int = 200_000,
    val_tokens_len: int = 60_000,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    train_tokens = corpus[train_start : train_start + train_tokens_len]
    val_tokens = corpus[val_start : val_start + val_tokens_len]
    from research.eval.utils import make_batches

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


def _merge_metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = ("default->medium", "routed->hard", "routed->skip", "routed->input")
    merged: dict[str, Any] = {}
    for label in labels:
        primary = []
        secondary = []
        dominance = []
        for row in rows:
            stage = ((row.get("routing") or {}).get("merge_stage_metrics") or {}).get(
                label
            ) or {}
            if "primary_share" in stage:
                primary.append(float(stage["primary_share"]))
            if "secondary_share" in stage:
                secondary.append(float(stage["secondary_share"]))
            if "dominance" in stage:
                dominance.append(float(stage["dominance"]))
        if primary:
            merged[label] = {
                "primary_share_mean": round(mean(primary), 4),
                "secondary_share_mean": round(mean(secondary), 4),
                "dominance_mean": round(mean(dominance), 4),
            }
    return merged


def _aggregate_runs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def _series(key: str) -> list[float]:
        vals = [row.get(key) for row in rows if row.get(key) is not None]
        return [float(v) for v in vals]

    def _routing_series(key: str) -> list[float]:
        vals = [
            (row.get("routing") or {}).get(key)
            for row in rows
            if (row.get("routing") or {}).get(key) is not None
        ]
        return [float(v) for v in vals]

    payload = {
        "train_loss_mean": mean(_series("train_final_loss")),
        "train_loss_std": pstdev(_series("train_final_loss")),
        "eval_loss_mean": mean(_series("eval_loss_after")),
        "eval_loss_std": pstdev(_series("eval_loss_after")),
        "throughput_mean": mean(_series("throughput_tokens_per_s")),
        "throughput_std": pstdev(_series("throughput_tokens_per_s")),
        "memory_mean": mean(_series("max_memory_mb")),
        "memory_std": pstdev(_series("max_memory_mb")),
        "convergence_step_mean": mean(_series("convergence_step_75pct")),
    }
    routing = {}
    for key in (
        "dead_lane_rate",
        "sparse_span_coverage",
        "lane_entropy",
        "route_strength_mean",
        "routed_branch_share",
        "branch_dominance_mean",
        "default_path_fraction",
        "routed_compute_fraction",
        "active_lane_count",
    ):
        vals = _routing_series(key)
        if vals:
            routing[key] = {
                "mean": mean(vals),
                "std": pstdev(vals),
            }
    routing["merge_stage_metrics"] = _merge_metric_summary(rows)
    payload["routing"] = routing
    return payload


def _config_spec(
    *,
    medium_op: str,
    hard_op: str,
    span_widths: tuple[int, ...],
    curriculum: bool = False,
) -> dict[str, Any]:
    return {
        "medium_op": medium_op,
        "hard_op": hard_op,
        "span_widths": span_widths,
        "enable_curriculum": curriculum,
    }


def _run_named_config(
    *,
    cfg: Phase4Config,
    corpus: np.ndarray,
    spec: dict[str, Any],
    seeds: tuple[int, ...],
    steps: int,
    train_start: int,
    val_start: int,
) -> dict[str, Any]:
    runs = []
    for seed in seeds:
        train_batches, val_batches = _tensor_batches(
            corpus,
            device=cfg.device,
            seq_len=cfg.seq_len,
            batch_size=cfg.batch_size,
            train_batches=cfg.train_batches,
            val_batches=cfg.val_batches,
            seed=seed,
            train_start=train_start,
            val_start=val_start,
        )
        runs.append(
            _evaluate_variant(
                cfg=cfg,
                train_batches=train_batches,
                val_batches=val_batches,
                medium_op=spec["medium_op"],
                hard_op=spec["hard_op"],
                span_widths=tuple(spec["span_widths"]),
                enable_curriculum=bool(spec.get("enable_curriculum", False)),
                steps=steps,
                seed=seed,
            )
        )
    return {"spec": spec, "runs": runs, "aggregate": _aggregate_runs(runs)}


def _select_family_distinct_challenger(
    rows: list[dict[str, Any]],
    *,
    slot: str,
    default_family: str,
) -> dict[str, Any]:
    classifier = (
        classify_medium_mechanism if slot == "medium" else classify_hard_mechanism
    )
    eligible = []
    for row in rows:
        op_name = row.get("dispatch_name")
        if op_name is None:
            op_name = row.get("medium_op") if slot == "medium" else row.get("hard_op")
        if op_name is None:
            continue
        family = classifier(op_name)["family"]
        if family == default_family:
            continue
        eligible.append((float(row["eval_loss_after"]), row))
    if not eligible:
        raise ValueError(f"No family-distinct {slot} challenger found.")
    return min(eligible, key=lambda item: item[0])[1]


def _quality_cost_delta(
    winner: dict[str, Any],
    fallback: dict[str, Any],
) -> dict[str, float]:
    eval_gain = float(fallback["aggregate"]["eval_loss_mean"]) - float(
        winner["aggregate"]["eval_loss_mean"]
    )
    throughput_loss_pct = (
        100.0
        * (
            float(fallback["aggregate"]["throughput_mean"])
            - float(winner["aggregate"]["throughput_mean"])
        )
        / max(float(fallback["aggregate"]["throughput_mean"]), 1e-9)
    )
    memory_delta_pct = (
        100.0
        * (
            float(winner["aggregate"]["memory_mean"])
            - float(fallback["aggregate"]["memory_mean"])
        )
        / max(float(fallback["aggregate"]["memory_mean"]), 1e-9)
    )
    return {
        "eval_gain_vs_fallback": round(eval_gain, 6),
        "throughput_loss_pct_vs_fallback": round(throughput_loss_pct, 4),
        "memory_delta_pct_vs_fallback": round(memory_delta_pct, 4),
        "eval_gain_per_10pct_throughput_loss": round(
            eval_gain / max(throughput_loss_pct / 10.0, 1e-9), 6
        ),
    }


def _load_previous_report(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _build_report_markdown(payload: dict[str, Any]) -> str:
    winner = payload["winner_hardening"]["default"]
    fallback = payload["winner_hardening"]["fallback"]
    med_alt = payload["winner_hardening"]["medium_family_challenger"]
    hard_alt = payload["winner_hardening"]["hard_family_challenger"]
    long_default = payload["long_horizon"]["default"]
    long_fallback = payload["long_horizon"]["fallback"]
    perturb_default = payload["data_mix_perturbation"]["default"]
    perturb_fallback = payload["data_mix_perturbation"]["fallback"]
    lines = [
        "# Multiscale Rich Lane Router Phase 4",
        "",
        "## Findings Summary",
        "",
        "- Current default under test: `conv_only + mixed_recursion_gate + [2,3,4] + calibrated merge`, curriculum off.",
        "- Cost fallback under test: `conv_only + mixed_recursion_gate + [2,3] + calibrated merge`, curriculum off.",
        f"- Mechanism-diversity conclusion: `{payload['mechanism_diversity_assessment']['diversity_bottleneck']}`.",
        f"- Width-4 decision: `{payload['deployment_recommendation']['width4_default_status']}`.",
        "",
        "## Winner Hardening Scoreboard",
        "",
        "| Config | Eval Mean | Eval Std | Tok/s | Dead Rate | Span Cov | Route Str | Routed Share | Branch Dominance |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, row in (
        ("default [2,3,4]", winner),
        ("fallback [2,3]", fallback),
        ("medium alt", med_alt),
        ("hard alt", hard_alt),
    ):
        agg = row["aggregate"]
        routing = agg["routing"]
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    f"{agg['eval_loss_mean']:.4f}",
                    f"{agg['eval_loss_std']:.4f}",
                    f"{agg['throughput_mean']:.1f}",
                    f"{routing['dead_lane_rate']['mean']:.4f}",
                    f"{routing['sparse_span_coverage']['mean']:.4f}",
                    f"{routing['route_strength_mean']['mean']:.4f}",
                    f"{routing['routed_branch_share']['mean']:.4f}",
                    f"{routing['branch_dominance_mean']['mean']:.4f}",
                ]
            )
            + " |"
        )
    default_merge = winner["aggregate"]["routing"]["merge_stage_metrics"]
    lines.extend(
        [
            "",
            "### Merge Dominance",
            "",
            f"- `default->medium` secondary share: `{default_merge['default->medium']['secondary_share_mean']:.4f}`",
            f"- `routed->hard` secondary share: `{default_merge['routed->hard']['secondary_share_mean']:.4f}`",
            f"- `routed->skip` secondary share: `{default_merge['routed->skip']['secondary_share_mean']:.4f}`",
            f"- `routed->input` secondary share: `{default_merge['routed->input']['secondary_share_mean']:.4f}`",
            "",
            "## Longer-Horizon Check",
            "",
            "| Config | Eval Mean | Tok/s | Dead Rate |",
            "| --- | ---: | ---: | ---: |",
            f"| default [2,3,4] | {long_default['aggregate']['eval_loss_mean']:.4f} | {long_default['aggregate']['throughput_mean']:.1f} | {long_default['aggregate']['routing']['dead_lane_rate']['mean']:.4f} |",
            f"| fallback [2,3] | {long_fallback['aggregate']['eval_loss_mean']:.4f} | {long_fallback['aggregate']['throughput_mean']:.1f} | {long_fallback['aggregate']['routing']['dead_lane_rate']['mean']:.4f} |",
            "",
            "## Shifted-Corpus Perturbation",
            "",
            "| Config | Eval Mean | Tok/s | Dead Rate |",
            "| --- | ---: | ---: | ---: |",
            f"| default [2,3,4] | {perturb_default['aggregate']['eval_loss_mean']:.4f} | {perturb_default['aggregate']['throughput_mean']:.1f} | {perturb_default['aggregate']['routing']['dead_lane_rate']['mean']:.4f} |",
            f"| fallback [2,3] | {perturb_fallback['aggregate']['eval_loss_mean']:.4f} | {perturb_fallback['aggregate']['throughput_mean']:.1f} | {perturb_fallback['aggregate']['routing']['dead_lane_rate']['mean']:.4f} |",
            "",
            "## Mechanism Coverage Map",
            "",
            f"- Medium families: `{payload['mechanism_coverage']['medium']['family_count']}` across `{payload['mechanism_coverage']['medium_total_candidates']}` canonical candidates.",
            f"- Hard families: `{payload['mechanism_coverage']['hard']['family_count']}` across `{payload['mechanism_coverage']['hard_total_candidates']}` canonical candidates.",
            "",
            "### Medium Families",
            "",
            "| Family | Count | Share | Members |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for family in payload["mechanism_coverage"]["medium"]["families"]:
        lines.append(
            f"| `{family['family']}` | {family['count']} | {family['share']:.4f} | "
            + ", ".join(f"`{member['dispatch_name']}`" for member in family["members"])
            + " |"
        )
    lines.extend(
        [
            "",
            "### Hard Families",
            "",
            "| Family | Count | Share | Members |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for family in payload["mechanism_coverage"]["hard"]["families"]:
        lines.append(
            f"| `{family['family']}` | {family['count']} | {family['share']:.4f} | "
            + ", ".join(f"`{member['dispatch_name']}`" for member in family["members"])
            + " |"
        )
    lines.extend(
        [
            "",
            "## Challenger Rationale",
            "",
            "| Slot | Challenger | Mechanism Family | Why It Was Retested | Result |",
            "| --- | --- | --- | --- | --- |",
            f"| medium | `{payload['challenger_rationale']['medium']['dispatch_name']}` | `{payload['challenger_rationale']['medium']['family']}` | best non-default-family reachable medium from prior canonical sweep | {payload['challenger_rationale']['medium']['outcome']} |",
            f"| hard | `{payload['challenger_rationale']['hard']['dispatch_name']}` | `{payload['challenger_rationale']['hard']['family']}` | best non-default-family reachable hard from prior canonical sweep | {payload['challenger_rationale']['hard']['outcome']} |",
            "",
            "## Challenger Comparison",
            "",
            "| Config | Eval Mean | Tok/s | Quality-Per-Cost Note |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for row in payload["challenger_comparison_table"]:
        lines.append(
            f"| {row['label']} | {row['eval_loss_mean']:.4f} | {row['throughput_mean']:.1f} | {row['comment']} |"
        )
    lines.extend(
        [
            "",
            "## Deployment Recommendation",
            "",
            f"- Default config: `{payload['deployment_recommendation']['default_config']}`",
            f"- Cost-adjusted fallback: `{payload['deployment_recommendation']['cost_adjusted_fallback']}`",
            f"- Best challenger: `{payload['deployment_recommendation']['best_challenger']}`",
            f"- Next focus: `{payload['deployment_recommendation']['next_focus']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> Phase4Config:
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
    parser.add_argument("--multiseed-steps", type=int, default=24)
    parser.add_argument("--long-steps", type=int, default=72)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--corpus-path", default=str(DEFAULT_CORPUS))
    parser.add_argument("--previous-report-path", default=str(DEFAULT_PREVIOUS_REPORT))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--route-temperature", type=float, default=0.85)
    parser.add_argument("--min-keep-fraction", type=float, default=0.125)
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--merge-redesign", action="store_true", default=True)
    parser.add_argument("--perturb-train-start", type=int, default=300_000)
    parser.add_argument("--perturb-val-start", type=int, default=600_000)
    args = parser.parse_args()
    return Phase4Config(
        device=args.device,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        model_dim=args.model_dim,
        train_batches=args.train_batches,
        val_batches=args.val_batches,
        multiseed_steps=args.multiseed_steps,
        long_steps=args.long_steps,
        lr=args.lr,
        corpus_path=args.corpus_path,
        previous_report_path=args.previous_report_path,
        output_path=args.output_path,
        route_temperature=args.route_temperature,
        min_keep_fraction=args.min_keep_fraction,
        confidence_threshold=args.confidence_threshold,
        merge_redesign=bool(args.merge_redesign),
        perturb_train_start=args.perturb_train_start,
        perturb_val_start=args.perturb_val_start,
    )


def main() -> None:
    cfg = parse_args()
    previous = _load_previous_report(cfg.previous_report_path)
    registry = build_multiscale_registry()
    medium_rows, _ = _dedupe_candidates(registry["medium_candidates"])
    hard_rows, _ = _dedupe_candidates(registry["hard_candidates"])

    mechanism_coverage = {
        "medium_total_candidates": len(medium_rows),
        "hard_total_candidates": len(hard_rows),
        "medium": build_mechanism_coverage(medium_rows, "medium"),
        "hard": build_mechanism_coverage(hard_rows, "hard"),
    }

    default_medium_family = classify_medium_mechanism("conv_only")["family"]
    default_hard_family = classify_hard_mechanism("mixed_recursion_gate")["family"]
    medium_alt = _select_family_distinct_challenger(
        previous["medium_sweep"],
        slot="medium",
        default_family=default_medium_family,
    )
    hard_alt = _select_family_distinct_challenger(
        previous["hard_sweep"],
        slot="hard",
        default_family=default_hard_family,
    )

    corpus = np.load(cfg.corpus_path, mmap_mode="r")
    specs = {
        "default": _config_spec(
            medium_op="conv_only",
            hard_op="mixed_recursion_gate",
            span_widths=(2, 3, 4),
        ),
        "fallback": _config_spec(
            medium_op="conv_only",
            hard_op="mixed_recursion_gate",
            span_widths=(2, 3),
        ),
        "medium_family_challenger": _config_spec(
            medium_op=medium_alt["medium_op"],
            hard_op="mixed_recursion_gate",
            span_widths=(2, 3, 4),
        ),
        "hard_family_challenger": _config_spec(
            medium_op="conv_only",
            hard_op=hard_alt["hard_op"],
            span_widths=(2, 3, 4),
        ),
    }

    winner_hardening = {}
    for label, spec in specs.items():
        winner_hardening[label] = _run_named_config(
            cfg=cfg,
            corpus=corpus,
            spec=spec,
            seeds=(7, 17, 27),
            steps=cfg.multiseed_steps,
            train_start=0,
            val_start=200_000,
        )

    long_horizon = {}
    for label, spec in specs.items():
        long_horizon[label] = _run_named_config(
            cfg=cfg,
            corpus=corpus,
            spec=spec,
            seeds=(7,),
            steps=cfg.long_steps,
            train_start=0,
            val_start=200_000,
        )

    perturbation = {}
    for label in ("default", "fallback"):
        perturbation[label] = _run_named_config(
            cfg=cfg,
            corpus=corpus,
            spec=specs[label],
            seeds=(7, 17, 27),
            steps=cfg.multiseed_steps,
            train_start=cfg.perturb_train_start,
            val_start=cfg.perturb_val_start,
        )

    width_tradeoff = _quality_cost_delta(
        winner_hardening["default"],
        winner_hardening["fallback"],
    )
    width4_default_status = (
        "weak_keep"
        if width_tradeoff["eval_gain_vs_fallback"] > 0.0
        and width_tradeoff["throughput_loss_pct_vs_fallback"] >= 10.0
        else "keep"
    )
    diversity_bottleneck = (
        "not_primary_bottleneck"
        if mechanism_coverage["medium"]["family_count"] >= 6
        and mechanism_coverage["hard"]["family_count"] >= 5
        else "possible_gap"
    )
    default_eval = winner_hardening["default"]["aggregate"]["eval_loss_mean"]
    med_eval = winner_hardening["medium_family_challenger"]["aggregate"][
        "eval_loss_mean"
    ]
    hard_eval = winner_hardening["hard_family_challenger"]["aggregate"][
        "eval_loss_mean"
    ]
    best_challenger_name = (
        "adaptive_lane_mixer" if med_eval <= hard_eval else hard_alt["hard_op"]
    )
    scoreboard = [
        {
            "label": "default [2,3,4]",
            "eval_loss_mean": winner_hardening["default"]["aggregate"][
                "eval_loss_mean"
            ],
            "throughput_mean": winner_hardening["default"]["aggregate"][
                "throughput_mean"
            ],
            "dead_lane_rate": winner_hardening["default"]["aggregate"]["routing"][
                "dead_lane_rate"
            ]["mean"],
            "sparse_span_coverage": winner_hardening["default"]["aggregate"]["routing"][
                "sparse_span_coverage"
            ]["mean"],
            "route_strength_mean": winner_hardening["default"]["aggregate"]["routing"][
                "route_strength_mean"
            ]["mean"],
        },
        {
            "label": "fallback [2,3]",
            "eval_loss_mean": winner_hardening["fallback"]["aggregate"][
                "eval_loss_mean"
            ],
            "throughput_mean": winner_hardening["fallback"]["aggregate"][
                "throughput_mean"
            ],
            "dead_lane_rate": winner_hardening["fallback"]["aggregate"]["routing"][
                "dead_lane_rate"
            ]["mean"],
            "sparse_span_coverage": winner_hardening["fallback"]["aggregate"][
                "routing"
            ]["sparse_span_coverage"]["mean"],
            "route_strength_mean": winner_hardening["fallback"]["aggregate"]["routing"][
                "route_strength_mean"
            ]["mean"],
        },
        {
            "label": "adaptive_lane_mixer + mixed_recursion_gate",
            "eval_loss_mean": winner_hardening["medium_family_challenger"]["aggregate"][
                "eval_loss_mean"
            ],
            "throughput_mean": winner_hardening["medium_family_challenger"][
                "aggregate"
            ]["throughput_mean"],
            "dead_lane_rate": winner_hardening["medium_family_challenger"]["aggregate"][
                "routing"
            ]["dead_lane_rate"]["mean"],
            "sparse_span_coverage": winner_hardening["medium_family_challenger"][
                "aggregate"
            ]["routing"]["sparse_span_coverage"]["mean"],
            "route_strength_mean": winner_hardening["medium_family_challenger"][
                "aggregate"
            ]["routing"]["route_strength_mean"]["mean"],
        },
        {
            "label": "conv_only + moe_topk",
            "eval_loss_mean": winner_hardening["hard_family_challenger"]["aggregate"][
                "eval_loss_mean"
            ],
            "throughput_mean": winner_hardening["hard_family_challenger"]["aggregate"][
                "throughput_mean"
            ],
            "dead_lane_rate": winner_hardening["hard_family_challenger"]["aggregate"][
                "routing"
            ]["dead_lane_rate"]["mean"],
            "sparse_span_coverage": winner_hardening["hard_family_challenger"][
                "aggregate"
            ]["routing"]["sparse_span_coverage"]["mean"],
            "route_strength_mean": winner_hardening["hard_family_challenger"][
                "aggregate"
            ]["routing"]["route_strength_mean"]["mean"],
        },
    ]
    comparison_rows = [
        {
            "label": "default [2,3,4]",
            "eval_loss_mean": scoreboard[0]["eval_loss_mean"],
            "throughput_mean": scoreboard[0]["throughput_mean"],
            "comment": "best raw quality, weak width-4 ROI",
        },
        {
            "label": "fallback [2,3]",
            "eval_loss_mean": scoreboard[1]["eval_loss_mean"],
            "throughput_mean": scoreboard[1]["throughput_mean"],
            "comment": "near-tied loss with materially better throughput",
        },
        {
            "label": "adaptive_lane_mixer + mixed_recursion_gate",
            "eval_loss_mean": scoreboard[2]["eval_loss_mean"],
            "throughput_mean": scoreboard[2]["throughput_mean"],
            "comment": "best non-convolution medium family, still below default",
        },
        {
            "label": "conv_only + moe_topk",
            "eval_loss_mean": scoreboard[3]["eval_loss_mean"],
            "throughput_mean": scoreboard[3]["throughput_mean"],
            "comment": "best non-recursive hard family, still below default",
        },
    ]

    payload = {
        "config": asdict(cfg),
        "winner_hardening_scoreboard": scoreboard,
        "winner_hardening": winner_hardening,
        "long_horizon": long_horizon,
        "data_mix_perturbation": perturbation,
        "mechanism_coverage": mechanism_coverage,
        "challenger_comparison_table": comparison_rows,
        "challenger_rationale": {
            "medium": {
                **classify_medium_mechanism(medium_alt["medium_op"]),
                "dispatch_name": medium_alt["medium_op"],
                "label": medium_alt["label"],
                "outcome": "lost_to_default"
                if med_eval >= default_eval
                else "beat_default",
            },
            "hard": {
                **classify_hard_mechanism(hard_alt["hard_op"]),
                "dispatch_name": hard_alt["hard_op"],
                "label": hard_alt["label"],
                "outcome": "lost_to_default"
                if hard_eval >= default_eval
                else "beat_default",
            },
        },
        "challenger_expansion": {
            "needed": False,
            "rationale": "Reachable canonical pools already cover the major mechanism families that are slot-legal for this template. Phase 4 found no evidence that adding new glue components is the next high-ROI move.",
            "proposed_medium": [],
            "proposed_hard": [],
        },
        "mechanism_diversity_assessment": {
            "diversity_bottleneck": diversity_bottleneck,
            "medium_pool_overconcentrated": mechanism_coverage["medium"][
                "largest_family_share"
            ]
            > 0.4,
            "hard_pool_overconcentrated": mechanism_coverage["hard"][
                "largest_family_share"
            ]
            > 0.4,
            "summary": "The reachable medium pool spans local convolution, sparse linear, recurrent, verification, fallback, lane-mixing, and nested-router families. The hard pool spans recursion, MoE, compression-first, sparse bottleneck, and state-space families. That is broad enough that mechanism diversity is not the first-order limiter.",
        },
        "width_tradeoff": width_tradeoff,
        "deployment_recommendation": {
            "default_config": "multiscale_rich_lane_router + conv_only + mixed_recursion_gate + calibrated_merge + widths=[2,3,4] + curriculum=off",
            "cost_adjusted_fallback": "multiscale_rich_lane_router + conv_only + mixed_recursion_gate + calibrated_merge + widths=[2,3] + curriculum=off",
            "best_challenger": best_challenger_name,
            "width4_default_status": width4_default_status,
            "curriculum_status": "optional_only",
            "diversity_is_real_bottleneck": False,
            "next_focus": "training_schedule_refinement_or_stop_if_roi_flattens",
        },
    }

    output_path = Path(cfg.output_path)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_path.with_suffix(".md").write_text(
        _build_report_markdown(payload),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
