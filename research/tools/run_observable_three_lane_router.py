#!/usr/bin/env python
"""Run an explicit observable three-lane router and log checkpoint math."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from research.eval.routing_telemetry import collect_routing_telemetry
from research.eval.training_core import make_optimizer
from research.eval.utils import clip_grad_norm, language_model_loss, make_batches
from research.synthesis.compiler import compile_model
from research.tools.audit_multiscale_rich_lane_router_phase2 import _memory_mb
from research.tools.routing_template_variants import build_observable_three_lane_router


DEFAULT_CORPUS = (
    Path(__file__).resolve().parents[1] / "corpus" / "wikitext103_train.npy"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "observable_three_lane_router.json"
)


@dataclass(slots=True)
class RunConfig:
    device: str
    vocab_size: int
    seq_len: int
    batch_size: int
    model_dim: int
    train_batches: int
    val_batches: int
    steps: int
    checkpoints: tuple[int, ...]
    lr: float
    clip_grad: float
    corpus_path: str
    output_path: str
    seed: int


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


def _mean_or_none(values: torch.Tensor | None) -> list[float] | None:
    if values is None:
        return None
    return [float(v) for v in values.detach().cpu().tolist()]


def _collect_observability(model: torch.nn.Module) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "difficulty": {},
        "merges": [],
        "hard_lane": {},
        "aggregate_routing": collect_routing_telemetry(model, False) or {},
    }
    merge_idx = 0
    for name, module in model.named_modules():
        op_name = getattr(module, "op_name", None)
        if op_name == "token_class_proj":
            scores = getattr(module, "_class_scores", None)
            if scores is not None:
                probs = torch.softmax(scores.float(), dim=-1)
                entropy = -(probs * torch.log(probs.clamp_min(1e-9))).sum(dim=-1).mean()
                payload["difficulty"] = {
                    "module": name,
                    "mean_prob": _mean_or_none(probs.mean(dim=(0, 1))),
                    "mean_max_prob": float(probs.max(dim=-1).values.mean().item()),
                    "entropy": float(entropy.item()),
                }
        elif op_name == "calibrated_branch_merge":
            telemetry = getattr(module, "routing_telemetry", None) or {}
            branch_count = max(int(telemetry.get("branch_weight_count", 0)), 1)
            branch_sum = telemetry.get("branch_weight_sum")
            branch_mean = None
            if isinstance(branch_sum, torch.Tensor):
                branch_mean = [
                    float(v)
                    for v in (branch_sum / branch_count).detach().cpu().tolist()
                ]
            gains = getattr(module, "branch_gain", None)
            gain_values = None
            if gains is not None:
                gain_values = [
                    float(v)
                    for v in (0.5 + torch.sigmoid(gains)).detach().cpu().tolist()
                ]
            bias = getattr(module, "branch_bias", None)
            bias_values = None
            if bias is not None:
                bias_values = [float(v) for v in bias.detach().cpu().tolist()]
            payload["merges"].append(
                {
                    "merge_index": merge_idx,
                    "module": name,
                    "branch_weight_mean": branch_mean,
                    "branch_gain_values": gain_values,
                    "branch_bias_values": bias_values,
                    "branch_dominance_mean": float(
                        telemetry.get("branch_dominance_sum", 0.0)
                    )
                    / branch_count,
                    "routed_branch_share_mean": float(
                        telemetry.get("routed_branch_share_sum", 0.0)
                    )
                    / branch_count,
                    "medium_branch_share_mean": float(
                        telemetry.get("medium_branch_share_sum", 0.0)
                    )
                    / branch_count,
                    "hard_branch_share_mean": float(
                        telemetry.get("hard_branch_share_sum", 0.0)
                    )
                    / branch_count,
                    "trace_payload": telemetry.get("trace_payload"),
                }
            )
            merge_idx += 1
        elif op_name == "moe_topk":
            telemetry = getattr(module, "routing_telemetry", None) or {}
            count = max(int(telemetry.get("count", 0)), 1)
            expert_counts = telemetry.get("expert_counts")
            payload["hard_lane"] = {
                "module": name,
                "lane_entropy": float(telemetry.get("entropy_sum", 0.0)) / count,
                "mean_confidence": float(telemetry.get("confidence_sum", 0.0))
                / max(int(telemetry.get("confidence_count", 0)), 1),
                "expert_count_histogram": _mean_or_none(expert_counts),
            }
    return payload


def _build_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Observable Three-Lane Router",
        "",
        "## Graph",
        "",
        "- `input -> rmsnorm -> token_class_proj -> signal_conditioned_compression`",
        "- `easy = cheap_verify_blend(stem)`",
        "- `medium = block_sparse_linear(routed_seed)`",
        "- `hard = moe_topk(routed_seed)`",
        "- `merge easy+medium -> merge + hard -> residual add input -> rmsnorm -> output`",
        "",
        "## Checkpoints",
        "",
        "| Step | Eval | Train Loss | Lane Entropy | Route Strength | Span Coverage | Difficulty Entropy | Difficulty Max Prob |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["checkpoints"]:
        routing = row["observability"]["aggregate_routing"]
        difficulty = row["observability"]["difficulty"]
        lines.append(
            f"| {row['step']} | {row['eval_loss']:.4f} | {row['train_loss']:.4f} | "
            f"{float(routing.get('lane_entropy', 0.0)):.4f} | {float(routing.get('route_strength_mean', 0.0)):.4f} | "
            f"{float(routing.get('sparse_span_coverage', 0.0)):.4f} | {float(difficulty.get('entropy', 0.0)):.4f} | "
            f"{float(difficulty.get('mean_max_prob', 0.0)):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Merge Math",
            "",
        ]
    )
    for row in payload["checkpoints"]:
        lines.append(f"### Step {row['step']}")
        for merge in row["observability"]["merges"]:
            lines.append(
                f"- Merge {merge['merge_index']}: weights={merge['branch_weight_mean']}, gains={merge['branch_gain_values']}, "
                f"bias={merge['branch_bias_values']}, dominance={merge['branch_dominance_mean']:.4f}"
            )
    return "\n".join(lines) + "\n"


def parse_args() -> RunConfig:
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
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--checkpoints", nargs="+", type=int, default=[300, 500, 1000])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--corpus-path", default=str(DEFAULT_CORPUS))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    return RunConfig(
        device=args.device,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        model_dim=args.model_dim,
        train_batches=args.train_batches,
        val_batches=args.val_batches,
        steps=args.steps,
        checkpoints=tuple(sorted(set(args.checkpoints))),
        lr=args.lr,
        clip_grad=args.clip_grad,
        corpus_path=args.corpus_path,
        output_path=args.output_path,
        seed=args.seed,
    )


def main() -> None:
    cfg = parse_args()
    _set_seed(cfg.seed)
    corpus = np.load(cfg.corpus_path, mmap_mode="r")
    train_batches, val_batches = _tensor_batches(
        corpus,
        device=cfg.device,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        train_batches=cfg.train_batches,
        val_batches=cfg.val_batches,
        seed=cfg.seed,
    )
    graph = build_observable_three_lane_router(model_dim=cfg.model_dim)
    model = compile_model(
        [graph], vocab_size=cfg.vocab_size, max_seq_len=cfg.seq_len
    ).to(cfg.device)
    optimizer = make_optimizer(model.parameters(), optimizer_name="adamw", lr=cfg.lr)
    if cfg.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device=cfg.device)

    checkpoints: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        _ = model(val_batches[0])
    checkpoints.append(
        {
            "step": 0,
            "train_loss": float("nan"),
            "eval_loss": _eval_loss(model, val_batches, cfg.vocab_size),
            "observability": _collect_observability(model),
        }
    )

    model.train()
    start = time.perf_counter()
    last_train_loss = float("nan")
    for step in range(1, cfg.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        batch = train_batches[(step - 1) % len(train_batches)]
        logits = model(batch)
        loss = language_model_loss(logits, batch, cfg.vocab_size)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward()
        if cfg.clip_grad > 0:
            clip_grad_norm(model.parameters(), cfg.clip_grad)
        optimizer.step()
        last_train_loss = float(loss.item())
        if step in cfg.checkpoints:
            model.eval()
            with torch.no_grad():
                _ = model(val_batches[0])
            checkpoints.append(
                {
                    "step": step,
                    "train_loss": last_train_loss,
                    "eval_loss": _eval_loss(model, val_batches, cfg.vocab_size),
                    "observability": _collect_observability(model),
                }
            )
            model.train()

    elapsed = time.perf_counter() - start
    payload = {
        "config": asdict(cfg),
        "throughput_tokens_per_s": (cfg.steps * cfg.batch_size * cfg.seq_len)
        / max(elapsed, 1e-9),
        "max_memory_mb": _memory_mb(cfg.device),
        "checkpoints": checkpoints,
    }
    output_path = Path(cfg.output_path)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output_path.with_suffix(".md").write_text(
        _build_markdown(payload), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
