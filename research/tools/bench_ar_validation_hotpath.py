#!/usr/bin/env python
"""Benchmark the AR Validation synthetic batch hot path.

This is intentionally read-only and model-free. It measures the data-generation
path that feeds the CUDA probe so regressions are visible without spending a
full backfill run.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from research.eval.ar_validation import (
    ARValidationConfig,
    build_ar_validation_pair_table,
    make_ar_validation_batch,
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda_unavailable")

    cfg = replace(
        ARValidationConfig(),
        seed=int(args.seed),
        batch_size=int(args.batch_size),
        pairs_per_example=int(args.pairs_per_example),
        n_key_tokens=int(args.key_tokens),
        n_value_tokens=int(args.value_tokens),
        n_train_pairs=int(args.train_pairs),
        n_held_pairs=int(args.held_pairs),
    )
    table = build_ar_validation_pair_table(cfg)
    table = type(table)(
        train_keys=table.train_keys.to(device),
        train_values=table.train_values.to(device),
        held_keys=table.held_keys.to(device),
        held_values=table.held_values.to(device),
        vocab_lo=table.vocab_lo,
        value_lo=table.value_lo,
        value_hi=table.value_hi,
        n_value_classes=table.n_value_classes,
    )
    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed))

    for _ in range(int(args.warmup_batches)):
        make_ar_validation_batch(
            table,
            split="train",
            batch_size=int(args.batch_size),
            pairs_per_example=int(args.pairs_per_example),
            sep_token=int(args.sep_token),
            ans_token=int(args.ans_token),
            device=device,
            generator=gen,
            episodic_values=bool(args.episodic_values),
        )
    _sync(device)

    start = time.perf_counter()
    for _ in range(int(args.batches)):
        ids, targets, classes = make_ar_validation_batch(
            table,
            split="train",
            batch_size=int(args.batch_size),
            pairs_per_example=int(args.pairs_per_example),
            sep_token=int(args.sep_token),
            ans_token=int(args.ans_token),
            device=device,
            generator=gen,
            episodic_values=bool(args.episodic_values),
        )
    _sync(device)
    elapsed_s = time.perf_counter() - start

    examples = int(args.batches) * int(args.batch_size)
    seq_len = int(ids.shape[1])
    payload = {
        "benchmark": "ar_validation_batch_hotpath",
        "device": str(device),
        "batches": int(args.batches),
        "warmup_batches": int(args.warmup_batches),
        "batch_size": int(args.batch_size),
        "seq_len": seq_len,
        "examples": examples,
        "elapsed_ms": round(elapsed_s * 1000.0, 3),
        "batches_per_s": round(int(args.batches) / max(elapsed_s, 1e-9), 3),
        "examples_per_s": round(examples / max(elapsed_s, 1e-9), 3),
        "tokens_per_s": round(examples * seq_len / max(elapsed_s, 1e-9), 3),
        "target_checksum": int(targets.sum().item()),
        "class_checksum": int(classes.sum().item()),
        "config": asdict(cfg),
    }
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batches", type=int, default=500)
    parser.add_argument("--warmup-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--pairs-per-example", type=int, default=9)
    parser.add_argument("--key-tokens", type=int, default=1024)
    parser.add_argument("--value-tokens", type=int, default=96)
    parser.add_argument("--train-pairs", type=int, default=256)
    parser.add_argument("--held-pairs", type=int, default=64)
    parser.add_argument("--sep-token", type=int, default=2)
    parser.add_argument("--ans-token", type=int, default=3)
    parser.add_argument(
        "--episodic-values",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_benchmark(args)
    text = json.dumps(payload, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
