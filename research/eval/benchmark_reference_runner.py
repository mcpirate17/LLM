"""Microbenchmark for the shared reference-model training paths."""

from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import asdict, dataclass

import torch

from ._reference_model_native import load_reference_model_native
from ._runner_native import load_runner_native
from .reference_training import BaselineTransformer
from .training_core import run_training_loop
from .utils import clip_grad_norm, language_model_loss


@dataclass(slots=True)
class RunnerBenchResult:
    label: str
    median_ms: float
    mean_ms: float
    stdev_ms: float
    final_loss: float


def _legacy_train_once(
    *,
    d_model: int,
    seq_len: int,
    batch_size: int,
    vocab_size: int,
    n_layers: int,
    n_steps: int,
    lr: float,
    seed: int,
) -> tuple[float, float]:
    device = torch.device("cpu")
    torch.manual_seed(seed)
    os.environ["ARIA_DISABLE_REFERENCE_MODEL_NATIVE"] = "1"
    model = BaselineTransformer(vocab_size, d_model, n_layers=n_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    t0 = time.perf_counter()
    final_loss = float("inf")
    for _ in range(n_steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens)
        loss = language_model_loss(logits, tokens, vocab_size)
        loss.backward()
        clip_grad_norm(model.parameters(), 1.0)
        optimizer.step()
        final_loss = float(loss.item())
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    os.environ.pop("ARIA_DISABLE_REFERENCE_MODEL_NATIVE", None)
    return elapsed_ms, final_loss


def _native_train_once(
    *,
    d_model: int,
    seq_len: int,
    batch_size: int,
    vocab_size: int,
    n_layers: int,
    n_steps: int,
    lr: float,
    seed: int,
) -> tuple[float, float]:
    device = torch.device("cpu")
    torch.manual_seed(seed)
    os.environ.pop("ARIA_DISABLE_REFERENCE_MODEL_NATIVE", None)
    model = BaselineTransformer(vocab_size, d_model, n_layers=n_layers).to(device)
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    def compute_loss(_step: int) -> torch.Tensor:
        return language_model_loss(model(tokens), tokens, vocab_size)

    t0 = time.perf_counter()
    result = run_training_loop(
        model.parameters(),
        compute_loss,
        n_steps=n_steps,
        optimizer_name="adamw",
        lr=lr,
        weight_decay=0.01,
        clip_grad=1.0,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return elapsed_ms, result.final_loss


def benchmark_reference_runner(
    *,
    d_model: int = 64,
    seq_len: int = 64,
    batch_size: int = 4,
    vocab_size: int = 256,
    n_layers: int = 2,
    n_steps: int = 8,
    repeats: int = 5,
    lr: float = 3e-4,
) -> dict:
    load_runner_native()
    load_reference_model_native()
    samples = {}
    for label, fn in (
        ("legacy_torch", _legacy_train_once),
        ("native_shared_runner", _native_train_once),
    ):
        timings = []
        last_loss = float("inf")
        for rep in range(repeats):
            elapsed_ms, last_loss = fn(
                d_model=d_model,
                seq_len=seq_len,
                batch_size=batch_size,
                vocab_size=vocab_size,
                n_layers=n_layers,
                n_steps=n_steps,
                lr=lr,
                seed=rep,
            )
            timings.append(elapsed_ms)
        samples[label] = RunnerBenchResult(
            label=label,
            median_ms=round(statistics.median(timings), 3),
            mean_ms=round(statistics.mean(timings), 3),
            stdev_ms=round(statistics.pstdev(timings), 3),
            final_loss=round(float(last_loss), 6),
        )

    legacy = samples["legacy_torch"]
    native = samples["native_shared_runner"]
    speedup = legacy.median_ms / native.median_ms if native.median_ms > 0 else None
    return {
        "config": {
            "d_model": d_model,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "vocab_size": vocab_size,
            "n_layers": n_layers,
            "n_steps": n_steps,
            "repeats": repeats,
            "lr": lr,
            "device": "cpu",
        },
        "results": {name: asdict(result) for name, result in samples.items()},
        "median_speedup_vs_legacy": round(float(speedup), 3) if speedup else None,
    }


if __name__ == "__main__":
    print(json.dumps(benchmark_reference_runner(), indent=2))
