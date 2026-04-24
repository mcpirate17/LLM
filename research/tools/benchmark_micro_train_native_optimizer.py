#!/usr/bin/env python
"""Compare current micro-train optimizer path against native C++ optimizer kernels."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import statistics
import time
from typing import Any, Dict, Iterator

import torch

from research.eval._runner_native import load_runner_native
from research.scientist.native_runner import compile_model_native_first as compile_model
from research.scientist.runner import ExperimentRunner, RunConfig
from research.tools.profile_component_scaffolds import build_gpt2_attn_scaffold


@contextmanager
def _env_flag(name: str, value: str) -> Iterator[None]:
    old = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


def _build_config(*, fixture: str) -> RunConfig:
    quick = fixture == "quick"
    return RunConfig(
        device="cpu",
        data_mode="random",
        model_dim=32 if quick else 48,
        n_layers=1,
        vocab_size=128 if quick else 256,
        max_seq_len=16 if quick else 24,
        stage1_steps=4 if quick else 12,
        stage1_batch_size=1 if quick else 2,
        stage1_lr=3e-4,
        optimizer_type="adamw",
        optimizer_betas=(0.9, 0.95),
        optimizer_weight_decay=0.01,
        enable_perf_tracing=False,
        collect_training_curve=True,
        profile_disable_post_eval=True,
        profile_disable_inflight_checks=True,
        stage1_compute_discovery_loss=False,
        stage1_compute_val_loss=False,
        skip_post_s1_fingerprint=True,
        skip_post_s1_triage=True,
        skip_binding_probes=True,
        skip_screening_wikitext=True,
        skip_screening_hellaswag=True,
        skip_screening_blimp=True,
    )


def _compile_seeded_model(config: RunConfig, *, seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    graph = build_gpt2_attn_scaffold("softmax_attention", model_dim=config.model_dim)
    return compile_model(
        [graph] * config.n_layers,
        vocab_size=config.vocab_size,
        max_seq_len=config.max_seq_len,
    )


def _run_one(
    *,
    runner: ExperimentRunner,
    config: RunConfig,
    seed: int,
    model_seed: int,
    native_optimizer: bool,
) -> tuple[Dict[str, Any], torch.nn.Module, float]:
    model = _compile_seeded_model(config, seed=model_seed)
    flag = "1" if native_optimizer else "0"
    started = time.perf_counter()
    with _env_flag("MICRO_TRAIN_NATIVE_OPTIMIZER", flag):
        result = runner._micro_train(model, config, torch.device("cpu"), seed=seed)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return result, model, elapsed_ms


def _max_param_delta(a: torch.nn.Module, b: torch.nn.Module) -> float:
    max_delta = 0.0
    with torch.no_grad():
        for left, right in zip(a.parameters(), b.parameters(), strict=True):
            delta = (left.detach() - right.detach()).abs().max().item()
            max_delta = max(max_delta, float(delta))
    return max_delta


def _compare_curve(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    curve_a = a.get("training_curve") or []
    curve_b = b.get("training_curve") or []
    n = min(len(curve_a), len(curve_b))
    max_loss_delta = 0.0
    max_grad_delta = 0.0
    for idx in range(n):
        max_loss_delta = max(
            max_loss_delta,
            abs(
                float(curve_a[idx].get("loss", 0.0))
                - float(curve_b[idx].get("loss", 0.0))
            ),
        )
        max_grad_delta = max(
            max_grad_delta,
            abs(
                float(curve_a[idx].get("grad_norm", 0.0))
                - float(curve_b[idx].get("grad_norm", 0.0))
            ),
        )
    return {
        "curve_len_reference": len(curve_a),
        "curve_len_native": len(curve_b),
        "curve_compared_steps": n,
        "max_curve_loss_delta": max_loss_delta,
        "max_curve_grad_norm_delta": max_grad_delta,
    }


def run_comparison(
    *,
    fixture: str,
    repeats: int,
    warmups: int,
    seed: int,
    model_seed: int,
) -> Dict[str, Any]:
    if fixture not in {"quick", "standard"}:
        raise ValueError(f"Unsupported fixture: {fixture}")

    # Keep extension build/load out of the timing comparison.
    load_runner_native()

    config = _build_config(fixture=fixture)
    runner = ExperimentRunner("research/lab_notebook.db")
    pairs = []
    reference_times = []
    native_times = []

    for idx in range(max(0, warmups)):
        run_seed = seed - idx - 1
        _run_one(
            runner=runner,
            config=config,
            seed=run_seed,
            model_seed=model_seed,
            native_optimizer=False,
        )
        _run_one(
            runner=runner,
            config=config,
            seed=run_seed,
            model_seed=model_seed,
            native_optimizer=True,
        )

    for idx in range(max(1, repeats)):
        run_seed = seed + idx
        reference, reference_model, reference_ms = _run_one(
            runner=runner,
            config=config,
            seed=run_seed,
            model_seed=model_seed,
            native_optimizer=False,
        )
        native, native_model, native_ms = _run_one(
            runner=runner,
            config=config,
            seed=run_seed,
            model_seed=model_seed,
            native_optimizer=True,
        )
        reference_times.append(reference_ms)
        native_times.append(native_ms)
        final_loss_ref = float(reference.get("final_loss") or float("nan"))
        final_loss_native = float(native.get("final_loss") or float("nan"))
        pairs.append(
            {
                "run": idx,
                "seed": run_seed,
                "reference_ms": reference_ms,
                "native_ms": native_ms,
                "reference_final_loss": final_loss_ref,
                "native_final_loss": final_loss_native,
                "final_loss_delta": abs(final_loss_ref - final_loss_native),
                "reference_passed": bool(reference.get("passed")),
                "native_passed": bool(native.get("passed")),
                "reference_steps": int(reference.get("n_train_steps") or 0),
                "native_steps": int(native.get("n_train_steps") or 0),
                "max_param_delta": _max_param_delta(reference_model, native_model),
                "curve_delta": _compare_curve(reference, native),
                "native_optimizer_active": native.get("native_optimizer_active"),
            }
        )

    ref_median = statistics.median(reference_times)
    native_median = statistics.median(native_times)
    return {
        "fixture": fixture,
        "repeats": max(1, repeats),
        "warmups": max(0, warmups),
        "config": {
            "stage1_steps": config.stage1_steps,
            "stage1_batch_size": config.stage1_batch_size,
            "max_seq_len": config.max_seq_len,
            "model_dim": config.model_dim,
            "vocab_size": config.vocab_size,
            "optimizer_type": config.optimizer_type,
            "optimizer_betas": list(config.optimizer_betas),
            "optimizer_weight_decay": config.optimizer_weight_decay,
        },
        "summary": {
            "reference_ms_median": ref_median,
            "native_ms_median": native_median,
            "speedup": ref_median / native_median if native_median > 0.0 else 0.0,
            "improvement_pct": (
                ((ref_median - native_median) / ref_median) * 100.0
                if ref_median > 0.0
                else 0.0
            ),
            "max_final_loss_delta": max(row["final_loss_delta"] for row in pairs),
            "max_param_delta": max(row["max_param_delta"] for row in pairs),
            "pass_mismatches": sum(
                1 for row in pairs if row["reference_passed"] != row["native_passed"]
            ),
            "step_mismatches": sum(
                1 for row in pairs if row["reference_steps"] != row["native_steps"]
            ),
        },
        "runs": pairs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", choices=("quick", "standard"), default="quick")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--model-seed", type=int, default=77)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    report = run_comparison(
        fixture=args.fixture,
        repeats=max(1, int(args.repeats)),
        warmups=max(0, int(args.warmups)),
        seed=int(args.seed),
        model_seed=int(args.model_seed),
    )
    text = json.dumps(report, indent=2)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(text)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
