#!/usr/bin/env python
"""Validate native micro-train optimizer cutover against the PyTorch path."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import logging
import os
from pathlib import Path
import statistics
import time
from typing import Any, Iterator

import torch

from research.eval._runner_native import load_runner_native
from research.scientist.native_runner import compile_model_native_first as compile_model
from research.scientist.runner import ExperimentRunner, RunConfig
from research.synthesis.serializer import graph_to_json
from research.tools.profile_component_scaffolds import (
    ScaffoldCase,
    build_scaffold,
    generate_cases,
)


@dataclass(frozen=True)
class CutoverThresholds:
    final_loss: float = 1e-6
    initial_loss: float = 1e-6
    curve_loss: float = 1e-6
    curve_grad_norm: float = 2e-5
    parameter: float = 2e-6


class _LogCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        if not record.name.startswith("research."):
            return
        self.records.append(
            {
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
        )


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


@contextmanager
def _capture_research_logs() -> Iterator[list[dict[str, Any]]]:
    handler = _LogCapture()
    root = logging.getLogger()
    old_level = root.level
    root.addHandler(handler)
    if old_level > logging.WARNING:
        root.setLevel(logging.WARNING)
    try:
        yield handler.records
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)


def _build_config(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        device="cpu",
        data_mode="random",
        model_dim=int(args.model_dim),
        n_layers=int(args.n_layers),
        vocab_size=int(args.vocab_size),
        max_seq_len=int(args.seq_len),
        stage1_steps=int(args.steps),
        stage1_batch_size=int(args.batch_size),
        stage1_lr=float(args.lr),
        optimizer_type=str(args.optimizer),
        optimizer_betas=(0.9, 0.95),
        optimizer_weight_decay=float(args.weight_decay),
        enable_perf_tracing=bool(args.perf_tracing),
        collect_training_curve=True,
        profile_disable_post_eval=not bool(args.post_eval),
        profile_disable_inflight_checks=bool(args.disable_inflight_checks),
        stage1_compute_discovery_loss=bool(args.post_eval),
        stage1_compute_val_loss=bool(args.post_eval),
        stage1_discovery_batches=1,
        stage1_val_batches=1,
        stage1_discovery_batch_size=min(2, int(args.batch_size)),
        stage1_val_batch_size=min(2, int(args.batch_size)),
        skip_post_s1_fingerprint=True,
        skip_post_s1_triage=True,
        skip_binding_probes=True,
        skip_screening_wikitext=True,
        skip_screening_hellaswag=True,
        skip_screening_blimp=True,
    )


def _selected_cases(args: argparse.Namespace) -> list[ScaffoldCase]:
    families = tuple(args.family)
    if args.fast_set:
        return [
            ScaffoldCase(
                "gpt2_attn", "gpt2_attn:softmax_attention", "softmax_attention"
            ),
            ScaffoldCase("gpt2_attn", "gpt2_attn:linear_attention", "linear_attention"),
            ScaffoldCase("gpt2_ffn", "gpt2_ffn:swiglu_mlp", "swiglu_mlp"),
            ScaffoldCase("gpt2_ffn", "gpt2_ffn:conv1d_seq", "conv1d_seq"),
            ScaffoldCase("gpt2_replace", "gpt2_replace:rwkv_channel", "rwkv_channel"),
            ScaffoldCase("mamba_mixer", "mamba_mixer:selective_scan", "selective_scan"),
            ScaffoldCase(
                "pair_residual",
                "pair_residual:swiglu_mlp+conv1d_seq",
                "swiglu_mlp",
                "conv1d_seq",
            ),
        ][: max(1, int(args.max_cases))]
    cases = generate_cases(
        families,
        args.ops or None,
        max_pairs=int(args.max_pairs),
        allow_arbitrary_ops=bool(args.allow_arbitrary_ops),
    )
    return cases[: max(1, int(args.max_cases))]


def _compile_case(
    case: ScaffoldCase,
    config: RunConfig,
    *,
    model_seed: int,
) -> tuple[torch.nn.Module, str, str]:
    torch.manual_seed(int(model_seed))
    graph = build_scaffold(case, model_dim=int(config.model_dim))
    model = compile_model(
        [graph] * int(config.n_layers),
        vocab_size=int(config.vocab_size),
        max_seq_len=int(config.max_seq_len),
    )
    return model, graph_to_json(graph), graph.fingerprint()


def _run_case_once(
    *,
    runner: ExperimentRunner,
    case: ScaffoldCase,
    config: RunConfig,
    seed: int,
    model_seed: int,
    native_optimizer: bool,
) -> dict[str, Any]:
    model, graph_json, fingerprint = _compile_case(case, config, model_seed=model_seed)
    env_value = "1" if native_optimizer else "0"
    started = time.perf_counter()
    with _capture_research_logs() as logs:
        with _env_flag("MICRO_TRAIN_NATIVE_OPTIMIZER", env_value):
            result = runner._micro_train(
                model,
                config,
                torch.device("cpu"),
                seed=int(seed),
                graph_json=graph_json,
            )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    params = [p.detach().cpu().clone() for p in model.parameters()]
    return {
        "result": result,
        "params": params,
        "elapsed_ms": elapsed_ms,
        "logs": list(logs),
        "graph_fingerprint": fingerprint,
    }


def _max_param_delta(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    max_delta = 0.0
    for left, right in zip(a, b, strict=True):
        max_delta = max(max_delta, float((left - right).abs().max().item()))
    return max_delta


def _float_value(result: dict[str, Any], key: str) -> float:
    value = result.get(key)
    return float("nan") if value is None else float(value)


def _curve_delta(reference: dict[str, Any], native: dict[str, Any]) -> dict[str, Any]:
    ref_curve = reference.get("training_curve") or []
    native_curve = native.get("training_curve") or []
    compared = min(len(ref_curve), len(native_curve))
    max_loss = 0.0
    max_grad = 0.0
    for idx in range(compared):
        max_loss = max(
            max_loss,
            abs(
                float(ref_curve[idx].get("loss", 0.0))
                - float(native_curve[idx].get("loss", 0.0))
            ),
        )
        max_grad = max(
            max_grad,
            abs(
                float(ref_curve[idx].get("grad_norm", 0.0))
                - float(native_curve[idx].get("grad_norm", 0.0))
            ),
        )
    return {
        "reference_len": len(ref_curve),
        "native_len": len(native_curve),
        "compared": compared,
        "max_loss_delta": max_loss,
        "max_grad_norm_delta": max_grad,
    }


def _research_error_logs(logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in logs if row.get("level") in {"ERROR", "CRITICAL"}]


def _evaluate_cutover(
    *,
    reference_run: dict[str, Any],
    native_run: dict[str, Any],
    thresholds: CutoverThresholds,
) -> dict[str, Any]:
    ref = reference_run["result"]
    native = native_run["result"]
    curve = _curve_delta(ref, native)
    final_delta = abs(
        _float_value(ref, "final_loss") - _float_value(native, "final_loss")
    )
    initial_delta = abs(
        _float_value(ref, "initial_loss") - _float_value(native, "initial_loss")
    )
    param_delta = _max_param_delta(reference_run["params"], native_run["params"])
    ref_errors = _research_error_logs(reference_run["logs"])
    native_errors = _research_error_logs(native_run["logs"])
    checks = {
        "final_loss": final_delta <= thresholds.final_loss,
        "initial_loss": initial_delta <= thresholds.initial_loss,
        "curve_loss": curve["max_loss_delta"] <= thresholds.curve_loss,
        "curve_grad_norm": curve["max_grad_norm_delta"] <= thresholds.curve_grad_norm,
        "parameters": param_delta <= thresholds.parameter,
        "passed": bool(ref.get("passed")) == bool(native.get("passed")),
        "steps": int(ref.get("n_train_steps") or 0)
        == int(native.get("n_train_steps") or 0),
        "error_type": (ref.get("error_type") or "") == (native.get("error_type") or ""),
        "native_active": str(native.get("native_optimizer_active") or "").startswith(
            "_Native"
        ),
        "native_error_logs": not native_errors,
        "reference_error_logs": not ref_errors,
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "deltas": {
            "final_loss": final_delta,
            "initial_loss": initial_delta,
            "curve_loss": curve["max_loss_delta"],
            "curve_grad_norm": curve["max_grad_norm_delta"],
            "parameters": param_delta,
        },
        "curve": curve,
        "native_error_logs": native_errors,
        "reference_error_logs": ref_errors,
    }


def _case_report(
    *,
    case: ScaffoldCase,
    seed: int,
    model_seed: int,
    reference_run: dict[str, Any],
    native_run: dict[str, Any],
    verdict: dict[str, Any],
) -> dict[str, Any]:
    ref = reference_run["result"]
    native = native_run["result"]
    return {
        "case": asdict(case),
        "seed": int(seed),
        "model_seed": int(model_seed),
        "graph_fingerprint": native_run["graph_fingerprint"],
        "ok": bool(verdict["ok"]),
        "reference_ms": reference_run["elapsed_ms"],
        "native_ms": native_run["elapsed_ms"],
        "speedup": (
            reference_run["elapsed_ms"] / native_run["elapsed_ms"]
            if native_run["elapsed_ms"] > 0.0
            else 0.0
        ),
        "reference": {
            "passed": bool(ref.get("passed")),
            "error_type": ref.get("error_type"),
            "initial_loss": ref.get("initial_loss"),
            "final_loss": ref.get("final_loss"),
            "n_train_steps": ref.get("n_train_steps"),
        },
        "native": {
            "passed": bool(native.get("passed")),
            "error_type": native.get("error_type"),
            "initial_loss": native.get("initial_loss"),
            "final_loss": native.get("final_loss"),
            "n_train_steps": native.get("n_train_steps"),
            "native_optimizer_active": native.get("native_optimizer_active"),
        },
        "verdict": verdict,
    }


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    load_runner_native()
    config = _build_config(args)
    thresholds = CutoverThresholds(
        final_loss=float(args.max_final_loss_delta),
        initial_loss=float(args.max_initial_loss_delta),
        curve_loss=float(args.max_curve_loss_delta),
        curve_grad_norm=float(args.max_curve_grad_norm_delta),
        parameter=float(args.max_param_delta),
    )
    runner = ExperimentRunner(str(args.db))
    cases = _selected_cases(args)
    for warmup_idx in range(max(0, int(args.warmups))):
        warmup_case = cases[warmup_idx % len(cases)]
        warmup_seed = int(args.seed) - warmup_idx - 1
        warmup_model_seed = int(args.model_seed) - warmup_idx - 1
        _run_case_once(
            runner=runner,
            case=warmup_case,
            config=config,
            seed=warmup_seed,
            model_seed=warmup_model_seed,
            native_optimizer=False,
        )
        _run_case_once(
            runner=runner,
            case=warmup_case,
            config=config,
            seed=warmup_seed,
            model_seed=warmup_model_seed,
            native_optimizer=True,
        )
    rows = []
    for case_idx, case in enumerate(cases):
        for repeat in range(max(1, int(args.repeats))):
            seed = int(args.seed) + case_idx * 1000 + repeat
            model_seed = int(args.model_seed) + case_idx
            reference_run = _run_case_once(
                runner=runner,
                case=case,
                config=config,
                seed=seed,
                model_seed=model_seed,
                native_optimizer=False,
            )
            native_run = _run_case_once(
                runner=runner,
                case=case,
                config=config,
                seed=seed,
                model_seed=model_seed,
                native_optimizer=True,
            )
            verdict = _evaluate_cutover(
                reference_run=reference_run,
                native_run=native_run,
                thresholds=thresholds,
            )
            rows.append(
                _case_report(
                    case=case,
                    seed=seed,
                    model_seed=model_seed,
                    reference_run=reference_run,
                    native_run=native_run,
                    verdict=verdict,
                )
            )
            if bool(args.stop_on_failure) and not verdict["ok"]:
                break
        if bool(args.stop_on_failure) and rows and not rows[-1]["ok"]:
            break

    reference_times = [float(row["reference_ms"]) for row in rows]
    native_times = [float(row["native_ms"]) for row in rows]
    failures = [row for row in rows if not row["ok"]]
    summary = {
        "ok": not failures,
        "eligible_for_default_native_optimizer": not failures,
        "cases": len(cases),
        "runs": len(rows),
        "failures": len(failures),
        "reference_ms_median": statistics.median(reference_times)
        if reference_times
        else 0.0,
        "native_ms_median": statistics.median(native_times) if native_times else 0.0,
        "speedup_median": (
            statistics.median(reference_times) / statistics.median(native_times)
            if native_times and statistics.median(native_times) > 0.0
            else 0.0
        ),
        "max_final_loss_delta": max(
            (row["verdict"]["deltas"]["final_loss"] for row in rows), default=0.0
        ),
        "max_curve_loss_delta": max(
            (row["verdict"]["deltas"]["curve_loss"] for row in rows), default=0.0
        ),
        "max_curve_grad_norm_delta": max(
            (row["verdict"]["deltas"]["curve_grad_norm"] for row in rows), default=0.0
        ),
        "max_param_delta": max(
            (row["verdict"]["deltas"]["parameters"] for row in rows), default=0.0
        ),
    }
    return {
        "summary": summary,
        "thresholds": asdict(thresholds),
        "config": {
            "model_dim": config.model_dim,
            "n_layers": config.n_layers,
            "vocab_size": config.vocab_size,
            "max_seq_len": config.max_seq_len,
            "stage1_steps": config.stage1_steps,
            "stage1_batch_size": config.stage1_batch_size,
            "optimizer_type": config.optimizer_type,
            "post_eval": bool(args.post_eval),
            "warmups": int(args.warmups),
        },
        "failures": failures,
        "runs": rows,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="research/runs.db")
    parser.add_argument("--json-out", default="")
    parser.add_argument(
        "--family",
        action="append",
        default=["gpt2_attn", "gpt2_ffn", "gpt2_replace", "mamba_mixer"],
    )
    parser.add_argument("--ops", action="append", default=[])
    parser.add_argument(
        "--fast-set", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--allow-arbitrary-ops", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--max-cases", type=int, default=7)
    parser.add_argument("--max-pairs", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=6101)
    parser.add_argument("--model-seed", type=int, default=9103)
    parser.add_argument("--model-dim", type=int, default=48)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--optimizer", choices=("adamw", "sgd"), default="adamw")
    parser.add_argument(
        "--post-eval", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--perf-tracing", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--disable-inflight-checks", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--stop-on-failure", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--max-final-loss-delta", type=float, default=1e-6)
    parser.add_argument("--max-initial-loss-delta", type=float, default=1e-6)
    parser.add_argument("--max-curve-loss-delta", type=float, default=1e-6)
    parser.add_argument("--max-curve-grad-norm-delta", type=float, default=2e-5)
    parser.add_argument("--max-param-delta", type=float, default=2e-6)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = run_validation(args)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out:
        Path(args.json_out).write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0 if report["summary"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
