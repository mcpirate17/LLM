#!/usr/bin/env python
"""Profile the current micro-train step boundary and native train-step ROI."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any, Iterator

import torch

from research.scientist.native_runner import compile_model_native_first as compile_model
from research.scientist.runner import ExperimentRunner, RunConfig
from research.scientist.runner.execution_training_native_boundary import (
    _collect_aux_modules,
)
from research.tools.profile_component_scaffolds import (
    ScaffoldCase,
    build_scaffold,
)


@contextmanager
def _env_flag(name: str, value: str | None) -> Iterator[None]:
    old = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


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
        collect_training_curve=False,
        enable_perf_tracing=False,
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


def _case_from_args(args: argparse.Namespace) -> ScaffoldCase:
    family = str(args.family)
    op = str(args.op)
    if family == "pair_residual":
        op_b = str(args.op_b or "conv1d_seq")
        return ScaffoldCase(family, f"{family}:{op}+{op_b}", op, op_b)
    return ScaffoldCase(family, f"{family}:{op}", op)


def _compile_model(case: ScaffoldCase, config: RunConfig, *, model_seed: int):
    torch.manual_seed(int(model_seed))
    graph = build_scaffold(case, model_dim=int(config.model_dim))
    model = compile_model(
        [graph] * int(config.n_layers),
        vocab_size=int(config.vocab_size),
        max_seq_len=int(config.max_seq_len),
    )
    return model, graph


def _setup_profile_context(
    *,
    runner: ExperimentRunner,
    model: torch.nn.Module,
    config: RunConfig,
    graph_json: str,
    seed: int,
):
    ctx = runner._micro_train_build_context(
        model,
        config,
        torch.device("cpu"),
        int(seed),
        graph_json,
    )
    optimizer, _ = runner._micro_train_setup_optimizer(
        model,
        config,
        torch.device("cpu"),
        int(seed),
        ctx.result,
        ctx.use_synthesized_training,
        ctx.tracer,
        ctx.trace_totals_ms,
        ctx.run_profiler,
    )
    ctx.optimizer = optimizer
    ctx.model_params = tuple(model.parameters())
    (
        ctx.routing_modules,
        ctx.early_exit_modules,
        ctx.lm_head,
        ctx.norm,
    ) = _collect_aux_modules(model)
    return ctx


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _total(values: list[float]) -> float:
    return float(sum(values))


def _pct(value: float, total: float) -> float:
    return (value / total * 100.0) if total > 0.0 else 0.0


def _profile_steps(
    *,
    runner: ExperimentRunner,
    ctx: Any,
    steps: int,
    warmups: int,
) -> dict[str, Any]:
    samples: list[dict[str, float]] = []
    loss_values: list[float] = []
    grad_values: list[float] = []
    total = max(1, int(steps)) + max(0, int(warmups))
    for step in range(total):
        data_started = time.perf_counter()
        input_ids, step_started = runner._micro_train_sample_data(ctx, step)
        data_ms = (time.perf_counter() - data_started) * 1000.0

        before = dict(ctx.trace_totals_ms)
        exec_started = time.perf_counter()
        step_state = runner._micro_train_execute_step(ctx, input_ids, step)
        exec_ms = (time.perf_counter() - exec_started) * 1000.0
        after = dict(ctx.trace_totals_ms)

        if step < warmups:
            continue

        measured_step_ms = (time.perf_counter() - step_started) * 1000.0
        forward_ms = float(
            after.get("forward_pass", 0.0) - before.get("forward_pass", 0.0)
        )
        backward_ms = float(
            after.get("backward_pass", 0.0) - before.get("backward_pass", 0.0)
        )
        optimizer_ms = float(
            after.get("optimizer_step", 0.0) - before.get("optimizer_step", 0.0)
        )
        backward_optimizer_ms = float(
            after.get("backward_optimizer_step", 0.0)
            - before.get("backward_optimizer_step", 0.0)
        )
        accounted_ms = (
            data_ms + forward_ms + backward_ms + optimizer_ms + backward_optimizer_ms
        )
        python_orchestration_ms = max(0.0, measured_step_ms - accounted_ms)
        loss = step_state.get("loss")
        if loss is not None:
            loss_values.append(float(loss.detach().item()))
        grad_values.append(float(step_state.get("grad_norm", 0.0)))
        samples.append(
            {
                "step": float(step - warmups),
                "data_sampling_ms": data_ms,
                "forward_ms": forward_ms,
                "backward_ms": backward_ms,
                "optimizer_ms": optimizer_ms,
                "backward_optimizer_ms": backward_optimizer_ms,
                "python_orchestration_ms": python_orchestration_ms,
                "execute_step_ms": exec_ms,
                "step_total_ms": measured_step_ms,
            }
        )
    totals = {
        "data_sampling_ms": _total([row["data_sampling_ms"] for row in samples]),
        "forward_ms": _total([row["forward_ms"] for row in samples]),
        "backward_ms": _total([row["backward_ms"] for row in samples]),
        "optimizer_ms": _total([row["optimizer_ms"] for row in samples]),
        "backward_optimizer_ms": _total(
            [row["backward_optimizer_ms"] for row in samples]
        ),
        "python_orchestration_ms": _total(
            [row["python_orchestration_ms"] for row in samples]
        ),
        "step_total_ms": _total([row["step_total_ms"] for row in samples]),
    }
    medians = {
        "data_sampling_ms": _median([row["data_sampling_ms"] for row in samples]),
        "forward_ms": _median([row["forward_ms"] for row in samples]),
        "backward_ms": _median([row["backward_ms"] for row in samples]),
        "optimizer_ms": _median([row["optimizer_ms"] for row in samples]),
        "backward_optimizer_ms": _median(
            [row["backward_optimizer_ms"] for row in samples]
        ),
        "python_orchestration_ms": _median(
            [row["python_orchestration_ms"] for row in samples]
        ),
        "step_total_ms": _median([row["step_total_ms"] for row in samples]),
    }
    total_ms = totals["step_total_ms"]
    shares = {
        key: _pct(value, total_ms)
        for key, value in totals.items()
        if key != "step_total_ms"
    }
    return {
        "samples": samples,
        "totals_ms": totals,
        "medians_ms": medians,
        "shares_pct": shares,
        "loss": {
            "first": loss_values[0] if loss_values else None,
            "last": loss_values[-1] if loss_values else None,
        },
        "grad_norm": {
            "median": _median(grad_values),
            "max": max(grad_values) if grad_values else 0.0,
        },
    }


def _abi_readiness(ctx: Any, graph: Any) -> dict[str, Any]:
    dispatcher_classes = []
    dispatcher_stats = []
    for layer in getattr(ctx.model, "layers", []):
        dispatcher = getattr(layer, "_subgraph_dispatcher", None)
        if dispatcher is None:
            continue
        dispatcher_classes.append(type(dispatcher).__name__)
        stats = dispatcher.stats if hasattr(dispatcher, "stats") else {}
        dispatcher_stats.append(stats)
    optimizer_class = type(ctx.optimizer).__name__
    native_optimizer = hasattr(ctx.optimizer, "step_with_grad_clip")
    bound_dispatch = any(
        name == "BoundNativeSubgraphDispatcher" for name in dispatcher_classes
    )
    unsupported = []
    if ctx.dev.type != "cpu":
        unsupported.append("device_not_cpu")
    if str(getattr(ctx.config, "optimizer_type", "")).lower() not in {"adamw", "sgd"}:
        unsupported.append("optimizer_not_adamw_sgd")
    if not native_optimizer:
        unsupported.append("optimizer_not_native")
    if not bound_dispatch:
        unsupported.append("no_bound_native_subgraph_dispatcher")
    if ctx.routing_modules:
        unsupported.append("routing_aux_modules_present")
    if ctx.early_exit_modules:
        unsupported.append("early_exit_modules_present")
    return {
        "native_train_step_candidate": not unsupported,
        "unsupported_reasons": unsupported,
        "optimizer_class": optimizer_class,
        "native_optimizer": native_optimizer,
        "dispatcher_classes": dispatcher_classes,
        "dispatcher_stats": dispatcher_stats,
        "graph_ops": sorted(
            {
                getattr(node, "op_name", "")
                for node in getattr(graph, "nodes", {}).values()
                if not getattr(node, "is_input", False)
            }
        ),
        "proposed_native_train_step_abi": {
            "session_state": [
                "compiled bound graph handle",
                "model parameter tensors",
                "optimizer state tensors",
                "optimizer hyperparameters",
                "step counter",
            ],
            "per_step_inputs": ["input_ids"],
            "per_step_outputs": [
                "loss",
                "grad_norm",
                "step_time_ms",
                "nonfinite_status",
            ],
            "native_owned_work": [
                "forward",
                "loss",
                "backward",
                "grad clipping",
                "optimizer update",
            ],
        },
    }


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    case = _case_from_args(args)
    config = _build_config(args)
    runner = ExperimentRunner(str(args.db))
    env_value = "0" if args.disable_native_optimizer else None
    with _env_flag("MICRO_TRAIN_NATIVE_OPTIMIZER", env_value):
        with _env_flag(
            "MICRO_TRAIN_NATIVE_BACKWARD_STEP",
            "1" if args.native_backward_step else "0",
        ):
            model, graph = _compile_model(case, config, model_seed=int(args.model_seed))
            ctx = _setup_profile_context(
                runner=runner,
                model=model,
                config=config,
                graph_json="",
                seed=int(args.seed),
            )
            ctx.run_profiler.__enter__()
            try:
                profile = _profile_steps(
                    runner=runner,
                    ctx=ctx,
                    steps=int(args.steps),
                    warmups=int(args.warmups),
                )
            finally:
                ctx.run_profiler.__exit__(None, None, None)
    return {
        "case": {
            "family": case.family,
            "name": case.name,
            "op_a": case.op_a,
            "op_b": case.op_b,
        },
        "config": {
            "model_dim": config.model_dim,
            "n_layers": config.n_layers,
            "vocab_size": config.vocab_size,
            "max_seq_len": config.max_seq_len,
            "batch_size": config.stage1_batch_size,
            "steps": int(args.steps),
            "warmups": int(args.warmups),
            "optimizer": config.optimizer_type,
            "native_backward_step": bool(args.native_backward_step),
        },
        "profile": profile,
        "abi_readiness": _abi_readiness(ctx, graph),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="research/lab_notebook.db")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--family", default="gpt2_attn")
    parser.add_argument("--op", default="softmax_attention")
    parser.add_argument("--op-b", default="")
    parser.add_argument("--model-dim", type=int, default=48)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--model-seed", type=int, default=888)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--optimizer", choices=("adamw", "sgd"), default="adamw")
    parser.add_argument(
        "--disable-native-optimizer",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--native-backward-step",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = run_profile(args)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out:
        Path(args.json_out).write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
