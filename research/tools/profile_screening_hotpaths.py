#!/usr/bin/env python
"""Targeted benchmark for experiment-screening hot paths.

Measures the parts that still matter after `research/eval` cleanup:
1. Stage 1 micro-train control-plane cost on a fixed scaffold
2. Lean candidate-screening config vs full config
3. Orchestrator preprocessing and worker-queue telemetry on a controlled batch
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict

import torch

from research.perf_contract import build_perf_contract, emit_perf_artifact
from research.scientist.native_runner import compile_model_native_first as compile_model
from research.scientist.runner import ExperimentRunner, RunConfig
from research.scientist.runner.execution_screening import _make_stage1_screening_config
from research.tools.profile_component_scaffolds import build_gpt2_attn_scaffold
from research.orchestrator.executor import WorkerPoolOrchestrator


def _build_config(*, fixture: str) -> RunConfig:
    quick = fixture == "quick"
    return RunConfig(
        device="cpu",
        data_mode="random",
        model_dim=32 if quick else 48,
        n_layers=1,
        vocab_size=128 if quick else 256,
        max_seq_len=16 if quick else 24,
        stage1_steps=1 if quick else 4,
        stage1_batch_size=1 if quick else 2,
        enable_perf_tracing=False,
        collect_training_curve=False,
    )


def _compile_scaffold(config: RunConfig) -> tuple[torch.nn.Module, float]:
    graph = build_gpt2_attn_scaffold("softmax_attention", model_dim=config.model_dim)
    started = time.perf_counter()
    model = compile_model(
        [graph] * config.n_layers,
        vocab_size=config.vocab_size,
        max_seq_len=config.max_seq_len,
    )
    compile_ms = (time.perf_counter() - started) * 1000.0
    return model, compile_ms


def _time_micro_train(
    runner: ExperimentRunner,
    model: torch.nn.Module,
    config: RunConfig,
) -> Dict[str, Any]:
    started = time.perf_counter()
    result = runner._micro_train(model, config, torch.device("cpu"), seed=101)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "elapsed_ms": elapsed_ms,
        "final_loss": result.get("final_loss"),
        "passed": bool(result.get("passed", False)),
        "n_train_steps": int(result.get("n_train_steps") or 0),
        "discovery_loss": result.get("discovery_loss"),
        "validation_loss": result.get("validation_loss"),
    }


def benchmark_stage1_hotpath(*, fixture: str, repeats: int) -> Dict[str, Any]:
    config = _build_config(fixture=fixture)
    lean_config = _make_stage1_screening_config(config)
    runner = ExperimentRunner("research/lab_notebook.db")
    base_runs = []
    lean_runs = []
    for _ in range(max(1, repeats)):
        base_model, base_compile_ms = _compile_scaffold(config)
        base_run = _time_micro_train(runner, base_model, config)
        base_run["compile_ms"] = base_compile_ms
        base_runs.append(base_run)

        lean_model, lean_compile_ms = _compile_scaffold(config)
        lean_run = _time_micro_train(runner, lean_model, lean_config)
        lean_run["compile_ms"] = lean_compile_ms
        lean_runs.append(lean_run)
    base_elapsed = [row["elapsed_ms"] for row in base_runs]
    lean_elapsed = [row["elapsed_ms"] for row in lean_runs]
    base_compile = [row["compile_ms"] for row in base_runs]
    lean_compile = [row["compile_ms"] for row in lean_runs]
    base = dict(base_runs[-1])
    lean = dict(lean_runs[-1])
    base["elapsed_ms_median"] = statistics.median(base_elapsed)
    lean["elapsed_ms_median"] = statistics.median(lean_elapsed)
    base["compile_ms_median"] = statistics.median(base_compile)
    lean["compile_ms_median"] = statistics.median(lean_compile)
    base["elapsed_ms_runs"] = base_elapsed
    lean["elapsed_ms_runs"] = lean_elapsed
    base["compile_ms_runs"] = base_compile
    lean["compile_ms_runs"] = lean_compile
    return {
        "base": base,
        "lean": lean,
        "speedup": (
            base["elapsed_ms_median"] / lean["elapsed_ms_median"]
            if lean["elapsed_ms_median"] > 0.0
            else 0.0
        ),
        "config": {
            "stage1_steps": config.stage1_steps,
            "stage1_batch_size": config.stage1_batch_size,
            "max_seq_len": config.max_seq_len,
            "model_dim": config.model_dim,
            "n_layers": config.n_layers,
            "vocab_size": config.vocab_size,
            "repeats": max(1, repeats),
        },
    }


def benchmark_orchestrator_hotpath(*, fixture: str) -> Dict[str, Any]:
    config = _build_config(fixture=fixture)
    lean_config = _make_stage1_screening_config(config)
    runner = ExperimentRunner("research/lab_notebook.db")
    graph = build_gpt2_attn_scaffold("softmax_attention", model_dim=config.model_dim)
    orchestrator = WorkerPoolOrchestrator(
        train_fn=lambda model, cfg, seed, dev: runner._micro_train(
            model, cfg, dev, seed
        ),
        num_workers=1,
        max_queue_size=2 if fixture == "quick" else 3,
        devices=["cpu"],
    )
    started = time.perf_counter()
    try:
        n_jobs = 2 if fixture == "quick" else 3
        for index in range(n_jobs):
            orchestrator.submit(
                index=index,
                graph=graph,
                config=lean_config,
                seed=101 + index,
                payload={
                    "metrics": {},
                    "graph": graph,
                    "queue_kind": "candidate_screening",
                },
            )

        results = []
        while (
            len(results) < n_jobs
            or orchestrator.job_queue.unfinished_tasks > 0
            or orchestrator.prep_queue.unfinished_tasks > 0
        ):
            results.extend(orchestrator.get_results(timeout=0.002))
        telemetry = orchestrator.get_telemetry()
    finally:
        orchestrator.shutdown()

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    passed = sum(1 for row in results if row.s1_result.get("passed"))
    return {
        "elapsed_ms": elapsed_ms,
        "n_jobs": len(results),
        "n_passed": passed,
        "queue_telemetry": telemetry,
    }


def build_screening_hotpath_report(
    *, fixture: str = "quick", repeats: int | None = None
) -> Dict[str, Any]:
    if fixture not in {"quick", "standard"}:
        raise ValueError(f"Unsupported fixture: {fixture}")
    stage1_repeats = (
        repeats if repeats is not None else (1 if fixture == "quick" else 2)
    )
    stage1 = benchmark_stage1_hotpath(fixture=fixture, repeats=stage1_repeats)
    orchestrator = benchmark_orchestrator_hotpath(fixture=fixture)
    metrics = {
        "stage1_base_ms": round(stage1["base"]["elapsed_ms_median"], 4),
        "stage1_lean_ms": round(stage1["lean"]["elapsed_ms_median"], 4),
        "stage1_speedup": round(stage1["speedup"], 4),
        "stage1_base_compile_ms": round(stage1["base"]["compile_ms_median"], 4),
        "stage1_lean_compile_ms": round(stage1["lean"]["compile_ms_median"], 4),
        "stage1_base_final_loss": stage1["base"]["final_loss"],
        "stage1_lean_final_loss": stage1["lean"]["final_loss"],
        "orchestrator_total_ms": round(orchestrator["elapsed_ms"], 4),
        "orchestrator_jobs": orchestrator["n_jobs"],
        "orchestrator_passed": orchestrator["n_passed"],
        "queue_submit_wait_ms": round(
            float(
                orchestrator["queue_telemetry"].get("submit_wait_avg_ms", 0.0) or 0.0
            ),
            4,
        ),
        "queue_prep_wait_ms": round(
            float(
                orchestrator["queue_telemetry"].get("prep_queue_wait_avg_ms", 0.0)
                or 0.0
            ),
            4,
        ),
        "queue_scheduling_wait_ms": round(
            float(
                orchestrator["queue_telemetry"].get("scheduling_wait_avg_ms", 0.0)
                or 0.0
            ),
            4,
        ),
        "queue_preprocessing_ms": round(
            float(
                orchestrator["queue_telemetry"].get("preprocessing_avg_ms", 0.0) or 0.0
            ),
            4,
        ),
        "queue_execution_ms": round(
            float(
                orchestrator["queue_telemetry"].get("job_execution_avg_ms", 0.0) or 0.0
            ),
            4,
        ),
        "total_time_ms": round(
            stage1["base"]["elapsed_ms_median"]
            + stage1["lean"]["elapsed_ms_median"]
            + orchestrator["elapsed_ms"],
            4,
        ),
    }
    return {
        "fixture": fixture,
        "stage1": stage1,
        "orchestrator": orchestrator,
        "metrics": metrics,
        # Compatibility aliases for existing audit readers.
        "stage1_base_median_ms": metrics["stage1_base_ms"],
        "stage09_gate_median_ms": metrics["stage1_lean_ms"],
        "stage09_speedup_x": metrics["stage1_speedup"],
        "orchestrator_total_ms": metrics["orchestrator_total_ms"],
        "queue_telemetry": orchestrator["queue_telemetry"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark experiment-screening hot paths"
    )
    parser.add_argument(
        "--fixture",
        choices=("quick", "standard"),
        default="quick",
        help="Benchmark fixture size",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=0,
        help="Override stage-1 timing repeats (default: 1 for quick, 3 for standard)",
    )
    parser.add_argument(
        "--json-out", type=str, default="", help="Optional JSON report path"
    )
    parser.add_argument(
        "--artifact-root",
        type=str,
        default="",
        help="Optional perf artifact root override",
    )
    args = parser.parse_args()

    report = build_screening_hotpath_report(
        fixture=str(args.fixture),
        repeats=(int(args.repeats) if int(args.repeats) > 0 else None),
    )
    contract = build_perf_contract(
        component="research",
        workload="experiment_screening_hotpaths",
        metrics=report["metrics"],
        identity={
            "fixture": report["fixture"],
            "orchestrator_jobs": report["orchestrator"]["n_jobs"],
            "stage1_repeats": report["stage1"]["config"]["repeats"],
        },
        warnings=[],
    )
    artifact_path = emit_perf_artifact(
        contract,
        root=args.artifact_root or None,
        slug=f"screening_hotpaths_{report['fixture']}",
    )
    report["perf_contract"] = contract
    report["perf_contract"]["artifact_path"] = artifact_path

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    else:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
