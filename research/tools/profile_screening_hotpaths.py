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
from contextlib import suppress
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


def _close_warmup_model(model: torch.nn.Module) -> None:
    session = getattr(model, "_native_runner_abi_session", None)
    if session is None or not hasattr(session, "close"):
        return
    with suppress(Exception):
        session.close()


def _safe_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _perf_summary_ms(result: Dict[str, Any]) -> Dict[str, float]:
    perf = result.get("perf_report") or result.get("perf_traces") or {}
    summary = perf.get("summary_ms") if isinstance(perf, dict) else {}
    if not isinstance(summary, dict):
        return {}
    out: Dict[str, float] = {}
    for key, value in summary.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _probe_timings_ms(result: Dict[str, Any]) -> Dict[str, float]:
    timings: Dict[str, float] = {}
    for key, value in result.items():
        if not str(key).endswith("_elapsed_ms"):
            continue
        try:
            timings[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return timings


def _probe_status_fields(result: Dict[str, Any]) -> Dict[str, Any]:
    statuses: Dict[str, Any] = {}
    for key, value in result.items():
        if value is None:
            continue
        key_str = str(key)
        if key_str.endswith("_status") or key_str.endswith("_timed_out"):
            statuses[key_str] = _safe_scalar(value)
    return statuses


def _rank_timing_sources(
    perf_summary: Dict[str, float],
    probe_timings: Dict[str, float],
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for key, value in perf_summary.items():
        rows.append({"source": f"perf_summary.{key}", "elapsed_ms": float(value)})
    for key, value in probe_timings.items():
        rows.append({"source": f"probe.{key}", "elapsed_ms": float(value)})
    rows.sort(key=lambda row: row["elapsed_ms"], reverse=True)
    return rows[:12]


def _extract_eval_diagnostics(result: Dict[str, Any]) -> Dict[str, Any]:
    perf_summary = _perf_summary_ms(result)
    probe_timings = _probe_timings_ms(result)
    return {
        "perf_summary_ms": perf_summary,
        "probe_timings_ms": probe_timings,
        "probe_status": _probe_status_fields(result),
        "top_timing_sources": _rank_timing_sources(perf_summary, probe_timings),
    }


def _time_micro_train(
    runner: ExperimentRunner,
    model: torch.nn.Module,
    config: RunConfig,
    *,
    seed: int = 101,
) -> Dict[str, Any]:
    started = time.perf_counter()
    result = runner._micro_train(model, config, torch.device("cpu"), seed=seed)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "elapsed_ms": elapsed_ms,
        "final_loss": result.get("final_loss"),
        "passed": bool(result.get("passed", False)),
        "n_train_steps": int(result.get("n_train_steps") or 0),
        "discovery_loss": result.get("discovery_loss"),
        "validation_loss": result.get("validation_loss"),
        "diagnostics": _extract_eval_diagnostics(result),
    }


def _variant_config(config: RunConfig, variant: str) -> RunConfig:
    variant_config = _make_stage1_screening_config(config)
    if variant == "current":
        return variant_config
    if variant == "train_only_no_post_eval":
        variant_config.profile_disable_post_eval = True
        return variant_config
    if variant == "language_only":
        variant_config.skip_binding_probes = True
        variant_config.skip_ar_probe = True
        variant_config.skip_ar_gate = True
        return variant_config
    if variant == "language_no_blimp":
        variant_config.skip_binding_probes = True
        variant_config.skip_ar_probe = True
        variant_config.skip_ar_gate = True
        variant_config.skip_screening_blimp = True
        return variant_config
    if variant == "binding_only_no_ar":
        variant_config.skip_screening_wikitext = True
        variant_config.skip_screening_hellaswag = True
        variant_config.skip_screening_blimp = True
        variant_config.skip_ar_probe = True
        variant_config.skip_ar_gate = True
        return variant_config
    if variant == "binding_plus_ar_gate":
        variant_config.skip_screening_wikitext = True
        variant_config.skip_screening_hellaswag = True
        variant_config.skip_screening_blimp = True
        variant_config.skip_ar_probe = True
        return variant_config
    raise ValueError(f"Unsupported gating variant: {variant}")


def _variant_flags(config: RunConfig) -> Dict[str, Any]:
    return {
        "profile_disable_post_eval": bool(config.profile_disable_post_eval),
        "skip_screening_wikitext": bool(config.skip_screening_wikitext),
        "skip_screening_hellaswag": bool(config.skip_screening_hellaswag),
        "skip_screening_blimp": bool(config.skip_screening_blimp),
        "skip_binding_probes": bool(config.skip_binding_probes),
        "skip_induction_probe": bool(config.skip_induction_probe),
        "skip_binding_probe": bool(config.skip_binding_probe),
        "skip_ar_probe": bool(config.skip_ar_probe),
        "skip_ar_gate": bool(config.skip_ar_gate),
    }


def _time_stage1_variant(
    runner: ExperimentRunner,
    config: RunConfig,
    *,
    model_seed: int,
    seed: int,
) -> Dict[str, Any]:
    torch.manual_seed(int(model_seed))
    model, compile_ms = _compile_scaffold(config)
    try:
        run = _time_micro_train(runner, model, config, seed=seed)
    finally:
        _close_warmup_model(model)
    run["compile_ms"] = compile_ms
    return run


def _find_passing_variant_seed(
    runner: ExperimentRunner,
    base_config: RunConfig,
    *,
    fixture: str,
) -> tuple[int, Dict[str, Any]]:
    train_only = _variant_config(base_config, "train_only_no_post_eval")
    candidates = (1701, 1702, 1703, 1704, 1705, 1706)
    last_run: Dict[str, Any] | None = None
    for model_seed in candidates:
        run = _time_stage1_variant(
            runner,
            train_only,
            model_seed=model_seed,
            seed=901,
        )
        last_run = run
        if run["passed"]:
            return model_seed, run
        if fixture == "quick":
            break
    return candidates[0], last_run or {}


def benchmark_gating_variants(*, fixture: str) -> Dict[str, Any]:
    base_config = _build_config(fixture=fixture)
    runner = ExperimentRunner("research/runs.db")
    variants = (
        "train_only_no_post_eval",
        "language_no_blimp",
        "language_only",
        "binding_only_no_ar",
        "binding_plus_ar_gate",
        "current",
    )

    model_seed, baseline_probe = _find_passing_variant_seed(
        runner,
        base_config,
        fixture=fixture,
    )

    # Warm the common import/tokenization/native paths so the matrix reflects
    # steady-state scheduling decisions rather than first-use module cost.
    _time_stage1_variant(
        runner,
        _variant_config(base_config, "current"),
        model_seed=model_seed,
        seed=701,
    )

    rows: list[Dict[str, Any]] = []
    for index, variant in enumerate(variants):
        config = _variant_config(base_config, variant)
        run = _time_stage1_variant(
            runner,
            config,
            model_seed=model_seed,
            seed=901,
        )
        diagnostics = run["diagnostics"]
        probe_timings = diagnostics.get("probe_timings_ms") or {}
        rows.append(
            {
                "variant": variant,
                "elapsed_ms": run["elapsed_ms"],
                "compile_ms": run["compile_ms"],
                "model_seed": model_seed,
                "train_seed": 901,
                "passed": run["passed"],
                "flags": _variant_flags(config),
                "probe_timings_ms": probe_timings,
                "top_timing_sources": diagnostics.get("top_timing_sources") or [],
            }
        )

    baseline_ms = next(
        row["elapsed_ms"] for row in rows if row["variant"] == "train_only_no_post_eval"
    )
    current_ms = next(row["elapsed_ms"] for row in rows if row["variant"] == "current")
    for row in rows:
        row["post_eval_overhead_ms"] = max(0.0, row["elapsed_ms"] - baseline_ms)
        row["vs_current_saved_ms"] = max(0.0, current_ms - row["elapsed_ms"])

    return {
        "variants": rows,
        "baseline_variant": "train_only_no_post_eval",
        "baseline_probe_passed": bool(baseline_probe.get("passed", False)),
        "current_variant": "current",
        "notes": {
            "language_only": "WikiText + HellaSwag + BLiMP; binding/induction/AR disabled.",
            "binding_only_no_ar": "Induction + zero-shot binding + curriculum binding; language and AR disabled.",
            "binding_plus_ar_gate": "Binding-only schedule with AR gate enabled and legacy AR disabled.",
        },
    }


def _warm_stage1_pair(
    runner: ExperimentRunner,
    config: RunConfig,
    lean_config: RunConfig,
) -> Dict[str, Any]:
    base_model, base_compile_ms = _compile_scaffold(config)
    try:
        base_run = _time_micro_train(runner, base_model, config)
    finally:
        _close_warmup_model(base_model)

    lean_model, lean_compile_ms = _compile_scaffold(lean_config)
    try:
        lean_run = _time_micro_train(runner, lean_model, lean_config)
    finally:
        _close_warmup_model(lean_model)

    return {
        "base_compile_ms": base_compile_ms,
        "base_elapsed_ms": base_run["elapsed_ms"],
        "base_diagnostics": base_run["diagnostics"],
        "lean_compile_ms": lean_compile_ms,
        "lean_elapsed_ms": lean_run["elapsed_ms"],
        "lean_diagnostics": lean_run["diagnostics"],
    }


def benchmark_stage1_hotpath(*, fixture: str, repeats: int) -> Dict[str, Any]:
    config = _build_config(fixture=fixture)
    lean_config = _make_stage1_screening_config(config)
    runner = ExperimentRunner("research/runs.db")
    warmup = _warm_stage1_pair(runner, config, lean_config)
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
    base["diagnostics_runs"] = [row["diagnostics"] for row in base_runs]
    lean["diagnostics_runs"] = [row["diagnostics"] for row in lean_runs]
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
            "warmup": warmup,
            "warmup_compile_ms": warmup["base_compile_ms"],
        },
    }


def benchmark_orchestrator_hotpath(*, fixture: str) -> Dict[str, Any]:
    config = _build_config(fixture=fixture)
    lean_config = _make_stage1_screening_config(config)
    runner = ExperimentRunner("research/runs.db")
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
        "job_diagnostics": [
            {
                "index": row.index,
                "passed": bool(row.s1_result.get("passed")),
                "diagnostics": _extract_eval_diagnostics(row.s1_result),
            }
            for row in sorted(results, key=lambda item: item.index)
        ],
    }


def build_screening_hotpath_report(
    *,
    fixture: str = "quick",
    repeats: int | None = None,
    include_gating: bool = False,
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
    report = {
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
    if include_gating:
        gating = benchmark_gating_variants(fixture=fixture)
        report["gating"] = gating
        current = next(row for row in gating["variants"] if row["variant"] == "current")
        baseline = next(
            row
            for row in gating["variants"]
            if row["variant"] == gating["baseline_variant"]
        )
        report["metrics"]["gating_current_ms"] = round(current["elapsed_ms"], 4)
        report["metrics"]["gating_train_only_ms"] = round(baseline["elapsed_ms"], 4)
        report["metrics"]["gating_post_eval_overhead_ms"] = round(
            current["post_eval_overhead_ms"], 4
        )
    return report


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
    parser.add_argument(
        "--include-gating",
        action="store_true",
        help="Also run a controlled scheduling/gating variant matrix",
    )
    args = parser.parse_args()

    report = build_screening_hotpath_report(
        fixture=str(args.fixture),
        repeats=(int(args.repeats) if int(args.repeats) > 0 else None),
        include_gating=bool(args.include_gating),
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
