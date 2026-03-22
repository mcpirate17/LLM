"""
Performance Profiler for aria_designer workflows.

Provides FLOPs estimation, memory profiling, and latency benchmarking
at both the graph level (static analysis) and runtime level (actual execution).

Usage:
    from runtime.profiler import profile_workflow

    report = profile_workflow(workflow_json, model_dim=256, device="cpu")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List

import torch

from .bridge import workflow_to_graph

from research.defaults import MODEL_DIM, VOCAB_SIZE
from research.eval.perf_budget import evaluate_perf_budget_gate
from research.perf_contract import build_duplicate_work_report, build_perf_contract
from research.synthesis.primitives import PRIMITIVE_REGISTRY, safe_eval_formula


# ── Static Analysis ──────────────────────────────────────────────────


@dataclass(slots=True)
class OpProfile:
    """Per-op performance profile."""

    node_id: int
    op_name: str
    params: int = 0
    flops: int = 0
    memory_bytes: int = 0
    has_native_kernel: bool = False


@dataclass(slots=True)
class ProfileReport:
    """Complete performance profile for a workflow."""

    # Graph-level static analysis
    total_params: int = 0
    total_flops_per_token: int = 0
    total_memory_bytes: int = 0
    op_profiles: List[OpProfile] = field(default_factory=list)

    # Category breakdown
    flops_by_category: Dict[str, int] = field(default_factory=dict)
    params_by_category: Dict[str, int] = field(default_factory=dict)

    # Runtime profiling (if executed)
    compile_time_ms: float = 0.0
    forward_time_ms: float = 0.0
    backward_time_ms: float = 0.0
    peak_memory_mb: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    total_time_ms: float = 0.0

    # Per-op latency (if profiled)
    op_latencies_ms: Dict[str, float] = field(default_factory=dict)

    # Bottleneck analysis
    bottleneck_ops: List[str] = field(default_factory=list)
    native_coverage: float = 0.0  # fraction of ops with native C kernels
    avoided_duplicate_conversions: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert numpy types
        for k, v in d.items():
            if hasattr(v, "item"):
                d[k] = v.item()
        metrics = {
            "total_params": d.get("total_params", 0),
            "total_flops_per_token": d.get("total_flops_per_token", 0),
            "total_memory_bytes": d.get("total_memory_bytes", 0),
            "forward_time_ms": d.get("forward_time_ms", 0.0),
            "backward_time_ms": d.get("backward_time_ms", 0.0),
            "peak_memory_mb": d.get("peak_memory_mb", 0.0),
            "throughput_tokens_per_sec": d.get("throughput_tokens_per_sec", 0.0),
            "native_coverage": d.get("native_coverage", 0.0),
            "compile_time_ms": d.get("compile_time_ms", 0.0),
            "total_time_ms": d.get("total_time_ms", 0.0),
        }
        duplicate_work = build_duplicate_work_report(
            repeated_keys={},
            avoided_keys={
                "workflow_to_graph": int(d.get("avoided_duplicate_conversions", 0) or 0)
            },
        )
        contract = build_perf_contract(
            component="aria_designer",
            workload="workflow_profile_runtime"
            if metrics["forward_time_ms"]
            else "workflow_profile_static",
            identity={},
            metrics=metrics,
            budget_profile="designer_interactive",
            budget_verdict=evaluate_perf_budget_gate(
                {
                    "metrics": metrics,
                    "duplicate_work": duplicate_work,
                },
                budget_profile="designer_interactive",
            ),
            duplicate_work=duplicate_work,
            warnings=d.get("bottleneck_ops", []),
        )
        d["perf_contract"] = contract
        return d


def _estimate_op_flops(op_name: str, model_dim: int, config: Dict) -> int:
    """Estimate FLOPs for a single operation per token."""
    D = model_dim
    if op_name not in PRIMITIVE_REGISTRY:
        return D  # conservative default

    op = PRIMITIVE_REGISTRY[op_name]

    if op.shape_rule == "linear":
        out_dim = config.get("out_dim", D)
        return 2 * D * out_dim  # matmul: 2*D*out_dim per token

    elif op.shape_rule == "matmul":
        return 2 * D * D

    elif op.shape_rule == "identity":
        # Elementwise ops: 1-5 FLOPs per element
        if op_name in ("gelu", "silu"):
            return 8 * D  # approximation needs several ops
        elif op_name in ("exp", "log", "sqrt", "tanh", "sigmoid"):
            return 5 * D
        elif op_name in ("rmsnorm",):
            return 5 * D  # norm + scale
        return D

    elif op.shape_rule == "binary_broadcast":
        return D

    elif op.shape_rule in ("reduce_last", "reduce_seq"):
        return D

    elif op.shape_rule == "rfft":
        # FFT: O(D log D)
        import math

        return int(D * math.log2(max(D, 2))) * 5

    elif op.shape_rule == "irfft":
        import math

        return int(D * math.log2(max(D, 2))) * 5

    elif op.shape_rule == "concat":
        return 0  # no compute, just memory

    elif op.shape_rule == "split":
        return 0  # no compute, just view

    return D  # fallback


def _estimate_op_params(op_name: str, model_dim: int, config: Dict) -> int:
    """Estimate learnable parameters for a single operation."""
    D = model_dim
    if op_name not in PRIMITIVE_REGISTRY:
        return 0

    op = PRIMITIVE_REGISTRY[op_name]
    if not op.has_params:
        return 0

    formula = op.param_formula.replace("D", str(D))
    if "out_dim" in formula:
        out_dim = config.get("out_dim", D)
        formula = formula.replace("out_dim", str(out_dim))

    try:
        return int(safe_eval_formula(formula))
    except Exception:
        return D * D  # conservative fallback


def _estimate_op_memory(
    op_name: str, model_dim: int, config: Dict, batch_size: int, seq_len: int
) -> int:
    """Estimate memory in bytes for activations of a single op."""
    D = model_dim
    B = batch_size
    S = seq_len
    bytes_per_elem = 4  # float32

    if op_name not in PRIMITIVE_REGISTRY:
        return B * S * D * bytes_per_elem

    op = PRIMITIVE_REGISTRY[op_name]

    if op.shape_rule == "linear":
        out_dim = config.get("out_dim", D)
        return B * S * out_dim * bytes_per_elem
    elif op.shape_rule == "split":
        n = 2 if op_name == "split2" else 3
        return B * S * (D // n) * bytes_per_elem
    elif op.shape_rule == "concat":
        return B * S * D * 2 * bytes_per_elem  # rough: 2 inputs concatenated
    else:
        return B * S * D * bytes_per_elem


# Native kernel availability check
_NATIVE_KERNELS = {
    "relu",
    "gelu",
    "silu",
    "sin",
    "cos",
    "add",
    "mul",
    "matmul",
    "linear",
    "rmsnorm",
}


def _has_native_kernel(op_name: str) -> bool:
    """Check if a native C kernel exists for this op."""
    # Strip suffixes like _proj, _f32, etc.
    base = op_name.split("_")[0] if "_" in op_name else op_name
    return op_name in _NATIVE_KERNELS or base in _NATIVE_KERNELS


# ── Main Profiling Functions ─────────────────────────────────────────


def profile_static(
    workflow_json: Dict[str, Any],
    model_dim: int = MODEL_DIM,
    batch_size: int = 2,
    seq_len: int = 128,
) -> ProfileReport:
    """Static performance analysis (no GPU needed).

    Analyzes the workflow graph to estimate FLOPs, params, memory, and
    identify bottleneck operations.
    """
    graph = workflow_to_graph(workflow_json, model_dim=model_dim)
    return profile_static_graph(
        graph, model_dim=model_dim, batch_size=batch_size, seq_len=seq_len
    )


def profile_static_graph(
    graph: Any,
    model_dim: int = MODEL_DIM,
    batch_size: int = 2,
    seq_len: int = 128,
) -> ProfileReport:
    """Static performance analysis for an already materialized ComputationGraph."""
    report = ProfileReport()

    native_count = 0
    total_ops = 0

    for node in graph.nodes.values():
        if node.is_input:
            continue

        total_ops += 1
        op_name = node.op_name
        config = node.config

        params = _estimate_op_params(op_name, model_dim, config)
        flops = _estimate_op_flops(op_name, model_dim, config)
        memory = _estimate_op_memory(op_name, model_dim, config, batch_size, seq_len)
        has_native = _has_native_kernel(op_name)

        if has_native:
            native_count += 1

        op_prof = OpProfile(
            node_id=node.id,
            op_name=op_name,
            params=params,
            flops=flops,
            memory_bytes=memory,
            has_native_kernel=has_native,
        )
        report.op_profiles.append(op_prof)

        report.total_params += params
        report.total_flops_per_token += flops
        report.total_memory_bytes += memory

        # Category breakdown
        if op_name in PRIMITIVE_REGISTRY:
            cat = PRIMITIVE_REGISTRY[op_name].category
            cat_name = cat.value if hasattr(cat, "value") else str(cat)
            report.flops_by_category[cat_name] = (
                report.flops_by_category.get(cat_name, 0) + flops
            )
            report.params_by_category[cat_name] = (
                report.params_by_category.get(cat_name, 0) + params
            )

    report.native_coverage = native_count / max(total_ops, 1)

    # Identify bottleneck ops (top 3 by FLOPs)
    sorted_ops = sorted(report.op_profiles, key=lambda p: p.flops, reverse=True)
    report.bottleneck_ops = [
        f"{p.op_name} (node {p.node_id}, {p.flops} FLOPs)" for p in sorted_ops[:3]
    ]

    return report


def profile_runtime(
    workflow_json: Dict[str, Any],
    model_dim: int = MODEL_DIM,
    vocab_size: int = VOCAB_SIZE,
    device: str = "cpu",
    batch_size: int = 2,
    seq_len: int = 128,
    warmup_iters: int = 3,
    bench_iters: int = 10,
) -> ProfileReport:
    """Runtime profiling with actual execution.

    Compiles the workflow and benchmarks forward/backward passes.
    """
    total_started = time.perf_counter()
    graph = workflow_to_graph(workflow_json, model_dim=model_dim)

    # Start with static analysis
    report = profile_static_graph(graph, model_dim, batch_size, seq_len)
    report.avoided_duplicate_conversions = 1
    report.compile_time_ms = 0.0
    report.total_time_ms = 0.0

    # Compile
    try:
        from research.synthesis.compiler import compile_model

        compile_started = time.perf_counter()
        model = compile_model([graph], vocab_size=vocab_size)
        report.compile_time_ms = (time.perf_counter() - compile_started) * 1000.0
        model = model.to(device)
        model.train()
    except Exception as e:
        report.forward_time_ms = -1
        report.bottleneck_ops.append(f"Compilation failed: {e}")
        report.total_time_ms = (time.perf_counter() - total_started) * 1000.0
        return report

    # Prepare input
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Warmup
    for _ in range(warmup_iters):
        with torch.no_grad():
            model(x)

    # Benchmark forward
    if device == "cuda":
        torch.cuda.synchronize()

    fwd_times = []
    for _ in range(bench_iters):
        t0 = time.perf_counter()
        logits = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        fwd_times.append((time.perf_counter() - t0) * 1000)

    report.forward_time_ms = sum(fwd_times) / len(fwd_times)

    # Benchmark backward
    bwd_times = []
    for _ in range(bench_iters):
        logits = model(x)
        loss = logits.sum()
        t0 = time.perf_counter()
        loss.backward()
        if device == "cuda":
            torch.cuda.synchronize()
        bwd_times.append((time.perf_counter() - t0) * 1000)
        model.zero_grad()

    report.backward_time_ms = sum(bwd_times) / len(bwd_times)

    # Memory
    if device == "cuda":
        report.peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # Throughput
    total_tokens = batch_size * seq_len
    total_time_s = (report.forward_time_ms + report.backward_time_ms) / 1000
    if total_time_s > 0:
        report.throughput_tokens_per_sec = total_tokens / total_time_s
    report.total_time_ms = (time.perf_counter() - total_started) * 1000.0

    return report


def profile_workflow(
    workflow_json: Dict[str, Any],
    model_dim: int = MODEL_DIM,
    device: str = "cpu",
    runtime: bool = True,
    **kwargs,
) -> ProfileReport:
    """Convenience function: static + optional runtime profiling."""
    if runtime:
        return profile_runtime(
            workflow_json, model_dim=model_dim, device=device, **kwargs
        )
    else:
        static_kwargs = {
            k: v for k, v in kwargs.items() if k in ("batch_size", "seq_len")
        }
        return profile_static(workflow_json, model_dim=model_dim, **static_kwargs)
