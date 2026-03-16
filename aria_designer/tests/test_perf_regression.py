"""
Performance regression tests.

These tests verify that profiling baselines don't regress:
- FLOPs estimates stay consistent
- Param count estimates are accurate
- Runtime latency doesn't blow up
- The bridge + profiler pipeline stays functional

These serve as a regression gate in CI.
"""

import time
import pytest

from aria_designer.runtime.bridge import validate_workflow_graph, evaluate_workflow, estimate_performance
from aria_designer.runtime.profiler import profile_static, profile_runtime


# ── Reference Workflows ──────────────────────────────────────────────

REFERENCE_MLP = {
    "nodes": [
        {"id": "n0", "component_type": "graph_input", "params": {}},
        {"id": "n1", "component_type": "linear_proj", "params": {"out_dim": 256}},
        {"id": "n2", "component_type": "gelu", "params": {}},
        {"id": "n3", "component_type": "linear_proj", "params": {"out_dim": 256}},
        {"id": "n4", "component_type": "graph_output", "params": {}},
    ],
    "edges": [
        {"id": "e0", "source": "n0", "target": "n1"},
        {"id": "e1", "source": "n1", "target": "n2"},
        {"id": "e2", "source": "n2", "target": "n3"},
        {"id": "e3", "source": "n3", "target": "n4"},
    ],
}

REFERENCE_ATTENTION = {
    "nodes": [
        {"id": "in", "component_type": "graph_input", "params": {}},
        {"id": "q", "component_type": "linear_proj", "params": {"out_dim": 256}},
        {"id": "k", "component_type": "linear_proj", "params": {"out_dim": 256}},
        {"id": "v", "component_type": "linear_proj", "params": {"out_dim": 256}},
        {"id": "attn", "component_type": "matmul", "params": {}},
        {"id": "sm", "component_type": "softmax_last", "params": {}},
        {"id": "av", "component_type": "matmul", "params": {}},
        {"id": "proj", "component_type": "linear_proj", "params": {"out_dim": 256}},
        {"id": "out", "component_type": "graph_output", "params": {}},
    ],
    "edges": [
        {"id": "e0", "source": "in", "target": "q"},
        {"id": "e1", "source": "in", "target": "k"},
        {"id": "e2", "source": "in", "target": "v"},
        {"id": "e3", "source": "q", "target": "attn"},
        {"id": "e4", "source": "k", "target": "attn"},
        {"id": "e5", "source": "attn", "target": "sm"},
        {"id": "e6", "source": "sm", "target": "av"},
        {"id": "e7", "source": "v", "target": "av"},
        {"id": "e8", "source": "av", "target": "proj"},
        {"id": "e9", "source": "proj", "target": "out"},
    ],
}

REFERENCE_RESIDUAL = {
    "nodes": [
        {"id": "in", "component_type": "graph_input", "params": {}},
        {"id": "l1", "component_type": "linear_proj", "params": {"out_dim": 256}},
        {"id": "act", "component_type": "gelu", "params": {}},
        {"id": "l2", "component_type": "linear_proj", "params": {"out_dim": 256}},
        {"id": "res", "component_type": "add", "params": {}},
        {"id": "out", "component_type": "graph_output", "params": {}},
    ],
    "edges": [
        {"id": "e0", "source": "in", "target": "l1"},
        {"id": "e1", "source": "l1", "target": "act"},
        {"id": "e2", "source": "act", "target": "l2"},
        {"id": "e3", "source": "l2", "target": "res"},
        {"id": "e4", "source": "in", "target": "res"},
        {"id": "e5", "source": "res", "target": "out"},
    ],
}


# ── Expected baselines (with generous tolerances for stability) ──────

# MLP: 2x linear_proj(256,256) = 2*256*256 = 131072 params
MLP_EXPECTED_PARAMS = 131072
MLP_EXPECTED_OPS = 3
MLP_EXPECTED_DEPTH = 3

# Attention: 4x linear_proj(256,256) + 2 matmul + 1 softmax = 7 ops
ATTN_EXPECTED_PARAMS = 262144  # 4 * 256*256
ATTN_EXPECTED_OPS = 7
ATTN_EXPECTED_DEPTH = 5

# Residual: 2x linear_proj + gelu + add = 4 ops
RES_EXPECTED_PARAMS = 131072
RES_EXPECTED_OPS = 4


# ── Param count regression ───────────────────────────────────────────

def test_mlp_param_count():
    result = validate_workflow_graph(REFERENCE_MLP, model_dim=256)
    assert result["valid"]
    # Param count should be exact (deterministic formula)
    assert result["graph_info"]["n_params_estimate"] == MLP_EXPECTED_PARAMS


def test_attention_param_count():
    result = validate_workflow_graph(REFERENCE_ATTENTION, model_dim=256)
    assert result["valid"]
    assert result["graph_info"]["n_params_estimate"] == ATTN_EXPECTED_PARAMS


def test_residual_param_count():
    result = validate_workflow_graph(REFERENCE_RESIDUAL, model_dim=256)
    assert result["valid"]
    assert result["graph_info"]["n_params_estimate"] == RES_EXPECTED_PARAMS


# ── Op count regression ──────────────────────────────────────────────

def test_mlp_op_count():
    result = validate_workflow_graph(REFERENCE_MLP, model_dim=256)
    assert result["graph_info"]["n_ops"] == MLP_EXPECTED_OPS


def test_attention_op_count():
    result = validate_workflow_graph(REFERENCE_ATTENTION, model_dim=256)
    assert result["graph_info"]["n_ops"] == ATTN_EXPECTED_OPS


def test_residual_op_count():
    result = validate_workflow_graph(REFERENCE_RESIDUAL, model_dim=256)
    assert result["graph_info"]["n_ops"] == RES_EXPECTED_OPS


# ── Depth regression ─────────────────────────────────────────────────

def test_mlp_depth():
    result = validate_workflow_graph(REFERENCE_MLP, model_dim=256)
    assert result["graph_info"]["depth"] == MLP_EXPECTED_DEPTH


def test_attention_depth():
    result = validate_workflow_graph(REFERENCE_ATTENTION, model_dim=256)
    assert result["graph_info"]["depth"] == ATTN_EXPECTED_DEPTH


# ── FLOPs estimation regression ──────────────────────────────────────

def test_mlp_flops_estimate():
    report = profile_static(REFERENCE_MLP, model_dim=256)
    # 2x linear: 2*256*256 = 131072 each, gelu: ~2048
    # Total should be in reasonable range
    assert report.total_flops_per_token > 200000
    assert report.total_flops_per_token < 500000


def test_attention_flops_estimate():
    report = profile_static(REFERENCE_ATTENTION, model_dim=256)
    # More compute-heavy: 4x linear + 2x matmul + softmax
    assert report.total_flops_per_token > 500000


def test_flops_monotonic():
    """Attention should use more FLOPs than simple MLP."""
    mlp = profile_static(REFERENCE_MLP, model_dim=256)
    attn = profile_static(REFERENCE_ATTENTION, model_dim=256)
    assert attn.total_flops_per_token > mlp.total_flops_per_token


# ── Native coverage regression ───────────────────────────────────────

def test_mlp_native_coverage():
    report = profile_static(REFERENCE_MLP, model_dim=256)
    # All ops in MLP should have native kernels
    assert report.native_coverage == 1.0


def test_attention_native_coverage():
    report = profile_static(REFERENCE_ATTENTION, model_dim=256)
    # Most ops should have native kernels
    assert report.native_coverage >= 0.5


# ── Runtime latency bounds ───────────────────────────────────────────

def test_mlp_forward_latency():
    """Forward pass should complete in reasonable time on CPU."""
    report = profile_runtime(
        REFERENCE_MLP,
        model_dim=256,
        device="cpu",
        warmup_iters=1,
        bench_iters=3,
        batch_size=1,
        seq_len=32,
    )
    # Should be under 500ms on CPU (very generous bound)
    assert report.forward_time_ms > 0
    assert report.forward_time_ms < 500


def test_mlp_backward_latency():
    """Backward pass should complete in reasonable time on CPU."""
    report = profile_runtime(
        REFERENCE_MLP,
        model_dim=256,
        device="cpu",
        warmup_iters=1,
        bench_iters=3,
        batch_size=1,
        seq_len=32,
    )
    assert report.backward_time_ms > 0
    assert report.backward_time_ms < 1000


# ── End-to-end evaluation regression ────────────────────────────────

def test_mlp_evaluation_succeeds():
    """Full evaluation pipeline should succeed for reference MLP."""
    result = evaluate_workflow(
        REFERENCE_MLP,
        model_dim=256,
        device="cpu",
        run_fingerprint=False,
        run_novelty=False,
        batch_size=1,
        seq_len=32,
    )
    assert result.status == "success"
    assert result.sandbox.passed
    assert result.param_count > 0


def test_evaluation_latency_bound():
    """Full evaluation should complete within time budget."""
    t0 = time.monotonic()
    evaluate_workflow(
        REFERENCE_MLP,
        model_dim=256,
        device="cpu",
        run_fingerprint=False,
        run_novelty=False,
        batch_size=1,
        seq_len=32,
    )
    elapsed = (time.monotonic() - t0) * 1000
    # Should be under 5 seconds on CPU
    assert elapsed < 5000


# ── Profiler consistency ─────────────────────────────────────────────

def test_profiler_idempotent():
    """Same workflow should produce same static profile."""
    r1 = profile_static(REFERENCE_MLP, model_dim=256)
    r2 = profile_static(REFERENCE_MLP, model_dim=256)
    assert r1.total_params == r2.total_params
    assert r1.total_flops_per_token == r2.total_flops_per_token
    assert r1.native_coverage == r2.native_coverage


def test_estimate_matches_profile():
    """Bridge estimate and profiler should agree on param counts."""
    est = estimate_performance(REFERENCE_MLP, model_dim=256)
    prof = profile_static(REFERENCE_MLP, model_dim=256)
    assert est["n_params_estimate"] == prof.total_params
