from __future__ import annotations

from types import SimpleNamespace

import torch

from research.eval.routing_ablations import rank_matched_budget_variants
from research.eval.sandbox import safe_eval
from research.synthesis.compiler import compile_model
from research.synthesis.compiler_ops_routing import (
    _op_calibrated_branch_merge,
    _op_hybrid_sparse_router,
)
from research.synthesis.graph import ComputationGraph


def _build_hybrid_sparse_router_graph() -> ComputationGraph:
    g = ComputationGraph(model_dim=32)
    inp = g.add_input()
    default_path = g.add_op("default_path", [inp], {})
    gate = g.add_op("hybrid_token_gate", [inp], {"threshold": 0.45})
    spans = g.add_op(
        "sparse_span_builder",
        [gate],
        {"span_width": 3, "fallback_behavior": "default_path"},
    )
    routed = g.add_op(
        "hybrid_sparse_router",
        [spans],
        {"span_width": 3, "lane_count": 3, "confidence_threshold": 0.45},
    )
    lane = g.add_op("lane_conditioned_block", [routed], {"lane_id": 1})
    merged = g.add_op("add", [default_path, lane], {})
    g.set_output(merged)
    return g


def test_hybrid_sparse_router_graph_safe_eval_emits_routing_metrics():
    graph = _build_hybrid_sparse_router_graph()
    model = compile_model([graph], vocab_size=64, max_seq_len=16)
    result = safe_eval(
        model,
        batch_size=2,
        seq_len=8,
        vocab_size=64,
        device="cpu",
        timeout_seconds=30,
    )
    assert result.passed, result.error
    assert result.routing_report is not None
    report = result.routing_report
    assert "routing_keep_drop_ratio" in report
    assert "sparse_span_coverage" in report
    assert "lane_utilization_histogram" in report
    assert "route_confidence_mean" in report
    assert "route_strength_mean" in report
    assert "dead_lane_count" in report


def test_rank_matched_budget_variants_prefers_quality_per_compute():
    ranked = rank_matched_budget_variants(
        [
            {"variant": "no_routing", "quality": 0.60, "compute": 1.0},
            {"variant": "single_token", "quality": 0.63, "compute": 1.02},
            {"variant": "dense_triplet", "quality": 0.67, "compute": 1.08},
            {"variant": "sparse_triplet", "quality": 0.69, "compute": 1.01},
        ],
        budget_tolerance=0.1,
    )
    assert ranked[0]["variant"] == "sparse_triplet"
    assert ranked[0]["matched_budget"] is True


def test_hybrid_sparse_router_does_not_write_future_span_signal_into_prefix():
    module = SimpleNamespace()
    config = {"span_width": 3, "lane_count": 3, "confidence_threshold": 0.4}

    base = torch.tensor([[[1.0], [2.0], [3.0], [4.0], [5.0]]], dtype=torch.float32)
    modified = base.clone()
    modified[:, 4, 0] = 50.0

    base_out = _op_hybrid_sparse_router(module, [base], config)
    modified_out = _op_hybrid_sparse_router(module, [modified], config)

    # Changing the final token may affect the final routed token, but must not
    # perturb earlier positions if span routing is causal.
    assert torch.allclose(base_out[:, :4], modified_out[:, :4])


def test_hybrid_sparse_router_does_not_route_zeroed_tokens():
    module = SimpleNamespace()
    config = {"span_width": 3, "lane_count": 3, "confidence_threshold": 0.4}
    x = torch.zeros((1, 8, 4), dtype=torch.float32)

    out = _op_hybrid_sparse_router(module, [x], config)

    assert torch.allclose(out, x)
    telemetry = getattr(module, "routing_telemetry", {})
    assert telemetry.get("keep_count", 0) == 0
    assert telemetry.get("sparse_span_count", 0) == 0


def test_hybrid_sparse_router_rescues_minimum_tokens_when_gate_is_overselective():
    module = SimpleNamespace()
    config = {
        "span_width": 3,
        "lane_count": 3,
        "confidence_threshold": 0.95,
        "min_keep_fraction": 0.25,
    }
    x = -torch.ones((1, 8, 4), dtype=torch.float32)

    out = _op_hybrid_sparse_router(module, [x], config)

    assert out.shape == x.shape
    telemetry = getattr(module, "routing_telemetry", {})
    assert telemetry.get("keep_count", 0) > 0
    assert telemetry.get("sparse_span_count", 0) > 0
    assert telemetry.get("routed_token_count", 0) > 0


def test_hybrid_sparse_router_curriculum_keeps_more_tokens_early_than_late():
    early = SimpleNamespace(_routing_progress=0.0)
    late = SimpleNamespace(_routing_progress=1.0)
    config = {
        "span_width": 3,
        "lane_count": 3,
        "confidence_threshold": 0.55,
        "min_keep_fraction": 0.125,
        "route_temperature": 0.85,
        "curriculum_enabled": True,
        "curriculum_warmup_frac": 0.25,
        "curriculum_mid_frac": 0.65,
        "confidence_threshold_start": 0.3,
        "confidence_threshold_mid": 0.44,
        "confidence_threshold_end": 0.55,
        "min_keep_fraction_start": 0.28,
        "min_keep_fraction_mid": 0.18,
        "min_keep_fraction_end": 0.125,
        "route_temperature_start": 1.35,
        "route_temperature_mid": 1.0,
        "route_temperature_end": 0.85,
    }
    x = -torch.ones((1, 8, 4), dtype=torch.float32)

    _op_hybrid_sparse_router(early, [x], config)
    _op_hybrid_sparse_router(late, [x], config)

    early_rt = getattr(early, "routing_telemetry", {})
    late_rt = getattr(late, "routing_telemetry", {})
    assert early_rt.get("keep_count", 0) >= late_rt.get("keep_count", 0)
    assert early_rt.get("routed_token_count", 0) >= late_rt.get("routed_token_count", 0)


def test_calibrated_branch_merge_protects_routed_share_and_emits_metrics():
    module = SimpleNamespace(_routing_progress=0.0)
    config = {
        "normalize_inputs": True,
        "primary_role": "routed",
        "secondary_role": "skip",
        "min_secondary_share": 0.08,
        "max_secondary_share": 0.22,
    }
    routed = torch.full((1, 4, 8), 0.5)
    skip = torch.full((1, 4, 8), 10.0)

    out = _op_calibrated_branch_merge(module, [routed, skip], config)

    assert out.shape == routed.shape
    telemetry = getattr(module, "routing_telemetry", {})
    assert telemetry.get("branch_weight_count", 0) > 0
    assert telemetry.get("routed_branch_share_sum", 0.0) > 0.0
    branch_weights = telemetry.get("branch_weight_sum")
    assert branch_weights is not None
    secondary_share = float(branch_weights[1].item())
    assert 0.08 <= secondary_share <= 0.22
