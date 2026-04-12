from __future__ import annotations

import json

from research.scientist.runner.failure_provenance import (
    infer_graph_failure_provenance,
)
from research.synthesis.graph import ComputationGraph
from research.synthesis.validator import validate_graph


def test_selective_scan_rejects_residual_add_predecessor_and_raw_projection_successor():
    g = ComputationGraph(64)
    inp = g.add_input()
    norm = g.add_op("rmsnorm", [inp])
    residual = g.add_op("add", [inp, norm])
    scan = g.add_op("selective_scan", [residual])
    proj = g.add_op("linear_proj", [scan], {"out_dim": 64})
    out = g.add_op("add", [inp, proj])
    g.set_output(out)

    result = validate_graph(g)

    assert not result.valid
    assert any("add" in err and "selective_scan" in err for err in result.errors)
    assert any(
        "selective_scan requires immediate norm/conv/silu predecessor context" in err
        for err in result.errors
    )


def test_selective_scan_rejects_ternary_projection_successor():
    g = ComputationGraph(64)
    inp = g.add_input()
    norm = g.add_op("rmsnorm", [inp])
    scan = g.add_op("selective_scan", [norm])
    ternary = g.add_op("ternary_projection", [scan])
    out = g.add_op("add", [inp, ternary])
    g.set_output(out)

    result = validate_graph(g)

    assert not result.valid
    assert any(
        "selective_scan" in err and "ternary_projection" in err for err in result.errors
    )


def test_adjacent_token_merge_rejects_selective_scan_successor():
    g = ComputationGraph(64)
    inp = g.add_input()
    norm = g.add_op("rmsnorm", [inp])
    merged = g.add_op("adjacent_token_merge", [norm], {"n_keep": 4})
    scan = g.add_op("selective_scan", [merged])
    out = g.add_op("add", [inp, scan])
    g.set_output(out)

    result = validate_graph(g)

    assert not result.valid
    assert any(
        "adjacent_token_merge" in err and "selective_scan" in err
        for err in result.errors
    )


def test_adjacent_token_merge_rejects_softmax_attention_successor():
    g = ComputationGraph(64)
    inp = g.add_input()
    norm = g.add_op("rmsnorm", [inp])
    merged = g.add_op("adjacent_token_merge", [norm], {"n_keep": 4})
    attn = g.add_op("softmax_attention", [merged])
    out = g.add_op("add", [inp, attn])
    g.set_output(out)

    result = validate_graph(g)

    assert not result.valid
    assert any(
        "adjacent_token_merge" in err and "softmax_attention" in err
        for err in result.errors
    )


def test_hybrid_token_gate_rejects_standalone_residual_use():
    g = ComputationGraph(64)
    inp = g.add_input()
    norm = g.add_op("rmsnorm", [inp])
    gate = g.add_op("hybrid_token_gate", [norm], {})
    out = g.add_op("add", [inp, gate])
    g.set_output(out)

    result = validate_graph(g)

    assert not result.valid
    assert any(
        "hybrid_token_gate must feed sparse_span_builder or hybrid_sparse_router" in err
        for err in result.errors
    )


def test_sparse_span_builder_requires_router_successor():
    g = ComputationGraph(64)
    inp = g.add_input()
    norm = g.add_op("rmsnorm", [inp])
    gate = g.add_op("hybrid_token_gate", [norm], {})
    spans = g.add_op("sparse_span_builder", [gate], {"span_width": 3})
    out = g.add_op("add", [inp, spans])
    g.set_output(out)

    result = validate_graph(g)

    assert not result.valid
    assert any(
        "sparse_span_builder" in err
        and ("hybrid_sparse_router successor" in err or "-> add" in err)
        for err in result.errors
    )


def test_hybrid_sparse_router_requires_full_causal_chain():
    g = ComputationGraph(64)
    inp = g.add_input()
    norm = g.add_op("rmsnorm", [inp])
    gate = g.add_op("hybrid_token_gate", [norm], {})
    spans = g.add_op("sparse_span_builder", [gate], {"span_width": 3})
    router = g.add_op("hybrid_sparse_router", [spans], {})
    lane = g.add_op("lane_conditioned_block", [router], {"lane_id": 1})
    out = g.add_op("add", [inp, lane])
    g.set_output(out)

    result = validate_graph(g)

    assert result.valid, result.errors


def test_hyp_distance_rejects_immediate_linear_proj_successor():
    g = ComputationGraph(64)
    inp = g.add_input()
    norm = g.add_op("rmsnorm", [inp])
    dist = g.add_op("hyp_distance", [norm, norm])
    proj = g.add_op("linear_proj", [dist], {"out_dim": 64})
    out = g.add_op("add", [inp, proj])
    g.set_output(out)

    result = validate_graph(g)

    assert not result.valid
    assert any("hyp_distance" in err and "linear_proj" in err for err in result.errors)


def test_moe_topk_rejects_linear_proj_up_predecessor():
    g = ComputationGraph(64)
    inp = g.add_input()
    up = g.add_op("linear_proj_up", [inp], {"out_dim": 64})
    moe = g.add_op("moe_topk", [up], {})
    out = g.add_op("add", [inp, moe])
    g.set_output(out)

    result = validate_graph(g)

    assert not result.valid
    assert any("linear_proj_up" in err and "moe_topk" in err for err in result.errors)


def test_relu_gated_moe_rejects_norm_predecessor():
    g = ComputationGraph(64)
    inp = g.add_input()
    norm = g.add_op("rmsnorm", [inp])
    moe = g.add_op("relu_gated_moe", [norm], {})
    out = g.add_op("add", [inp, moe])
    g.set_output(out)

    result = validate_graph(g)

    assert not result.valid
    assert any("rmsnorm" in err and "relu_gated_moe" in err for err in result.errors)


def test_swiglu_mlp_rejects_immediate_rmsnorm_successor():
    g = ComputationGraph(64)
    inp = g.add_input()
    mlp = g.add_op("swiglu_mlp", [inp], {})
    norm = g.add_op("rmsnorm", [mlp])
    out = g.add_op("add", [inp, norm])
    g.set_output(out)

    result = validate_graph(g)

    assert not result.valid
    assert any("swiglu_mlp" in err and "rmsnorm" in err for err in result.errors)


def test_linear_proj_down_rejects_neg_predecessor():
    g = ComputationGraph(64)
    inp = g.add_input()
    up = g.add_op("linear_proj_up", [inp], {"out_dim": 64})
    neg = g.add_op("neg", [up])
    down = g.add_op("linear_proj_down", [neg], {"out_dim": 64})
    out = g.add_op("add", [inp, down])
    g.set_output(out)

    result = validate_graph(g)

    assert not result.valid
    assert any("neg" in err and "linear_proj_down" in err for err in result.errors)


def test_validate_graph_includes_dim_flow_skip_only_errors():
    g = ComputationGraph(64)
    inp = g.add_input()
    g.set_output(inp)

    result = validate_graph(g)

    assert not result.valid
    assert any("skip-only" in err for err in result.errors)


def test_failure_provenance_identifies_stale_routing_bias_state():
    g = ComputationGraph(64)
    inp = g.add_input()
    norm = g.add_op("rmsnorm", [inp])
    router = g.add_op("compute_budget_router", [norm])
    moe = g.add_op("moe_2expert", [router])
    out = g.add_op("add", [inp, moe])
    g.set_output(out)

    provenance = infer_graph_failure_provenance(
        g,
        error_type="RuntimeError",
        error_message="The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 2",
    )

    assert provenance["failure_op"] == "moe_2expert"
    assert "stale_routing_bias_state" in provenance["failure_details_json"]


def test_failure_provenance_identifies_split_branch_restore_contract():
    g = ComputationGraph(256)
    inp = g.add_input()
    norm = g.add_op("rmsnorm", [inp])
    split_a = g.add_op("split2", [norm], {"part": 0})
    split_b = g.add_op("split2", [norm], {"part": 1})
    restored_a = g.add_op("linear_proj", [split_a], {"out_dim": 256})
    merged = g.add_op("concat", [restored_a, split_b])
    proj = g.add_op("linear_proj", [merged], {"out_dim": 256})
    out = g.add_op("add", [inp, proj])
    g.set_output(out)

    provenance = infer_graph_failure_provenance(
        g,
        error_type="RuntimeError",
        error_message="The size of tensor a (128) must match the size of tensor b (256) at non-singleton dimension 0",
    )

    assert provenance["failure_op"] == "split2"
    assert "split_branch_restore_contract" in provenance["failure_details_json"]


def test_failure_provenance_identifies_routing_telemetry_state_mismatch():
    g = ComputationGraph(256)
    inp = g.add_input()
    norm = g.add_op("rmsnorm", [inp])
    router = g.add_op("compute_budget_router", [norm])
    moe = g.add_op("moe_topk", [router], {"num_experts": 4, "top_k": 2})
    out = g.add_op("add", [inp, moe])
    g.set_output(out)

    provenance = infer_graph_failure_provenance(
        g,
        error_type="RuntimeError",
        error_message="The size of tensor a (3) must match the size of tensor b (4) at non-singleton dimension 0",
    )

    assert provenance["failure_op"] == "moe_topk"
    assert "routing_telemetry_state_mismatch" in provenance["failure_details_json"]


def test_failure_provenance_identifies_rapid_grad_explosion():
    g = ComputationGraph(256)
    inp = g.add_input()
    norm = g.add_op("rmsnorm", [inp])
    gate = g.add_op("hybrid_token_gate", [norm])
    router = g.add_op("hybrid_sparse_router", [gate])
    out = g.add_op("add", [inp, router])
    g.set_output(out)

    provenance = infer_graph_failure_provenance(
        g,
        error_type="rapid_screening_error",
        error_message="Grad norm 971.9 > 500.0 at step 124",
    )

    assert provenance["failure_op"] == "hybrid_token_gate"
    assert "grad_explosion" in provenance["failure_details_json"]


def test_failure_provenance_reports_upstream_source_op_not_terminal_victim():
    g = ComputationGraph(256)
    inp = g.add_input()
    norm = g.add_op("layernorm", [inp])
    identity = g.add_op("identity", [norm])
    routed = g.add_op("hybrid_sparse_router", [identity])
    out = g.add_op("add", [inp, routed])
    g.set_output(out)

    provenance = infer_graph_failure_provenance(
        g,
        error_type="s1_causality_violation",
        error_message="Strict Causality Gate Failed: Model looks ahead at future tokens.",
    )
    details = json.loads(provenance["failure_details_json"])

    assert details["error_type"] == "causality_violation"
    assert details["root_cause_code"] == "hybrid_routing_assembly"
    assert details["source_op"] == "hybrid_sparse_router"


def test_failure_provenance_identifies_rapid_no_learning_signal():
    g = ComputationGraph(256)
    inp = g.add_input()
    norm = g.add_op("rmsnorm", [inp])
    mlp = g.add_op("swiglu_mlp", [norm])
    out = g.add_op("add", [inp, mlp])
    g.set_output(out)

    provenance = infer_graph_failure_provenance(
        g,
        error_type="rapid_screening_error",
        error_message="No learning after 150 steps: init=10.627 final=10.454 (threshold=10.414, rate=0.020)",
    )

    assert provenance["failure_op"] == "swiglu_mlp"
    assert "rapid_no_learning_signal" in provenance["failure_details_json"]


def test_failure_provenance_identifies_generalization_failure():
    g = ComputationGraph(256)
    inp = g.add_input()
    norm = g.add_op("rmsnorm", [inp])
    mlp = g.add_op("swiglu_mlp", [norm])
    out = g.add_op("add", [inp, mlp])
    g.set_output(out)

    provenance = infer_graph_failure_provenance(
        g,
        error_type="insufficient_learning",
        error_message="Validation loss ratio 0.7136 > 0.60 — model memorized training but failed to generalize",
    )

    assert provenance["failure_op"] == "swiglu_mlp"
    assert "generalization_failure" in provenance["failure_details_json"]
