from __future__ import annotations

import numpy as np
import pytest

from research.synthesis.dim_flow_opcode_tables import (
    FULL_DIM_OPS as CANONICAL_FULL_DIM_OPS,
    KV_CACHE_BREAKING_OPS as CANONICAL_KV_CACHE_BREAKING_OPS,
)
from research.synthesis.dim_flow_support import (
    FULL_DIM_OPS,
    KV_CACHE_BREAKING_OPS,
    build_dim_flow_inputs,
    ensure_dim_flow_flags,
)
from research.synthesis.graph import ComputationGraph
from research.synthesis.native_analysis import (
    summarize_dim_flow,
    validate_edges,
    validate_packed_ir_batch_natively,
    validate_packed_ir_natively,
)
from research.synthesis.validation_opcode_tables import validation_opcode_tables
from research.synthesis.validator import compute_effective_depth
from research.synthesis.native_dim_flow_flags import build_dim_flow_flags_natively


def test_dim_flow_support_keeps_legacy_opcode_exports():
    assert FULL_DIM_OPS is CANONICAL_FULL_DIM_OPS
    assert KV_CACHE_BREAKING_OPS is CANONICAL_KV_CACHE_BREAKING_OPS


def test_build_dim_flow_flags_natively_uses_runtime(monkeypatch):
    calls = {}

    class FakeLib:
        def aria_graph_build_dim_flow_flags(
            self,
            n_nodes,
            op_codes_ptr,
            param_estimates_ptr,
            opcode_has_params_ptr,
            opcode_nontrivial_ptr,
            opcode_kv_breaking_ptr,
            opcode_kind_ptr,
            opcode_full_dim_ptr,
            has_params_ptr,
            nontrivial_ptr,
            kv_breaking_ptr,
            op_kind_ptr,
            full_dim_ptr,
        ):
            calls["n_nodes"] = n_nodes
            np.ctypeslib.as_array(has_params_ptr, shape=(n_nodes,))[:] = [0, 1]
            np.ctypeslib.as_array(nontrivial_ptr, shape=(n_nodes,))[:] = [0, 1]
            np.ctypeslib.as_array(kv_breaking_ptr, shape=(n_nodes,))[:] = [0, 0]
            np.ctypeslib.as_array(op_kind_ptr, shape=(n_nodes,))[:] = [0, 3]
            np.ctypeslib.as_array(full_dim_ptr, shape=(n_nodes,))[:] = [0, 1]
            return 0

    monkeypatch.setattr(
        "research.synthesis.native_dim_flow_flags.load_native_graph_analysis_lib",
        lambda: FakeLib(),
    )

    result = build_dim_flow_flags_natively(
        op_codes=np.array([0, 7], dtype=np.int32),
        param_estimates=np.array([0, 64], dtype=np.int64),
        opcode_has_params=np.array([0, 1, 1, 1, 1, 1, 1, 1], dtype=np.int32),
        opcode_nontrivial=np.array([0, 1, 1, 1, 1, 1, 1, 1], dtype=np.int32),
        opcode_kv_breaking=np.zeros(8, dtype=np.int32),
        opcode_kind=np.zeros(8, dtype=np.int32),
        opcode_full_dim=np.zeros(8, dtype=np.int32),
    )

    assert calls["n_nodes"] == 2
    assert result is not None
    assert result["has_params_flags"].tolist() == [0, 1]
    assert result["op_kind_flags"].tolist() == [0, 3]


def test_build_dim_flow_inputs_reuses_ir_param_estimates(monkeypatch):
    graph = ComputationGraph(8)
    inp = graph.add_input()
    proj = graph.add_op("linear_proj", [inp], config={"out_dim": 8})
    graph.set_output(proj)

    analysis_ir = graph._analysis_ir()
    analysis_ir.param_estimates = np.array([0, 123], dtype=np.int64)
    monkeypatch.setattr(graph, "_analysis_ir", lambda: analysis_ir)
    monkeypatch.setattr(
        "research.synthesis.dim_flow_support.build_dim_flow_flags_natively",
        lambda **kwargs: None,
        raising=False,
    )

    result = build_dim_flow_inputs(
        graph,
        op_kind_default=0,
        op_kind_irfft=1,
        op_kind_identity=2,
        op_kind_binary_broadcast=3,
    )

    assert result.param_estimates.tolist() == [0, 123]
    assert result.has_params_flags.tolist() == [0, 1]


def test_build_dim_flow_inputs_can_defer_flags():
    graph = ComputationGraph(8)
    inp = graph.add_input()
    proj = graph.add_op("linear_proj", [inp], config={"out_dim": 8})
    graph.set_output(proj)

    deferred = build_dim_flow_inputs(
        graph,
        op_kind_default=0,
        op_kind_irfft=1,
        op_kind_identity=2,
        op_kind_binary_broadcast=3,
        build_flags=False,
    )
    eager = build_dim_flow_inputs(
        graph,
        op_kind_default=0,
        op_kind_irfft=1,
        op_kind_identity=2,
        op_kind_binary_broadcast=3,
    )

    assert not deferred.flags_ready
    assert deferred.has_params_flags.size == 0

    ensure_dim_flow_flags(
        deferred,
        op_kind_default=0,
        op_kind_irfft=1,
        op_kind_identity=2,
        op_kind_binary_broadcast=3,
    )

    assert deferred.flags_ready
    assert deferred.has_params_flags.tolist() == eager.has_params_flags.tolist()
    assert deferred.op_kind_flags.tolist() == eager.op_kind_flags.tolist()


def test_validate_packed_ir_native_matches_split_reference():
    graph = ComputationGraph(8)
    inp = graph.add_input()
    proj = graph.add_op("linear_proj", [inp], config={"out_dim": 8})
    norm = graph.add_op("rmsnorm", [proj])
    graph.set_output(norm)

    dim_inputs = build_dim_flow_inputs(
        graph,
        op_kind_default=0,
        op_kind_irfft=1,
        op_kind_identity=2,
        op_kind_binary_broadcast=3,
    )
    tables = validation_opcode_tables()
    analysis = dim_inputs.analysis
    reachable_mask = np.asarray(analysis.reachable_mask).astype(np.int32, copy=False)
    summary_mask = reachable_mask.copy()
    summary_mask[dim_inputs.node_id_to_analysis_idx[inp]] = 0

    packed = validate_packed_ir_natively(
        op_codes=dim_inputs.analysis_ir.op_codes,
        input_indices=dim_inputs.analysis_ir.input_indices,
        output_node_idx=int(dim_inputs.analysis_ir.output_node_idx),
        param_estimates=dim_inputs.param_estimates,
        has_params_flags=dim_inputs.has_params_flags,
        nontrivial_flags=dim_inputs.nontrivial_flags,
        kv_breaking_flags=dim_inputs.kv_breaking_flags,
        node_dims=dim_inputs.node_dims,
        node_seq_flags=dim_inputs.node_seq_flags,
        op_kind_flags=dim_inputs.op_kind_flags,
        full_dim_flags=dim_inputs.full_dim_flags,
        model_dim=graph.model_dim,
        input_node_idx=dim_inputs.node_id_to_analysis_idx[inp],
        effective_depth_weights=tables.effective_depth_weight,
        discount_successor_u8=tables.discount_successor_u8,
    )
    if packed is None:
        pytest.skip("packed graph-validation runtime unavailable")

    split_summary = summarize_dim_flow(
        reachable_mask=summary_mask,
        has_params_flags=dim_inputs.has_params_flags,
        param_estimates=dim_inputs.param_estimates,
        nontrivial_flags=dim_inputs.nontrivial_flags,
        kv_breaking_flags=dim_inputs.kv_breaking_flags,
    )
    split_edges = validate_edges(
        reachable_mask=reachable_mask,
        input_indices=dim_inputs.analysis_ir.input_indices,
        node_dims=dim_inputs.node_dims,
        node_seq_flags=dim_inputs.node_seq_flags,
        op_kind_flags=dim_inputs.op_kind_flags,
        full_dim_flags=dim_inputs.full_dim_flags,
        model_dim=graph.model_dim,
    )

    assert packed.dim_flow.reachable_param_count == split_summary.reachable_param_count
    assert (
        packed.dim_flow.reachable_param_estimate
        == split_summary.reachable_param_estimate
    )
    assert (
        packed.dim_flow.reachable_nontrivial_ops
        == split_summary.reachable_nontrivial_ops
    )
    assert packed.dim_flow.reachable_ops == split_summary.reachable_ops
    assert packed.edge_validation.freq_mismatch_bits.tolist() == (
        split_edges.freq_mismatch_bits.tolist()
    )
    assert packed.edge_validation.full_dim_input_bits.tolist() == (
        split_edges.full_dim_input_bits.tolist()
    )
    assert packed.dead_parameterized_mask.tolist() == [0, 0, 0]
    assert packed.effective_depth == pytest.approx(
        compute_effective_depth(dim_inputs.analysis_ir)
    )


def _build_packed_ir_test_graphs():
    graphs = []
    for out_dim in (8, 4, 8):
        graph = ComputationGraph(8)
        inp = graph.add_input()
        proj = graph.add_op("linear_proj", [inp], config={"out_dim": out_dim})
        if out_dim != 8:
            proj = graph.add_op("linear_proj", [proj], config={"out_dim": 8})
        norm = graph.add_op("rmsnorm", [proj])
        graph.set_output(norm)
        graphs.append(graph)
    inputs = [
        build_dim_flow_inputs(
            graph,
            op_kind_default=0,
            op_kind_irfft=1,
            op_kind_identity=2,
            op_kind_binary_broadcast=3,
        )
        for graph in graphs
    ]
    return graphs, inputs


def _validate_single_packed(graph, item, tables):
    return validate_packed_ir_natively(
        op_codes=item.analysis_ir.op_codes,
        input_indices=item.analysis_ir.input_indices,
        output_node_idx=int(item.analysis_ir.output_node_idx),
        param_estimates=item.param_estimates,
        has_params_flags=item.has_params_flags,
        nontrivial_flags=item.nontrivial_flags,
        kv_breaking_flags=item.kv_breaking_flags,
        node_dims=item.node_dims,
        node_seq_flags=item.node_seq_flags,
        op_kind_flags=item.op_kind_flags,
        full_dim_flags=item.full_dim_flags,
        model_dim=graph.model_dim,
        input_node_idx=item.node_id_to_analysis_idx[graph._input_node_id],
        effective_depth_weights=tables.effective_depth_weight,
        discount_successor_u8=tables.discount_successor_u8,
    )


def _validate_batch_packed(graphs, inputs, tables):
    return validate_packed_ir_batch_natively(
        op_codes=[item.analysis_ir.op_codes for item in inputs],
        input_indices=[item.analysis_ir.input_indices for item in inputs],
        output_node_indices=[int(item.analysis_ir.output_node_idx) for item in inputs],
        param_estimates=[item.param_estimates for item in inputs],
        has_params_flags=[item.has_params_flags for item in inputs],
        nontrivial_flags=[item.nontrivial_flags for item in inputs],
        kv_breaking_flags=[item.kv_breaking_flags for item in inputs],
        node_dims=[item.node_dims for item in inputs],
        node_seq_flags=[item.node_seq_flags for item in inputs],
        op_kind_flags=[item.op_kind_flags for item in inputs],
        full_dim_flags=[item.full_dim_flags for item in inputs],
        model_dims=[graph.model_dim for graph in graphs],
        input_node_indices=[
            item.node_id_to_analysis_idx[graph._input_node_id]
            for graph, item in zip(graphs, inputs)
        ],
        effective_depth_weights=tables.effective_depth_weight,
        discount_successor_u8=tables.discount_successor_u8,
    )


def _assert_packed_results_match(batched, single):
    assert batched.analysis.depth == single.analysis.depth
    assert batched.analysis.reachable_count == single.analysis.reachable_count
    assert batched.analysis.param_estimate == single.analysis.param_estimate
    assert batched.dim_flow.reachable_param_count == (
        single.dim_flow.reachable_param_count
    )
    assert batched.edge_error_count == single.edge_error_count
    assert batched.dead_parameterized_count == single.dead_parameterized_count
    assert batched.reachable_mask.tolist() == single.reachable_mask.tolist()
    assert batched.dead_parameterized_mask.tolist() == (
        single.dead_parameterized_mask.tolist()
    )
    assert batched.edge_validation.freq_mismatch_bits.tolist() == (
        single.edge_validation.freq_mismatch_bits.tolist()
    )
    assert batched.effective_depth == pytest.approx(single.effective_depth)


def test_validate_packed_ir_batch_native_matches_single_graph_results():
    graphs, inputs = _build_packed_ir_test_graphs()
    tables = validation_opcode_tables()
    singles = [
        _validate_single_packed(graph, item, tables)
        for graph, item in zip(graphs, inputs)
    ]
    if any(result is None for result in singles):
        pytest.skip("packed graph-validation runtime unavailable")
    batch = _validate_batch_packed(graphs, inputs, tables)
    if batch is None:
        pytest.skip("packed graph-validation batch runtime unavailable")
    assert len(batch) == len(singles)
    for batched, single in zip(batch, singles):
        _assert_packed_results_match(batched, single)
