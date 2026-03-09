"""Tests for Rust scheduler dispatch of 6 reference architecture ops.

Each test builds a graph with a single input node, calls
aria_scheduler.execute_graph(), and verifies output shape and finite values.

Since execute_graph() feeds the same data to all input nodes, ops that need
multiple distinct inputs (e.g. embedding_lookup) reference the single input
node multiple times via input_ids. The test verifies dispatch correctness
rather than mathematical accuracy.
"""

import json
import struct
import pytest
import numpy as np

try:
    import aria_scheduler  # type: ignore[import-untyped]
    HAS_RUST_SCHEDULER = True
except ImportError:
    try:
        from scientist import aria_scheduler  # type: ignore[import-untyped]
        HAS_RUST_SCHEDULER = True
    except ImportError:
        HAS_RUST_SCHEDULER = False

pytestmark = [pytest.mark.native, pytest.mark.skipif(
    not HAS_RUST_SCHEDULER,
    reason="aria_scheduler Rust module not available",
)]


def _make_graph_json(nodes, edges, output_node_id):
    """Helper to build a native IR JSON string."""
    return json.dumps({
        "schema_version": "0.1",
        "model_dim": 4,
        "nodes": nodes,
        "edges": edges,
        "output_node_id": output_node_id,
        "metadata": None,
    })


class TestRopeRotate:
    """rope_rotate: single input, single output — simplest case."""

    def test_dispatch(self):
        batch, seq, dim = 1, 2, 4
        x = [float(i) * 0.1 for i in range(batch * seq * dim)]

        nodes = [
            {"id": 0, "op_name": "input", "input_ids": [], "config": {},
             "is_input": True, "is_output": False},
            {"id": 1, "op_name": "rope_rotate", "input_ids": [0],
             "config": {"batch": batch, "seq": seq, "dim": dim, "theta_base": 10000.0},
             "is_input": False, "is_output": False},
            {"id": 2, "op_name": "output", "input_ids": [1], "config": {},
             "is_input": False, "is_output": True},
        ]
        edges = [
            {"source": 0, "target": 1, "source_port": None, "target_port": None},
            {"source": 1, "target": 2, "source_port": None, "target_port": None},
        ]
        graph_json = _make_graph_json(nodes, edges, 2)
        result = aria_scheduler.execute_graph(graph_json, x)
        assert len(result) == batch * seq * dim
        assert all(np.isfinite(result))


class TestCosineSimilarity:
    """cosine_similarity: two inputs (same data) → output is all 1.0."""

    def test_dispatch(self):
        batch, seq, dim = 1, 2, 4
        # Use non-zero data so cosine_sim is well-defined
        x = [float(i) * 0.1 + 0.1 for i in range(batch * seq * dim)]

        nodes = [
            {"id": 0, "op_name": "input", "input_ids": [], "config": {},
             "is_input": True, "is_output": False},
            {"id": 1, "op_name": "cosine_similarity", "input_ids": [0, 0],
             "config": {"batch": batch, "seq": seq, "dim": dim},
             "is_input": False, "is_output": False},
            {"id": 2, "op_name": "output", "input_ids": [1], "config": {},
             "is_input": False, "is_output": True},
        ]
        edges = [
            {"source": 0, "target": 1, "source_port": None, "target_port": None},
            {"source": 1, "target": 2, "source_port": None, "target_port": None},
        ]
        graph_json = _make_graph_json(nodes, edges, 2)
        result = aria_scheduler.execute_graph(graph_json, x)
        assert len(result) == batch * seq
        # cos(x, x) == 1.0
        np.testing.assert_allclose(result, [1.0] * (batch * seq), atol=1e-5)


class TestEmbeddingLookup:
    """embedding_lookup: table + indices (reinterpreted from f32)."""

    def test_dispatch(self):
        batch, dim, vocab_size = 2, 4, 8
        # Build table as input data: [vocab_size * dim] floats
        # The indices will also come from the same input (reinterpreted as i32),
        # but the first batch*1 i32 values happen to be whatever bits the first
        # few f32 values have. Instead, use a graph where input_ids=[0,0]
        # so both table and indices point to the same buffer.
        # vocab_size*dim = 32 elements. The first 2 elements (batch=2) as i32
        # are the bit patterns of the first 2 floats.
        # This will work as long as index < vocab_size (may be 0 for 0.0).
        table = [0.0] * (vocab_size * dim)
        # Fill with values — index 0 will be used since 0.0 as i32 = 0
        for i in range(vocab_size * dim):
            table[i] = float(i) * 0.1

        nodes = [
            {"id": 0, "op_name": "input", "input_ids": [], "config": {},
             "is_input": True, "is_output": False},
            {"id": 1, "op_name": "embedding_lookup", "input_ids": [0, 0],
             "config": {"batch": batch, "dim": dim, "vocab_size": vocab_size},
             "is_input": False, "is_output": False},
            {"id": 2, "op_name": "output", "input_ids": [1], "config": {},
             "is_input": False, "is_output": True},
        ]
        edges = [
            {"source": 0, "target": 1, "source_port": None, "target_port": None},
            {"source": 1, "target": 2, "source_port": None, "target_port": None},
        ]
        graph_json = _make_graph_json(nodes, edges, 2)
        result = aria_scheduler.execute_graph(graph_json, table)
        assert len(result) == batch * dim
        assert all(np.isfinite(result))


class TestGatedLinear:
    """gated_linear: x, W, b, W_gate, b_gate — all from same input."""

    def test_dispatch(self):
        # Use small dims so the shared input buffer covers all needed data
        batch, dim_in, dim_out = 1, 4, 4
        # Input size = batch*dim_in = 4. This is reused for all 5 input slots.
        # W needs dim_out*dim_in=16 but we only have 4 elements, which is fine
        # for dispatch testing (C kernel reads from valid memory).
        # Use a larger input to be safe.
        n = max(batch * dim_in, dim_out * dim_in, dim_out)
        x = [float(i) * 0.01 for i in range(n)]

        nodes = [
            {"id": 0, "op_name": "input", "input_ids": [], "config": {},
             "is_input": True, "is_output": False},
            {"id": 1, "op_name": "gated_linear", "input_ids": [0, 0, 0, 0, 0],
             "config": {"batch": batch, "dim_in": dim_in, "dim_out": dim_out},
             "is_input": False, "is_output": False},
            {"id": 2, "op_name": "output", "input_ids": [1], "config": {},
             "is_input": False, "is_output": True},
        ]
        edges = [
            {"source": 0, "target": 1, "source_port": None, "target_port": None},
            {"source": 1, "target": 2, "source_port": None, "target_port": None},
        ]
        graph_json = _make_graph_json(nodes, edges, 2)
        result = aria_scheduler.execute_graph(graph_json, x)
        assert len(result) == batch * dim_out
        assert all(np.isfinite(result))


class TestGatherTopk:
    """gather_topk: scores + values from same input."""

    def test_dispatch(self):
        batch, n_items, dim, k = 1, 4, 3, 2
        # Need at least n_items (scores) + n_items*dim (values) = 4+12 = 16
        n = batch * n_items * dim + batch * n_items
        x = [float(i) * 0.1 for i in range(n)]

        nodes = [
            {"id": 0, "op_name": "input", "input_ids": [], "config": {},
             "is_input": True, "is_output": False},
            {"id": 1, "op_name": "gather_topk", "input_ids": [0, 0],
             "config": {"batch": batch, "n_items": n_items, "dim": dim, "k": k},
             "is_input": False, "is_output": False},
            {"id": 2, "op_name": "output", "input_ids": [1], "config": {},
             "is_input": False, "is_output": True},
        ]
        edges = [
            {"source": 0, "target": 1, "source_port": None, "target_port": None},
            {"source": 1, "target": 2, "source_port": None, "target_port": None},
        ]
        graph_json = _make_graph_json(nodes, edges, 2)
        result = aria_scheduler.execute_graph(graph_json, x)
        assert len(result) == batch * k * dim
        assert all(np.isfinite(result))


class TestRwkvTimeMixing:
    """rwkv_time_mixing: x, w_decay, u_bonus, W_k, W_v, W_r from same input."""

    def test_dispatch(self):
        batch, seq, dim = 1, 2, 4
        # Largest buffer needed: dim*dim = 16 for W_k/W_v/W_r
        n = max(batch * seq * dim, dim * dim, dim)
        x = [float(i) * 0.01 for i in range(n)]

        nodes = [
            {"id": 0, "op_name": "input", "input_ids": [], "config": {},
             "is_input": True, "is_output": False},
            {"id": 1, "op_name": "rwkv_time_mixing", "input_ids": [0, 0, 0, 0, 0, 0],
             "config": {"batch": batch, "seq": seq, "dim": dim},
             "is_input": False, "is_output": False},
            {"id": 2, "op_name": "output", "input_ids": [1], "config": {},
             "is_input": False, "is_output": True},
        ]
        edges = [
            {"source": 0, "target": 1, "source_port": None, "target_port": None},
            {"source": 1, "target": 2, "source_port": None, "target_port": None},
        ]
        graph_json = _make_graph_json(nodes, edges, 2)
        result = aria_scheduler.execute_graph(graph_json, x)
        assert len(result) == batch * seq * dim
        assert all(np.isfinite(result))
