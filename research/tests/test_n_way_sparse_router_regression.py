from __future__ import annotations

import pytest

from research.eval.sandbox import safe_eval
from research.synthesis.compiler import compile_model
from research.synthesis.graph import ComputationGraph


pytestmark = pytest.mark.unit


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="CUDA required")
def test_n_way_sparse_router_safe_eval_survives_autocast_dtype_mix():
    g = ComputationGraph(256)
    inp = g.add_input()
    ln = g.add_op("rmsnorm", [inp])
    left = g.add_op("linear_proj", [ln], {"out_dim": 256})
    right = g.add_op("linear_proj", [ln], {"out_dim": 256})
    gate = g.add_op("sigmoid", [right])
    mixed = g.add_op("outer_product", [left, gate])
    res = g.add_op("add", [inp, mixed])
    norm = g.add_op("layernorm", [res])
    ternary = g.add_op("ternary_projection", [norm], {"out_dim": 256})
    trig = g.add_op("cos", [ternary])
    router = g.add_op("n_way_sparse_router", [trig], {"n_ways": 4, "top_k": 2})
    routed = g.add_op("rmsnorm", [router])
    out = g.add_op("add", [res, routed])
    g.set_output(out)

    model = compile_model([g] * 4, vocab_size=32000, max_seq_len=128)
    result = safe_eval(
        model,
        batch_size=2,
        seq_len=128,
        vocab_size=32000,
        device="cuda",
        timeout_seconds=30,
        run_stability_probe=True,
    )

    assert "Expected self.dtype to be equal to src.dtype" not in str(result.error)
