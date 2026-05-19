"""Tests for table-driven compiler op dispatch (6a) and xxHash64 graph fingerprint (6b)."""

import torch
import torch.nn.functional as F

from research.synthesis.compiler import _OP_DISPATCH
from research.synthesis.compiler_ops_math import OP_IMPLS, _TABLE_OPS
from research.synthesis.graph import ComputationGraph


# ── 6a: Table-driven op dispatch ─────────────────────────────────────

# Every op name that must be present after the table-driven refactor
_EXPECTED_MATH_OPS = {
    "identity",
    "neg",
    "abs",
    "exp",
    "log",
    "sin",
    "cos",
    "tanh",
    "sigmoid",
    "relu",
    "gelu",
    "silu",
    "sqrt",
    "square",
    "sign_ste",
    "reciprocal",
    "add",
    "mul",
    "sub",
    "div_safe",
    "maximum",
    "minimum",
    "sum_last",
    "mean_last",
    "max_last",
    "norm_last",
    "cumsum",
    "cumprod_safe",
    "matmul",
    "outer_product",
    "transpose_sd",
    "split2",
    "split3",
    "split4",
    "concat",
    "roll_seq",
    "roll_neg",
    "multi_head_mix",
    "linear_proj",
    "linear_proj_down",
    "linear_proj_up",
    "fused_linear_gelu",
    "learnable_scale",
    "learnable_bias",
}

# Ops that should be generated from the dispatch tables
_TABLE_GENERATED_OPS = {
    "abs",
    "sin",
    "cos",
    "tanh",
    "sigmoid",
    "relu",
    "gelu",
    "silu",
    "maximum",
    "minimum",
    "sub",
}


def test_all_math_ops_registered():
    """All previously-defined op names are still in _OP_DISPATCH."""
    missing = _EXPECTED_MATH_OPS - set(_OP_DISPATCH.keys())
    assert not missing, f"Missing ops from _OP_DISPATCH: {missing}"


def test_all_math_ops_in_op_impls():
    """All expected ops are exported from compiler_ops_math.OP_IMPLS."""
    missing = _EXPECTED_MATH_OPS - set(OP_IMPLS.keys())
    assert not missing, f"Missing ops from OP_IMPLS: {missing}"


def test_table_generated_ops_present():
    """Table-driven ops are generated and present."""
    missing = _TABLE_GENERATED_OPS - set(_TABLE_OPS.keys())
    assert not missing, f"Missing table ops: {missing}"


def test_unary_table_ops_produce_correct_output():
    """Table-generated unary ops match torch reference."""
    # requires_grad=True bypasses _c() so we test the torch fallback path
    x = torch.randn(2, 4, 8, requires_grad=True)
    expected = {
        "abs": torch.abs(x),
        "sin": torch.sin(x),
        "cos": torch.cos(x),
        "tanh": torch.tanh(x),
        "sigmoid": torch.sigmoid(x),
        "relu": F.relu(x),
        "gelu": F.gelu(x),
        "silu": F.silu(x),
    }
    for name, ref in expected.items():
        op_fn = _OP_DISPATCH[name]
        result = op_fn(None, (x,), {})
        assert torch.allclose(result, ref, atol=1e-6), f"Op '{name}' output mismatch"


def test_binary_table_ops_produce_correct_output():
    """Table-generated binary ops match torch reference."""
    a = torch.randn(2, 4, 8, requires_grad=True)
    b = torch.randn(2, 4, 8, requires_grad=True)
    expected = {
        "maximum": torch.maximum(a, b),
        "minimum": torch.minimum(a, b),
        "sub": a - b,
    }
    for name, ref in expected.items():
        op_fn = _OP_DISPATCH[name]
        result = op_fn(None, (a, b), {})
        assert torch.allclose(result, ref, atol=1e-6), f"Op '{name}' output mismatch"


def test_custom_ops_still_work():
    """Non-table ops with custom logic still produce correct output."""
    x = torch.randn(2, 4, 8, requires_grad=True)
    # exp with clamp
    result = _OP_DISPATCH["exp"](None, (x,), {})
    assert torch.allclose(result, torch.exp(torch.clamp(x, -20, 20)))
    # neg
    result = _OP_DISPATCH["neg"](None, (x,), {})
    assert torch.allclose(result, -x)
    # square
    result = _OP_DISPATCH["square"](None, (x,), {})
    assert torch.allclose(result, x * x)


# ── 6b: xxHash64 graph fingerprint ──────────────────────────────────


def test_fingerprint_is_hex_string():
    """Fingerprint produces a stable hex string."""
    g = ComputationGraph(model_dim=64)
    inp = g.add_input()
    relu_id = g.add_op("relu", [inp])
    g.set_output(relu_id)
    fp = g.fingerprint()
    assert isinstance(fp, str)
    assert len(fp) == 16  # xxh64 produces 16 hex chars
    int(fp, 16)  # must be valid hex


def test_fingerprint_stable():
    """Same graph structure produces same fingerprint."""

    def _make():
        g = ComputationGraph(model_dim=64)
        inp = g.add_input()
        r = g.add_op("relu", [inp])
        g.set_output(r)
        return g.fingerprint()

    assert _make() == _make()


def test_fingerprint_differs_for_different_graphs():
    """Different graphs produce different fingerprints."""
    g1 = ComputationGraph(model_dim=64)
    inp1 = g1.add_input()
    g1.add_op("relu", [inp1])
    g1.set_output(g1._next_id - 1)

    g2 = ComputationGraph(model_dim=64)
    inp2 = g2.add_input()
    g2.add_op("gelu", [inp2])
    g2.set_output(g2._next_id - 1)

    assert g1.fingerprint() != g2.fingerprint()
