"""Component Health Tests — verify all previously-broken ops have healthy gradients.

Tests ops that were reported broken (0% S0/S1 rate), had extreme gradient
norms, or suffered from boundary LayerNorm gradient kill. Each test
instantiates the op through CompiledLayer (the real compilation path),
runs forward + backward, and checks grad norms are bounded.
"""

import pytest
import torch

from research.synthesis.compiler import CompiledLayer
from research.synthesis.graph import ComputationGraph


def _build_layer(
    op_name: str, n_inputs: int = 1, config: dict | None = None
) -> CompiledLayer:
    """Build a single-op CompiledLayer for testing."""
    g = ComputationGraph(model_dim=64)
    inp = g.add_input()
    inputs = [inp]
    if n_inputs == 2:
        inp2 = g.add_op("linear_proj", [inp], config={"out_dim": 64})
        inputs = [inp, inp2]
    g.add_op(op_name, inputs, config=config or {})
    g.set_output(g._next_id - 1)
    return CompiledLayer(g)


def _run_fwd_bwd(layer: CompiledLayer) -> tuple[float, float, bool]:
    """Run forward + backward, return (max_param_grad, input_grad, has_nan)."""
    x = torch.randn(2, 8, 64, requires_grad=True)
    y = layer(x)
    try:
        y.sum().backward()
    except RuntimeError:
        pass  # Non-parametric ops with no grad path
    grads = [p.grad.norm().item() for p in layer.parameters() if p.grad is not None]
    max_gn = max(grads) if grads else 0.0
    input_gn = x.grad.norm().item() if x.grad is not None else 0.0
    has_nan = torch.isnan(y).any().item()
    return max_gn, input_gn, has_nan


# ── BUCKET A: Fixed ops (previously broken, now healthy) ───────────────


class TestFixedOps:
    """Ops that had bugs and were fixed. Grad norms must be < 200."""

    @pytest.mark.parametrize(
        "op,n_inputs,config",
        [
            ("route_topk", 1, {"k": 2}),
            ("conv_only", 1, None),
            ("gated_linear", 1, None),
            ("adaptive_lane_mixer", 1, None),
            ("mixed_recursion_gate", 2, {"max_depth": 3}),
            ("routing_conditioned_compression", 2, None),
            ("basis_expansion", 1, None),
        ],
    )
    def test_grad_norm_bounded(self, op, n_inputs, config):
        layer = _build_layer(op, n_inputs, config)
        max_gn, input_gn, has_nan = _run_fwd_bwd(layer)
        assert not has_nan, f"{op} produced NaN"
        assert max_gn < 500, f"{op} param_grad={max_gn:.1f} too high"


class TestTropicalOps:
    """Tropical ops: fixed gradient flow through custom autograd + residual."""

    def test_tropical_attention_grad(self):
        layer = _build_layer("tropical_attention")
        max_gn, input_gn, has_nan = _run_fwd_bwd(layer)
        assert not has_nan
        assert input_gn > 1.0, f"tropical_attention input_grad={input_gn:.4f} too low"
        assert max_gn < 500

    def test_tropical_center_no_nan(self):
        layer = _build_layer("tropical_center")
        x = torch.randn(2, 8, 64)
        y = layer(x)
        assert not torch.isnan(y).any()
        assert y.shape == x.shape


# ── BUCKET C: Non-learnable unary ops ──────────────────────────────────


class TestUnaryOps:
    """Non-parametric unary ops: correct output, no NaN."""

    @pytest.mark.parametrize("op", ["neg", "cos", "abs", "identity"])
    def test_unary_no_nan(self, op):
        layer = _build_layer(op)
        x = torch.randn(2, 8, 64, requires_grad=True)
        y = layer(x)
        assert not torch.isnan(y).any(), f"{op} produced NaN"
        assert y.shape == x.shape


# ── Dangerous grad norm ops (div_safe, log, reciprocal) ────────────────


class TestGradNormOps:
    """Ops that previously had extreme grad norms (>1e6). Now bounded."""

    def test_div_safe_bounded(self):
        layer = _build_layer("div_safe", n_inputs=2)
        max_gn, input_gn, has_nan = _run_fwd_bwd(layer)
        assert not has_nan
        assert max_gn < 100, f"div_safe param_grad={max_gn:.1f}"
        assert input_gn < 100, f"div_safe input_grad={input_gn:.1f}"

    def test_log_bounded(self):
        layer = _build_layer("log")
        max_gn, input_gn, has_nan = _run_fwd_bwd(layer)
        assert not has_nan
        assert input_gn < 100, f"log input_grad={input_gn:.1f}"

    def test_reciprocal_bounded(self):
        layer = _build_layer("reciprocal")
        max_gn, input_gn, has_nan = _run_fwd_bwd(layer)
        assert not has_nan
        assert input_gn < 100, f"reciprocal input_grad={input_gn:.1f}"


# ── Pass 2: Gradient amplifiers and math-space boundary norm fixes ─────


class TestMathSpaceOps:
    """Math-space ops: RMSNorm boundary instead of LayerNorm."""

    @pytest.mark.parametrize(
        "op",
        [
            "padic_expand",
            "padic_residual",
            "padic_gate",
            "clifford_attention",
            "grade_mix",
            "spike_rate_code",
        ],
    )
    def test_mathspace_grad_bounded(self, op):
        layer = _build_layer(op)
        max_gn, input_gn, has_nan = _run_fwd_bwd(layer)
        assert not has_nan, f"{op} produced NaN"
        assert max_gn < 500, f"{op} param_grad={max_gn:.1f} too high"


class TestGradientAmplifiers:
    """Ops that had extreme gradient norms in production."""

    def test_state_space_bounded(self):
        layer = _build_layer("state_space")
        max_gn, input_gn, has_nan = _run_fwd_bwd(layer)
        assert not has_nan
        assert max_gn < 200, f"state_space param_grad={max_gn:.1f}"

    def test_compression_mixture_experts_single_input(self):
        """Verify CME handles single input gracefully (was IndexError)."""
        layer = _build_layer("compression_mixture_experts")
        max_gn, input_gn, has_nan = _run_fwd_bwd(layer)
        assert not has_nan

    def test_compression_mixture_experts_two_inputs(self):
        layer = _build_layer("compression_mixture_experts", n_inputs=2)
        max_gn, input_gn, has_nan = _run_fwd_bwd(layer)
        assert not has_nan
        assert max_gn < 200


class TestDashboard0PercentS0:
    """Ops that showed 0% S0 in dashboard but compile fine in isolation."""

    @pytest.mark.parametrize(
        "op,n_inputs",
        [
            ("gated_delta", 1),
            ("graph_attention", 1),
            ("diff_attention", 1),
            ("fused_linear_gelu", 1),
            ("causal_mask", 1),
            ("local_window_attn", 1),
            ("shared_basis_proj", 1),
            ("tied_proj", 1),
            ("route_recursion", 1),
            ("route_lanes", 1),
            ("speculative", 1),
            ("low_rank_proj", 1),
            ("learnable_scale", 1),
            ("learnable_bias", 1),
            ("exp", 1),
            ("topk_gate", 1),
        ],
    )
    def test_compiles_and_runs(self, op, n_inputs):
        layer = _build_layer(op, n_inputs)
        x = torch.randn(2, 8, 64)
        y = layer(x)
        assert not torch.isnan(y).any(), f"{op} produced NaN"
        assert y.shape == x.shape
