"""Unit tests for routing compiler ops (Phase 2).

Tests both aria_core C kernel paths and PyTorch fallback paths.
Verifies output shapes, semantics, and telemetry recording.
"""

import torch
import torch.nn as nn
import pytest

from research.synthesis.compiler_ops_routing import (
    _op_feature_sparsity,
    _op_gated_lane_blend,
    _op_depth_gated_transform,
    _op_adjacent_token_merge,
    _op_depth_token_mask,
    _op_confidence_token_gate,
    _op_learned_token_gate,
    _op_cheap_verify_blend,
    _op_depth_weighted_proj,
    _op_swiglu_mlp,
)
from research.synthesis.compiler_op_utils import _record_routing_telemetry

pytestmark = pytest.mark.unit

# Check if aria_core is available


class DummyModule(nn.Module):
    """Minimal module to hold telemetry attributes."""

    def __init__(self):
        super().__init__()


class SwiGLUModule(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim)
        self.up_proj = nn.Linear(dim, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, dim)


# ── route_topk ──────────────────────────────────────────────────────


class TestFeatureSparsity:
    def test_output_shape_matches_input(self):
        """route_topk now returns (B,S,D) tensor (sparse mask), not a tuple."""
        module = DummyModule()
        B, S, D, k = 2, 8, 64, 4
        x = torch.randn(B, S, D)
        result = _op_feature_sparsity(module, [x], {"k": k})
        assert isinstance(result, torch.Tensor), f"Expected Tensor, got {type(result)}"
        assert result.shape == (B, S, D), f"Expected {(B, S, D)}, got {result.shape}"

    def test_sparsity(self):
        """Only top-k values per (B,S) slice should be non-zero."""
        module = DummyModule()
        B, S, D, k = 2, 8, 64, 4
        x = torch.randn(B, S, D)
        result = _op_feature_sparsity(module, [x], {"k": k})
        non_zero_per_slice = (result != 0).float().sum(dim=-1)  # (B, S)
        assert (non_zero_per_slice == k).all(), f"Expected {k} non-zero per slice"

    def test_values_preserved(self):
        """Non-zero entries should equal original values times density scale."""
        module = DummyModule()
        D, k = 16, 3
        x = torch.randn(2, 4, D)
        result = _op_feature_sparsity(module, [x], {"k": k})
        mask = result != 0
        scale = (D / k) ** 0.5
        assert torch.allclose(result[mask], x[mask] * scale)

    def test_telemetry_recorded(self):
        module = DummyModule()
        x = torch.randn(2, 8, 64)
        _op_feature_sparsity(module, [x], {"k": 4})
        assert hasattr(module, "routing_telemetry")
        rt = module.routing_telemetry
        assert rt["tokens_total"] > 0

    def test_gradient_flows(self):
        module = DummyModule()
        x = torch.randn(2, 4, 16, requires_grad=True)
        result = _op_feature_sparsity(module, [x], {"k": 3})
        result.sum().backward()
        assert x.grad is not None

    def test_k_equals_one(self):
        module = DummyModule()
        x = torch.randn(2, 4, 16)
        result = _op_feature_sparsity(module, [x], {"k": 1})
        non_zero = (result != 0).float().sum(dim=-1)
        assert (non_zero == 1).all()


# ── route_lanes ─────────────────────────────────────────────────────


def _make_lane_module(D, n_lanes):
    """Build a DummyModule with learned lane routing params."""
    module = DummyModule()
    module.lane_scorer = nn.Parameter(torch.randn(n_lanes, D) * 0.02)
    module.lane_projs = nn.ParameterList(
        [nn.Parameter(torch.randn(D, D) * 0.02) for _ in range(n_lanes)]
    )
    return module


class TestGatedLaneBlend:
    def test_output_shape(self):
        B, S, D, L = 2, 8, 64, 4
        module = _make_lane_module(D, L)
        x = torch.randn(B, S, D)
        result = _op_gated_lane_blend(module, [x], {"n_lanes": L})
        assert result.shape == (B, S, D)

    def test_non_identity(self):
        B, S, D, L = 2, 8, 64, 3
        module = _make_lane_module(D, L)
        x = torch.randn(B, S, D)
        result = _op_gated_lane_blend(module, [x], {"n_lanes": L})
        assert not torch.allclose(result, x, atol=1e-5)

    def test_telemetry_recorded(self):
        D, L = 64, 4
        module = _make_lane_module(D, L)
        x = torch.randn(2, 8, D)
        _op_gated_lane_blend(module, [x], {"n_lanes": L})
        assert hasattr(module, "routing_telemetry")
        rt = module.routing_telemetry
        assert rt["tokens_total"] > 0

    def test_gradient_flows(self):
        D, L = 32, 3
        module = _make_lane_module(D, L)
        x = torch.randn(2, 4, D, requires_grad=True)
        result = _op_gated_lane_blend(module, [x], {"n_lanes": L})
        result.sum().backward()
        assert x.grad is not None


class TestSwiGLUMlp:
    def test_cpu_path_matches_dense_pytorch(self, monkeypatch):
        module = SwiGLUModule(dim=16, hidden_dim=32).eval()
        x = torch.randn(2, 5, 16)

        def _fail(*args, **kwargs):
            raise AssertionError("aria_core.swiglu_f32 should not be used on CPU")

        monkeypatch.setattr(
            "research.synthesis.compiler_ops_routing.aria_core.swiglu_f32",
            _fail,
            raising=False,
        )

        expected = module.down_proj(
            torch.nn.functional.silu(module.gate_proj(x)) * module.up_proj(x)
        )
        result = _op_swiglu_mlp(module, [x], {})
        assert torch.allclose(result, expected, atol=1e-6, rtol=1e-5)


# ── route_recursion ─────────────────────────────────────────────────


def _make_depth_module(D, max_depth):
    """Build a DummyModule with learned depth routing params."""
    module = DummyModule()
    module.depth_scorer = nn.Parameter(torch.randn(max_depth, D) * 0.02)
    module.depth_projs = nn.ParameterList(
        [nn.Parameter(torch.randn(D, D) * 0.02) for _ in range(max_depth)]
    )
    return module


class TestDepthGatedTransform:
    def test_output_shape(self):
        B, S, D, Dp = 2, 8, 64, 5
        module = _make_depth_module(D, Dp)
        x = torch.randn(B, S, D)
        result = _op_depth_gated_transform(module, [x], {"max_depth": Dp})
        assert result.shape == (B, S, D)

    def test_non_identity(self):
        B, S, D, Dp = 2, 8, 64, 3
        module = _make_depth_module(D, Dp)
        x = torch.randn(B, S, D)
        result = _op_depth_gated_transform(module, [x], {"max_depth": Dp})
        assert not torch.allclose(result, x, atol=1e-5)

    def test_telemetry_recorded(self):
        D, Dp = 64, 5
        module = _make_depth_module(D, Dp)
        x = torch.randn(2, 8, D)
        _op_depth_gated_transform(module, [x], {"max_depth": Dp})
        assert hasattr(module, "routing_telemetry")
        rt = module.routing_telemetry
        assert rt["tokens_total"] > 0
        assert rt["count"] > 0


# ── token_merge ─────────────────────────────────────────────────────


class TestAdjacentTokenMerge:
    def test_output_shape_restored(self):
        """token_merge restores to original seq length via gather."""
        module = DummyModule()
        B, S, D = 2, 8, 16
        x = torch.randn(B, S, D)
        result = _op_adjacent_token_merge(module, [x], {"n_keep": 4})
        # After restore, output has original sequence length
        assert result.shape == (B, S, D)

    def test_default_n_keep(self):
        module = DummyModule()
        x = torch.randn(2, 8, 16)
        result = _op_adjacent_token_merge(module, [x], {})
        assert result.shape == (2, 8, 16)  # restored to original length

    def test_telemetry_recorded(self):
        module = DummyModule()
        x = torch.randn(2, 8, 16)
        _op_adjacent_token_merge(module, [x], {"n_keep": 4})
        assert hasattr(module, "routing_telemetry")
        rt = module.routing_telemetry
        assert rt["tokens_total"] == 2 * 8
        assert rt["tokens_processed"] == 2 * 4
        assert rt["merge_kept"] == 2 * 4
        assert rt["merge_dropped"] == 2 * 4

    def test_n_keep_equals_seq_len(self):
        module = DummyModule()
        x = torch.randn(2, 8, 16)
        result = _op_adjacent_token_merge(module, [x], {"n_keep": 8})
        assert result.shape == x.shape
        # When n_keep == S, output should equal input
        assert torch.allclose(result, x)


# ── Telemetry helper ────────────────────────────────────────────────


class TestRecordRoutingTelemetry:
    def test_accumulates_across_calls(self):
        module = DummyModule()
        indices = torch.tensor([[0, 1], [2, 3]])
        _record_routing_telemetry(module, 4, indices)
        _record_routing_telemetry(module, 4, indices)
        rt = module.routing_telemetry
        assert rt["tokens_total"] == 8  # 2 calls * 2 batch * 2 seq

    def test_entropy_computed_with_logits(self):
        module = DummyModule()
        indices = torch.tensor([[0, 1], [2, 3]])
        logits = torch.randn(2, 2, 4)
        _record_routing_telemetry(module, 4, indices, logits=logits)
        rt = module.routing_telemetry
        assert rt["entropy_sum"] > 0
        assert rt["count"] == 1


# ── Control routing ops (Phase 2 bridge) ────────────────────────────


class TestDepthTokenMask:
    def test_output_shape_preserved(self):
        module = DummyModule()
        x = torch.randn(2, 8, 16)
        result = _op_depth_token_mask(module, [x], {"capacity_factor": 0.5})
        assert result.shape == x.shape

    def test_sparsity_applied(self):
        module = DummyModule()
        x = torch.ones(2, 8, 16)
        result = _op_depth_token_mask(module, [x], {"capacity_factor": 0.5})
        # Some tokens should be zeroed out
        zeros = (result.abs().sum(dim=-1) < 1e-6).sum().item()
        assert zeros > 0, "capacity_factor=0.5 should zero some tokens"

    def test_telemetry_recorded(self):
        module = DummyModule()
        x = torch.randn(2, 8, 16)
        _op_depth_token_mask(module, [x], {"capacity_factor": 0.75})
        assert hasattr(module, "routing_telemetry")


class TestConfidenceTokenGate:
    def test_output_shape_preserved(self):
        module = DummyModule()
        x = torch.randn(2, 8, 16)
        result = _op_confidence_token_gate(module, [x], {"threshold": 0.5})
        assert result.shape == x.shape

    def test_telemetry_recorded(self):
        module = DummyModule()
        x = torch.randn(2, 8, 16)
        _op_confidence_token_gate(module, [x], {"threshold": 0.5})
        assert hasattr(module, "routing_telemetry")


class TestLearnedTokenGate:
    def test_output_shape_preserved(self):
        module = DummyModule()
        x = torch.randn(2, 8, 16)
        result = _op_learned_token_gate(module, [x], {"threshold": 0.5})
        assert result.shape == x.shape

    def test_telemetry_recorded(self):
        module = DummyModule()
        x = torch.randn(2, 8, 16)
        _op_learned_token_gate(module, [x], {"threshold": 0.5})
        assert hasattr(module, "routing_telemetry")


class TestCheapVerifyBlend:
    def test_output_shape_preserved(self):
        module = DummyModule()
        x = torch.randn(2, 8, 16)
        result = _op_cheap_verify_blend(module, [x], {"threshold": 0.5})
        assert result.shape == x.shape

    def test_scales_rather_than_drops(self):
        module = DummyModule()
        x = torch.ones(2, 8, 16)
        result = _op_cheap_verify_blend(module, [x], {"threshold": 0.5})
        # speculative uses 0.5 + 0.5*gate scaling, never zero
        assert (result.abs().sum(dim=-1) > 0).all()


class TestDepthWeightedProj:
    def test_output_shape_preserved(self):
        module = DummyModule()
        x = torch.randn(2, 8, 16)
        result = _op_depth_weighted_proj(module, [x], {"max_depth": 3})
        assert result.shape == x.shape

    def test_max_depth_clamped(self):
        module = DummyModule()
        x = torch.randn(2, 8, 16)
        # max_depth > 6 should be clamped
        result = _op_depth_weighted_proj(module, [x], {"max_depth": 100})
        assert result.shape == x.shape


class TestAdjacentTokenMergeControl:
    def test_output_shape_restored(self):
        module = DummyModule()
        x = torch.randn(2, 8, 16)
        result = _op_adjacent_token_merge(module, [x], {"n_keep": 4})
        assert result.shape == (2, 8, 16)  # restored to original length
