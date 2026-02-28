"""Unit tests for routing compiler ops (Phase 2).

Tests both aria-core C kernel paths and PyTorch fallback paths.
Verifies output shapes, semantics, and telemetry recording.
"""
import torch
import torch.nn as nn
import pytest

from research.synthesis.compiler import (
    _op_route_topk, _op_route_lanes, _op_route_recursion, _op_token_merge,
    _op_mod_topk, _op_early_exit, _op_cascade, _op_speculative,
    _op_adaptive_recursion, _op_token_merging,
    _record_routing_telemetry,
)

# Check if aria-core is available
try:
    import aria_core
    HAS_ARIA_CORE = True
except ImportError:
    HAS_ARIA_CORE = False


class DummyModule(nn.Module):
    """Minimal module to hold telemetry attributes."""
    def __init__(self):
        super().__init__()


# ── route_topk ──────────────────────────────────────────────────────

class TestRouteTopk:
    def test_output_shapes(self):
        module = DummyModule()
        B, S, k = 2, 8, 3
        scores = torch.randn(B, S)
        indices, weights = _op_route_topk(module, [scores], {"k": k})
        assert indices.shape == (B, k), f"Expected ({B},{k}), got {indices.shape}"
        assert weights.shape == (B, k), f"Expected ({B},{k}), got {weights.shape}"

    def test_weights_are_normalized(self):
        module = DummyModule()
        scores = torch.randn(2, 8)
        _, weights = _op_route_topk(module, [scores], {"k": 3})
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_indices_are_valid(self):
        module = DummyModule()
        B, S, k = 2, 10, 4
        scores = torch.randn(B, S)
        indices, _ = _op_route_topk(module, [scores], {"k": k})
        assert (indices >= 0).all() and (indices < S).all()

    def test_telemetry_recorded(self):
        module = DummyModule()
        scores = torch.randn(2, 8)
        _op_route_topk(module, [scores], {"k": 3})
        assert hasattr(module, "routing_telemetry")
        rt = module.routing_telemetry
        assert rt["tokens_total"] > 0
        assert rt["tokens_processed"] > 0

    def test_k_equals_one(self):
        module = DummyModule()
        scores = torch.randn(2, 8)
        indices, weights = _op_route_topk(module, [scores], {"k": 1})
        assert indices.shape == (2, 1)

    @pytest.mark.skipif(not HAS_ARIA_CORE, reason="aria_core not available")
    def test_aria_core_path_matches_fallback(self):
        B, S, k = 2, 8, 3
        scores = torch.randn(B, S)
        # aria-core path
        m1 = DummyModule()
        idx1, w1 = _op_route_topk(m1, [scores], {"k": k})
        # fallback path (force by using non-float32)
        m2 = DummyModule()
        idx2, w2 = _op_route_topk(m2, [scores.double()], {"k": k})
        # Should select same top-k indices
        assert torch.equal(idx1.sort(dim=-1).values, idx2.sort(dim=-1).values)


# ── route_lanes ─────────────────────────────────────────────────────

class TestRouteLanes:
    def test_output_shape(self):
        module = DummyModule()
        B, S, L = 2, 8, 4
        scores = torch.randn(B, S, L)
        result = _op_route_lanes(module, [scores], {"n_lanes": L})
        assert result.shape == (B, S)

    def test_values_in_range(self):
        module = DummyModule()
        B, S, L = 2, 8, 4
        scores = torch.randn(B, S, L)
        result = _op_route_lanes(module, [scores], {"n_lanes": L})
        assert (result >= 0).all() and (result < L).all()

    def test_telemetry_recorded(self):
        module = DummyModule()
        scores = torch.randn(2, 8, 4)
        _op_route_lanes(module, [scores], {"n_lanes": 4})
        assert hasattr(module, "routing_telemetry")
        rt = module.routing_telemetry
        assert rt["tokens_total"] > 0

    def test_deterministic(self):
        module = DummyModule()
        scores = torch.randn(2, 8, 4)
        r1 = _op_route_lanes(module, [scores], {"n_lanes": 4})
        module2 = DummyModule()
        r2 = _op_route_lanes(module2, [scores], {"n_lanes": 4})
        assert torch.equal(r1, r2)


# ── route_recursion ─────────────────────────────────────────────────

class TestRouteRecursion:
    def test_output_shape(self):
        module = DummyModule()
        B, S, Dp = 2, 8, 5
        scores = torch.randn(B, S, Dp)
        result = _op_route_recursion(module, [scores], {"max_depth": Dp})
        assert result.shape == (B, S)

    def test_depth_values_in_range(self):
        module = DummyModule()
        B, S, Dp = 2, 8, 5
        scores = torch.randn(B, S, Dp)
        result = _op_route_recursion(module, [scores], {"max_depth": Dp})
        # argmax + 1 means values are 1..Dp
        assert (result >= 1).all() and (result <= Dp).all()

    def test_telemetry_recorded(self):
        module = DummyModule()
        scores = torch.randn(2, 8, 5)
        _op_route_recursion(module, [scores], {"max_depth": 5})
        assert hasattr(module, "routing_telemetry")
        rt = module.routing_telemetry
        assert rt["tokens_total"] > 0
        assert rt["count"] > 0


# ── token_merge ─────────────────────────────────────────────────────

class TestTokenMerge:
    def test_output_shape_restored(self):
        """token_merge restores to original seq length via gather."""
        module = DummyModule()
        B, S, D = 2, 8, 16
        x = torch.randn(B, S, D)
        result = _op_token_merge(module, [x], {"n_keep": 4})
        # After restore, output has original sequence length
        assert result.shape == (B, S, D)

    def test_default_n_keep(self):
        module = DummyModule()
        x = torch.randn(2, 8, 16)
        result = _op_token_merge(module, [x], {})
        assert result.shape == (2, 8, 16)  # restored to original length

    def test_telemetry_recorded(self):
        module = DummyModule()
        x = torch.randn(2, 8, 16)
        _op_token_merge(module, [x], {"n_keep": 4})
        assert hasattr(module, "routing_telemetry")
        rt = module.routing_telemetry
        assert rt["tokens_total"] == 2 * 8
        assert rt["tokens_processed"] == 2 * 4
        assert rt["merge_kept"] == 2 * 4
        assert rt["merge_dropped"] == 2 * 4

    def test_n_keep_equals_seq_len(self):
        module = DummyModule()
        x = torch.randn(2, 8, 16)
        result = _op_token_merge(module, [x], {"n_keep": 8})
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

class TestModTopk:
    def test_output_shape_preserved(self):
        module = DummyModule()
        x = torch.randn(2, 8, 16)
        result = _op_mod_topk(module, [x], {"capacity_factor": 0.5})
        assert result.shape == x.shape

    def test_sparsity_applied(self):
        module = DummyModule()
        x = torch.ones(2, 8, 16)
        result = _op_mod_topk(module, [x], {"capacity_factor": 0.5})
        # Some tokens should be zeroed out
        zeros = (result.abs().sum(dim=-1) < 1e-6).sum().item()
        assert zeros > 0, "capacity_factor=0.5 should zero some tokens"

    def test_telemetry_recorded(self):
        module = DummyModule()
        x = torch.randn(2, 8, 16)
        _op_mod_topk(module, [x], {"capacity_factor": 0.75})
        assert hasattr(module, "routing_telemetry")


class TestEarlyExit:
    def test_output_shape_preserved(self):
        module = DummyModule()
        x = torch.randn(2, 8, 16)
        result = _op_early_exit(module, [x], {"threshold": 0.5})
        assert result.shape == x.shape

    def test_telemetry_recorded(self):
        module = DummyModule()
        x = torch.randn(2, 8, 16)
        _op_early_exit(module, [x], {"threshold": 0.5})
        assert hasattr(module, "routing_telemetry")


class TestCascade:
    def test_output_shape_preserved(self):
        module = DummyModule()
        x = torch.randn(2, 8, 16)
        result = _op_cascade(module, [x], {"threshold": 0.5})
        assert result.shape == x.shape

    def test_telemetry_recorded(self):
        module = DummyModule()
        x = torch.randn(2, 8, 16)
        _op_cascade(module, [x], {"threshold": 0.5})
        assert hasattr(module, "routing_telemetry")


class TestSpeculative:
    def test_output_shape_preserved(self):
        module = DummyModule()
        x = torch.randn(2, 8, 16)
        result = _op_speculative(module, [x], {"threshold": 0.5})
        assert result.shape == x.shape

    def test_scales_rather_than_drops(self):
        module = DummyModule()
        x = torch.ones(2, 8, 16)
        result = _op_speculative(module, [x], {"threshold": 0.5})
        # speculative uses 0.5 + 0.5*gate scaling, never zero
        assert (result.abs().sum(dim=-1) > 0).all()


class TestAdaptiveRecursion:
    def test_output_shape_preserved(self):
        module = DummyModule()
        x = torch.randn(2, 8, 16)
        result = _op_adaptive_recursion(module, [x], {"max_depth": 3})
        assert result.shape == x.shape

    def test_max_depth_clamped(self):
        module = DummyModule()
        x = torch.randn(2, 8, 16)
        # max_depth > 6 should be clamped
        result = _op_adaptive_recursion(module, [x], {"max_depth": 100})
        assert result.shape == x.shape


class TestTokenMergingControl:
    def test_output_shape_restored(self):
        module = DummyModule()
        x = torch.randn(2, 8, 16)
        result = _op_token_merging(module, [x], {"n_keep": 4})
        assert result.shape == (2, 8, 16)  # restored to original length
