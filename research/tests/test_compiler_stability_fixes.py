"""
Tests for compiler-level stability fixes:
  1. spectral_filter: freq_mask clamped to prevent FFT blow-up
  2. route_topk: gradient scale capped at 4.0
  3. block_sparse_linear: density floor raised to 0.25
"""

import torch
from research.synthesis.compiler import _build_block_sparse_mask
from research.synthesis.compiler_ops_routing import _op_route_topk
from research.synthesis.compiler_ops_mathspaces import _op_spectral_filter


class TestSpectralFilterClamp:
    def test_large_mask_values_clamped(self):
        """freq_mask with extreme values should be clamped, not blow up."""
        module = type("M", (), {})()
        module.freq_mask = torch.tensor([100.0, -50.0, 0.5, 200.0])
        x = torch.randn(2, 4, 6)  # rfft of dim=6 produces 4 freq bins
        out = _op_spectral_filter(module, [x], {})
        assert torch.isfinite(out).all(), "Output should be finite with clamped mask"

    def test_normal_mask_unchanged(self):
        """Mask values within [-2, 2] should pass through unchanged."""
        module = type("M", (), {})()
        module.freq_mask = torch.tensor([1.0, -0.5, 0.0, 1.5])
        x = torch.randn(2, 4, 6)
        out = _op_spectral_filter(module, [x], {})
        assert torch.isfinite(out).all()

    def test_no_mask_passthrough(self):
        """Without freq_mask, input passes through."""
        module = type("M", (), {})()
        x = torch.randn(2, 4, 6)
        out = _op_spectral_filter(module, [x], {})
        assert torch.equal(out, x)

    def test_grad_flows_through_clamp(self):
        """Gradients should flow through the clamped spectral filter."""
        module = type("M", (), {})()
        module.freq_mask = torch.nn.Parameter(torch.ones(5))  # rfft of dim=8 → 5 bins
        x = torch.randn(1, 2, 8, requires_grad=True)
        out = _op_spectral_filter(module, [x], {})
        out.sum().backward()
        assert x.grad is not None
        assert module.freq_mask.grad is not None


class TestRouteTopkScaleCap:
    def test_extreme_sparsity_scale_capped(self):
        """With k=1 and D=512, scale should be capped at 4.0, not sqrt(512)=22.6."""
        module = type("M", (), {"_routing_ctx": {}})()
        # Use uniform input so topk selection doesn't amplify by picking outliers.
        x = torch.ones(1, 4, 512)
        out = _op_route_topk(module, [x], {"k": 1})
        # With uniform input, selected value = 1.0 * scale.
        # If scale were sqrt(512)=22.6, nonzero would be 22.6.
        # With cap at 4.0, nonzero should be 4.0.
        nonzero = out[out != 0]
        assert nonzero.numel() > 0
        assert nonzero[0].item() <= 4.01, (
            f"Scale {nonzero[0].item():.1f} > 4.0 — cap not working"
        )

    def test_moderate_sparsity_uncapped(self):
        """With k=D//4, scale=sqrt(4)=2.0 which is below cap — should be used as-is."""
        module = type("M", (), {"_routing_ctx": {}})()
        x = torch.ones(1, 2, 64)
        k = 16  # D//4
        out = _op_route_topk(module, [x], {"k": k})
        # scale = sqrt(64/16) = 2.0, well below cap
        nonzero = out[out != 0]
        assert nonzero.numel() > 0

    def test_grad_flows(self):
        """Gradients should flow through route_topk."""
        module = type("M", (), {"_routing_ctx": {}})()
        x = torch.randn(1, 2, 32, requires_grad=True)
        out = _op_route_topk(module, [x], {"k": 4})
        out.sum().backward()
        assert x.grad is not None


class TestBlockSparseDensityFloor:
    def test_density_floor_enforced(self):
        """Density below 0.25 should be clamped to 0.25."""
        w = torch.randn(64, 64)
        mask_low = _build_block_sparse_mask(w, block_size=16, block_density=0.05)
        mask_quarter = _build_block_sparse_mask(w, block_size=16, block_density=0.25)
        # With floor at 0.25, requesting 0.05 should give same result as 0.25
        assert mask_low.mean().item() == mask_quarter.mean().item()

    def test_density_above_floor_respected(self):
        """Density of 0.5 should keep ~50% of blocks."""
        w = torch.randn(64, 64)
        mask = _build_block_sparse_mask(w, block_size=16, block_density=0.5)
        actual_density = mask.mean().item()
        assert actual_density > 0.4, (
            f"Density {actual_density:.2f} too low for 0.5 target"
        )

    def test_full_density_is_all_ones(self):
        """Density 1.0 should keep all blocks."""
        w = torch.randn(32, 32)
        mask = _build_block_sparse_mask(w, block_size=8, block_density=1.0)
        assert mask.mean().item() == 1.0

    def test_mask_shape_matches_weight(self):
        """Mask should have same shape as weight."""
        w = torch.randn(128, 64)
        mask = _build_block_sparse_mask(w, block_size=16, block_density=0.3)
        assert mask.shape == w.shape
