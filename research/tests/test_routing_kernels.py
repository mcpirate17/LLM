import torch
import torch.nn.functional as F
import pytest
import numpy as np

pytestmark = pytest.mark.native

try:
    import aria_core
    HAS_ARIA_CORE = True
except ImportError:
    HAS_ARIA_CORE = False

def test_route_topk():
    batch, seq, experts = 2, 8, 4
    k = 2
    scores = torch.randn(batch, seq, experts, dtype=torch.float32)
    
    # 1. Test aria_core native if available
    if HAS_ARIA_CORE:
        # Kernel expects 2D: [n_tokens, n_experts]
        scores_2d = scores.view(-1, experts)
        indices_2d, weights_2d = aria_core.route_topk_indices_f32(scores_2d, k)
        indices = indices_2d.view(batch, seq, k)
        weights = weights_2d.view(batch, seq, k)
        
        assert indices.shape == (batch, seq, k)
        assert weights.shape == (batch, seq, k)
        # Weights should be sorted desc by default or at least match indices
        # Check softmax sum
        sums = weights.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums))

    # 2. Test manual fallback logic (replicating compiler.py fallback)
    weights_fb, indices_fb = scores.topk(k, dim=-1)
    weights_fb = F.softmax(weights_fb, dim=-1)
    
    assert indices_fb.shape == (batch, seq, k)
    assert weights_fb.shape == (batch, seq, k)

def test_route_lanes():
    batch, seq, lanes = 2, 16, 3
    scores = torch.randn(batch, seq, lanes, dtype=torch.float32)
    
    if HAS_ARIA_CORE:
        # These kernels currently expect 3D in bindings.cpp
        lane_indices = aria_core.route_lane_argmax_f32(scores)
        assert lane_indices.shape == (batch, seq)
        # Check argmax property
        expected = scores.argmax(dim=-1)
        torch.testing.assert_close(lane_indices.to(torch.int64), expected)

def test_route_recursion():
    batch, seq, max_dp_opt = 2, 8, 5
    scores = torch.randn(batch, seq, max_dp_opt, dtype=torch.float32)
    
    if HAS_ARIA_CORE:
        depth = aria_core.route_recursion_depth_f32(scores)
        assert depth.shape == (batch, seq)
        # argmax + 1
        expected = scores.argmax(dim=-1) + 1
        torch.testing.assert_close(depth.to(torch.int64), expected)

def test_token_merge():
    batch, seq, dim = 2, 16, 32
    n_keep = 8
    x = torch.randn(batch, seq, dim, dtype=torch.float32)
    
    if HAS_ARIA_CORE:
        y, restore_map = aria_core.token_merge_simple_f32(x, n_keep)
        assert y.shape == (batch, n_keep, dim)
        assert restore_map.shape == (batch, seq)
        # Simple merge keeps first n_keep
        torch.testing.assert_close(y, x[:, :n_keep, :])

if __name__ == "__main__":
    pytest.main([__file__])
