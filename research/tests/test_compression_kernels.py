import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
import numpy as np
from research.mathspaces.compression import (
    execute_low_rank_proj,
    execute_grouped_linear,
    execute_bottleneck_proj,
    execute_shared_basis_proj,
    execute_tied_proj
)

try:
    import aria_core
    HAS_ARIA_CORE = True
except ImportError:
    HAS_ARIA_CORE = False

class MockModule(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)

def test_low_rank_proj_equivalence():
    if not HAS_ARIA_CORE: pytest.skip("aria_core not available")
    B, S, D = 2, 4, 16
    rank = 4
    x = torch.randn(B, S, D)
    U = torch.randn(rank, D)
    V = torch.randn(D, rank)
    
    module = MockModule(U=U.t().contiguous(), V=V.t().contiguous())
    expected = execute_low_rank_proj(module, x)
    
    x_flat = x.view(-1, D).contiguous()
    actual_flat = aria_core.linear_low_rank_f32(x_flat, U.contiguous(), V.contiguous(), None)
    actual = actual_flat.view(B, S, D)
    
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-2)

def test_grouped_linear_equivalence():
    if not HAS_ARIA_CORE: pytest.skip("aria_core not available")
    B, S, D = 2, 4, 16
    groups = 4
    dg = D // groups
    x = torch.randn(B, S, D)
    W = torch.randn(groups, dg, dg)
    
    module = MockModule(weight=W.contiguous(), n_groups=groups)
    expected = execute_grouped_linear(module, x)
    
    x_flat = x.view(-1, D).contiguous()
    W_cpp = W.transpose(1, 2).contiguous()
    actual_flat = aria_core.linear_grouped_f32(x_flat, W_cpp, None, groups)
    actual = actual_flat.view(B, S, D)
    
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-2)

def test_bottleneck_proj_equivalence():
    if not HAS_ARIA_CORE: pytest.skip("aria_core not available")
    B, S, D = 2, 4, 16
    rank = 4
    x = torch.randn(B, S, D)
    W_down = torch.randn(rank, D)
    W_up = torch.randn(D, rank)
    
    module = MockModule(down=W_down.contiguous(), up=W_up.contiguous())
    expected = execute_bottleneck_proj(module, x)
    
    x_flat = x.view(-1, D).contiguous()
    actual_flat = aria_core.linear_bottleneck_f32(x_flat, W_down.contiguous(), W_up.contiguous(), None, None)
    actual = actual_flat.view(B, S, D)
    
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-2)

def test_shared_basis_proj_equivalence():
    if not HAS_ARIA_CORE: pytest.skip("aria_core not available")
    B, S, D = 2, 4, 16
    k = 4
    x = torch.randn(B, S, D)
    mixing = torch.randn(D, k)
    basis = torch.randn(k, D)
    
    module = MockModule(mixing=mixing.contiguous(), basis=basis.contiguous())
    expected = execute_shared_basis_proj(module, x)
    
    W_mixing = mixing.t().contiguous()
    W_basis = basis.t().contiguous()
    
    x_flat = x.view(-1, D).contiguous()
    actual_flat = aria_core.linear_shared_basis_f32(x_flat, W_mixing, W_basis)
    actual = actual_flat.view(B, S, D)
    
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-2)

def test_tied_proj_equivalence():
    if not HAS_ARIA_CORE: pytest.skip("aria_core not available")
    B, S, D = 2, 4, 16
    rank = 4
    x = torch.randn(B, S, D)
    W = torch.randn(rank, D)
    
    module = MockModule(tied_weight=W.contiguous())
    expected = execute_tied_proj(module, x)
    
    x_flat = x.view(-1, D).contiguous()
    actual_flat = aria_core.linear_tied_f32(x_flat, W.contiguous(), None, None)
    actual = actual_flat.view(B, S, D)
    
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-2)

if __name__ == "__main__":
    pytest.main([__file__])
