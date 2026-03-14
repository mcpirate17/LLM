
import torch
from research.env import aria_core, HAS_ARIA_CORE as _HAS_ARIA_CORE
import pytest
from research.mathspaces.tropical import tropical_add, tropical_matmul, execute_tropical_center, execute_tropical_attention, execute_tropical_gate
from research.mathspaces.tropical_routing import TropicalRouter

def test_tropical_add_parity():
    shapes = [(1, 4, 32), (2, 16, 64), (4, 32, 128)]
    for shape in shapes:
        a = torch.randn(shape)
        b = torch.randn(shape)
        
        # Fallback
        expected = torch.minimum(a, b)
        
        # Native
        actual = aria_core.tropical_add_f32(a.contiguous(), b.contiguous())
        
        assert torch.allclose(expected, actual, atol=1e-5), f"Parity failed for shape {shape}"

def test_tropical_matmul_parity():
    # Test cases: (B, S, D) and (B, D, S2)
    test_configs = [
        ((1, 4, 32), (1, 32, 4)),
        ((2, 16, 64), (2, 64, 16)),
        ((4, 32, 128), (4, 128, 32)),
        ((1, 1, 32), (1, 32, 1)), # Edge case S=1, B=1
    ]
    
    for shape_a, shape_b in test_configs:
        a = torch.randn(shape_a)
        b = torch.randn(shape_b)
        
        # Fallback (manual min-plus)
        expanded_a = a.unsqueeze(3) # (B, S, D, 1)
        expanded_b = b.unsqueeze(1) # (B, 1, D, S2)
        # We want min_k(a_ik + b_kj)
        # a: (B, S, D), b: (B, D, S2)
        # expected[b, i, j] = min_k(a[b, i, k] + b[b, k, j])
        
        B, S, D = shape_a
        _, _, S2 = shape_b
        expected = torch.zeros((B, S, S2))
        for b_idx in range(B):
            for i in range(S):
                for j in range(S2):
                    expected[b_idx, i, j] = torch.min(a[b_idx, i, :] + b[b_idx, :, j])
        
        # Native
        actual = tropical_matmul(a.contiguous(), b.contiguous())
        
        assert torch.allclose(expected, actual, atol=1e-5), f"Parity failed for shapes {shape_a}, {shape_b}"


def test_tropical_matmul_self_attention_layout_parity():
    x = torch.randn(2, 8, 16)
    expected = torch.zeros((2, 8, 8))
    for b in range(2):
        for i in range(8):
            for j in range(8):
                expected[b, i, j] = torch.min(x[b, i, :] + x[b, j, :])

    actual = tropical_matmul(x.contiguous(), x.contiguous())

    assert torch.allclose(expected, actual, atol=1e-5)

def test_tropical_center_parity():
    shapes = [(1, 4, 32), (2, 16, 64), (4, 32, 128)]
    for shape in shapes:
        x = torch.randn(shape)
        
        # Fallback
        cmin = torch.cummin(x, dim=1).values
        expected = x - cmin
        
        # Native
        actual = aria_core.tropical_center_f32(x.contiguous())
        
        assert torch.allclose(expected, actual, atol=1e-5), f"Parity failed for shape {shape}"

def test_tropical_attention_parity():
    shapes = [(1, 4, 32), (2, 8, 64)]
    for shape in shapes:
        x = torch.randn(shape)
        
        # We need a dummy module without weights to trigger native path
        class DummyModule(torch.nn.Module):
            pass
        module = DummyModule()
        
        # Native
        actual = execute_tropical_attention(module, x.contiguous())
        
        # Expected (manual)
        B, S, D = shape
        # 1. Distances (min-plus)
        dist = torch.zeros((B, S, S))
        for b in range(B):
            for i in range(S):
                for j in range(i + 1):
                    dist[b, i, j] = torch.min(x[b, i, :] + x[b, j, :])
                for j in range(i + 1, S):
                    dist[b, i, j] = float('inf')
        
        # 2. Softmax
        weights = torch.softmax(-dist / 0.1, dim=-1)
        
        # 3. Weighted sum
        expected = torch.bmm(weights, x)
        
        assert torch.allclose(expected, actual, atol=1e-5), f"Parity failed for shape {shape}"

def test_tropical_gate_parity():
    shapes = [(1, 4, 32), (2, 8, 64)]
    for shape in shapes:
        x = torch.randn(shape)
        
        class DummyModule(torch.nn.Module):
            pass
        module = DummyModule()
        
        # Native
        actual = execute_tropical_gate(module, x.contiguous())
        
        # Expected
        B, S, D = shape
        dist = torch.zeros((B, S, S))
        for b in range(B):
            for i in range(S):
                for j in range(i + 1):
                    dist[b, i, j] = torch.min(x[b, i, :] + x[b, j, :])
                for j in range(i + 1, S):
                    dist[b, i, j] = float('inf')
        
        weights = torch.softmax(-dist / 0.1, dim=-1)
        gated = torch.bmm(weights, x)
        gate = torch.sigmoid(gated)
        expected = x * gate
        
        assert torch.allclose(expected, actual, atol=1e-5), f"Parity failed for shape {shape}"

def test_tropical_router_parity():
    B, S, D, E = 2, 8, 64, 16
    x = torch.randn(B, S, D)
    router = TropicalRouter(D, E, temperature=0.1)
    
    # Native
    actual = router(x.contiguous())
    
    # Expected
    x_min = torch.min(x, dim=-1, keepdim=True).values
    x_norm = x - x_min
    
    expected_scores = torch.zeros((B, S, E))
    for b in range(B):
        for s in range(S):
            for e in range(E):
                expected_scores[b, s, e] = torch.min(x_norm[b, s, :] + router.centroids[e, :])
    
    expected = torch.softmax(-expected_scores / 0.1, dim=-1)
    
    assert torch.allclose(expected, actual, atol=1e-5), "Router parity failed"
