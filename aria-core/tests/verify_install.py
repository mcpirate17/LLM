import torch
import aria_core
import time
import math

def test_relu():
    print("Testing ReLU...")
    x = torch.randn(1024, 1024, dtype=torch.float32)
    y = aria_core.relu_f32(x)
    y_ref = torch.relu(x)
    assert torch.allclose(y, y_ref)
    print("ReLU: OK")

def test_rmsnorm():
    print("Testing RMSNorm...")
    batch, dim = 32, 512
    x = torch.randn(batch, dim, dtype=torch.float32)
    weight = torch.ones(dim, dtype=torch.float32)
    y = aria_core.rmsnorm_f32(x, weight, 1e-6)
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
    y_ref = (x / rms) * weight
    assert torch.allclose(y, y_ref, atol=1e-5)
    print("RMSNorm: OK")

def test_clifford():
    print("Testing Clifford Cl(3,0) Geometric Product...")
    B, S, K = 2, 64, 32
    a = torch.randn(B, S, K, 8, dtype=torch.float32)
    b = torch.randn(B, S, K, 8, dtype=torch.float32)
    y = aria_core.clifford_geometric_product_cl30_f32(a, b)
    a0, a1, a2, a3 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    a12, a13, a23, a123 = a[..., 4], a[..., 5], a[..., 6], a[..., 7]
    b0, b1, b2, b3 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    b12, b13, b23, b123 = b[..., 4], b[..., 5], b[..., 6], b[..., 7]
    r0 = (a0*b0 + a1*b1 + a2*b2 + a3*b3 - a12*b12 - a13*b13 - a23*b23 - a123*b123)
    r1 = (a0*b1 + a1*b0 - a2*b12 + a12*b2 - a3*b13 + a13*b3 + a23*b123 - a123*b23)
    r2 = (a0*b2 + a1*b12 + a2*b0 - a12*b1 - a3*b23 - a13*b123 + a23*b3 + a123*b13)
    r3 = (a0*b3 - a1*b13 + a2*b23 + a3*b0 + a12*b123 + a13*b1 - a23*b2 - a123*b12)
    r12 = (a0*b12 + a1*b2 - a2*b1 + a12*b0 + a3*b123 - a13*b23 + a23*b13 + a123*b3)
    r13 = (a0*b13 + a1*b3 - a3*b1 + a13*b0 - a2*b123 + a12*b23 - a23*b12 - a123*b2)
    r23 = (a0*b23 + a2*b3 - a3*b2 + a23*b0 + a1*b123 - a12*b13 + a13*b12 + a123*b1)
    r123 = (a0*b123 + a1*b23 - a2*b13 + a3*b12 + a12*b3 - a13*b2 + a23*b1 + a123*b0)
    y_ref = torch.stack([r0, r1, r2, r3, r12, r13, r23, r123], dim=-1)
    assert torch.allclose(y, y_ref, atol=1e-5)
    print("Clifford GP: OK")

def test_hyperbolic():
    print("Testing Hyperbolic Distance...")
    for i in range(10):
        batch, dim = 32, 64
        c = 1.0
        x = torch.randn(batch, dim, dtype=torch.float32) * 0.2
        y = torch.randn(batch, dim, dtype=torch.float32) * 0.2
        out = aria_core.hyperbolic_distance_f32(x, y, c)
        
        def _clamp_norm(x, max_norm=1.0 - 1e-3):
            norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-15)
            mask = norm > max_norm
            res = x.clone()
            res[mask.expand_as(x)] = (x / norm * max_norm)[mask.expand_as(x)]
            return res

        def mobius_add(x, v, c):
            x = _clamp_norm(x)
            v = _clamp_norm(v)
            x2 = torch.sum(x * x, dim=-1, keepdim=True)
            v2 = torch.sum(v * v, dim=-1, keepdim=True)
            xv = torch.sum(x * v, dim=-1, keepdim=True)
            num = (1 + 2 * c * xv + c * v2) * x + (1 - c * x2) * v
            den = 1 + 2 * c * xv + c**2 * x2 * v2
            res = num / den
            return _clamp_norm(res)

        diff = mobius_add(-x, y, c)
        res_norm = torch.norm(diff, dim=-1)
        arg = (math.sqrt(c) * res_norm).clamp_max(1 - 1e-7)
        out_ref = 2 * torch.atanh(arg) / math.sqrt(c)
        
        if not torch.allclose(out, out_ref, atol=5e-4):
            print(f"Iteration {i} failed! Max diff: {(out - out_ref).abs().max():.8f}")
            assert False
    print("Hyperbolic Distance: OK")

if __name__ == "__main__":
    test_relu()
    test_rmsnorm()
    test_clifford()
    test_hyperbolic()
