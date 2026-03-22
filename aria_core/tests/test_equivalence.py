"""Numerical equivalence tests: aria_core C kernels vs PyTorch reference implementations.

Verifies that every exposed kernel produces results within tolerance of the
equivalent PyTorch operation. This is the Phase 4 unified test suite from
DRY_HIGH_PERF_TODO.md.
"""

import torch
import torch.nn.functional as F
import pytest
import math

import aria_core


ATOL = 1e-5
RTOL = 1e-4


# ═══════════════════════════════════════════════════════════════════════
# Elementwise unary
# ═══════════════════════════════════════════════════════════════════════


class TestUnaryOps:
    def setup_method(self):
        self.x = torch.randn(32, 64)

    def test_relu(self):
        torch.testing.assert_close(
            aria_core.relu_f32(self.x), F.relu(self.x), atol=ATOL, rtol=RTOL
        )

    def test_gelu(self):
        # C kernel uses tanh approximation
        expected = (
            0.5
            * self.x
            * (1 + torch.tanh(math.sqrt(2 / math.pi) * (self.x + 0.044715 * self.x**3)))
        )
        torch.testing.assert_close(
            aria_core.gelu_f32(self.x), expected, atol=1e-4, rtol=1e-3
        )

    def test_silu(self):
        torch.testing.assert_close(
            aria_core.silu_f32(self.x), F.silu(self.x), atol=1e-4, rtol=1e-3
        )

    def test_sigmoid(self):
        torch.testing.assert_close(
            aria_core.sigmoid_f32(self.x), torch.sigmoid(self.x), atol=1e-4, rtol=1e-3
        )

    def test_tanh(self):
        torch.testing.assert_close(
            aria_core.tanh_f32(self.x), torch.tanh(self.x), atol=1e-4, rtol=1e-3
        )

    def test_exp(self):
        x = self.x.clamp(-10, 10)
        torch.testing.assert_close(
            aria_core.exp_f32(x), torch.exp(x), atol=1e-4, rtol=1e-3
        )

    def test_square(self):
        torch.testing.assert_close(
            aria_core.square_f32(self.x), self.x**2, atol=ATOL, rtol=RTOL
        )

    def test_abs(self):
        torch.testing.assert_close(
            aria_core.abs_f32(self.x), self.x.abs(), atol=ATOL, rtol=RTOL
        )

    def test_neg(self):
        torch.testing.assert_close(
            aria_core.neg_f32(self.x), -self.x, atol=ATOL, rtol=RTOL
        )

    def test_sin(self):
        torch.testing.assert_close(
            aria_core.sin_f32(self.x), torch.sin(self.x), atol=1e-4, rtol=1e-3
        )

    def test_cos(self):
        torch.testing.assert_close(
            aria_core.cos_f32(self.x), torch.cos(self.x), atol=1e-4, rtol=1e-3
        )

    def test_log(self):
        x = self.x.abs() + 0.01
        torch.testing.assert_close(
            aria_core.log_f32(x), torch.log(x), atol=1e-4, rtol=1e-3
        )

    def test_sqrt(self):
        x = self.x.abs() + 0.01
        torch.testing.assert_close(
            aria_core.sqrt_f32(x), torch.sqrt(x), atol=1e-4, rtol=1e-3
        )

    def test_reciprocal(self):
        x = self.x.abs() + 0.1
        torch.testing.assert_close(
            aria_core.reciprocal_f32(x), 1.0 / x, atol=1e-4, rtol=1e-3
        )


# ═══════════════════════════════════════════════════════════════════════
# Elementwise binary
# ═══════════════════════════════════════════════════════════════════════


class TestBinaryOps:
    def setup_method(self):
        self.a = torch.randn(16, 32)
        self.b = torch.randn(16, 32)

    def test_add(self):
        torch.testing.assert_close(
            aria_core.add_f32(self.a, self.b), self.a + self.b, atol=ATOL, rtol=RTOL
        )

    def test_mul(self):
        torch.testing.assert_close(
            aria_core.mul_f32(self.a, self.b), self.a * self.b, atol=ATOL, rtol=RTOL
        )

    def test_sub(self):
        torch.testing.assert_close(
            aria_core.sub_f32(self.a, self.b), self.a - self.b, atol=ATOL, rtol=RTOL
        )

    def test_maximum(self):
        torch.testing.assert_close(
            aria_core.maximum_f32(self.a, self.b),
            torch.maximum(self.a, self.b),
            atol=ATOL,
            rtol=RTOL,
        )

    def test_minimum(self):
        torch.testing.assert_close(
            aria_core.minimum_f32(self.a, self.b),
            torch.minimum(self.a, self.b),
            atol=ATOL,
            rtol=RTOL,
        )

    def test_tropical_add(self):
        # tropical add = elementwise min
        torch.testing.assert_close(
            aria_core.tropical_add_f32(self.a, self.b),
            torch.minimum(self.a, self.b),
            atol=ATOL,
            rtol=RTOL,
        )

    def test_div_safe(self):
        b = self.b.abs() + 0.1
        torch.testing.assert_close(
            aria_core.div_safe_f32(self.a, b), self.a / b, atol=1e-4, rtol=1e-3
        )


# ═══════════════════════════════════════════════════════════════════════
# Reductions
# ═══════════════════════════════════════════════════════════════════════


class TestReductions:
    def test_sum(self):
        x = torch.randn(256)
        assert abs(aria_core.sum_f32(x) - x.sum().item()) < 1e-2

    def test_mean(self):
        x = torch.randn(256)
        assert abs(aria_core.mean_f32(x) - x.mean().item()) < 1e-4


# ═══════════════════════════════════════════════════════════════════════
# Linear algebra
# ═══════════════════════════════════════════════════════════════════════


class TestLinAlg:
    def test_matmul(self):
        A = torch.randn(8, 16)
        B = torch.randn(16, 4)
        torch.testing.assert_close(
            aria_core.matmul_f32(A, B), A @ B, atol=1e-3, rtol=1e-3
        )

    def test_linear_with_bias(self):
        x = torch.randn(4, 8)
        W = torch.randn(16, 8)
        b = torch.randn(16)
        torch.testing.assert_close(
            aria_core.linear_f32(x, W, b), F.linear(x, W, b), atol=1e-3, rtol=1e-3
        )

    def test_linear_no_bias(self):
        x = torch.randn(4, 8)
        W = torch.randn(16, 8)
        torch.testing.assert_close(
            aria_core.linear_f32(x, W, None), F.linear(x, W), atol=1e-3, rtol=1e-3
        )


# ═══════════════════════════════════════════════════════════════════════
# Causality helpers
# ═══════════════════════════════════════════════════════════════════════


class TestCausalMask:
    def test_causal_mask_f32(self):
        x = torch.randn(1, 4, 4)
        y = aria_core.causal_mask_f32(x)
        for i in range(4):
            for j in range(4):
                if j <= i:
                    torch.testing.assert_close(
                        y[0, i, j], x[0, i, j], atol=ATOL, rtol=RTOL
                    )
                else:
                    assert y[0, i, j].item() < -1e8


# ═══════════════════════════════════════════════════════════════════════
# Normalization
# ═══════════════════════════════════════════════════════════════════════


class TestNorm:
    def test_rmsnorm(self):
        x = torch.randn(4, 32)
        w = torch.ones(32)
        result = aria_core.rmsnorm_f32(x, w, 1e-5)
        rms = (x**2).mean(dim=-1, keepdim=True).sqrt()
        expected = x / (rms + 1e-5) * w
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-3)

    def test_layernorm(self):
        x = torch.randn(4, 32)
        w = torch.ones(32)
        b = torch.zeros(32)
        result = aria_core.layernorm_f32(x, w, b, 1e-5)
        expected = F.layer_norm(x, [32], w, b, eps=1e-5)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-3)


# ═══════════════════════════════════════════════════════════════════════
# Softmax
# ═══════════════════════════════════════════════════════════════════════


class TestSoftmax:
    def test_softmax(self):
        x = torch.randn(4, 16)
        torch.testing.assert_close(
            aria_core.softmax_f32(x), F.softmax(x, dim=-1), atol=1e-4, rtol=1e-3
        )


# ═══════════════════════════════════════════════════════════════════════
# Fused kernels
# ═══════════════════════════════════════════════════════════════════════


class TestFused:
    def test_matmul_relu(self):
        A = torch.randn(4, 8)
        B = torch.randn(8, 16)
        torch.testing.assert_close(
            aria_core.matmul_relu_f32(A, B), F.relu(A @ B), atol=1e-3, rtol=1e-3
        )

    def test_matmul_gelu(self):
        A = torch.randn(4, 8)
        B = torch.randn(8, 16)
        result = aria_core.matmul_gelu_f32(A, B)
        mm = A @ B
        expected = (
            0.5
            * mm
            * (1 + torch.tanh(math.sqrt(2 / math.pi) * (mm + 0.044715 * mm**3)))
        )
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-2)

    def test_fused_linear_gelu(self):
        x = torch.randn(4, 8)
        W = torch.randn(16, 8)
        b = torch.randn(16)
        result = aria_core.fused_linear_gelu_f32(x, W, b)
        mm = F.linear(x, W, b)
        expected = (
            0.5
            * mm
            * (1 + torch.tanh(math.sqrt(2 / math.pi) * (mm + 0.044715 * mm**3)))
        )
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-2)


# ═══════════════════════════════════════════════════════════════════════
# Backward kernels
# ═══════════════════════════════════════════════════════════════════════


class TestBackward:
    def test_relu_backward(self):
        x = torch.randn(32)
        go = torch.randn(32)
        expected = go * (x > 0).float()
        torch.testing.assert_close(
            aria_core.relu_backward_f32(go, x), expected, atol=ATOL, rtol=RTOL
        )

    def test_add_backward(self):
        go = torch.randn(32)
        ga, gb = aria_core.add_backward_f32(go)
        torch.testing.assert_close(ga, go, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(gb, go, atol=ATOL, rtol=RTOL)

    def test_matmul_backward(self):
        A = torch.randn(4, 8, requires_grad=True)
        B = torch.randn(8, 3, requires_grad=True)
        go = torch.randn(4, 3)
        gA, gB = aria_core.matmul_backward_f32(go, A.detach(), B.detach())
        # grad_A = go @ B^T, grad_B = A^T @ go
        torch.testing.assert_close(gA, go @ B.detach().T, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(gB, A.detach().T @ go, atol=1e-3, rtol=1e-3)


# ═══════════════════════════════════════════════════════════════════════
# Math space: Clifford
# ═══════════════════════════════════════════════════════════════════════


class TestClifford:
    def test_geometric_product_identity(self):
        """e * 1 = e for any multivector e."""
        mv = torch.randn(8, 8)  # 8 multivectors
        identity = torch.zeros(8, 8)
        identity[:, 0] = 1.0  # scalar part = 1
        result = aria_core.clifford_geometric_product_cl30_f32(mv, identity)
        torch.testing.assert_close(result, mv, atol=1e-5, rtol=1e-4)

    def test_rotor_identity(self):
        """Identity rotor (1,0,0,0,0,0,0,0) should not change vectors."""
        x = torch.randn(4, 8)
        rotor = torch.zeros(4, 8)
        rotor[:, 0] = 1.0
        result = aria_core.clifford_rotor_transform_cl30_f32(x, rotor)
        torch.testing.assert_close(result, x, atol=1e-4, rtol=1e-3)


# ═══════════════════════════════════════════════════════════════════════
# Math space: Hyperbolic
# ═══════════════════════════════════════════════════════════════════════


class TestHyperbolic:
    def test_mobius_add_zero(self):
        """x + 0 = x in Poincare ball."""
        x = torch.randn(4, 8) * 0.1  # small to stay in ball
        v = torch.zeros(4, 8)
        result = aria_core.hyperbolic_mobius_add_f32(x, v, 1.0)
        torch.testing.assert_close(result, x, atol=1e-4, rtol=1e-3)

    def test_exp_log_roundtrip(self):
        """log(exp(x)) ≈ x for small x."""
        x = torch.randn(64) * 0.1
        result = aria_core.log_map_f32(aria_core.exp_map_f32(x, 1.0), 1.0)
        torch.testing.assert_close(result, x, atol=1e-3, rtol=1e-2)


# ═══════════════════════════════════════════════════════════════════════
# Math space: Tropical
# ═══════════════════════════════════════════════════════════════════════


class TestTropical:
    def test_tropical_matmul(self):
        """Tropical matmul: C[i,j] = min_k(A[i,k] + B[k,j])"""
        A = torch.randn(4, 8)
        B = torch.randn(8, 3)
        result = aria_core.tropical_matmul_f32(A, B)
        # Reference: min-plus
        expected = torch.full((4, 3), float("inf"))
        for i in range(4):
            for j in range(3):
                for k in range(8):
                    expected[i, j] = min(
                        expected[i, j].item(), A[i, k].item() + B[k, j].item()
                    )
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-3)


# ═══════════════════════════════════════════════════════════════════════
# Reference architecture ops
# ═══════════════════════════════════════════════════════════════════════


class TestRefArch:
    def test_embedding_lookup(self):
        table = torch.randn(100, 32)
        indices = torch.randint(0, 100, (8,), dtype=torch.int32)
        result = aria_core.embedding_lookup_f32(table, indices, None)
        expected = table[indices.long()]
        torch.testing.assert_close(result, expected, atol=ATOL, rtol=RTOL)

    def test_cosine_similarity(self):
        a = torch.randn(2, 4, 8)
        b = torch.randn(2, 4, 8)
        result = aria_core.cosine_similarity_f32(a, b)
        expected = F.cosine_similarity(a, b, dim=-1)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
