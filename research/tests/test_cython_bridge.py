"""Tests for the native Cython bridge (aria_bridge).

Verifies that C kernel dispatch produces correct results for all ops
exposed through the aria_bridge Cython extension.
"""
import sys
import os
import pytest
import numpy as np

# Add the Cython build directory to the path
_cython_dir = os.path.join(os.path.dirname(__file__), '..', 'runtime', 'native', 'cython')
sys.path.insert(0, os.path.abspath(_cython_dir))

try:
    import aria_bridge
    HAS_BRIDGE = True
except ImportError:
    HAS_BRIDGE = False

pytestmark = [pytest.mark.native, pytest.mark.skipif(not HAS_BRIDGE, reason="aria_bridge not built")]


# ── Unary ops ────────────────────────────────────────────────────────

class TestUnaryOps:
    def _rand(self, n=1024):
        return np.random.randn(n).astype(np.float32)

    def test_relu(self):
        x = self._rand()
        y = aria_bridge.dispatch_unary('relu', x)
        np.testing.assert_allclose(y, np.maximum(x, 0), atol=1e-6)

    def test_gelu(self):
        x = self._rand()
        y = aria_bridge.dispatch_unary('gelu', x)
        # Approximate GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        expected = x * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
        np.testing.assert_allclose(y, expected, atol=1e-4)

    def test_silu(self):
        x = self._rand()
        y = aria_bridge.dispatch_unary('silu', x)
        expected = x / (1.0 + np.exp(-x))
        np.testing.assert_allclose(y, expected, atol=1e-5)

    def test_sigmoid(self):
        x = self._rand()
        y = aria_bridge.dispatch_unary('sigmoid', x)
        expected = 1.0 / (1.0 + np.exp(-x))
        np.testing.assert_allclose(y, expected, atol=1e-5)

    def test_tanh(self):
        x = self._rand()
        y = aria_bridge.dispatch_unary('tanh', x)
        np.testing.assert_allclose(y, np.tanh(x), atol=1e-6)

    def test_exp(self):
        x = self._rand()
        y = aria_bridge.dispatch_unary('exp', x)
        np.testing.assert_allclose(y, np.exp(x), rtol=1e-4)

    def test_sin(self):
        x = self._rand()
        y = aria_bridge.dispatch_unary('sin', x)
        np.testing.assert_allclose(y, np.sin(x), atol=1e-6)

    def test_cos(self):
        x = self._rand()
        y = aria_bridge.dispatch_unary('cos', x)
        np.testing.assert_allclose(y, np.cos(x), atol=1e-6)

    def test_square(self):
        x = self._rand()
        y = aria_bridge.dispatch_unary('square', x)
        np.testing.assert_allclose(y, x * x, atol=1e-6)

    def test_abs(self):
        x = self._rand()
        y = aria_bridge.dispatch_unary('abs', x)
        np.testing.assert_allclose(y, np.abs(x), atol=1e-6)

    def test_neg(self):
        x = self._rand()
        y = aria_bridge.dispatch_unary('neg', x)
        np.testing.assert_allclose(y, -x, atol=1e-6)

    def test_reciprocal(self):
        x = np.random.rand(1024).astype(np.float32) + 1e-3
        y = aria_bridge.dispatch_unary('reciprocal', x)
        np.testing.assert_allclose(y, 1.0 / x, atol=1e-6)

    def test_reciprocal_zero_domain_behavior(self):
        x = np.array([1.0, 0.0, -2.0], dtype=np.float32)
        y = aria_bridge.dispatch_unary('reciprocal', x)
        assert np.isinf(y[1])
        np.testing.assert_allclose(y[[0, 2]], np.array([1.0, -0.5], dtype=np.float32), atol=1e-6)

    def test_log(self):
        x = np.random.rand(1024).astype(np.float32) + 1e-3
        y = aria_bridge.dispatch_unary('log', x)
        np.testing.assert_allclose(y, np.log(x), atol=1e-6)

    def test_sqrt(self):
        x = np.random.rand(1024).astype(np.float32)
        y = aria_bridge.dispatch_unary('sqrt', x)
        np.testing.assert_allclose(y, np.sqrt(x), atol=1e-6)

    def test_unsupported_raises(self):
        with pytest.raises(ValueError, match="Unsupported unary op"):
            aria_bridge.dispatch_unary('nonexistent', self._rand())


# ── Binary ops ───────────────────────────────────────────────────────

class TestBinaryOps:
    def _pair(self, n=512):
        return (np.random.randn(n).astype(np.float32),
                np.random.randn(n).astype(np.float32))

    def test_add(self):
        a, b = self._pair()
        np.testing.assert_allclose(aria_bridge.dispatch_binary('add', a, b), a + b, atol=1e-6)

    def test_mul(self):
        a, b = self._pair()
        np.testing.assert_allclose(aria_bridge.dispatch_binary('mul', a, b), a * b, atol=1e-6)

    def test_sub(self):
        a, b = self._pair()
        np.testing.assert_allclose(aria_bridge.dispatch_binary('sub', a, b), a - b, atol=1e-6)

    def test_unsupported_raises(self):
        a, b = self._pair()
        with pytest.raises(ValueError, match="Unsupported binary op"):
            aria_bridge.dispatch_binary('div', a, b)


# ── Linear algebra ───────────────────────────────────────────────────

class TestLinAlg:
    def test_matmul(self):
        A = np.random.randn(8, 16).astype(np.float32)
        B = np.random.randn(16, 12).astype(np.float32)
        C = aria_bridge.dispatch_matmul(A, B)
        np.testing.assert_allclose(C, A @ B, atol=1e-3)

    def test_matmul_shape_mismatch(self):
        A = np.random.randn(4, 8).astype(np.float32)
        B = np.random.randn(4, 6).astype(np.float32)  # B rows != A cols
        with pytest.raises(AssertionError):
            aria_bridge.dispatch_matmul(A, B)

    def test_linear_no_bias(self):
        x = np.random.randn(4, 16).astype(np.float32)
        W = np.random.randn(8, 16).astype(np.float32)
        y = aria_bridge.dispatch_linear(x, W, bias=None)
        expected = x @ W.T
        np.testing.assert_allclose(y, expected, atol=1e-3)

    def test_linear_with_bias(self):
        x = np.random.randn(4, 16).astype(np.float32)
        W = np.random.randn(8, 16).astype(np.float32)
        bias = np.random.randn(8).astype(np.float32)
        y = aria_bridge.dispatch_linear(x, W, bias=bias)
        expected = x @ W.T + bias
        np.testing.assert_allclose(y, expected, atol=1e-3)


# ── Normalization ────────────────────────────────────────────────────

class TestNormalization:
    def test_rmsnorm(self):
        x = np.random.randn(4, 32).astype(np.float32)
        w = np.ones(32, dtype=np.float32)
        y = aria_bridge.dispatch_rmsnorm(x, w, eps=1e-5)
        # Manual RMSNorm
        rms = np.sqrt(np.mean(x ** 2, axis=1, keepdims=True) + 1e-5)
        expected = x / rms * w
        np.testing.assert_allclose(y, expected, atol=1e-4)

    def test_layernorm_identity(self):
        """LayerNorm with weight=1, bias=0 should center and normalize."""
        x = np.random.randn(4, 32).astype(np.float32)
        w = np.ones(32, dtype=np.float32)
        b = np.zeros(32, dtype=np.float32)
        y = aria_bridge.dispatch_layernorm(x, w, b, eps=1e-5)
        mean = x.mean(axis=1, keepdims=True)
        var = x.var(axis=1, keepdims=True)
        expected = (x - mean) / np.sqrt(var + 1e-5)
        np.testing.assert_allclose(y, expected, atol=1e-4)

    def test_layernorm_with_affine(self):
        """LayerNorm with non-trivial weight and bias."""
        x = np.random.randn(4, 16).astype(np.float32)
        w = np.random.randn(16).astype(np.float32) * 0.5 + 1.0
        b = np.random.randn(16).astype(np.float32) * 0.1
        y = aria_bridge.dispatch_layernorm(x, w, b, eps=1e-5)
        mean = x.mean(axis=1, keepdims=True)
        var = x.var(axis=1, keepdims=True)
        expected = (x - mean) / np.sqrt(var + 1e-5) * w + b
        np.testing.assert_allclose(y, expected, atol=1e-4)


# ── Softmax ──────────────────────────────────────────────────────────

class TestSoftmax:
    def test_softmax_basic(self):
        x = np.random.randn(4, 16).astype(np.float32)
        y = aria_bridge.dispatch_softmax(x)
        e = np.exp(x - x.max(axis=1, keepdims=True))
        expected = e / e.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(y, expected, atol=1e-5)

    def test_softmax_sums_to_one(self):
        x = np.random.randn(8, 64).astype(np.float32)
        y = aria_bridge.dispatch_softmax(x)
        row_sums = y.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(8), atol=1e-5)

    def test_softmax_nonnegative(self):
        x = np.random.randn(4, 32).astype(np.float32)
        y = aria_bridge.dispatch_softmax(x)
        assert np.all(y >= 0)


# ── Structural ops ───────────────────────────────────────────────────

class TestStructuralOps:
    def test_transpose2d(self):
        x = np.random.randn(5, 7).astype(np.float32)
        y = aria_bridge.dispatch_transpose2d(x)
        assert y.shape == (7, 5)
        np.testing.assert_allclose(y, x.T, atol=1e-6)

    def test_transpose2d_square(self):
        x = np.random.randn(4, 4).astype(np.float32)
        y = aria_bridge.dispatch_transpose2d(x)
        np.testing.assert_allclose(y, x.T, atol=1e-6)


# ── Reductions ───────────────────────────────────────────────────────

class TestReductions:
    def test_sum(self):
        x = np.random.randn(1024).astype(np.float32)
        s = aria_bridge.native_sum(x)
        np.testing.assert_allclose(s, x.sum(), atol=0.05)

    def test_mean(self):
        x = np.random.randn(1024).astype(np.float32)
        m = aria_bridge.native_mean(x)
        np.testing.assert_allclose(m, x.mean(), atol=0.001)


# ── Registry ─────────────────────────────────────────────────────────

class TestRegistry:
    def test_list_native_ops(self):
        ops = aria_bridge.list_native_ops()
        assert isinstance(ops, list)
        assert len(ops) >= 15
        for name in ['relu', 'gelu', 'square', 'abs', 'neg', 'reciprocal', 'log', 'sqrt', 'sin', 'cos', 'add', 'matmul', 'softmax', 'layernorm', 'transpose2d']:
            assert name in ops, f"{name} missing from native ops"

    def test_is_native_true(self):
        for op in ['relu', 'square', 'abs', 'neg', 'reciprocal', 'log', 'sqrt', 'sin', 'cos', 'add', 'matmul', 'softmax', 'layernorm', 'transpose2d', 'rmsnorm']:
            assert aria_bridge.is_native(op), f"{op} should be native"

    def test_is_native_false(self):
        assert not aria_bridge.is_native('nonexistent')
        assert not aria_bridge.is_native('attention')
