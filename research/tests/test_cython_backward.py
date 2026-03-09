"""Tests for the native Cython backward (gradient) kernels.

Verifies that backward dispatch through aria_bridge produces correct
gradients for all supported ops, validated against numpy reference
implementations.
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


# ── Unary backward ops ───────────────────────────────────────────────

class TestUnaryBackward:
    def _rand(self, n=512):
        return np.random.randn(n).astype(np.float32)

    def test_relu_backward(self):
        x = self._rand()
        grad_out = self._rand()
        grad_in = aria_bridge.dispatch_unary_backward('relu', grad_out, x)
        # relu'(x) = 1 if x > 0, else 0
        expected = grad_out * (x > 0).astype(np.float32)
        np.testing.assert_allclose(grad_in, expected, atol=1e-6)

    def test_relu_backward_zeros(self):
        """ReLU backward at zero should produce zero gradient."""
        x = np.zeros(64, dtype=np.float32)
        grad_out = np.ones(64, dtype=np.float32)
        grad_in = aria_bridge.dispatch_unary_backward('relu', grad_out, x)
        np.testing.assert_allclose(grad_in, np.zeros(64), atol=1e-6)

    def test_sigmoid_backward(self):
        x = self._rand()
        # Sigmoid backward uses forward *output*
        sig_out = 1.0 / (1.0 + np.exp(-x))
        sig_out = sig_out.astype(np.float32)
        grad_out = self._rand()
        grad_in = aria_bridge.dispatch_unary_backward('sigmoid', grad_out, sig_out)
        # sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        expected = grad_out * sig_out * (1.0 - sig_out)
        np.testing.assert_allclose(grad_in, expected, atol=1e-5)

    def test_tanh_backward(self):
        x = self._rand()
        # Tanh backward uses forward *output*
        tanh_out = np.tanh(x).astype(np.float32)
        grad_out = self._rand()
        grad_in = aria_bridge.dispatch_unary_backward('tanh', grad_out, tanh_out)
        # tanh'(x) = 1 - tanh(x)^2
        expected = grad_out * (1.0 - tanh_out ** 2)
        np.testing.assert_allclose(grad_in, expected, atol=1e-5)

    def test_gelu_backward(self):
        x = self._rand()
        grad_out = self._rand()
        grad_in = aria_bridge.dispatch_unary_backward('gelu', grad_out, x)
        # GELU'(x) = 0.5 * (1 + tanh(a)) + 0.5 * x * (1 - tanh(a)^2) * a'
        # where a = sqrt(2/pi) * (x + 0.044715 * x^3)
        # Numerical gradient check instead of exact formula
        eps = 1e-3
        x_plus = x + eps
        x_minus = x - eps
        gelu_plus = x_plus * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x_plus + 0.044715 * x_plus**3)))
        gelu_minus = x_minus * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x_minus + 0.044715 * x_minus**3)))
        numerical_grad = (gelu_plus - gelu_minus) / (2 * eps)
        expected = grad_out * numerical_grad
        np.testing.assert_allclose(grad_in, expected, atol=1e-2)

    def test_silu_backward(self):
        x = self._rand()
        grad_out = self._rand()
        grad_in = aria_bridge.dispatch_unary_backward('silu', grad_out, x)
        # SiLU'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        #          = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        sig = 1.0 / (1.0 + np.exp(-x))
        expected = grad_out * (sig + x * sig * (1.0 - sig))
        np.testing.assert_allclose(grad_in, expected, atol=1e-4)

    def test_unsupported_raises(self):
        with pytest.raises(ValueError, match="Unsupported unary backward op"):
            aria_bridge.dispatch_unary_backward('exp', self._rand(), self._rand())

    def test_shape_mismatch_raises(self):
        with pytest.raises(AssertionError):
            aria_bridge.dispatch_unary_backward('relu',
                                                 np.ones(10, dtype=np.float32),
                                                 np.ones(20, dtype=np.float32))


# ── Binary backward ops ──────────────────────────────────────────────

class TestBinaryBackward:
    def _triple(self, n=256):
        return (np.random.randn(n).astype(np.float32),
                np.random.randn(n).astype(np.float32),
                np.random.randn(n).astype(np.float32))

    def test_add_backward(self):
        grad_out, a, b = self._triple()
        grad_a, grad_b = aria_bridge.dispatch_binary_backward('add', grad_out, a, b)
        # d/da (a+b) = 1, d/db (a+b) = 1
        np.testing.assert_allclose(grad_a, grad_out, atol=1e-6)
        np.testing.assert_allclose(grad_b, grad_out, atol=1e-6)

    def test_sub_backward(self):
        grad_out, a, b = self._triple()
        grad_a, grad_b = aria_bridge.dispatch_binary_backward('sub', grad_out, a, b)
        # d/da (a-b) = 1, d/db (a-b) = -1
        np.testing.assert_allclose(grad_a, grad_out, atol=1e-6)
        np.testing.assert_allclose(grad_b, -grad_out, atol=1e-6)

    def test_mul_backward(self):
        grad_out, a, b = self._triple()
        grad_a, grad_b = aria_bridge.dispatch_binary_backward('mul', grad_out, a, b)
        # d/da (a*b) = b, d/db (a*b) = a
        np.testing.assert_allclose(grad_a, grad_out * b, atol=1e-5)
        np.testing.assert_allclose(grad_b, grad_out * a, atol=1e-5)

    def test_unsupported_raises(self):
        n = 16
        with pytest.raises(ValueError, match="Unsupported binary backward op"):
            aria_bridge.dispatch_binary_backward(
                'div',
                np.ones(n, dtype=np.float32),
                np.ones(n, dtype=np.float32),
                np.ones(n, dtype=np.float32),
            )

    def test_returns_tuple(self):
        """Binary backward should always return a tuple of two arrays."""
        n = 32
        grad_out = np.ones(n, dtype=np.float32)
        a = np.ones(n, dtype=np.float32)
        b = np.ones(n, dtype=np.float32)
        result = aria_bridge.dispatch_binary_backward('add', grad_out, a, b)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0].shape == (n,)
        assert result[1].shape == (n,)


# ── Matmul backward ──────────────────────────────────────────────────

class TestMatmulBackward:
    def test_matmul_backward_basic(self):
        M, K, N = 8, 16, 12
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        grad_out = np.random.randn(M, N).astype(np.float32)

        grad_A, grad_B = aria_bridge.dispatch_matmul_backward(grad_out, A, B)

        # grad_A = grad_out @ B^T
        expected_grad_A = grad_out @ B.T
        # grad_B = A^T @ grad_out
        expected_grad_B = A.T @ grad_out

        np.testing.assert_allclose(grad_A, expected_grad_A, atol=1e-3)
        np.testing.assert_allclose(grad_B, expected_grad_B, atol=1e-3)

    def test_matmul_backward_square(self):
        n = 8
        A = np.random.randn(n, n).astype(np.float32)
        B = np.random.randn(n, n).astype(np.float32)
        grad_out = np.random.randn(n, n).astype(np.float32)

        grad_A, grad_B = aria_bridge.dispatch_matmul_backward(grad_out, A, B)
        np.testing.assert_allclose(grad_A, grad_out @ B.T, atol=1e-3)
        np.testing.assert_allclose(grad_B, A.T @ grad_out, atol=1e-3)

    def test_matmul_backward_shapes(self):
        M, K, N = 4, 6, 10
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        grad_out = np.random.randn(M, N).astype(np.float32)

        grad_A, grad_B = aria_bridge.dispatch_matmul_backward(grad_out, A, B)
        assert grad_A.shape == (M, K)
        assert grad_B.shape == (K, N)

    def test_matmul_backward_returns_tuple(self):
        A = np.eye(4, dtype=np.float32)
        B = np.eye(4, dtype=np.float32)
        grad_out = np.ones((4, 4), dtype=np.float32)
        result = aria_bridge.dispatch_matmul_backward(grad_out, A, B)
        assert isinstance(result, tuple)
        assert len(result) == 2


# ── has_backward registry ────────────────────────────────────────────

class TestHasBackward:
    def test_supported_ops(self):
        for op in ['relu', 'gelu', 'silu', 'sigmoid', 'tanh', 'add', 'mul', 'sub', 'matmul']:
            assert aria_bridge.has_backward(op), f"{op} should have backward"

    def test_unsupported_ops(self):
        for op in ['exp', 'linear', 'nonexistent']:
            assert not aria_bridge.has_backward(op), f"{op} should not have backward"

    def test_norm_ops_have_backward(self):
        for op in ['softmax', 'layernorm', 'rmsnorm']:
            assert aria_bridge.has_backward(op), f"{op} should have backward"


# ── dispatch_op_backward_native (Python-level dispatch) ──────────────

class TestDispatchOpBackwardNative:
    """Test the Python-level dispatch in native_runner.py."""

    @pytest.fixture(autouse=True)
    def _import_dispatch(self):
        # Add research/ to path for scientist module
        research_dir = os.path.join(os.path.dirname(__file__), '..')
        sys.path.insert(0, os.path.abspath(research_dir))
        try:
            from scientist.native_runner import dispatch_op_backward_native
            self.dispatch = dispatch_op_backward_native
            self.available = True
        except ImportError:
            self.available = False

    def test_relu_backward_via_dispatch(self):
        if not self.available:
            pytest.skip("dispatch_op_backward_native not importable")
        x = np.random.randn(128).astype(np.float32)
        grad_out = np.random.randn(128).astype(np.float32)
        grad_in = self.dispatch('relu', grad_out, x)
        expected = grad_out * (x > 0).astype(np.float32)
        np.testing.assert_allclose(grad_in, expected, atol=1e-6)

    def test_mul_backward_via_dispatch(self):
        if not self.available:
            pytest.skip("dispatch_op_backward_native not importable")
        a = np.random.randn(64).astype(np.float32)
        b = np.random.randn(64).astype(np.float32)
        grad_out = np.random.randn(64).astype(np.float32)
        grad_a, grad_b = self.dispatch('mul', grad_out, a, b)
        np.testing.assert_allclose(grad_a, grad_out * b, atol=1e-5)
        np.testing.assert_allclose(grad_b, grad_out * a, atol=1e-5)

    def test_matmul_backward_via_dispatch(self):
        if not self.available:
            pytest.skip("dispatch_op_backward_native not importable")
        A = np.random.randn(4, 8).astype(np.float32)
        B = np.random.randn(8, 6).astype(np.float32)
        grad_out = np.random.randn(4, 6).astype(np.float32)
        grad_A, grad_B = self.dispatch('matmul', grad_out, A, B)
        np.testing.assert_allclose(grad_A, grad_out @ B.T, atol=1e-3)
        np.testing.assert_allclose(grad_B, A.T @ grad_out, atol=1e-3)

    def test_unsupported_op_raises(self):
        if not self.available:
            pytest.skip("dispatch_op_backward_native not importable")
        with pytest.raises(ValueError, match="Unsupported op for native backward"):
            self.dispatch('nonexistent_op', np.ones(4, dtype=np.float32))

    def test_wrong_saved_tensor_count_raises(self):
        if not self.available:
            pytest.skip("dispatch_op_backward_native not importable")
        with pytest.raises(ValueError, match="expects 1 saved tensor"):
            self.dispatch('relu', np.ones(4, dtype=np.float32),
                          np.ones(4, dtype=np.float32),
                          np.ones(4, dtype=np.float32))
