"""Tests for Cython bridge integration in native_runner.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from research.scientist.native_runner import (
    _reset_cython_bridge_cache,
    _try_import_cython_bridge,
    _check_native_op_support,
    _activate_selective_native_dispatch,
    dispatch_op_native,
)
from research.tests.conftest import make_fake_graph

pytestmark = pytest.mark.native


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_bridge():
    """Create a mock that mimics the aria_bridge module API."""
    bridge = MagicMock()
    bridge.is_native = MagicMock(
        side_effect=lambda op: (
            op
            in {
                "relu",
                "gelu",
                "silu",
                "square",
                "abs",
                "neg",
                "reciprocal",
                "log",
                "sqrt",
                "sin",
                "cos",
                "sigmoid",
                "tanh",
                "exp",
                "add",
                "mul",
                "sub",
                "matmul",
                "linear",
                "rmsnorm",
                "layernorm",
                "softmax",
                "transpose2d",
                "sum",
                "mean",
                "concat",
                "split",
            }
        )
    )
    bridge.dispatch_unary = MagicMock(
        side_effect=lambda op, x: (
            np.maximum(x, 0) if op == "relu" else (x * x if op == "square" else x)
        )
    )
    bridge.dispatch_binary = MagicMock(
        side_effect=lambda op, a, b: a + b if op == "add" else a * b
    )
    bridge.dispatch_matmul = MagicMock(side_effect=lambda A, B: A @ B)
    bridge.dispatch_softmax = MagicMock(
        side_effect=lambda x: x  # stub
    )
    bridge.dispatch_transpose2d = MagicMock(side_effect=lambda x: x.T)
    bridge.dispatch_linear = MagicMock(return_value=np.zeros((2, 4), dtype=np.float32))
    bridge.dispatch_rmsnorm = MagicMock(return_value=np.zeros((2, 4), dtype=np.float32))
    bridge.dispatch_layernorm = MagicMock(
        return_value=np.zeros((2, 4), dtype=np.float32)
    )
    return bridge


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCythonBridgeImport:
    def teardown_method(self):
        _reset_cython_bridge_cache()

    def test_cython_bridge_import(self):
        """_try_import_cython_bridge returns a module (or None if .so not built)."""
        _reset_cython_bridge_cache()
        result = _try_import_cython_bridge()
        # The .so is built, so we expect a module; but even if not, it must
        # return either a module or None (never raise).
        assert result is None or hasattr(result, "is_native")

    def test_cython_bridge_import_caches(self):
        """Repeated calls return the same cached result."""
        _reset_cython_bridge_cache()
        r1 = _try_import_cython_bridge()
        r2 = _try_import_cython_bridge()
        assert r1 is r2


class TestOpSupportUsesCython:
    def teardown_method(self):
        _reset_cython_bridge_cache()

    def test_op_support_uses_cython(self):
        """When Cython bridge is available, _check_native_op_support uses it
        instead of ctypes nk_is_registered."""
        fake_bridge = _make_fake_bridge()
        _reset_cython_bridge_cache()

        graphs = [make_fake_graph(["relu", "matmul", "custom_op"])]

        with patch(
            "research.scientist.native.dispatch._try_import_cython_bridge",
            return_value=fake_bridge,
        ):
            result = _check_native_op_support(graphs, native_lib=None)

        assert "relu" in result["supported"]
        assert "matmul" in result["supported"]
        assert "custom_op" in result["unsupported"]
        assert result["native_coverage"] == pytest.approx(2.0 / 3.0)
        # relu and matmul are in _all_known_native quick-check set, so only
        # custom_op falls through to bridge.is_native().
        assert fake_bridge.is_native.call_count == 1

    def test_op_support_resolves_aliases_and_composites(self):
        """Alias/composite ops should count as supported when backing kernels exist."""
        fake_bridge = _make_fake_bridge()
        _reset_cython_bridge_cache()

        graphs = [
            make_fake_graph(["linear_proj", "softmax_last", "square", "custom_op"])
        ]

        with patch(
            "research.scientist.native.dispatch._try_import_cython_bridge",
            return_value=fake_bridge,
        ):
            result = _check_native_op_support(graphs, native_lib=None)

        assert "linear_proj" in result["supported"]
        assert "softmax_last" in result["supported"]
        assert "square" in result["supported"]
        assert result["unsupported"] == ["custom_op"]

    def test_op_support_falls_back_to_ctypes_when_no_cython(self):
        """When Cython bridge returns None, ctypes path is used."""
        fake_lib = MagicMock()
        fake_lib.nk_is_registered = MagicMock(return_value=1)
        _reset_cython_bridge_cache()

        # Use an op NOT in _all_known_native to force ctypes query.
        graphs = [make_fake_graph(["exotic_custom_op"])]

        with patch(
            "research.scientist.native.dispatch._try_import_cython_bridge",
            return_value=None,
        ):
            result = _check_native_op_support(graphs, native_lib=fake_lib)

        assert "exotic_custom_op" in result["supported"]
        assert fake_lib.nk_is_registered.called


class TestDispatchOpNative:
    def teardown_method(self):
        _reset_cython_bridge_cache()

    def test_dispatch_op_native_unary(self):
        """dispatch_op_native routes relu through the Cython bridge."""
        fake_bridge = _make_fake_bridge()
        x = np.array([-1.0, 0.0, 2.0, 3.5], dtype=np.float32)

        with patch(
            "research.scientist.native.dispatch._try_import_cython_bridge",
            return_value=fake_bridge,
        ):
            result = dispatch_op_native("relu", x)

        np.testing.assert_array_equal(
            result, np.array([0.0, 0.0, 2.0, 3.5], dtype=np.float32)
        )
        fake_bridge.dispatch_unary.assert_called_once_with("relu", x)

    def test_dispatch_op_native_sin(self):
        """dispatch_op_native routes sin through the Cython bridge."""
        fake_bridge = _make_fake_bridge()
        x = np.array([0.0, np.pi / 2], dtype=np.float32)

        with patch(
            "research.scientist.native.dispatch._try_import_cython_bridge",
            return_value=fake_bridge,
        ):
            dispatch_op_native("sin", x)

        fake_bridge.dispatch_unary.assert_called_with("sin", x)

    def test_dispatch_op_native_log(self):
        """dispatch_op_native routes log through the Cython bridge."""
        fake_bridge = _make_fake_bridge()
        x = np.array([1.0, 2.0], dtype=np.float32)

        with patch(
            "research.scientist.native.dispatch._try_import_cython_bridge",
            return_value=fake_bridge,
        ):
            dispatch_op_native("log", x)

        fake_bridge.dispatch_unary.assert_called_with("log", x)

    def test_dispatch_op_native_sqrt(self):
        """dispatch_op_native routes sqrt through the Cython bridge."""
        fake_bridge = _make_fake_bridge()
        x = np.array([0.0, 4.0], dtype=np.float32)

        with patch(
            "research.scientist.native.dispatch._try_import_cython_bridge",
            return_value=fake_bridge,
        ):
            dispatch_op_native("sqrt", x)

        fake_bridge.dispatch_unary.assert_called_with("sqrt", x)

    def test_dispatch_op_native_abs(self):
        """dispatch_op_native routes abs through the Cython bridge."""
        fake_bridge = _make_fake_bridge()
        x = np.array([-2.0, 3.0], dtype=np.float32)

        with patch(
            "research.scientist.native.dispatch._try_import_cython_bridge",
            return_value=fake_bridge,
        ):
            dispatch_op_native("abs", x)

        fake_bridge.dispatch_unary.assert_called_with("abs", x)

    def test_dispatch_op_native_neg(self):
        """dispatch_op_native routes neg through the Cython bridge."""
        fake_bridge = _make_fake_bridge()
        x = np.array([-2.0, 3.0], dtype=np.float32)

        with patch(
            "research.scientist.native.dispatch._try_import_cython_bridge",
            return_value=fake_bridge,
        ):
            dispatch_op_native("neg", x)

        fake_bridge.dispatch_unary.assert_called_with("neg", x)

    def test_dispatch_op_native_reciprocal(self):
        """dispatch_op_native routes reciprocal through the Cython bridge."""
        fake_bridge = _make_fake_bridge()
        x = np.array([1.0, 2.0], dtype=np.float32)

        with patch(
            "research.scientist.native.dispatch._try_import_cython_bridge",
            return_value=fake_bridge,
        ):
            dispatch_op_native("reciprocal", x)

        fake_bridge.dispatch_unary.assert_called_with("reciprocal", x)

    def test_dispatch_op_native_binary(self):
        """dispatch_op_native routes add through the Cython bridge."""
        fake_bridge = _make_fake_bridge()
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([10.0, 20.0], dtype=np.float32)

        with patch(
            "research.scientist.native.dispatch._try_import_cython_bridge",
            return_value=fake_bridge,
        ):
            result = dispatch_op_native("add", a, b)

        np.testing.assert_array_equal(result, np.array([11.0, 22.0], dtype=np.float32))
        fake_bridge.dispatch_binary.assert_called_once_with("add", a, b)

    def test_dispatch_op_native_matmul(self):
        """dispatch_op_native routes matmul through the Cython bridge."""
        fake_bridge = _make_fake_bridge()
        A = np.eye(3, dtype=np.float32)
        B = np.ones((3, 2), dtype=np.float32)

        with patch(
            "research.scientist.native.dispatch._try_import_cython_bridge",
            return_value=fake_bridge,
        ):
            result = dispatch_op_native("matmul", A, B)

        np.testing.assert_array_equal(result, A @ B)
        fake_bridge.dispatch_matmul.assert_called_once_with(A, B)

    def test_dispatch_op_native_linear_proj_alias(self):
        """linear_proj should dispatch through the linear native kernel."""
        fake_bridge = _make_fake_bridge()
        x = np.ones((2, 4), dtype=np.float32)
        w = np.ones((3, 4), dtype=np.float32)

        with patch(
            "research.scientist.native.dispatch._try_import_cython_bridge",
            return_value=fake_bridge,
        ):
            dispatch_op_native("linear_proj", x, w)

        fake_bridge.dispatch_linear.assert_called_once_with(x, w, bias=None)

    def test_dispatch_op_native_square_uses_unary_kernel(self):
        """square should use the dedicated unary square kernel when available."""
        fake_bridge = _make_fake_bridge()
        x = np.array([2.0, -3.0], dtype=np.float32)

        with patch(
            "research.scientist.native.dispatch._try_import_cython_bridge",
            return_value=fake_bridge,
        ):
            result = dispatch_op_native("square", x)

        fake_bridge.dispatch_unary.assert_called_with("square", x)
        np.testing.assert_array_equal(result, x * x)

    def test_dispatch_op_native_square_falls_back_to_mul(self):
        """square falls back to mul(x, x) for older bridges lacking unary square."""
        fake_bridge = _make_fake_bridge()
        x = np.array([2.0, -3.0], dtype=np.float32)

        def _dispatch_unary(op, tensor):
            if op == "square":
                raise ValueError("Unsupported unary op: square")
            if op == "relu":
                return np.maximum(tensor, 0)
            return tensor

        fake_bridge.dispatch_unary = MagicMock(side_effect=_dispatch_unary)

        with patch(
            "research.scientist.native.dispatch._try_import_cython_bridge",
            return_value=fake_bridge,
        ):
            result = dispatch_op_native("square", x)

        fake_bridge.dispatch_binary.assert_called_once_with("mul", x, x)
        np.testing.assert_array_equal(result, x * x)

    def test_dispatch_op_native_unsupported_raises(self):
        """dispatch_op_native raises ValueError for unknown ops."""
        fake_bridge = _make_fake_bridge()

        with patch(
            "research.scientist.native.dispatch._try_import_cython_bridge",
            return_value=fake_bridge,
        ):
            with pytest.raises(ValueError, match="Unsupported op.*fancy_attention"):
                dispatch_op_native("fancy_attention", np.zeros(4, dtype=np.float32))

    def test_dispatch_op_native_raises_when_no_bridge(self):
        """dispatch_op_native raises RuntimeError when Cython bridge is unavailable."""
        _reset_cython_bridge_cache()
        with patch(
            "research.scientist.native.dispatch._try_import_cython_bridge",
            return_value=None,
        ):
            with pytest.raises(RuntimeError, match="Cython bridge.*not available"):
                dispatch_op_native("relu", np.zeros(4, dtype=np.float32))


class TestSelectiveActivationUsesCython:
    def teardown_method(self):
        _reset_cython_bridge_cache()

    def test_selective_activation_uses_cython(self):
        """_activate_selective_native_dispatch prefers Cython bridge over ctypes."""
        fake_bridge = _make_fake_bridge()
        _reset_cython_bridge_cache()

        with (
            patch(
                "research.scientist.native.dispatch._try_import_cython_bridge",
                return_value=fake_bridge,
            ),
            patch(
                "research.scientist.native.dispatch._try_import_rust_scheduler",
                return_value=None,
            ),
        ):
            result = _activate_selective_native_dispatch(native_lib=None)

        assert result["activated"] is True
        assert result["reason"] == "ok"
        assert result.get("dispatch_backend") == "cython"
        # Verify Cython dispatch was used (not ctypes).
        assert fake_bridge.dispatch_unary.called
        assert fake_bridge.dispatch_binary.called

    def test_selective_activation_falls_back_to_ctypes(self):
        """When Cython bridge is None, ctypes path is used."""
        _reset_cython_bridge_cache()

        class FakeNativeLib:
            @staticmethod
            def aria_relu_f32(x, y, n):
                for i in range(int(n)):
                    y[i] = x[i] if x[i] > 0 else 0.0

            @staticmethod
            def aria_add_f32(a, b, y, n):
                for i in range(int(n)):
                    y[i] = a[i] + b[i]

        with patch(
            "research.scientist.native.dispatch._try_import_cython_bridge",
            return_value=None,
        ):
            result = _activate_selective_native_dispatch(FakeNativeLib())

        assert result["activated"] is True
        assert result["reason"] == "ok"
        assert result.get("dispatch_backend") == "ctypes"
