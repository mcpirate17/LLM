"""Native C kernel autograd integration for training support.

Provides ``torch.autograd.Function`` subclasses that route forward ops through
the Cython bridge (aria_bridge) and backward ops through the corresponding C
gradient kernels.  This enables gradient computation to flow through native
kernels during training -- not just inference.

Usage::

    from scientist.native_autograd import native_autograd_dispatch

    # Returns a torch tensor with grad_fn attached when inputs require grad.
    result = native_autograd_dispatch("relu", x)

The dispatch function ``native_autograd_dispatch`` selects the correct
autograd Function subclass, converts torch tensors to contiguous numpy
arrays for the C kernels, and wraps the results back into torch tensors.
"""

from __future__ import annotations

import numpy as np
import torch

from .native_runner import dispatch_op_native, dispatch_op_backward_native


# ── Helpers ──────────────────────────────────────────────────────────


def _to_np(t: torch.Tensor) -> np.ndarray:
    """Detach a torch tensor and return a contiguous float32 numpy array."""
    return t.detach().cpu().contiguous().numpy().astype(np.float32)


def _to_np_flat(t: torch.Tensor) -> np.ndarray:
    """Detach and flatten to a 1-D float32 numpy array.

    The Cython bridge's ``dispatch_unary`` / ``dispatch_binary`` /
    ``dispatch_unary_backward`` / ``dispatch_binary_backward`` all expect
    1-D buffers.  We flatten here and reshape back to the original tensor
    shape after the C kernel call.
    """
    return t.detach().cpu().contiguous().numpy().astype(np.float32).ravel()


def _to_tensor(arr: np.ndarray, *, device: torch.device) -> torch.Tensor:
    """Wrap a numpy array as a torch tensor on the given device."""
    return torch.from_numpy(np.asarray(arr, dtype=np.float32)).to(device)


# ── Factory functions for common autograd patterns ─────────────────


def _make_unary_save_input(op_name: str) -> type:
    """Create a torch.autograd.Function for a unary op that saves the input."""

    class _F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            shape = x.shape
            x_np = _to_np_flat(x)
            ctx.save_for_backward(x)
            y_np = dispatch_op_native(op_name, x_np)
            return _to_tensor(y_np, device=x.device).reshape(shape)

        @staticmethod
        def backward(ctx, grad_output):
            (x,) = ctx.saved_tensors
            shape = grad_output.shape
            grad_np = _to_np_flat(grad_output)
            x_np = _to_np_flat(x)
            grad_in_np = dispatch_op_backward_native(op_name, grad_np, x_np)
            return _to_tensor(grad_in_np, device=grad_output.device).reshape(shape)

    _F.__name__ = _F.__qualname__ = f"Native{op_name.capitalize()}"
    return _F


def _make_unary_save_output(op_name: str) -> type:
    """Create a torch.autograd.Function for a unary op that saves the output."""

    class _F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            shape = x.shape
            x_np = _to_np_flat(x)
            y_np = dispatch_op_native(op_name, x_np)
            y = _to_tensor(y_np, device=x.device).reshape(shape)
            ctx.save_for_backward(y)
            return y

        @staticmethod
        def backward(ctx, grad_output):
            (y,) = ctx.saved_tensors
            shape = grad_output.shape
            grad_np = _to_np_flat(grad_output)
            y_np = _to_np_flat(y)
            grad_in_np = dispatch_op_backward_native(op_name, grad_np, y_np)
            return _to_tensor(grad_in_np, device=grad_output.device).reshape(shape)

    _F.__name__ = _F.__qualname__ = f"Native{op_name.capitalize()}"
    return _F


def _make_binary_flat(op_name: str) -> type:
    """Create a torch.autograd.Function for a binary op using flat buffers."""

    class _F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a, b):
            shape = a.shape
            a_np = _to_np_flat(a)
            b_np = _to_np_flat(b)
            ctx.save_for_backward(a, b)
            y_np = dispatch_op_native(op_name, a_np, b_np)
            return _to_tensor(y_np, device=a.device).reshape(shape)

        @staticmethod
        def backward(ctx, grad_output):
            a, b = ctx.saved_tensors
            shape = grad_output.shape
            grad_np = _to_np_flat(grad_output)
            a_np = _to_np_flat(a)
            b_np = _to_np_flat(b)
            grad_a_np, grad_b_np = dispatch_op_backward_native(
                op_name, grad_np, a_np, b_np
            )
            dev = grad_output.device
            return _to_tensor(grad_a_np, device=dev).reshape(shape), _to_tensor(
                grad_b_np, device=dev
            ).reshape(shape)

    _F.__name__ = _F.__qualname__ = f"Native{op_name.capitalize()}"
    return _F


# ── Unary ops (save input) ────────────────────────────────────────────

NativeRelu = _make_unary_save_input("relu")
NativeGelu = _make_unary_save_input("gelu")
NativeSilu = _make_unary_save_input("silu")

# ── Unary ops (save output) ───────────────────────────────────────────

NativeSigmoid = _make_unary_save_output("sigmoid")
NativeTanh = _make_unary_save_output("tanh")

# ── Binary ops (flat buffers) ─────────────────────────────────────────

NativeAdd = _make_binary_flat("add")
NativeMul = _make_binary_flat("mul")
NativeSub = _make_binary_flat("sub")
NativeMaximum = _make_binary_flat("maximum")
NativeMinimum = _make_binary_flat("minimum")
NativeDivSafe = _make_binary_flat("div_safe")


# ── Special ops (unique forward/backward patterns) ────────────────────


class NativeMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        A_np = _to_np(A)
        B_np = _to_np(B)
        ctx.save_for_backward(A, B)
        C_np = dispatch_op_native("matmul", A_np, B_np)
        return _to_tensor(C_np, device=A.device)

    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        grad_np = _to_np(grad_output)
        A_np = _to_np(A)
        B_np = _to_np(B)
        grad_A_np, grad_B_np = dispatch_op_backward_native(
            "matmul", grad_np, A_np, B_np
        )
        dev = grad_output.device
        return _to_tensor(grad_A_np, device=dev), _to_tensor(grad_B_np, device=dev)


class NativeSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x_np = _to_np(x)
        y_np = dispatch_op_native("softmax", x_np)
        y = _to_tensor(y_np, device=x.device)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        (y,) = ctx.saved_tensors
        grad_np = _to_np(grad_output)
        y_np = _to_np(y)
        grad_in_np = dispatch_op_backward_native("softmax", grad_np, y_np)
        return _to_tensor(grad_in_np, device=grad_output.device)


class NativeLayernorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, beta):
        x_np = _to_np(x)
        gamma_np = _to_np(gamma)
        beta_np = _to_np(beta)
        y_np = dispatch_op_native("layernorm", x_np, gamma_np, beta_np)
        y = _to_tensor(y_np, device=x.device)
        ctx.save_for_backward(x, gamma)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, gamma = ctx.saved_tensors
        grad_np = _to_np(grad_output)
        x_np = _to_np(x)
        gamma_np = _to_np(gamma)
        grad_in_np, grad_gamma_np, grad_beta_np = dispatch_op_backward_native(
            "layernorm", grad_np, x_np, gamma_np
        )
        dev = grad_output.device
        return (
            _to_tensor(grad_in_np, device=dev),
            _to_tensor(grad_gamma_np, device=dev),
            _to_tensor(grad_beta_np, device=dev),
        )


class NativeRmsnorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma):
        x_np = _to_np(x)
        gamma_np = _to_np(gamma)
        y_np = dispatch_op_native("rmsnorm", x_np, gamma_np)
        y = _to_tensor(y_np, device=x.device)
        ctx.save_for_backward(x, gamma)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, gamma = ctx.saved_tensors
        grad_np = _to_np(grad_output)
        x_np = _to_np(x)
        gamma_np = _to_np(gamma)
        grad_in_np, grad_gamma_np = dispatch_op_backward_native(
            "rmsnorm", grad_np, x_np, gamma_np
        )
        dev = grad_output.device
        return (
            _to_tensor(grad_in_np, device=dev),
            _to_tensor(grad_gamma_np, device=dev),
        )


class NativeSignSte(torch.autograd.Function):
    """Sign with straight-through estimator: forward = sign, backward = identity."""

    @staticmethod
    def forward(ctx, x):
        shape = x.shape
        x_np = _to_np_flat(x)
        y_np = dispatch_op_native("sign_ste", x_np)
        return _to_tensor(y_np, device=x.device).reshape(shape)

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through: gradient passes through unchanged
        return grad_output


# ── Registry / dispatch ─────────────────────────────────────────────

_NATIVE_AUTOGRAD_OPS = {
    "relu": NativeRelu,
    "sigmoid": NativeSigmoid,
    "tanh": NativeTanh,
    "gelu": NativeGelu,
    "silu": NativeSilu,
    "add": NativeAdd,
    "mul": NativeMul,
    "sub": NativeSub,
    "maximum": NativeMaximum,
    "minimum": NativeMinimum,
    "div_safe": NativeDivSafe,
    "sign_ste": NativeSignSte,
    "matmul": NativeMatmul,
    "softmax": NativeSoftmax,
    "layernorm": NativeLayernorm,
    "rmsnorm": NativeRmsnorm,
}

# Ops whose backward kernels exist in the C library.
NATIVE_AUTOGRAD_SUPPORTED_OPS = frozenset(_NATIVE_AUTOGRAD_OPS.keys())


def native_autograd_dispatch(op_name: str, *inputs: torch.Tensor) -> torch.Tensor:
    """Dispatch an op through the native autograd Function.

    Args:
        op_name: Name of the primitive op (e.g. "relu", "add", "matmul").
        *inputs: One or more torch tensors.

    Returns:
        Result tensor with a grad_fn if any input requires grad.

    Raises:
        ValueError: If ``op_name`` has no native autograd Function.
    """
    fn_cls = _NATIVE_AUTOGRAD_OPS.get(op_name)
    if fn_cls is None:
        raise ValueError(
            f"No native autograd Function for op '{op_name}'. "
            f"Supported: {sorted(NATIVE_AUTOGRAD_SUPPORTED_OPS)}"
        )
    return fn_cls.apply(*inputs)
