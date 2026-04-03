"""Native C kernel autograd integration for training support."""

from __future__ import annotations

import torch

from .native.tensor_bridge import to_device_tensor
from .native_runner import dispatch_op_native, dispatch_op_backward_native


# ── Factory functions for common autograd patterns ─────────────────


def _make_unary_save_input(op_name: str) -> type:
    """Create a torch.autograd.Function for a unary op that saves the input."""

    class _F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            shape = x.shape
            ctx.save_for_backward(x)
            y_np = dispatch_op_native(op_name, x)
            return to_device_tensor(y_np, reference=x).reshape(shape)

        @staticmethod
        def backward(ctx, grad_output):
            (x,) = ctx.saved_tensors
            shape = grad_output.shape
            grad_in_np = dispatch_op_backward_native(op_name, grad_output, x)
            return to_device_tensor(grad_in_np, reference=grad_output).reshape(shape)

    _F.__name__ = _F.__qualname__ = f"Native{op_name.capitalize()}"
    return _F


def _make_unary_save_output(op_name: str) -> type:
    """Create a torch.autograd.Function for a unary op that saves the output."""

    class _F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            shape = x.shape
            y_np = dispatch_op_native(op_name, x)
            y = to_device_tensor(y_np, reference=x).reshape(shape)
            ctx.save_for_backward(y)
            return y

        @staticmethod
        def backward(ctx, grad_output):
            (y,) = ctx.saved_tensors
            shape = grad_output.shape
            grad_in_np = dispatch_op_backward_native(op_name, grad_output, y)
            return to_device_tensor(grad_in_np, reference=grad_output).reshape(shape)

    _F.__name__ = _F.__qualname__ = f"Native{op_name.capitalize()}"
    return _F


def _make_binary_flat(op_name: str) -> type:
    """Create a torch.autograd.Function for a binary op using flat buffers."""

    class _F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a, b):
            shape = a.shape
            ctx.save_for_backward(a, b)
            y_np = dispatch_op_native(op_name, a, b)
            return to_device_tensor(y_np, reference=a).reshape(shape)

        @staticmethod
        def backward(ctx, grad_output):
            a, b = ctx.saved_tensors
            shape = grad_output.shape
            grad_a_np, grad_b_np = dispatch_op_backward_native(op_name, grad_output, a, b)
            return to_device_tensor(
                grad_a_np, reference=grad_output
            ).reshape(shape), to_device_tensor(grad_b_np, reference=grad_output).reshape(
                shape
            )

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
        ctx.save_for_backward(A, B)
        C_np = dispatch_op_native("matmul", A, B)
        return to_device_tensor(C_np, reference=A)

    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        grad_A_np, grad_B_np = dispatch_op_backward_native("matmul", grad_output, A, B)
        return to_device_tensor(grad_A_np, reference=grad_output), to_device_tensor(
            grad_B_np, reference=grad_output
        )


class NativeSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y_np = dispatch_op_native("softmax", x)
        y = to_device_tensor(y_np, reference=x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        (y,) = ctx.saved_tensors
        grad_in_np = dispatch_op_backward_native("softmax", grad_output, y)
        return to_device_tensor(grad_in_np, reference=grad_output)


class NativeLayernorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, beta):
        y_np = dispatch_op_native("layernorm", x, gamma, beta)
        y = to_device_tensor(y_np, reference=x)
        ctx.save_for_backward(x, gamma)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, gamma = ctx.saved_tensors
        grad_in_np, grad_gamma_np, grad_beta_np = dispatch_op_backward_native(
            "layernorm", grad_output, x, gamma
        )
        return (
            to_device_tensor(grad_in_np, reference=grad_output),
            to_device_tensor(grad_gamma_np, reference=grad_output),
            to_device_tensor(grad_beta_np, reference=grad_output),
        )


class NativeRmsnorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma):
        y_np = dispatch_op_native("rmsnorm", x, gamma)
        y = to_device_tensor(y_np, reference=x)
        ctx.save_for_backward(x, gamma)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, gamma = ctx.saved_tensors
        grad_in_np, grad_gamma_np = dispatch_op_backward_native(
            "rmsnorm", grad_output, x, gamma
        )
        return (
            to_device_tensor(grad_in_np, reference=grad_output),
            to_device_tensor(grad_gamma_np, reference=grad_output),
        )


class NativeSignSte(torch.autograd.Function):
    """Sign with straight-through estimator: forward = sign, backward = identity."""

    @staticmethod
    def forward(ctx, x):
        shape = x.shape
        y_np = dispatch_op_native("sign_ste", x)
        return to_device_tensor(y_np, reference=x).reshape(shape)

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
