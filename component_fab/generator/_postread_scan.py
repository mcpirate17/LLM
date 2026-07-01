"""Autograd wrappers for the CPU-native post-write-read surprise scans.

``native_postread_surprise.cpp`` is an exact port of the pure-Python
``_SurpriseMemoryBase._delta_step`` loop (tropical max-plus and tempered
log-sum-exp reads, readout AFTER the delta write). It exists so the fab's
CPU grading path stops paying ~20 Python kernel launches per token; the math
is unchanged. One deliberate edge difference: on exactly-tied maxima the
tropical backward routes the gradient to the first argmax, where PyTorch's
``amax`` splits it equally — ties are measure-zero for real activations.

``native_postread_supported`` is the dispatch guard: the C++ scans are
CPU-only and AT_DISPATCH_FLOATING_TYPES covers float32/float64.
"""

from __future__ import annotations

import torch

from research.runtime.native.torch_extension_loader import load_local_cpp_extension


def _ext():
    return load_local_cpp_extension(
        __file__,
        "native_postread_surprise.cpp",
        "component_fab_native_postread_surprise",
    )


def native_postread_supported(x: torch.Tensor) -> bool:
    return x.device.type == "cpu" and x.dtype in (torch.float32, torch.float64)


class TropicalPostreadScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, write, forget, momentum):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        write = write.contiguous()
        forget = forget.contiguous()
        momentum = momentum.contiguous()
        y, mem_prev, surprise_prev = _ext().tropical_forward(
            q, k, v, write, forget, momentum
        )
        ctx.save_for_backward(q, k, v, write, forget, momentum, mem_prev, surprise_prev)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        q, k, v, write, forget, momentum, mem_prev, surprise_prev = ctx.saved_tensors
        return tuple(
            _ext().tropical_backward(
                q,
                k,
                v,
                write,
                forget,
                momentum,
                grad_y.contiguous(),
                mem_prev,
                surprise_prev,
            )
        )


class SemiringPostreadScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, write, forget, momentum, beta):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        write = write.contiguous()
        forget = forget.contiguous()
        momentum = momentum.contiguous()
        beta = beta.contiguous()
        y, mem_prev, surprise_prev = _ext().semiring_forward(
            q, k, v, write, forget, momentum, beta
        )
        ctx.save_for_backward(
            q, k, v, write, forget, momentum, beta, mem_prev, surprise_prev
        )
        return y

    @staticmethod
    def backward(ctx, grad_y):
        q, k, v, write, forget, momentum, beta, mem_prev, surprise_prev = (
            ctx.saved_tensors
        )
        return tuple(
            _ext().semiring_backward(
                q,
                k,
                v,
                write,
                forget,
                momentum,
                beta,
                grad_y.contiguous(),
                mem_prev,
                surprise_prev,
            )
        )
