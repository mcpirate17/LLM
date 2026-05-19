"""
Tropical Semiring Operations

In tropical algebra, addition becomes min (or max) and multiplication
becomes addition. This gives shortest-path semantics:
"tropical matmul" computes shortest-path distances between tokens.

The tropical semiring (R ∪ {+∞}, min, +) replaces:
- Standard addition → min
- Standard multiplication → +

Applications: sequence alignment, shortest paths, parsing.

Gradient fix (2026-03-12): Hard min/max kills gradient flow on the
non-selected branch.  We use log-sum-exp smooth-min:
  softmin(x, y, τ) = -τ · log(exp(-x/τ) + exp(-y/τ))
which converges to exact min as τ→0 while giving both branches
gradient proportional to their softmin weight.  τ=0.1 is small
enough to preserve tropical semantics while enabling learning.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from research.env import aria_core, HAS_ARIA_CORE as _HAS_ARIA_CORE
from research.mathspaces._utils import causal_mask

try:
    from ..synthesis.kernels import triton_tropical_matmul

    _HAS_TRITON_KERNELS = True
except ImportError:
    _HAS_TRITON_KERNELS = False


# ── Smooth min/max primitives ────────────────────────────────────────
# Hard min: gradient is 1 for selected branch, 0 for all others.
# Over a chain of tropical ops, gradient info is multiplicatively lost.
# Smooth min via log-sum-exp preserves gradient flow to both branches.

_SMOOTH_TAU: float = (
    1.0  # Temperature for smooth min/max (was 0.1 — too small, recreated hard min)
)
_SMOOTH_S_REF: float = 128.0


def _disable_torch_compile(fn):
    try:
        return torch.compiler.disable(fn)
    except Exception:
        try:
            return torch._dynamo.disable(fn)
        except Exception:
            return fn


@_disable_torch_compile
def _logcumsumexp_dim1_eager(x: torch.Tensor) -> torch.Tensor:
    """Run scan in eager mode; Inductor scan codegen is unstable for this op."""
    return torch.logcumsumexp(x, dim=1)


def _adaptive_temperature(base_tau: float, size: int) -> float:
    scale = max(1.0, (max(int(size), 1) / _SMOOTH_S_REF) ** 0.5)
    return max(base_tau * scale, 1e-4)


def _smooth_min(
    x: torch.Tensor, y: torch.Tensor, tau: float = _SMOOTH_TAU
) -> torch.Tensor:
    """Smooth element-wise minimum via log-sum-exp.

    softmin(x, y, τ) = -τ · logsumexp(-x/τ, -y/τ)
    Converges to min(x, y) as τ→0.  With τ=0.1, both inputs receive
    gradient proportional to exp(-x_i/τ) / (exp(-x_i/τ) + exp(-y_i/τ)).
    """
    adaptive_tau = _adaptive_temperature(tau, x.shape[-1] if x.ndim else 1)
    inv_tau = 1.0 / adaptive_tau
    # Stack for logsumexp along new dim 0: shape (2, *input_shape)
    stacked = torch.stack([-x * inv_tau, -y * inv_tau], dim=0)
    return -adaptive_tau * torch.logsumexp(stacked, dim=0)


def tropical_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Tropical addition: element-wise smooth minimum."""
    if (
        _HAS_ARIA_CORE
        and x.is_contiguous()
        and y.is_contiguous()
        and x.device.type == "cpu"
        and not x.requires_grad
        and not y.requires_grad
    ):
        return aria_core.tropical_add_f32(x, y)
    return _smooth_min(x, y)


class _TropicalMatmulFn(torch.autograd.Function):
    """Memory-efficient tropical matmul: min_k(a_ik + b_kj) via chunking.

    Saves only the inputs a, b and the output result. Recomputes the
    chunked pairwise sums + smooth-min in backward, avoiding retention
    of O(chunks * B * chunk * S * D) intermediate tensors.
    """

    @staticmethod
    def forward(
        ctx, a: torch.Tensor, b_val: torch.Tensor, chunk_size: int, tau: float
    ) -> torch.Tensor:
        B, S1, D = a.shape
        S2 = b_val.shape[1]

        result = torch.empty((B, S1, S2), device=a.device, dtype=a.dtype)
        b_expanded = b_val.unsqueeze(1)  # (B, 1, S2, D)

        adaptive_tau = _adaptive_temperature(tau, D)
        inv_tau = 1.0 / adaptive_tau

        for i in range(0, S1, chunk_size):
            end = min(i + chunk_size, S1)
            a_chunk = a[:, i:end, :].unsqueeze(2)  # (B, c, 1, D)
            with torch.no_grad():
                pairwise = a_chunk + b_expanded  # (B, c, S2, D)
                result[:, i:end, :] = -adaptive_tau * torch.logsumexp(
                    -pairwise * inv_tau, dim=-1
                )

        ctx.save_for_backward(a, b_val)
        ctx.chunk_size = chunk_size
        ctx.tau = tau
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        a, b_val = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        B, S1, D = a.shape

        adaptive_tau = _adaptive_temperature(ctx.tau, D)

        if (
            _HAS_ARIA_CORE
            and a.device.type == "cpu"
            and hasattr(aria_core, "tropical_matmul_batched_backward_f32")
        ):
            # grad_output is (B, M, N)
            grad_a, grad_b = aria_core.tropical_matmul_batched_backward_f32(
                grad_output.contiguous(),
                a.contiguous(),
                b_val.contiguous(),
                float(adaptive_tau),
            )
            return grad_a, grad_b, None, None

        inv_tau = 1.0 / adaptive_tau

        grad_a = torch.zeros_like(a)
        grad_b = torch.zeros_like(b_val)
        b_expanded = b_val.unsqueeze(1)  # (B, 1, S2, D)

        bwd_chunk = min(chunk_size, 16)
        for i in range(0, S1, bwd_chunk):
            end = min(i + bwd_chunk, S1)
            a_chunk = a[:, i:end, :].unsqueeze(2)  # (B, c, 1, D)
            pairwise = a_chunk + b_expanded  # (B, c, S2, D)

            # Recompute smooth min weights
            neg_pw_scaled = -pairwise * inv_tau  # (B, c, S2, D)
            lse = torch.logsumexp(neg_pw_scaled, dim=-1, keepdim=True)  # (B, c, S2, 1)
            # softmin weights: exp(neg_pw_scaled - lse) = contribution of each D to the min
            sm_weights = torch.exp(neg_pw_scaled - lse)  # (B, c, S2, D)

            # grad_output[:, i:end, :] is (B, c, S2)
            # result = -tau * lse.squeeze(-1)
            # d(result)/d(pairwise) = sm_weights  (since d(-tau*lse)/d(x) = exp(x - lse))
            g_out = grad_output[:, i:end, :].unsqueeze(-1)  # (B, c, S2, 1)
            g_pairwise = g_out * sm_weights  # (B, c, S2, D)

            # pairwise = a_chunk + b_expanded
            grad_a[:, i:end, :] += g_pairwise.sum(dim=2)  # sum over S2
            grad_b += g_pairwise.sum(dim=1)  # sum over chunk

            del pairwise, neg_pw_scaled, sm_weights, g_pairwise

        return grad_a, grad_b, None, None


def tropical_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Tropical matrix multiplication.

    Instead of sum(a_ik * b_kj), computes min_k(a_ik + b_kj).
    Dispatch order: Triton (GPU) -> aria_core (CPU) -> memory-efficient torch fallback.

    Input: a (B, S, D), b (B, D, S) or (B, S, D)
    Output: (B, S, S) or similar
    """
    # GPU fast path: Triton kernel
    if _HAS_TRITON_KERNELS and a.is_cuda and b.is_cuda:
        try:
            return triton_tropical_matmul(a, b)
        except Exception as e:
            import logging

            logging.getLogger(__name__).debug("triton_tropical_matmul fallback: %s", e)

    # CPU fast path: native C kernel (inference only — no autograd support).
    # 2026-05-10 SEGV: aria_core.tropical_matmul_batched_f32 crashed when the
    # wrapper let through inputs the kernel's stride assumptions didn't cover.
    # Be defensive — only dispatch with verified shapes + freshly contiguous tensors.
    if (
        _HAS_ARIA_CORE
        and not a.requires_grad
        and not b.requires_grad
        and a.ndim == 3
        and b.ndim == 3
        and a.device.type == "cpu"
        and b.device.type == "cpu"
        and a.dtype == torch.float32
        and b.dtype == torch.float32
        and a.shape[0] == b.shape[0]  # batch must match
    ):
        B, S, D = a.shape
        native_b = None
        if b.shape[1] == D and b.shape[2] != D:
            # b is (B, D, S2) layout — kernel-native.
            native_b = b
        elif b.shape[2] == D and b.shape[1] != D:
            # b is (B, S2, D) — transpose to (B, D, S2) for the kernel.
            native_b = b.transpose(1, 2)
        elif b.shape == a.shape:
            # Square ambiguous case: assume (B, S, D), transpose to (B, D, S).
            native_b = b.transpose(1, 2)
        if native_b is not None:
            # Force a fresh contiguous copy on BOTH inputs. is_contiguous() is
            # not sufficient — the kernel assumes a specific stride layout that
            # transposed-then-contiguous views may not satisfy.
            a_c = a.contiguous()
            b_c = native_b.contiguous()
            S2 = b_c.shape[2]
            # Final invariants: catch anything weird in Python before native call.
            if (
                a_c.shape == (B, S, D)
                and b_c.shape == (B, D, S2)
                and a_c.is_contiguous()
                and b_c.is_contiguous()
                and S > 0
                and S2 > 0
                and D > 0
            ):
                result = aria_core.tropical_matmul_batched_f32(a_c, b_c)
                if result.shape == (B, S, S2):
                    return result
                # C kernel returned wrong shape; fall through to Python path.

    # Normalize b to (B, S2, D) layout.
    # Only transpose when b is explicitly (B, D, S) — i.e. shape[1] matches
    # a's feature dim AND shape[2] does NOT (avoids ambiguity when S == D).
    B, S1, D1 = a.shape
    if b.ndim == 3 and b.shape[1] == D1 and b.shape[2] != D1:
        b_val = b.transpose(1, 2)
    else:
        b_val = b

    # Memory-efficient path via custom autograd
    return _TropicalMatmulFn.apply(a, b_val, 32, _SMOOTH_TAU)


def tropical_softmax(
    x: torch.Tensor, dim: int = -1, temperature: float = 1.0
) -> torch.Tensor:
    """Smooth approximation of tropical (min) using softmax.

    Higher temperature preserves gradient flow while maintaining
    approximate tropical (softmin) semantics. τ=1.0 is the sweet spot
    between gradient health and tropical fidelity.
    """
    reduce_dim = dim if dim >= 0 else x.ndim + dim
    reduce_size = x.shape[reduce_dim] if x.ndim and 0 <= reduce_dim < x.ndim else 1
    adaptive_t = _adaptive_temperature(temperature, reduce_size)
    return torch.softmax(-x / adaptive_t, dim=dim)


def execute_tropical_softmax(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Primitive form: gradient-friendly softmin over the last dim.

    Composable substitute for vanilla softmax in templates where the
    semantics call for "attend to smallest score" (tropical / shortest-path
    semantics). Preserves input shape.

    Per external_research_2026-05-10.md §3.5 (Tropical geometry stability):
    hard max is non-differentiable at ties; LogSumExp/softmin with temperature
    converges to tropical max as t→0 while keeping gradients alive everywhere.
    """
    return tropical_softmax(x, dim=-1, temperature=_SMOOTH_TAU)


def tropical_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Tropical attention: shortest-path distance as attention weights.

    Instead of softmax(QK^T/sqrt(d))V, computes:
    1. Tropical distance matrix between Q and K
    2. Softmin to get weights (closest = highest weight)
    3. Standard weighted sum of V

    This makes tokens attend to their "nearest neighbors" in a
    shortest-path sense rather than highest-dot-product sense.
    """
    # Distance matrix via tropical matmul
    distances = tropical_matmul(q, k)  # (B, S, S)

    # Apply causal mask if S > 1
    S = q.shape[1]
    if S > 1:
        distances.masked_fill_(causal_mask(S, q.device), float("inf"))

    # Softmin: attend to closest tokens
    weights = tropical_softmax(distances, dim=-1)  # (B, S, S)
    # Standard value aggregation
    return torch.bmm(weights, v)  # (B, S, D)


# ── Primitive execution functions ─────────────────────────────────────


def execute_tropical_matmul(
    module: nn.Module, x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """Tropical matmul then project back to D dim."""
    B, S, D = x.shape
    # Ensure y matches x's shape for valid tropical matmul
    if y.shape != x.shape:
        if y.shape[-1] != D:
            # Project y to match D via truncation/padding
            y = (
                y[..., :D]
                if y.shape[-1] > D
                else torch.nn.functional.pad(y, (0, D - y.shape[-1]))
            )
        if y.shape[1] != S:
            y = y[:, :S] if y.shape[1] > S else y
    x = x.contiguous()
    y = y.contiguous()

    scores = tropical_matmul(x, y)  # (B, S, S)

    # Apply causal mask if S > 1
    if S > 1:
        scores.masked_fill_(causal_mask(S, x.device), float("inf"))

    weights = tropical_softmax(scores, dim=-1)
    return torch.bmm(weights, y)  # (B, S, D)


def execute_tropical_add(
    module: nn.Module, x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """Element-wise tropical addition (min)."""
    return tropical_add(x, y)


def execute_tropical_attention(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Self-attention using tropical geometry with residual.

    Residual prevents the weighted-average output from collapsing
    per-token variation, which would kill gradient flow through LayerNorm.
    """
    if hasattr(module, "weight"):
        q = torch.nn.functional.linear(x, module.weight.to(x.dtype))
    else:
        q = x
    return x + tropical_attention(q, x, x)


def execute_tropical_center(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Center features by tropical (min) sequence baseline.

    Uses smooth cumulative min to preserve gradient flow to all
    preceding tokens, not just the argmin position.
    τ=1.0 keeps gradients healthy while preserving tropical semantics.
    """
    if (
        _HAS_ARIA_CORE
        and x.is_contiguous()
        and x.ndim == 3
        and x.device.type == "cpu"
        and not x.requires_grad
    ):
        return aria_core.tropical_center_f32(x)
    B, S, D = x.shape
    tau = _SMOOTH_TAU
    inv_tau = 1.0 / tau
    neg_x_scaled = -x * inv_tau  # (B, S, D)
    # Keep this scan outside torch.compile. PyTorch 2.11 Inductor can fail
    # codegen for SplitScan(logcumsumexp/cumsum) at training shapes such as
    # [B=16, S=256, D=640] with "tensor_dim=None" during Triton scheduling.
    cmin_smooth = -tau * _logcumsumexp_dim1_eager(neg_x_scaled)  # (B, S, D)
    return x - cmin_smooth


def execute_tropical_gate(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Shortest-path routing as a gating mechanism.

    Tropical distance scores → sigmoid gate → elementwise multiply
    with linear projection. Routes information based on tropical
    (shortest-path) proximity rather than learned attention.
    """
    B, S, D = x.shape
    # Tropical distance scores: pairwise min-plus distances
    # Transpose second arg so matmul produces (B, S, S) attention map
    distances = tropical_matmul(x, x.transpose(1, 2))  # (B, S, S)

    # Apply causal mask if S > 1
    if S > 1:
        distances.masked_fill_(causal_mask(S, x.device), float("inf"))

    gate_scores = tropical_softmax(distances, dim=-1)  # (B, S, S)
    gated = torch.bmm(gate_scores, x)  # (B, S, D)
    # Linear projection if params available
    if hasattr(module, "weight"):
        gated = torch.nn.functional.linear(gated, module.weight.to(gated.dtype))
    # Sigmoid gate blending with residual
    gate = torch.sigmoid(gated)
    return x * gate
