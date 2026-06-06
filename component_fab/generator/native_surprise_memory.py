"""CPU-native surprise-memory lanes.

The scan/update math lives in ``native_surprise_memory.cpp``. Python is only the
outer module shell: projections, gates, and output layers are regular PyTorch
native ops; the recurrent memory loop and associative read/write are C++.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from component_fab.harness.rope import RotaryEmbedding, apply_rope
from research.runtime.native.torch_extension_loader import (
    load_local_cpp_extension,
    load_local_cuda_extension,
)


def _native_ext():
    return load_local_cpp_extension(
        __file__,
        "native_surprise_memory.cpp",
        "component_fab_native_surprise_memory",
    )


def _native_cuda_ext():
    return load_local_cuda_extension(
        __file__,
        "native_surprise_memory_cuda.cu",
        "component_fab_native_surprise_memory_cuda",
    )


class _NativeSurpriseScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, write, forget, momentum):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        write = write.contiguous()
        forget = forget.contiguous()
        momentum = momentum.contiguous()
        if q.is_cuda:
            y, mem_prev, surprise_prev = _native_cuda_ext().plain_forward(
                q, k, v, write, forget, float(momentum)
            )
        else:
            y, mem_prev, surprise_prev = _native_ext().forward(
                q, k, v, write, forget, momentum
            )
        ctx.save_for_backward(q, k, v, write, forget, momentum, mem_prev, surprise_prev)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        q, k, v, write, forget, momentum, mem_prev, surprise_prev = ctx.saved_tensors
        grad_y = grad_y.contiguous()
        if q.is_cuda:
            grad_q, grad_k, grad_v, grad_write, grad_forget, grad_momentum = (
                _native_cuda_ext().plain_backward(
                    q,
                    k,
                    v,
                    write,
                    forget,
                    float(momentum),
                    grad_y,
                    mem_prev,
                    surprise_prev,
                )
            )
        else:
            grad_q, grad_k, grad_v, grad_write, grad_forget, grad_momentum = (
                _native_ext().backward(
                    q, k, v, write, forget, momentum, grad_y, mem_prev, surprise_prev
                )
            )
        return grad_q, grad_k, grad_v, grad_write, grad_forget, grad_momentum


class _NativeSemiringSurpriseScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, write, forget, momentum, beta, balance):
        if q.device.type != "cpu":
            raise RuntimeError("native semiring surprise-memory scan is CPU-only")
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        write = write.contiguous()
        forget = forget.contiguous()
        momentum = momentum.contiguous()
        beta = beta.contiguous()
        balance = balance.contiguous()
        y, mem_prev, surprise_prev = _native_ext().semiring_forward(
            q, k, v, write, forget, momentum, beta, balance
        )
        ctx.save_for_backward(
            q, k, v, write, forget, momentum, beta, balance, mem_prev, surprise_prev
        )
        return y

    @staticmethod
    def backward(ctx, grad_y):
        q, k, v, write, forget, momentum, beta, balance, mem_prev, surprise_prev = (
            ctx.saved_tensors
        )
        (
            grad_q,
            grad_k,
            grad_v,
            grad_write,
            grad_forget,
            grad_momentum,
            grad_beta,
            grad_balance,
        ) = _native_ext().semiring_backward(
            q,
            k,
            v,
            write,
            forget,
            momentum,
            beta,
            balance,
            grad_y.contiguous(),
            mem_prev,
            surprise_prev,
        )
        return (
            grad_q,
            grad_k,
            grad_v,
            grad_write,
            grad_forget,
            grad_momentum,
            grad_beta,
            grad_balance,
        )


class _NativeAdaptiveSemiringSurpriseScan(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        write,
        forget,
        momentum,
        beta,
        balance,
        low_threshold,
        high_threshold,
        max_steps,
    ):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        write = write.contiguous()
        forget = forget.contiguous()
        momentum = momentum.contiguous()
        beta = beta.contiguous()
        balance = balance.contiguous()
        low_threshold = low_threshold.contiguous()
        high_threshold = high_threshold.contiguous()
        if q.is_cuda:
            y, mem_prev, surprise_prev, depth_counts = (
                _native_cuda_ext().adaptive_forward(
                    q,
                    k,
                    v,
                    write,
                    forget,
                    float(momentum),
                    float(beta),
                    float(balance),
                    float(low_threshold),
                    float(high_threshold),
                    int(max_steps),
                )
            )
        else:
            y, mem_prev, surprise_prev, depth_counts = (
                _native_ext().adaptive_semiring_forward(
                    q,
                    k,
                    v,
                    write,
                    forget,
                    momentum,
                    beta,
                    balance,
                    low_threshold,
                    high_threshold,
                    int(max_steps),
                )
            )
        ctx.max_steps = int(max_steps)
        ctx.save_for_backward(
            q,
            k,
            v,
            write,
            forget,
            momentum,
            beta,
            balance,
            low_threshold,
            high_threshold,
            mem_prev,
            surprise_prev,
        )
        ctx.mark_non_differentiable(depth_counts)
        return y, depth_counts

    @staticmethod
    def backward(ctx, grad_y, _grad_depth_counts):
        (
            q,
            k,
            v,
            write,
            forget,
            momentum,
            beta,
            balance,
            low_threshold,
            high_threshold,
            mem_prev,
            surprise_prev,
        ) = ctx.saved_tensors
        grad_y = grad_y.contiguous()
        if q.is_cuda:
            (
                grad_q,
                grad_k,
                grad_v,
                grad_write,
                grad_forget,
                grad_momentum,
                grad_beta,
                grad_balance,
            ) = _native_cuda_ext().adaptive_backward(
                q,
                k,
                v,
                write,
                forget,
                float(momentum),
                float(beta),
                float(balance),
                float(low_threshold),
                float(high_threshold),
                ctx.max_steps,
                grad_y,
                mem_prev,
                surprise_prev,
            )
            grad_low = None
            grad_high = None
        else:
            (
                grad_q,
                grad_k,
                grad_v,
                grad_write,
                grad_forget,
                grad_momentum,
                grad_beta,
                grad_balance,
                grad_low,
                grad_high,
            ) = _native_ext().adaptive_semiring_backward(
                q,
                k,
                v,
                write,
                forget,
                momentum,
                beta,
                balance,
                low_threshold,
                high_threshold,
                ctx.max_steps,
                grad_y,
                mem_prev,
                surprise_prev,
            )
        return (
            grad_q,
            grad_k,
            grad_v,
            grad_write,
            grad_forget,
            grad_momentum,
            grad_beta,
            grad_balance,
            grad_low,
            grad_high,
            None,
        )


class _NativeTwoLaneBlend(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, logit):
        y, gate = _native_ext().two_lane_blend_forward(
            a.contiguous(), b.contiguous(), logit.contiguous()
        )
        ctx.save_for_backward(a, b, gate)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        a, b, gate = ctx.saved_tensors
        grad_a, grad_b, grad_logit = _native_ext().two_lane_blend_backward(
            grad_y.contiguous(), a.contiguous(), b.contiguous(), gate.contiguous()
        )
        return grad_a, grad_b, grad_logit


class _NativeThreeLaneBlend(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, c, logits):
        y, weights = _native_ext().three_lane_blend_forward(
            a.contiguous(), b.contiguous(), c.contiguous(), logits.contiguous()
        )
        ctx.save_for_backward(a, b, c, y, weights)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        a, b, c, y, weights = ctx.saved_tensors
        grad_a, grad_b, grad_c, grad_logits = _native_ext().three_lane_blend_backward(
            grad_y.contiguous(),
            a.contiguous(),
            b.contiguous(),
            c.contiguous(),
            y.contiguous(),
            weights.contiguous(),
        )
        return grad_a, grad_b, grad_c, grad_logits


def _unit(t: torch.Tensor) -> torch.Tensor:
    return t / t.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def _causal_softmax_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dim: int
) -> torch.Tensor:
    """Standard causal QK^T-softmax attention (shared by the Titans-MAC lanes)."""
    scores = torch.matmul(q, k.transpose(1, 2)) * (dim**-0.5)
    seq_len = q.shape[1]
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device).triu(1)
    scores = scores.masked_fill(mask, float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v)


def _softplus_inverse(value: float) -> float:
    return math.log(math.expm1(float(value)))


class NativeReadBeforeWriteSurpriseMemoryLane(nn.Module):
    """Native C++ read-before-write surprise memory.

    Fixes the most suspicious issue in the earlier surprise lanes: output at
    position ``t`` reads ``M_{t-1}``, then token ``t`` updates memory for future
    positions. That matches next-token prediction and avoids the query token
    overwriting the association it is trying to retrieve.
    """

    def __init__(self, dim: int, memory_dim: int | None = None) -> None:
        super().__init__()
        memory_dim = memory_dim or min(dim, 24)
        self.q = nn.Linear(dim, memory_dim, bias=False)
        self.k = nn.Linear(dim, memory_dim, bias=False)
        self.v = nn.Linear(dim, memory_dim, bias=False)
        self.write_gate = nn.Linear(dim, 1)
        self.forget_gate = nn.Linear(dim, memory_dim)
        self.out = nn.Linear(memory_dim, dim, bias=False)
        self.momentum_logit = nn.Parameter(torch.tensor(0.0))
        nn.init.constant_(self.write_gate.bias, 1.0)
        nn.init.constant_(self.forget_gate.bias, -4.0)
        self.dim = dim
        self.memory_dim = memory_dim

    def _features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = _unit(torch.tanh(self.q(x)))
        k = _unit(torch.tanh(self.k(x)))
        return q, k

    def _scan(self, x: torch.Tensor) -> torch.Tensor:
        q, k = self._features(x)
        v = self.v(x)
        write = torch.sigmoid(self.write_gate(x)).squeeze(-1)
        forget = torch.sigmoid(self.forget_gate(x))
        momentum = torch.sigmoid(self.momentum_logit).reshape(())
        return _NativeSurpriseScan.apply(q, k, v, write, forget, momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self._scan(x))


class NativeContextGatedSurpriseMemoryLane(NativeReadBeforeWriteSurpriseMemoryLane):
    """Titans/MIRAS-style gated sidecar memory lane.

    The long-term native memory is not forced to be the whole mixer output. A
    learned native-backed local value branch can carry short-term/token-local
    information while the memory branch contributes retrieved context.
    """

    def __init__(self, dim: int, memory_dim: int | None = None) -> None:
        super().__init__(dim, memory_dim=memory_dim)
        self.local = nn.Linear(dim, self.memory_dim, bias=False)
        self.mix_gate = nn.Linear(dim, self.memory_dim)
        nn.init.constant_(self.mix_gate.bias, -1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mem = self._scan(x)
        local = torch.tanh(self.local(x))
        gate = torch.sigmoid(self.mix_gate(x))
        return self.out(local + gate * mem)


class NativeAtlasPolySurpriseMemoryLane(NativeContextGatedSurpriseMemoryLane):
    """ATLAS/OmegaNet-inspired higher-capacity feature-map memory.

    ATLAS identifies limited key/value feature capacity as a bottleneck and
    motivates higher-order feature maps. This lane expands q/k with a quadratic
    feature map before the native scan, while keeping the same read-before-write
    and gated sidecar behavior.
    """

    def __init__(self, dim: int, base_memory_dim: int | None = None) -> None:
        base = base_memory_dim or min(dim, 16)
        super().__init__(dim, memory_dim=base * 2)
        self.q = nn.Linear(dim, base, bias=False)
        self.k = nn.Linear(dim, base, bias=False)
        self.forget_gate = nn.Linear(dim, base * 2)
        self.local = nn.Linear(dim, base * 2, bias=False)
        self.mix_gate = nn.Linear(dim, base * 2)
        nn.init.constant_(self.forget_gate.bias, -4.0)
        nn.init.constant_(self.mix_gate.bias, -1.0)
        self.base_memory_dim = base

    def _poly(self, z: torch.Tensor) -> torch.Tensor:
        z = torch.tanh(z)
        return _unit(torch.cat([z, z * z], dim=-1))

    def _features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._poly(self.q(x)), self._poly(self.k(x))


class NativeTitansMACSurpriseMemoryLane(NativeAtlasPolySurpriseMemoryLane):
    """Titans MAC-style lane: native long-term memory plus attention core.

    This is the SOTA-faithful variant for this repo. Titans/MIRAS do not use
    long-term memory as the entire sequence processor; they pair it with precise
    short-term attention and let a gate decide how much memory context matters.
    Attention is implemented with PyTorch's native matmul/softmax kernels; the
    long-term memory scan remains the C++ extension.
    """

    def __init__(self, dim: int, base_memory_dim: int | None = None) -> None:
        super().__init__(dim, base_memory_dim=base_memory_dim or min(dim, 8))
        self.attn_q = nn.Linear(dim, dim, bias=False)
        self.attn_k = nn.Linear(dim, dim, bias=False)
        self.attn_v = nn.Linear(dim, dim, bias=False)
        self.mem_to_dim = nn.Linear(self.memory_dim, dim, bias=False)
        self.dim_gate = nn.Linear(dim, dim)
        self.out_dim = nn.Linear(dim, dim, bias=False)
        nn.init.constant_(self.dim_gate.bias, -1.0)

    def _attention_core(self, x: torch.Tensor) -> torch.Tensor:
        return _causal_softmax_attention(
            self.attn_q(x), self.attn_k(x), self.attn_v(x), self.dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        core = self._attention_core(x)
        mem = self.mem_to_dim(self._scan(x))
        gate = torch.sigmoid(self.dim_gate(x))
        return self.out_dim(core + gate * mem)


class NativeSemiringSurpriseMemoryLane(NativeReadBeforeWriteSurpriseMemoryLane):
    """Native tempered-semiring surprise memory.

    Same delta-rule write as the surprise-memory family, but retrieval is the
    learnable tempered log-sum-exp semiring used by
    ``SemiringSurpriseMemoryLane``. The scan and semiring read/backward are C++.
    """

    def __init__(
        self,
        dim: int,
        memory_dim: int | None = None,
        *,
        use_rope: bool = False,
        max_seq_len: int = 1024,
        semiring_temp_init: float = 4.0,
        recursive_balance_init: float = 0.0,
    ) -> None:
        super().__init__(dim, memory_dim=memory_dim)
        self.semiring_temp = nn.Parameter(torch.tensor(float(semiring_temp_init)))
        if recursive_balance_init > 0.0:
            self.recursive_balance_logit = nn.Parameter(
                torch.tensor(_softplus_inverse(recursive_balance_init))
            )
            self.register_buffer("_recursive_balance_fixed", torch.tensor(0.0))
        else:
            self.register_parameter("recursive_balance_logit", None)
            self.register_buffer("_recursive_balance_fixed", torch.tensor(0.0))
        self.rope = (
            RotaryEmbedding(self.memory_dim, max_seq_len=max_seq_len)
            if use_rope and self.memory_dim % 2 == 0
            else None
        )

    def _features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q, k = super()._features(x)
        if self.rope is not None:
            cos, sin = self.rope(x.shape[1], device=x.device, dtype=x.dtype)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        return q, k

    def _recursive_balance(self) -> torch.Tensor:
        if self.recursive_balance_logit is None:
            return self._recursive_balance_fixed
        return torch.nn.functional.softplus(self.recursive_balance_logit).clamp(
            0.0, 30.0
        )

    def _scan_params(self, x: torch.Tensor):
        """Shared q/k/v/gate/semiring inputs for the semiring scan family."""
        q, k = self._features(x)
        v = self.v(x)
        write = torch.sigmoid(self.write_gate(x)).squeeze(-1)
        forget = torch.sigmoid(self.forget_gate(x))
        momentum = torch.sigmoid(self.momentum_logit).reshape(())
        beta = torch.nn.functional.softplus(self.semiring_temp).clamp(1e-2, 30.0)
        balance = self._recursive_balance()
        return q, k, v, write, forget, momentum, beta, balance

    def _scan(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v, write, forget, momentum, beta, balance = self._scan_params(x)
        return _NativeSemiringSurpriseScan.apply(
            q, k, v, write, forget, momentum, beta, balance
        )


class NativeSemiringRopeSurpriseMemoryLane(NativeSemiringSurpriseMemoryLane):
    """Native semiring surprise memory with RoPE on q/k addressing."""

    def __init__(self, dim: int, memory_dim: int | None = None) -> None:
        super().__init__(dim, memory_dim=memory_dim, use_rope=True)


class NativeSemiringTitansMACSurpriseMemoryLane(NativeSemiringSurpriseMemoryLane):
    """Titans MAC-style attention core with native semiring long-term memory."""

    def __init__(
        self,
        dim: int,
        memory_dim: int | None = None,
        *,
        gate_bias: float = -1.0,
        semiring_temp_init: float = 4.0,
        qk_norm: bool = False,
        recursive_balance_init: float = 0.0,
    ) -> None:
        super().__init__(
            dim,
            memory_dim=memory_dim or min(dim, 16),
            semiring_temp_init=semiring_temp_init,
            recursive_balance_init=recursive_balance_init,
        )
        self.attn_q = nn.Linear(dim, dim, bias=False)
        self.attn_k = nn.Linear(dim, dim, bias=False)
        self.attn_v = nn.Linear(dim, dim, bias=False)
        self.mem_to_dim = nn.Linear(self.memory_dim, dim, bias=False)
        self.dim_gate = nn.Linear(dim, dim)
        self.out_dim = nn.Linear(dim, dim, bias=False)
        self.qk_norm = qk_norm
        nn.init.constant_(self.dim_gate.bias, float(gate_bias))

    def _attention_core(self, x: torch.Tensor) -> torch.Tensor:
        q = self.attn_q(x)
        k = self.attn_k(x)
        v = self.attn_v(x)
        if self.qk_norm:
            q = _unit(q)
            k = _unit(k)
        scores = torch.matmul(q, k.transpose(1, 2)) * (self.dim**-0.5)
        seq_len = x.shape[1]
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))
        return torch.matmul(torch.softmax(scores, dim=-1), v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        core = self._attention_core(x)
        mem = self.mem_to_dim(self._scan(x))
        gate = torch.sigmoid(self.dim_gate(x))
        return self.out_dim(core + gate * mem)


class NativeSemiringRopeTitansMACSurpriseMemoryLane(
    NativeSemiringTitansMACSurpriseMemoryLane
):
    """Titans MAC-style native semiring memory with RoPE addressing."""

    def __init__(
        self,
        dim: int,
        memory_dim: int | None = None,
        *,
        gate_bias: float = -1.0,
        semiring_temp_init: float = 4.0,
        qk_norm: bool = False,
        recursive_balance_init: float = 0.0,
    ) -> None:
        super().__init__(
            dim,
            memory_dim=memory_dim,
            gate_bias=gate_bias,
            semiring_temp_init=semiring_temp_init,
            qk_norm=qk_norm,
            recursive_balance_init=recursive_balance_init,
        )
        self.rope = (
            RotaryEmbedding(self.memory_dim, max_seq_len=1024)
            if self.memory_dim % 2 == 0
            else None
        )


class NativeBalancedSemiringTitansMACSurpriseMemoryLane(
    NativeSemiringTitansMACSurpriseMemoryLane
):
    """Semiring MAC lane with recursive surprise-write balancing enabled."""

    def __init__(
        self,
        dim: int,
        memory_dim: int | None = None,
        *,
        gate_bias: float = -1.0,
        semiring_temp_init: float = 4.0,
        qk_norm: bool = False,
        recursive_balance_init: float = 1.0,
    ) -> None:
        super().__init__(
            dim,
            memory_dim=memory_dim,
            gate_bias=gate_bias,
            semiring_temp_init=semiring_temp_init,
            qk_norm=qk_norm,
            recursive_balance_init=recursive_balance_init,
        )


class NativeBalancedSemiringRopeTitansMACSurpriseMemoryLane(
    NativeSemiringRopeTitansMACSurpriseMemoryLane
):
    """RoPE semiring MAC lane with recursive surprise-write balancing enabled."""

    def __init__(
        self,
        dim: int,
        memory_dim: int | None = None,
        *,
        gate_bias: float = -1.0,
        semiring_temp_init: float = 4.0,
        qk_norm: bool = False,
        recursive_balance_init: float = 1.0,
    ) -> None:
        super().__init__(
            dim,
            memory_dim=memory_dim,
            gate_bias=gate_bias,
            semiring_temp_init=semiring_temp_init,
            qk_norm=qk_norm,
            recursive_balance_init=recursive_balance_init,
        )


class NativeAdaptiveSemiringRopeTitansMACSurpriseMemoryLane(
    NativeBalancedSemiringRopeTitansMACSurpriseMemoryLane
):
    """RoPE semiring MAC lane with surprise-adaptive native recursion depth."""

    def __init__(
        self,
        dim: int,
        memory_dim: int | None = None,
        *,
        gate_bias: float = -1.0,
        semiring_temp_init: float = 1.0,
        qk_norm: bool = False,
        recursive_balance_init: float = 1.0,
        low_threshold: float = 0.01,
        high_threshold: float = 0.05,
        max_recursive_steps: int = 4,
    ) -> None:
        super().__init__(
            dim,
            memory_dim=memory_dim,
            gate_bias=gate_bias,
            semiring_temp_init=semiring_temp_init,
            qk_norm=qk_norm,
            recursive_balance_init=recursive_balance_init,
        )
        self.register_buffer(
            "_adaptive_low_threshold", torch.tensor(float(low_threshold))
        )
        self.register_buffer(
            "_adaptive_high_threshold", torch.tensor(float(high_threshold))
        )
        self.max_recursive_steps = int(max_recursive_steps)
        self.last_depth_counts: torch.Tensor | None = None

    def _scan(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v, write, forget, momentum, beta, balance = self._scan_params(x)
        y, depth_counts = _NativeAdaptiveSemiringSurpriseScan.apply(
            q,
            k,
            v,
            write,
            forget,
            momentum,
            beta,
            balance,
            self._adaptive_low_threshold.reshape(()),
            self._adaptive_high_threshold.reshape(()),
            self.max_recursive_steps,
        )
        self.last_depth_counts = depth_counts.detach()
        return y


class _NativeBalancedComposite(nn.Module):
    """Shared base for the gated multi-lane composites.

    Holds the common __init__ (lane_a + gate + bookkeeping); subclasses override
    ``_make_lane_a`` (the primary lane), ``_build_aux_lanes`` (lane_b/lane_c), the
    ``gate_width`` (1 logit for the 2-lane sigmoid blend, 3 for the tri-lane), and
    ``forward`` (the blend).
    """

    gate_width = 1

    def __init__(
        self,
        dim: int,
        memory_dim: int | None = None,
        *,
        gate_bias: float = 0.0,
        semiring_temp_init: float = 1.0,
        recursive_balance_init: float = 1.0,
        low_threshold: float = 0.01,
        high_threshold: float = 0.05,
        max_recursive_steps: int = 4,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.memory_dim = memory_dim or min(dim, 32)
        self.recursive_balance_init = float(recursive_balance_init)
        self.lane_a = self._make_lane_a(
            dim,
            self.memory_dim,
            gate_bias,
            semiring_temp_init,
            recursive_balance_init,
            low_threshold,
            high_threshold,
            max_recursive_steps,
        )
        self._build_aux_lanes(dim, self.memory_dim, gate_bias, semiring_temp_init)
        self.gate = nn.Linear(dim, self.gate_width)
        nn.init.constant_(self.gate.bias, 0.0)

    def _make_lane_a(
        self,
        dim,
        memory_dim,
        gate_bias,
        semiring_temp_init,
        recursive_balance_init,
        low_threshold,
        high_threshold,
        max_recursive_steps,
    ) -> nn.Module:
        return NativeBalancedSemiringRopeTitansMACSurpriseMemoryLane(
            dim,
            memory_dim=memory_dim,
            gate_bias=gate_bias,
            semiring_temp_init=semiring_temp_init,
            recursive_balance_init=recursive_balance_init,
        )

    def _build_aux_lanes(self, dim, memory_dim, gate_bias, semiring_temp_init) -> None:
        raise NotImplementedError


class NativeBalancedSemiringBiLaneSurpriseMemoryLane(_NativeBalancedComposite):
    """Native two-lane gated composite, matching mixer_fingerprint's bi-lane form."""

    def _build_aux_lanes(self, dim, memory_dim, gate_bias, semiring_temp_init) -> None:
        self.lane_b = NativeTitansMACSurpriseMemoryLane(
            dim, base_memory_dim=max(4, memory_dim // 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b, logit = self.lane_a(x), self.lane_b(x), self.gate(x)
        if x.is_cuda:
            gate = torch.sigmoid(logit)
            return gate * a + (1.0 - gate) * b
        return _NativeTwoLaneBlend.apply(a, b, logit)


class NativeAdaptiveSemiringBiLaneSurpriseMemoryLane(
    NativeBalancedSemiringBiLaneSurpriseMemoryLane
):
    """Native bi-lane where the semiring branch uses surprise-adaptive recursion."""

    def _make_lane_a(
        self,
        dim,
        memory_dim,
        gate_bias,
        semiring_temp_init,
        recursive_balance_init,
        low_threshold,
        high_threshold,
        max_recursive_steps,
    ) -> nn.Module:
        return NativeAdaptiveSemiringRopeTitansMACSurpriseMemoryLane(
            dim,
            memory_dim=memory_dim,
            gate_bias=gate_bias,
            semiring_temp_init=semiring_temp_init,
            recursive_balance_init=recursive_balance_init,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            max_recursive_steps=max_recursive_steps,
        )


class NativeBalancedSemiringTriLaneSurpriseMemoryLane(_NativeBalancedComposite):
    """Native three-lane gated composite, matching mixer_fingerprint's tri-lane form."""

    gate_width = 3

    def _build_aux_lanes(self, dim, memory_dim, gate_bias, semiring_temp_init) -> None:
        self.lane_b = NativeSemiringRopeTitansMACSurpriseMemoryLane(
            dim,
            memory_dim=memory_dim,
            gate_bias=gate_bias,
            semiring_temp_init=semiring_temp_init,
        )
        self.lane_c = NativeTitansMACSurpriseMemoryLane(
            dim, base_memory_dim=max(4, memory_dim // 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _NativeThreeLaneBlend.apply(
            self.lane_a(x), self.lane_b(x), self.lane_c(x), self.gate(x)
        )
