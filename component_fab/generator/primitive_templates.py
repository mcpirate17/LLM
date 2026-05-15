"""Novel primitive templates the fab can synthesize from ProposalSpecs.

Each class is a small, self-contained ``nn.Module`` that materializes one
math-axis choice the property miner surfaced as unrealized:

- ``TropicalAttention`` — replaces softmax+sum with max-plus over
  (affinity + value), giving sparse winner-take-all attention. Semiring
  swap from the standard euclidean inner product.
- ``TropicalStateSpace`` — combines max-plus algebra with an SSM-style
  running state: ``s[t] = max(A + s[t-1], B @ x[t])``. To our knowledge
  this combination is not in the literature; it sits in the unbuilt
  ``(tropical, O(L), has_state)`` corner of property space.
- ``TopKLinear`` — projects then keeps only the ``k`` largest activations
  per position. Projection swap from dense to top-k sparsity.
- ``FourierBasisLane`` — applies a learned complex linear along the
  sequence axis in the rFFT basis. Basis swap from content to frequency.

All four preserve ``[B, L, D]`` shape and produce finite gradients at init.
"""

from __future__ import annotations

import torch
from torch import nn


class TropicalAttention(nn.Module):
    """Max-plus attention with optional causal mask.

    ``out[b, i, d] = max_{j<=i} ( scale * (Q[b, i] . K[b, j]) + V[b, j, d] )``

    No softmax. Each output position takes the elementwise max across
    causal positions of (similarity + value). Sparse by construction —
    the argmax dominates, so it behaves like a hard top-1 router in the
    sequence dimension while remaining differentiable through the max.

    ``causal=True`` (default) enables the upper-triangular ``-inf`` mask so
    positions can only attend to themselves and earlier — required to pass
    the S0.5 causality gate.
    """

    def __init__(self, dim: int, causal: bool = True) -> None:
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.scale = float(dim) ** -0.5
        self.dim = dim
        self.causal = causal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        affinity = torch.einsum("bid,bjd->bij", q, k) * self.scale
        if self.causal:
            mask = torch.triu(
                torch.full(
                    (seq_len, seq_len), float("-inf"), device=x.device, dtype=x.dtype
                ),
                diagonal=1,
            )
            affinity = affinity + mask
        combined = affinity.unsqueeze(-1) + v.unsqueeze(1)
        return combined.max(dim=2).values


class TropicalStateSpace(nn.Module):
    """Max-plus recurrent kernel: ``s[t] = max(A + s[t-1], B(x[t]))``.

    The "+" inside the max is elementwise on the state vector, so the
    state evolves under tropical algebra. Output is ``C(s[t]) + x[t]``
    (a residual restore). ``state_dim`` defaults to ``dim``.

    Sits in the unbuilt ``(tropical, O(L), has_state)`` corner of property
    space identified by the miner.
    """

    def __init__(self, dim: int, state_dim: int | None = None) -> None:
        super().__init__()
        state_dim = state_dim or dim
        self.A = nn.Parameter(torch.randn(state_dim) * 0.1)
        self.B = nn.Linear(dim, state_dim, bias=False)
        self.C = nn.Linear(state_dim, dim, bias=False)
        self.state_dim = state_dim
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        Bx = self.B(x)
        state = torch.full(
            (batch_size, self.state_dim),
            float("-inf"),
            device=x.device,
            dtype=x.dtype,
        )
        state = torch.maximum(state, Bx[:, 0])
        outputs = [self.C(state)]
        for t in range(1, seq_len):
            state = torch.maximum(self.A.unsqueeze(0) + state, Bx[:, t])
            outputs.append(self.C(state))
        return torch.stack(outputs, dim=1) + x


class TopKLinear(nn.Module):
    """Linear projection followed by per-position top-k sparsity gate.

    Computes the dense output then keeps only the ``k`` largest activations
    per token (others zeroed). Differentiable through the surviving
    entries via straight-through on the mask.
    """

    def __init__(self, in_dim: int, out_dim: int, k: int) -> None:
        super().__init__()
        if k <= 0 or k > out_dim:
            raise ValueError(f"k={k} must be in [1, out_dim={out_dim}]")
        self.proj = nn.Linear(in_dim, out_dim)
        self.k = k
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.proj(x)
        _, topk_indices = projected.topk(self.k, dim=-1)
        mask = torch.zeros_like(projected).scatter_(-1, topk_indices, 1.0)
        return projected * mask


class FourierBasisLane(nn.Module):
    """Per-frequency complex channel mixing in the rFFT basis (FNO-style).

    ``rFFT(x)[f] -> W[f] @ x[f]`` for each frequency ``f``, then ``irFFT``.
    Per-frequency weights make the operator non-shift-invariant, so a
    position-localized perturbation spreads across all positions. Spectral
    basis swap from content to frequency.
    """

    def __init__(self, dim: int, max_seq_len: int = 128) -> None:
        super().__init__()
        max_freqs = max_seq_len // 2 + 1
        scale = 1.0 / float(dim)
        self.weight_real = nn.Parameter(torch.randn(max_freqs, dim, dim) * scale)
        self.weight_imag = nn.Parameter(torch.randn(max_freqs, dim, dim) * scale)
        self.dim = dim
        self.max_freqs = max_freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        n_freqs = seq_len // 2 + 1
        if n_freqs > self.max_freqs:
            raise ValueError(
                f"sequence length {seq_len} exceeds capacity "
                f"max_seq_len={2 * (self.max_freqs - 1)}"
            )
        spectrum = torch.fft.rfft(x, dim=1)
        wr = self.weight_real[:n_freqs]
        wi = self.weight_imag[:n_freqs]
        sr = spectrum.real
        si = spectrum.imag
        out_real = torch.einsum("fde,bfd->bfe", wr, sr) - torch.einsum(
            "fde,bfd->bfe", wi, si
        )
        out_imag = torch.einsum("fde,bfd->bfe", wr, si) + torch.einsum(
            "fde,bfd->bfe", wi, sr
        )
        return torch.fft.irfft(torch.complex(out_real, out_imag), n=seq_len, dim=1)


class FiniteDifferenceCalculusLane(nn.Module):
    """Causal calculus-inspired lane using finite differences plus integrals.

    The lane computes a backward difference ``dx[t] = x[t] - x[t-1]`` and a
    causal running integral ``mean(x[:t])``. A learned gate blends derivative
    and integral features before a final projection. This gives the generator
    an actual calculus knob instead of only algebra labels.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.derivative = nn.Linear(dim, dim, bias=False)
        self.integral = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(dim * 2, dim)
        self.out = nn.Linear(dim, dim, bias=False)
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prev = torch.zeros_like(x)
        prev[:, 1:] = x[:, :-1]
        dx = x - prev
        seq_len = x.shape[1]
        denom = torch.arange(1, seq_len + 1, dtype=x.dtype, device=x.device).view(
            1, -1, 1
        )
        running_integral = x.cumsum(dim=1) / denom
        gate = torch.sigmoid(self.gate(torch.cat([x, dx], dim=-1)))
        mixed = gate * self.derivative(dx) + (1.0 - gate) * self.integral(
            running_integral
        )
        return x + self.out(mixed)


class LowRankFactorizedLane(nn.Module):
    """Low-rank linear-algebra lane with factorized feature mixing.

    Uses two learned low-rank factors ``D -> rank -> D`` plus a causal
    low-rank running context. This turns a linear-algebra knob into a
    different parameterization and inductive bias, not only a metadata tag.
    """

    def __init__(self, dim: int, rank: int | None = None) -> None:
        super().__init__()
        rank = rank or max(1, dim // 4)
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        self.context_up = nn.Linear(rank, dim, bias=False)
        self.rank = rank
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.down(x)
        seq_len = x.shape[1]
        denom = torch.arange(1, seq_len + 1, dtype=x.dtype, device=x.device).view(
            1, -1, 1
        )
        context = z.cumsum(dim=1) / denom
        return x + self.up(z) + self.context_up(context)


class SparseBandedMatrixLane(nn.Module):
    """Causal block-sparse banded sequence matrix.

    Applies a learnable lower-banded sparse matrix over sequence positions:
    each output token sees only the current token and a small fixed number of
    previous offsets, with a separate feature projection per band. This is the
    first explicit sparse-matrix math knob in the fab generator.
    """

    def __init__(self, dim: int, bandwidth: int = 4) -> None:
        super().__init__()
        if bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        self.projections = nn.ModuleList(
            [nn.Linear(dim, dim, bias=False) for _ in range(bandwidth)]
        )
        self.bandwidth = bandwidth
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        for offset, projection in enumerate(self.projections):
            projected = projection(x)
            if offset == 0:
                out = out + projected
            else:
                out[:, offset:] = out[:, offset:] + projected[:, :-offset]
        return x + out / float(self.bandwidth)


class CalculusAugmentedLane(nn.Module):
    """Wrap a base lane with causal derivative/integral post-processing."""

    def __init__(self, base: nn.Module, dim: int) -> None:
        super().__init__()
        self.base = base
        self.calculus = FiniteDifferenceCalculusLane(dim)
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.calculus(self.base(x))


class LowRankAdapterLane(nn.Module):
    """Wrap a base lane with a low-rank residual adapter."""

    def __init__(self, base: nn.Module, dim: int, rank: int | None = None) -> None:
        super().__init__()
        self.base = base
        self.adapter = LowRankFactorizedLane(dim, rank=rank)
        self.dim = dim
        self.rank = self.adapter.rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        return y + (self.adapter(y) - y)


class SparseBandedAdapterLane(nn.Module):
    """Wrap a base lane with a causal sparse-banded residual adapter."""

    def __init__(self, base: nn.Module, dim: int, bandwidth: int = 4) -> None:
        super().__init__()
        self.base = base
        self.adapter = SparseBandedMatrixLane(dim, bandwidth=bandwidth)
        self.dim = dim
        self.bandwidth = bandwidth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        return y + (self.adapter(y) - y)


class RandomFeatureKernelLane(nn.Module):
    """Causal random-feature kernel mixer.

    Projects tokens into a positive random-feature map and computes causal
    linear attention from cumulative feature/value statistics. This supplies a
    kernel-method knob distinct from dot-product attention.
    """

    def __init__(self, dim: int, n_features: int | None = None) -> None:
        super().__init__()
        n_features = n_features or max(4, dim // 2)
        self.q_features = nn.Linear(dim, n_features, bias=False)
        self.k_features = nn.Linear(dim, n_features, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.n_features = n_features
        self.dim = dim

    def _positive_features(self, projection: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.elu(projection) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self._positive_features(self.q_features(x))
        k = self._positive_features(self.k_features(x))
        v = self.v(x)
        kv = torch.einsum("blf,bld->blfd", k, v).cumsum(dim=1)
        k_sum = k.cumsum(dim=1)
        numerator = torch.einsum("blf,blfd->bld", q, kv)
        denominator = (q * k_sum).sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return x + self.out(numerator / denominator)


class MultiscaleWaveletLane(nn.Module):
    """Haar-like causal multiscale lane.

    Blends local differences with causal averages at powers-of-two scales.
    This is a cheap wavelet/multiresolution knob without an FFT dependency.
    """

    def __init__(self, dim: int, n_scales: int = 3) -> None:
        super().__init__()
        if n_scales <= 0:
            raise ValueError("n_scales must be positive")
        self.projections = nn.ModuleList(
            [nn.Linear(dim, dim, bias=False) for _ in range(n_scales)]
        )
        self.mix = nn.Linear(dim * n_scales, dim, bias=False)
        self.n_scales = n_scales
        self.dim = dim

    def _causal_boxcar(self, x: torch.Tensor, width: int) -> torch.Tensor:
        seq_len = x.shape[1]
        csum = torch.nn.functional.pad(x.cumsum(dim=1), (0, 0, 1, 0))
        positions = torch.arange(seq_len, device=x.device)
        start = (positions - width + 1).clamp_min(0)
        total = csum[:, positions + 1] - csum[:, start]
        denom = (positions - start + 1).to(dtype=x.dtype).view(1, -1, 1)
        return total / denom

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features: list[torch.Tensor] = []
        prev_avg = x
        for scale, projection in enumerate(self.projections):
            width = 2**scale
            avg = self._causal_boxcar(x, width)
            detail = x - prev_avg if scale > 0 else x
            features.append(projection(avg + detail))
            prev_avg = avg
        return x + self.mix(torch.cat(features, dim=-1))


class GraphDiffusionLane(nn.Module):
    """Causal graph/Laplacian diffusion over token neighborhoods."""

    def __init__(self, dim: int, diffusion_steps: int = 2) -> None:
        super().__init__()
        if diffusion_steps <= 0:
            raise ValueError("diffusion_steps must be positive")
        self.self_proj = nn.Linear(dim, dim, bias=False)
        self.neighbor_proj = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Parameter(torch.tensor(0.5))
        self.diffusion_steps = diffusion_steps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = x
        alpha = torch.sigmoid(self.gate)
        for _ in range(self.diffusion_steps):
            prev = torch.zeros_like(state)
            prev[:, 1:] = state[:, :-1]
            state = (1.0 - alpha) * self.self_proj(state) + alpha * self.neighbor_proj(
                prev
            )
        return x + state


class RandomFeatureKernelAdapterLane(nn.Module):
    """Wrap a base lane with a random-feature kernel residual adapter."""

    def __init__(
        self, base: nn.Module, dim: int, n_features: int | None = None
    ) -> None:
        super().__init__()
        self.base = base
        self.adapter = RandomFeatureKernelLane(dim, n_features=n_features)
        self.dim = dim
        self.n_features = self.adapter.n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        return y + (self.adapter(y) - y)


class MultiscaleWaveletAdapterLane(nn.Module):
    """Wrap a base lane with causal multiscale residual mixing."""

    def __init__(self, base: nn.Module, dim: int, n_scales: int = 3) -> None:
        super().__init__()
        self.base = base
        self.adapter = MultiscaleWaveletLane(dim, n_scales=n_scales)
        self.dim = dim
        self.n_scales = n_scales

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        return y + (self.adapter(y) - y)


class GraphDiffusionAdapterLane(nn.Module):
    """Wrap a base lane with causal graph-diffusion residual mixing."""

    def __init__(self, base: nn.Module, dim: int, diffusion_steps: int = 2) -> None:
        super().__init__()
        self.base = base
        self.adapter = GraphDiffusionLane(dim, diffusion_steps=diffusion_steps)
        self.dim = dim
        self.diffusion_steps = diffusion_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        return y + (self.adapter(y) - y)


class CliffordAttention(nn.Module):
    """Cl(2,0) geometric-product attention.

    Splits the feature dim into ``dim // 4`` multivectors. Each multivector
    has 4 components ``(scalar, e1, e2, e12)`` where ``e1^2 = e2^2 = 1``
    and ``e12^2 = -1``. The attention affinity is the **scalar part** of
    the geometric product ``Q[i] * K[j]`` summed over multivectors:
    ``a*e + b*f + c*g - d*h``. Critically, the bivector term gets the
    opposite sign — this is the metric signature ``(+, +, +, -)`` and
    is what distinguishes Cl(2,0) from a pure euclidean dot product.
    """

    def __init__(self, dim: int, causal: bool = True) -> None:
        if dim % 4 != 0:
            raise ValueError(f"dim {dim} must be divisible by 4 for Cl(2,0)")
        super().__init__()
        self.dim = dim
        self.n_mv = dim // 4
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.scale = float(self.n_mv) ** -0.5
        self.causal = causal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.q(x).view(batch_size, seq_len, self.n_mv, 4)
        k = self.k(x).view(batch_size, seq_len, self.n_mv, 4)
        v = self.v(x).view(batch_size, seq_len, self.n_mv, 4)
        scalar_aff = (
            torch.einsum("bin,bjn->bij", q[..., 0], k[..., 0])
            + torch.einsum("bin,bjn->bij", q[..., 1], k[..., 1])
            + torch.einsum("bin,bjn->bij", q[..., 2], k[..., 2])
            - torch.einsum("bin,bjn->bij", q[..., 3], k[..., 3])
        ) * self.scale
        if self.causal:
            mask = torch.triu(
                torch.full(
                    (seq_len, seq_len), float("-inf"), device=x.device, dtype=x.dtype
                ),
                diagonal=1,
            )
            scalar_aff = scalar_aff + mask
        weights = torch.softmax(scalar_aff, dim=-1)
        out = torch.einsum("bij,bjnc->binc", weights, v)
        return out.reshape(batch_size, seq_len, self.dim)


class _SurrogateSpike(torch.autograd.Function):
    """Forward: hard step at threshold. Backward: arctan surrogate.

    The surrogate ``beta / (pi * (1 + (beta * (x - threshold))^2))``
    gives a smooth bump centered at the threshold so gradients flow
    through the spike step. Standard recipe in surrogate-gradient SNN
    literature (Neftci et al.).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, threshold: float, beta: float) -> torch.Tensor:  # type: ignore[override]
        ctx.save_for_backward(x)
        ctx.threshold = threshold
        ctx.beta = beta
        return (x > threshold).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (x,) = ctx.saved_tensors
        beta = ctx.beta
        threshold = ctx.threshold
        surrogate = beta / (1.0 + (beta * (x - threshold)) ** 2) / 3.141592653589793
        return grad_output * surrogate, None, None


class SpikingActivationGate(nn.Module):
    """Stateless surrogate-gradient spike threshold gate.

    ``proj_in -> SurrogateSpike(threshold) -> proj_out``. The output is
    a hard {0, 1} mask at runtime but gets a smooth gradient via the
    arctan surrogate. Useful as a discrete-activation lane that still
    trains end-to-end.
    """

    def __init__(self, dim: int, threshold: float = 0.5, beta: float = 2.0) -> None:
        super().__init__()
        self.proj_in = nn.Linear(dim, dim)
        self.proj_out = nn.Linear(dim, dim)
        self.threshold = float(threshold)
        self.beta = float(beta)
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        membrane = self.proj_in(x)
        spikes = _SurrogateSpike.apply(membrane, self.threshold, self.beta)
        return self.proj_out(spikes)


class PadicProjection(nn.Module):
    """Hierarchical block-shared projection — ultrametric-inspired.

    The feature dim is grouped into nested blocks at scales
    ``p^1, p^2, ..., p^n_levels``. At each level the projection is a
    learned linear shared across all blocks of that size, so two
    features in the same level-k block interact more strongly than
    features in different level-k blocks. That's the ultrametric: the
    "distance" between two features is determined by the smallest block
    containing both.

    Requires ``dim`` divisible by ``p^n_levels``.
    """

    def __init__(self, dim: int, p: int = 2, n_levels: int = 3) -> None:
        super().__init__()
        if dim % (p**n_levels) != 0:
            raise ValueError(
                f"dim {dim} must be divisible by p^n_levels = {p**n_levels}"
            )
        self.dim = dim
        self.p = p
        self.n_levels = n_levels
        self.projections = nn.ModuleList(
            [nn.Linear(p**k, p**k, bias=False) for k in range(1, n_levels + 1)]
        )
        self.gate = nn.Parameter(torch.ones(n_levels) / n_levels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        out = torch.zeros_like(x)
        for level, proj in enumerate(self.projections):
            block_size = self.p ** (level + 1)
            if dim % block_size != 0:
                continue
            n_blocks = dim // block_size
            reshaped = x.view(batch_size, seq_len, n_blocks, block_size)
            projected = proj(reshaped).view(batch_size, seq_len, dim)
            out = out + self.gate[level] * projected
        return out


class TropicalTopKStateSpace(nn.Module):
    """Max-plus recurrent kernel with top-k sparse state ("extra sparse compressed top-k state space").

    Combines three axes simultaneously:
    - tropical (max-plus) algebra
    - O(L) recurrent state
    - top-k sparse activation

    At each step the state evolves under ``s[t] = max(A + s[t-1], B(x[t]))``
    (tropical recurrence), then the top-k largest state components are
    kept and the rest zeroed (sparse compression). Output is
    ``C(s[t]) + x[t]``.

    Sits in the unbuilt ``(tropical, O(L), state, top_k)`` corner of
    property space.
    """

    def __init__(
        self, dim: int, state_dim: int | None = None, k: int | None = None
    ) -> None:
        super().__init__()
        state_dim = state_dim or dim
        k = k or max(1, state_dim // 4)
        if k > state_dim:
            raise ValueError(f"k={k} must be <= state_dim={state_dim}")
        self.A = nn.Parameter(torch.randn(state_dim) * 0.1)
        self.B = nn.Linear(dim, state_dim, bias=False)
        self.C = nn.Linear(state_dim, dim, bias=False)
        self.state_dim = state_dim
        self.dim = dim
        self.k = k

    def _topk_mask(self, state: torch.Tensor) -> torch.Tensor:
        _, indices = state.topk(self.k, dim=-1)
        mask = torch.zeros_like(state).scatter_(-1, indices, 1.0)
        return state * mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        Bx = self.B(x)
        state = torch.full(
            (batch_size, self.state_dim),
            float("-inf"),
            device=x.device,
            dtype=x.dtype,
        )
        state = self._topk_mask(torch.maximum(state, Bx[:, 0]))
        outputs = [self.C(state)]
        for t in range(1, seq_len):
            state = self._topk_mask(
                torch.maximum(self.A.unsqueeze(0) + state, Bx[:, t])
            )
            outputs.append(self.C(state))
        return torch.stack(outputs, dim=1) + x


class LinearStateSpaceLane(nn.Module):
    """Diagonal linear SSM with contractive per-channel recurrence.

    ``h[t] = a * h[t-1] + B(x[t])``, ``y[t] = C(h[t]) + x[t]``.
    Per-channel ``a`` passes through sigmoid so the recurrence is
    bounded in ``[0, 1]`` per channel — long-distance mixing without
    exploding state.

    Generic state-kernel primitive for the (any-algebra, O(L), has_state)
    corner that does not match a domain-specific state-space module
    (Tropical, etc.). Selected by ``_dispatch_state_kernel`` when
    ``op_dynamical_has_state=1`` and no algebra-specific dispatcher fires.
    """

    def __init__(self, dim: int, state_dim: int | None = None) -> None:
        super().__init__()
        state_dim = state_dim or dim
        self.a_raw = nn.Parameter(torch.zeros(state_dim))
        self.B = nn.Linear(dim, state_dim, bias=False)
        self.C = nn.Linear(state_dim, dim, bias=False)
        self.state_dim = state_dim
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        a = torch.sigmoid(self.a_raw)
        Bx = self.B(x)
        h = torch.zeros(batch_size, self.state_dim, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(seq_len):
            h = a.unsqueeze(0) * h + Bx[:, t]
            outputs.append(self.C(h))
        return torch.stack(outputs, dim=1) + x


class CausalFastWeightMemoryLane(nn.Module):
    """Causal fast-weight memory lane.

    Maintains a per-example fast-weight matrix ``M[t]`` updated from the
    current token's learned key/value outer product, then reads it with the
    current learned query. This is an invention-track primitive: a stateful
    content-addressed mixer with explicit write decay and no softmax over
    prior token positions.
    """

    def __init__(self, dim: int, memory_dim: int | None = None) -> None:
        super().__init__()
        memory_dim = memory_dim or min(dim, 32)
        self.q = nn.Linear(dim, memory_dim, bias=False)
        self.k = nn.Linear(dim, memory_dim, bias=False)
        self.v = nn.Linear(dim, memory_dim, bias=False)
        self.write_gate = nn.Linear(dim, 1)
        self.out = nn.Linear(memory_dim, dim, bias=False)
        self.decay_logit = nn.Parameter(torch.tensor(1.5))
        self.dim = dim
        self.memory_dim = memory_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = torch.tanh(self.q(x))
        k = torch.tanh(self.k(x))
        v = torch.tanh(self.v(x))
        gates = torch.sigmoid(self.write_gate(x)).squeeze(-1)
        decay = torch.sigmoid(self.decay_logit)
        memory = torch.zeros(
            batch_size,
            self.memory_dim,
            self.memory_dim,
            device=x.device,
            dtype=x.dtype,
        )
        outputs = []
        scale = float(self.memory_dim) ** -0.5
        for t in range(seq_len):
            write = torch.einsum("bi,bj->bij", k[:, t], v[:, t]) * scale
            memory = decay * memory + gates[:, t].view(batch_size, 1, 1) * write
            read = torch.einsum("bi,bij->bj", q[:, t], memory)
            outputs.append(self.out(read))
        return torch.stack(outputs, dim=1)


class CausalSlotRouterMemoryLane(nn.Module):
    """Small causal slot-memory router.

    Each token softly selects one of ``n_slots`` persistent memory slots,
    writes a gated candidate into the selected slots, then reads a weighted
    slot mixture. It is meant to test routing-as-memory rather than routing
    over existing expert lanes.
    """

    def __init__(self, dim: int, n_slots: int = 4) -> None:
        super().__init__()
        if n_slots <= 0:
            raise ValueError("n_slots must be positive")
        self.route = nn.Linear(dim, n_slots)
        self.write = nn.Linear(dim, dim)
        self.write_gate = nn.Linear(dim, n_slots)
        self.out = nn.Linear(dim, dim, bias=False)
        self.n_slots = n_slots
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        slots = torch.zeros(
            batch_size, self.n_slots, dim, device=x.device, dtype=x.dtype
        )
        outputs = []
        for t in range(seq_len):
            token = x[:, t]
            route = torch.softmax(self.route(token), dim=-1)
            gate = torch.sigmoid(self.write_gate(token))
            candidate = torch.tanh(self.write(token))
            write_weight = (route * gate).unsqueeze(-1)
            slots = slots * (1.0 - write_weight) + write_weight * candidate.unsqueeze(1)
            read = torch.einsum("bs,bsd->bd", route, slots)
            outputs.append(self.out(read))
        return torch.stack(outputs, dim=1)


class HierarchicalResidualCompressorLane(nn.Module):
    """Causal multi-timescale residual compressor.

    Keeps a small stack of learned summaries updated at powers-of-two
    intervals. The output reads all summaries through learned gates. This
    gives the fab an explicit compression candidate whose state budget is
    fixed in the number of levels rather than growing with sequence length.
    """

    def __init__(self, dim: int, n_levels: int = 4) -> None:
        super().__init__()
        if n_levels <= 0:
            raise ValueError("n_levels must be positive")
        self.updates = nn.ModuleList([nn.Linear(dim * 2, dim) for _ in range(n_levels)])
        self.gates = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_levels)])
        self.read = nn.Linear(dim * n_levels, dim, bias=False)
        self.n_levels = n_levels
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        summaries = [
            torch.zeros(batch_size, dim, device=x.device, dtype=x.dtype)
            for _ in range(self.n_levels)
        ]
        outputs = []
        for t in range(seq_len):
            token = x[:, t]
            for level, update in enumerate(self.updates):
                period = 2**level
                if t % period != 0:
                    continue
                candidate = torch.tanh(
                    update(torch.cat([summaries[level], token], dim=-1))
                )
                gate = torch.sigmoid(self.gates[level](token))
                summaries[level] = (1.0 - gate) * summaries[level] + gate * candidate
            outputs.append(self.read(torch.cat(summaries, dim=-1)))
        return torch.stack(outputs, dim=1)


class SymplecticResidualMixerLane(nn.Module):
    """Causal symplectic-style residual mixer.

    Splits channels into two halves ``(q, p)`` and applies an alternating
    update resembling a Hamiltonian step, with a causal running context as
    the sequence memory. The structure is deliberately different from
    attention/conv/SSM while staying shape-preserving and causal.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("dim must be even for SymplecticResidualMixerLane")
        half = dim // 2
        self.q_update = nn.Linear(dim, half, bias=False)
        self.p_update = nn.Linear(dim, half, bias=False)
        self.context_gate = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim, bias=False)
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        denom = torch.arange(1, seq_len + 1, dtype=x.dtype, device=x.device).view(
            1, -1, 1
        )
        context = x.cumsum(dim=1) / denom
        gated_context = torch.sigmoid(self.context_gate(x)) * context
        q, p = gated_context.chunk(2, dim=-1)
        q_next = q + torch.tanh(self.q_update(torch.cat([q, p], dim=-1)))
        p_next = p - torch.tanh(self.p_update(torch.cat([q_next, p], dim=-1)))
        return self.out(torch.cat([q_next, p_next], dim=-1))
