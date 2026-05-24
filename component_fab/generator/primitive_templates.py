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

from component_fab.harness.rope import RotaryEmbedding, apply_rope


def _disable_torch_compile(fn):
    try:
        return torch.compiler.disable(fn)
    except Exception:
        try:
            return torch._dynamo.disable(fn)
        except Exception:
            return fn


@_disable_torch_compile
def _cumsum_dim1_eager(x: torch.Tensor) -> torch.Tensor:
    """Keep sequence scans out of Inductor's unstable SplitScan lowering."""
    return x.cumsum(dim=1)


class _QKVRopeAttentionBase(nn.Module):
    def __init__(
        self,
        dim: int,
        causal: bool = True,
        *,
        use_rope: bool = False,
        max_seq_len: int = 1024,
    ) -> None:
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.scale = float(dim) ** -0.5
        self.dim = dim
        self.causal = causal
        self.rope = RotaryEmbedding(dim, max_seq_len=max_seq_len) if use_rope else None

    def _project_affinity_values(
        self,
        x: torch.Tensor,
        *,
        causal_mask_value: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[1]
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        if self.rope is not None:
            cos, sin = self.rope(seq_len, device=x.device, dtype=x.dtype)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        affinity = torch.einsum("bid,bjd->bij", q, k) * self.scale
        if self.causal:
            mask = torch.triu(
                torch.full(
                    (seq_len, seq_len),
                    causal_mask_value,
                    device=x.device,
                    dtype=x.dtype,
                ),
                diagonal=1,
            )
            affinity = affinity + mask
        return affinity, v


class TropicalAttention(_QKVRopeAttentionBase):
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

    def __init__(
        self,
        dim: int,
        causal: bool = True,
        *,
        use_rope: bool = False,
        max_seq_len: int = 1024,
    ) -> None:
        super().__init__(
            dim,
            causal=causal,
            use_rope=use_rope,
            max_seq_len=max_seq_len,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        affinity, v = self._project_affinity_values(
            x,
            causal_mask_value=float("-inf"),
        )
        combined = affinity.unsqueeze(-1) + v.unsqueeze(1)
        return combined.max(dim=2).values


def _causal_sparsemax(logits: torch.Tensor) -> torch.Tensor:
    """Sparsemax along the last dim with a causal mask already baked in.

    Inputs are expected to have masked positions set to a large negative
    value (not ``-inf`` — sparsemax's cumsum is NaN-fragile). Returns a
    sparse probability tensor whose nonzero entries sum to 1 along the
    last axis.
    """
    sorted_logits, _ = torch.sort(logits, dim=-1, descending=True)
    k = torch.arange(1, logits.size(-1) + 1, device=logits.device, dtype=logits.dtype)
    cumsum = sorted_logits.cumsum(dim=-1)
    support = 1 + k * sorted_logits > cumsum
    k_star = support.to(logits.dtype).sum(dim=-1, keepdim=True).clamp_min(1.0)
    gather_idx = (k_star.long() - 1).clamp_min(0)
    cumsum_at_star = cumsum.gather(-1, gather_idx)
    tau = (cumsum_at_star - 1) / k_star
    return (logits - tau).clamp_min(0.0)


class SparsemaxAttention(_QKVRopeAttentionBase):
    """Causal attention with sparsemax instead of softmax.

    Identity to standard scaled-dot-product attention everywhere except
    the weight normalization step: sparsemax(QK^T / sqrt(d)) projects the
    affinities onto the probability simplex and naturally zeros out all
    but the top-K positions (K is content-dependent, learned per query).

    Sits architecturally between ``TropicalAttention`` (max — keeps 1
    position, hard winner) and ``SoftmaxCausalAttention`` (softmax — keeps
    all positions, dense weighted average). Useful as the "integration"
    lane in a 3-lane gate alongside tropical.
    """

    _NEG_LARGE: float = -1e4

    def __init__(
        self,
        dim: int,
        causal: bool = True,
        *,
        use_rope: bool = False,
        max_seq_len: int = 1024,
    ) -> None:
        super().__init__(
            dim,
            causal=causal,
            use_rope=use_rope,
            max_seq_len=max_seq_len,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        affinity, v = self._project_affinity_values(
            x,
            causal_mask_value=self._NEG_LARGE,
        )
        weights = _causal_sparsemax(affinity)
        return torch.einsum("bij,bjd->bid", weights, v)


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
        running_integral = _cumsum_dim1_eager(x) / denom
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
        context = _cumsum_dim1_eager(z) / denom
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
        kv = _cumsum_dim1_eager(torch.einsum("blf,bld->blfd", k, v))
        k_sum = _cumsum_dim1_eager(k)
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
        csum = torch.nn.functional.pad(_cumsum_dim1_eager(x), (0, 0, 1, 0))
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
        from research.synthesis.compiler_ops_sequence import _parallel_associative_scan

        batch_size, seq_len, _ = x.shape
        a = torch.sigmoid(self.a_raw).clamp(1e-6, 1.0 - 1e-6)  # avoid log(0)
        Bx = self.B(x)  # [B, L, state]
        # Parallel Kogge-Stone scan over h[t] = a * h[t-1] + Bx[t].
        # log_a is constant across L (no selectivity here) but scan expects [..., L].
        log_a = torch.log(a).view(1, -1, 1).expand(batch_size, -1, seq_len).contiguous()
        b_seq = Bx.transpose(-1, -2).contiguous()  # [B, state, L]
        h_t = _parallel_associative_scan(log_a, b_seq)  # [B, state, L]
        h_seq = h_t.transpose(-1, -2)  # [B, L, state]
        return self.C(h_seq) + x


class FisherAttention(nn.Module):
    """Fisher-information attention (information_geometry knob).

    Replaces the Euclidean dot-product affinity with a Fisher-information
    metric: instead of treating Q, K as raw vectors, treats them as
    parameters of a Gaussian distribution (mean Q, diagonal cov K^2)
    and uses KL divergence as inverse affinity. This is geometry-aware
    on the simplex of distributions, not on Euclidean space.

    ``affinity[i, j] = -KL(N(q_i, I) || N(k_j, I)) = -0.5 * ||q_i - k_j||^2``
    plus a learned scale. With Q, K parameterizing distributions, the
    optimization landscape respects the manifold of probability measures.
    """

    def __init__(self, dim: int, causal: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.log_scale = nn.Parameter(torch.zeros(1))
        self.causal = causal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        # Squared distance per pair (the Fisher metric for unit-variance Gaussians).
        diff = q.unsqueeze(2) - k.unsqueeze(1)
        sq = (diff * diff).sum(dim=-1)
        affinity = -0.5 * sq * torch.exp(self.log_scale)
        if self.causal:
            mask = torch.triu(
                torch.full((l, l), float("-inf"), device=x.device, dtype=x.dtype),
                diagonal=1,
            )
            affinity = affinity + mask
        weights = torch.softmax(affinity, dim=-1)
        return torch.einsum("bij,bjd->bid", weights, v)


class ChebyshevSpectralLane(nn.Module):
    """Spectral-graph knob using Chebyshev polynomial features along sequence.

    Computes Chebyshev polynomials T_0..T_K of normalized sequence-position
    indices, then linearly combines them per channel. T_k is recursively
    defined by T_0(x)=1, T_1(x)=x, T_k(x)=2x*T_{k-1}(x)-T_{k-2}(x). The
    polynomial is causal-by-construction since T_k(t) depends only on t.

    Useful when token-position is meaningful (most language modeling):
    Chebyshev basis is the conditioning-optimal polynomial family on
    [-1, 1], outperforming Fourier on aperiodic signals.
    """

    def __init__(self, dim: int, n_terms: int = 5, max_seq_len: int = 512) -> None:
        super().__init__()
        if n_terms < 1:
            raise ValueError("n_terms must be >= 1")
        self.dim = dim
        self.n_terms = n_terms
        self.max_seq_len = max_seq_len
        # Learned mixing per (term, in_channel, out_channel).
        self.mix = nn.Parameter(torch.randn(n_terms, dim, dim) / (dim**0.5))
        self.gate = nn.Linear(dim, dim, bias=False)

    def _chebyshev_basis(self, seq_len: int, device, dtype) -> torch.Tensor:
        # Map position [0, L-1] -> [-1, 1].
        t = torch.linspace(-1.0, 1.0, seq_len, device=device, dtype=dtype)
        basis = [torch.ones_like(t), t]
        for _ in range(2, self.n_terms):
            basis.append(2.0 * t * basis[-1] - basis[-2])
        return torch.stack(basis[: self.n_terms], dim=0)  # [n_terms, L]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, d = x.shape
        cheb = self._chebyshev_basis(l, x.device, x.dtype)  # [n_terms, L]
        # Project each Chebyshev component through its own dim×dim mix matrix.
        # Output = sum_k cheb[k, t] * (x @ mix[k]). Aggregate over k.
        projected = torch.einsum("bld,kde->bkle", x, self.mix)  # [B, n_terms, L, D]
        weighted = projected * cheb.view(1, self.n_terms, l, 1)
        summed = weighted.sum(dim=1)  # [B, L, D]
        return torch.sigmoid(self.gate(x)) * summed


class TuckerDecompLane(nn.Module):
    """Tensor-decomp knob: Tucker decomposition of the channel-mix tensor.

    A standard linear lane is parameterized by a single ``D × D`` matrix
    (``D²`` parameters). Tucker decomposes this as a small core tensor
    contracted with mode matrices: ``W = sum_{r,s} core[r, s] * U[r] ⊗ V[s]``
    where ``U, V`` are ``D × rank``. This drops parameter count from
    ``D²`` to ``2 * D * rank + rank²`` while preserving the bilinear
    structure — different from low-rank-factorized which forces
    rank-1-product structure.

    Useful when the channel-mixing operator has structure that low-rank
    misses (e.g. block-diagonal, banded, or interaction terms).
    """

    def __init__(self, dim: int, rank: int | None = None) -> None:
        super().__init__()
        rank = rank or max(2, dim // 4)
        self.dim = dim
        self.rank = rank
        # Mode matrices.
        self.u = nn.Parameter(torch.randn(dim, rank) / (dim**0.5))
        self.v = nn.Parameter(torch.randn(dim, rank) / (dim**0.5))
        # Core tensor.
        self.core = nn.Parameter(torch.randn(rank, rank) / (rank**0.5))
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compose Tucker mix: x -> x @ U -> @ core -> @ V^T -> out.
        z = x @ self.u  # [B, L, rank]
        z = z @ self.core  # [B, L, rank]
        z = z @ self.v.t()  # [B, L, dim]
        return self.out(z)


class FisherAdapterLane(nn.Module):
    """Wrap a base lane with a Fisher-affinity residual adapter."""

    def __init__(self, base: nn.Module, dim: int) -> None:
        super().__init__()
        self.base = base
        self.adapter = FisherAttention(dim)
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        return y + (self.adapter(y) - y)


class ChebyshevAdapterLane(nn.Module):
    """Wrap a base lane with Chebyshev spectral residual mixing."""

    def __init__(self, base: nn.Module, dim: int, n_terms: int = 5) -> None:
        super().__init__()
        self.base = base
        self.adapter = ChebyshevSpectralLane(dim, n_terms=n_terms)
        self.dim = dim
        self.n_terms = n_terms

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        return y + (self.adapter(y) - y)


class TuckerAdapterLane(nn.Module):
    """Wrap a base lane with Tucker-decomposed channel mixing."""

    def __init__(self, base: nn.Module, dim: int, rank: int | None = None) -> None:
        super().__init__()
        self.base = base
        self.adapter = TuckerDecompLane(dim, rank=rank)
        self.dim = dim
        self.rank = self.adapter.rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        return y + (self.adapter(y) - y)


class QuaternionAttention(nn.Module):
    """Quaternion-valued causal attention via Hamilton product affinity.

    Splits ``dim`` into ``dim/4`` quaternions ``(w, x, y, z)``. The
    quaternion-product affinity uses all four components symmetrically:
    ``Re(Q[i] * conj(K[j]))`` which equals ``w1*w2 + x1*x2 + y1*y2 + z1*z2``
    (the quaternion inner product). The bivector terms cancel in the real
    part, so for affinity this is identical to euclidean dot product *but*
    the value side is updated in quaternion algebra:

    ``V_out[i, q] = sum_j attn[i, j] * (Q[i] * V[j])``

    Where the multiplication is the full Hamilton product, mixing all
    four components per quaternion. This is fundamentally different from
    real-valued attention: it composes rotations, not just sums.

    Requires ``dim % 4 == 0``.
    """

    def __init__(self, dim: int, causal: bool = True) -> None:
        if dim % 4 != 0:
            raise ValueError(f"dim {dim} must be divisible by 4 for quaternions")
        super().__init__()
        self.dim = dim
        self.n_q = dim // 4
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.scale = float(self.n_q) ** -0.5
        self.causal = causal

    @staticmethod
    def _ham(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Hamilton product of two ``[..., n_q, 4]`` quaternion tensors."""
        a_w, a_x, a_y, a_z = a.unbind(dim=-1)
        b_w, b_x, b_y, b_z = b.unbind(dim=-1)
        w = a_w * b_w - a_x * b_x - a_y * b_y - a_z * b_z
        x = a_w * b_x + a_x * b_w + a_y * b_z - a_z * b_y
        y = a_w * b_y - a_x * b_z + a_y * b_w + a_z * b_x
        z = a_w * b_z + a_x * b_y - a_y * b_x + a_z * b_w
        return torch.stack((w, x, y, z), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        q = self.q(x).view(b, l, self.n_q, 4)
        k = self.k(x).view(b, l, self.n_q, 4)
        v = self.v(x).view(b, l, self.n_q, 4)
        # Quaternion inner product affinity: sum over (n_q, 4 components).
        affinity = torch.einsum("binc,bjnc->bij", q, k) * self.scale
        if self.causal:
            mask = torch.triu(
                torch.full((l, l), float("-inf"), device=x.device, dtype=x.dtype),
                diagonal=1,
            )
            affinity = affinity + mask
        weights = torch.softmax(affinity, dim=-1)
        # Quaternion-multiply each query by aggregated value, then flatten.
        aggregated = torch.einsum("bij,bjnc->binc", weights, v)
        # Compose query with aggregated value via Hamilton product (rotation).
        composed = self._ham(q, aggregated)
        return composed.reshape(b, l, self.dim)


class PoincareAttention(nn.Module):
    """Causal attention with Poincaré-ball hyperbolic affinity.

    Maps Q/K projections through the exponential map at origin into the
    open unit ball (curvature c=1), measures affinity via the negative
    squared hyperbolic distance ``-d_H(q, k)^2 = -arcosh(1 + 2 ||q-k||^2 /
    ((1 - ||q||^2)(1 - ||k||^2)))^2``, then mixes V back in euclidean
    space. Hyperbolic geometry is exponentially-spread, so it naturally
    encodes tree-like / hierarchical token relations at small dim.

    The ``c=1`` curvature is fixed; ``project_to_ball`` clamps norms below
    ``1 - eps`` to keep arcosh finite.
    """

    def __init__(self, dim: int, causal: bool = True, eps: float = 1e-4) -> None:
        super().__init__()
        self.dim = dim
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.scale = float(dim) ** -0.5
        self.causal = causal
        self.eps = float(eps)

    def _project_to_ball(self, x: torch.Tensor) -> torch.Tensor:
        # Euclidean -> Poincaré: tanh of half-norm preserves direction and
        # shrinks norm into (0, 1).
        norm = x.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        scaled = torch.tanh(norm * 0.5) / norm
        return x * scaled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        q_ball = self._project_to_ball(self.q(x))
        k_ball = self._project_to_ball(self.k(x))
        v = self.v(x)
        q_norm_sq = (q_ball * q_ball).sum(dim=-1).clamp_max(1.0 - self.eps)
        k_norm_sq = (k_ball * k_ball).sum(dim=-1).clamp_max(1.0 - self.eps)
        diff = q_ball.unsqueeze(2) - k_ball.unsqueeze(1)
        diff_norm_sq = (diff * diff).sum(dim=-1)
        denom = (1.0 - q_norm_sq).unsqueeze(2) * (1.0 - k_norm_sq).unsqueeze(1)
        denom = denom.clamp_min(self.eps)
        dist_arg = 1.0 + 2.0 * diff_norm_sq / denom
        dist_arg = dist_arg.clamp_min(1.0 + self.eps)
        hyp_dist = torch.acosh(dist_arg)
        affinity = -(hyp_dist * hyp_dist) * self.scale
        if self.causal:
            mask = torch.triu(
                torch.full((l, l), float("-inf"), device=x.device, dtype=x.dtype),
                diagonal=1,
            )
            affinity = affinity + mask
        weights = torch.softmax(affinity, dim=-1)
        return torch.einsum("bij,bjd->bid", weights, v)


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
        context = _cumsum_dim1_eager(x) / denom
        gated_context = torch.sigmoid(self.context_gate(x)) * context
        q, p = gated_context.chunk(2, dim=-1)
        q_next = q + torch.tanh(self.q_update(torch.cat([q, p], dim=-1)))
        p_next = p - torch.tanh(self.p_update(torch.cat([q_next, p], dim=-1)))
        return self.out(torch.cat([q_next, p_next], dim=-1))
