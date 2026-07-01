"""Novel-math invention lanes for component_fab (Tier-1 + long-tail expansion).

Genuinely new, non-QKV sequence mixers, all anti-softmax-twin by structure:

- ``FractionalIntegralMemoryLane`` (NM-T1-3) — a causal depthwise convolution
  whose kernel is the Grünwald–Letnikov *fractional-integral* weight sequence
  ``w_k = Γ(k+α) / (Γ(k+1) Γ(α))`` (positive, ``∝ k**(α-1)``). Unlike the
  exponential decay of SSM / linear-attention memory, this is a **power-law**
  memory: a token at lag ``τ`` still contributes ``∝ τ**(α-1)``. ``α`` is a
  learnable per-channel parameter in ``(0, 1)``: ``α → 1`` flattens the kernel
  toward a running average (longest memory), ``α → 0`` collapses to the current
  token (near identity). This is the fractional *integral* (accumulating,
  low-pass) — NOT the fractional derivative (a differencer that does not
  accumulate memory).

- ``SheafDiffusionMixerLane`` (NM-T1-2) — causal windowed **sheaf-Laplacian
  diffusion**. Each token stalk is pulled to agree, under a learned restriction
  map ``R``, with its causal-window neighbours: ``k`` gradient-descent steps on
  the sheaf Dirichlet energy ``Σ_{s∈window} ‖R x_t − R x_s‖²``. Mixing is
  *overlap-agreement*, not score-weighted aggregation — there is no softmax
  analog. ``R`` is a learned non-identity linear map (the anti-collapse guard is
  keeping ``R`` non-degenerate; a training-time restriction-consistency penalty
  can be added on top).

- ``SignedExpanderMixerLane`` (MiniMax-M3-align M3X-M2) — a compact signed
  channel-expander mixer. It uses a deterministic regular circulant expander over
  channels, learned signed edge weights, and a causal decayed token context. The
  mechanism spreads information globally with ``O(D * degree)`` work/footprint
  rather than a dense ``O(D**2)`` matrix, while staying far from convex
  softmax-style token averaging.

Both preserve ``[B, L, D]`` shape and produce finite gradients at init.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from component_fab.generator._causal_scan import causal_decay_context


def _canonical_offset(raw: int, dim: int) -> int:
    offset = int(raw) % dim
    if offset == 0:
        return 0
    return min(offset, dim - offset)


def _expander_offsets(dim: int, n_offsets: int) -> tuple[int, ...]:
    """Deterministic coprime circulant offsets for a sparse channel expander."""
    max_offset = max(1, (dim - 1) // 2)
    offsets: list[int] = []

    def try_add(raw: int, *, require_coprime: bool = False) -> None:
        offset = _canonical_offset(raw, dim)
        if offset <= 0 or offset > max_offset or offset in offsets:
            return
        if not require_coprime or math.gcd(offset, dim) == 1:
            offsets.append(offset)

    try_add(1, require_coprime=True)
    for frac in (0.17, 0.25, 0.31, 0.43, 0.37, 0.23, 0.41):
        try_add(round(dim * frac))
        if len(offsets) >= n_offsets:
            break
    candidate = 2
    while len(offsets) < n_offsets and candidate <= max_offset:
        try_add(candidate)
        candidate += 1
    candidate = 2
    while len(offsets) < n_offsets and candidate <= max_offset:
        offset = _canonical_offset(candidate, dim)
        if offset and offset not in offsets:
            offsets.append(offset)
        candidate += 1
    return tuple(offsets)


class SignedExpanderMixerLane(nn.Module):
    """Signed regular-expander causal mixer (MiniMax-M3-align M3X-M2).

    The channel mixer is a sparse circulant expander: every channel reads the
    same small set of positive and negative offset neighbours, with learned
    signed edge weights shared across channels. A causal decayed token context
    supplies the sequence dimension, so token ``t`` reads only tokens ``<= t``.

    This is not attention: there are no query/key scores, no exp/normalization,
    and no convex aggregation over tokens. The compactness comes from storing
    ``O(D)`` per-channel scales/decays plus ``O(degree)`` edge weights rather
    than a dense ``D x D`` matrix.
    """

    def __init__(self, dim: int, degree: int = 8, decay_init: float = 0.75) -> None:
        super().__init__()
        if dim < 4:
            raise ValueError("SignedExpanderMixerLane requires dim >= 4")
        if degree < 2:
            raise ValueError("degree must be >= 2")
        if not 0.0 < decay_init < 1.0:
            raise ValueError("decay_init must be in (0, 1)")
        n_offsets = min(max(1, degree // 2), max(1, (dim - 1) // 2))
        offsets = _expander_offsets(dim, n_offsets)
        if not offsets:
            raise ValueError("could not build non-empty expander offsets")
        self.dim = dim
        self.degree = 2 * len(offsets)
        self.register_buffer(
            "_offsets", torch.tensor(offsets, dtype=torch.long), persistent=False
        )
        init_weights = torch.empty(len(offsets))
        init_weights[0::2] = 0.8
        init_weights[1::2] = -0.8
        self.edge_logits = nn.Parameter(torch.atanh(init_weights))
        self.input_scale = nn.Parameter(torch.ones(dim))
        self.output_scale = nn.Parameter(torch.full((dim,), 12.0))
        self.mix_gate = nn.Parameter(torch.full((dim,), 0.5))
        logit = torch.logit(torch.tensor(decay_init))
        self.log_decay = nn.Parameter(torch.full((dim,), float(logit)))

    def edge_weights(self) -> torch.Tensor:
        """Learned signed edge weights, one per undirected offset."""
        return torch.tanh(self.edge_logits)

    def channel_adjacency(self) -> torch.Tensor:
        """Absolute normalized channel adjacency used by tests/diagnostics."""
        weights = self.edge_weights().abs()
        rows = torch.arange(self.dim, device=weights.device)
        adj = weights.new_zeros(self.dim, self.dim)
        for idx, raw_offset in enumerate(self._offsets.tolist()):
            offset = int(raw_offset)
            adj[rows, (rows + offset) % self.dim] = weights[idx]
            adj[rows, (rows - offset) % self.dim] = weights[idx]
        denom = adj.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return adj / denom

    def spectral_gap(self) -> torch.Tensor:
        """Gap ``1 - |lambda_2|`` of the normalized unsigned expander graph."""
        eigvals = torch.linalg.eigvals(self.channel_adjacency()).abs()
        ordered = torch.sort(eigvals, descending=True).values
        if ordered.numel() < 2:
            return ordered.new_tensor(0.0)
        return 1.0 - ordered[1].real

    def _mix_channels(self, context: torch.Tensor) -> torch.Tensor:
        weights = self.edge_weights().to(context.dtype)
        mixed = torch.zeros_like(context)
        for idx, raw_offset in enumerate(self._offsets.tolist()):
            offset = int(raw_offset)
            neighbours = torch.roll(context, offset, dims=-1) + torch.roll(
                context, -offset, dims=-1
            )
            mixed = mixed + weights[idx] * neighbours
        return mixed / math.sqrt(float(self.degree))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        decay = torch.sigmoid(self.log_decay).clamp(1e-4, 1 - 1e-4)
        context = causal_decay_context(x * self.input_scale, decay)
        mixed = self._mix_channels(context)
        # Signed expander high-pass: subtracting the causal context
        # prevents the lane from becoming a non-negative averaging operator.
        update = (mixed - context) * self.output_scale
        return x + torch.tanh(self.mix_gate) * update


class FractionalIntegralMemoryLane(nn.Module):
    """Causal fractional-integral memory (Riemann–Liouville ``I**α``).

    Applies a per-channel causal convolution with the Grünwald–Letnikov
    fractional-integral kernel. The kernel is recomputed from the learnable
    per-channel order ``α ∈ (0, 1)`` on every forward via ``lgamma`` (so the
    memory horizon is trainable), then normalized to sum to 1 per channel for
    numerical stability — normalization preserves the relative power-law profile
    (the novel inductive bias) while bounding the output to input scale.
    """

    def __init__(self, dim: int, kernel_len: int = 256) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if kernel_len <= 1:
            raise ValueError("kernel_len must be > 1")
        self.dim = dim
        self.kernel_len = kernel_len
        self.in_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        # α = sigmoid(alpha_logit) ∈ (0, 1); init at 0 → α ≈ 0.5.
        self.alpha_logit = nn.Parameter(torch.zeros(dim))
        # Lag indices k = 0 .. K-1 (kept as a buffer so device/dtype follow x).
        self.register_buffer(
            "_k_idx", torch.arange(kernel_len, dtype=torch.float32), persistent=False
        )

    def alphas(self) -> torch.Tensor:
        """Per-channel fractional order ``α ∈ (eps, 1-eps)``."""
        return torch.sigmoid(self.alpha_logit).clamp(1e-3, 1.0 - 1e-3)

    def kernel(self) -> torch.Tensor:
        """Normalized per-channel GL fractional-integral kernel ``[dim, K]``."""
        alpha = self.alphas().unsqueeze(-1)  # [D, 1]
        k = self._k_idx.to(alpha.dtype).unsqueeze(0)  # [1, K]
        # log Γ(k+α) - log Γ(k+1) - log Γ(α); positive, ∝ k**(α-1).
        log_w = torch.lgamma(k + alpha) - torch.lgamma(k + 1.0) - torch.lgamma(alpha)
        w = torch.exp(log_w)  # [D, K], w[:, 0] == 1
        return w / w.sum(dim=-1, keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = x.shape
        h = self.in_proj(x).transpose(1, 2)  # [B, D, L]
        w = self.kernel().to(h.dtype)  # [D, K]
        # Truncate the kernel to the sequence length (no future leakage anyway).
        k_use = min(self.kernel_len, seq_len)
        weight = w[:, :k_use].flip(-1).unsqueeze(1)  # [D, 1, k_use], causal
        h_pad = F.pad(h, (k_use - 1, 0))
        y = F.conv1d(h_pad, weight, groups=self.dim)  # [B, D, L]
        return self.out_proj(y.transpose(1, 2))


class SheafDiffusionMixerLane(nn.Module):
    """Causal windowed sheaf-Laplacian diffusion mixer.

    For each token ``t`` the stalk is pulled toward agreement, under a learned
    restriction map ``R``, with the mean of its causal-window predecessors:

        x_t ← x_t − α · Rᵀ ( R x_t − mean_{s∈[t-w, t-1]} R x_s )

    repeated for ``n_steps`` — i.e. gradient descent on the sheaf Dirichlet
    energy ``Σ ‖R x_t − R x_s‖²``. This is overlap-agreement, not score-weighted
    aggregation: there is no exp / normalize over keys, so it is structurally not
    a softmax twin. Strictly causal (only predecessors contribute) and finite at
    init.
    """

    def __init__(
        self, dim: int, window: int = 8, n_steps: int = 3, max_alpha: float = 0.5
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if window <= 0:
            raise ValueError("window must be positive")
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        self.dim = dim
        self.window = window
        self.n_steps = n_steps
        self.max_alpha = float(max_alpha)
        # Restriction map R (kept non-identity — the anti-collapse guard).
        self.restrict = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.alpha_logit = nn.Parameter(torch.zeros(1))

    def _causal_prev_mean(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean of ``z[s]`` over ``s ∈ [t-window, t-1]`` (0 where none)."""
        batch, seq_len, dim = z.shape
        cs = torch.cumsum(z, dim=1)
        prefix = torch.cat([z.new_zeros(batch, 1, dim), cs], dim=1)  # S[:, t] = Σ_{<t}
        t_idx = torch.arange(seq_len, device=z.device)
        lower = (t_idx - self.window).clamp(min=0)
        window_sum = prefix[:, t_idx, :] - prefix[:, lower, :]  # Σ_{lower..t-1}
        count = (t_idx - lower).to(z.dtype).view(1, seq_len, 1)
        valid = (count > 0).to(z.dtype)
        prev_mean = window_sum / count.clamp(min=1.0)
        return prev_mean, valid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = torch.sigmoid(self.alpha_logit) * self.max_alpha
        h = x
        for _ in range(self.n_steps):
            rh = self.restrict(h)  # R h
            prev_mean, valid = self._causal_prev_mean(rh)
            disagree = (rh - prev_mean) * valid  # R x_t − mean R x_s
            update = disagree @ self.restrict.weight  # Rᵀ applied
            h = h - alpha * update
        return self.out(h)


class MeraRenormMixerLane(nn.Module):
    """Causal MERA-style multi-scale renormalization mixer (NM-T1-4).

    A dilated binary-tree renormalization group: at level ``l`` each token is
    paired with its ``2**l``-ago predecessor, a learned **disentangler** ``U``
    removes the cross-scale correlation, and a learned **isometry** ``W``
    coarse-grains the disentangled pair to one site — so the receptive field
    doubles per level (Vidal-style MERA: alternating disentanglers + isometries,
    the strict tensor-network renorm group, not level-wise gated summaries). The
    per-token readout concatenates every scale. At init ``U = identity`` and
    ``W = average``, so the lane is a stable causal multi-scale moving-average
    pyramid; training moves ``U``/``W`` off that. Strictly causal (each level
    only looks back ``2**l``) and finite at init.
    """

    def __init__(self, dim: int, n_levels: int = 3) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if n_levels <= 0:
            raise ValueError("n_levels must be positive")
        self.dim = dim
        self.n_levels = n_levels
        self.disentanglers = nn.ModuleList(
            nn.Linear(2 * dim, 2 * dim) for _ in range(n_levels)
        )
        self.isometries = nn.ModuleList(
            nn.Linear(2 * dim, dim) for _ in range(n_levels)
        )
        self.read = nn.Linear(dim * (n_levels + 1), dim, bias=False)
        self._init_renorm()

    def _init_renorm(self) -> None:
        """U = identity (no disentangling), W = average (coarse = mean of pair)."""
        with torch.no_grad():
            eye2 = torch.eye(2 * self.dim)
            avg = torch.cat(
                [0.5 * torch.eye(self.dim), 0.5 * torch.eye(self.dim)], dim=1
            )
            for u in self.disentanglers:
                u.weight.copy_(eye2)
                u.bias.zero_()
            for w in self.isometries:
                w.weight.copy_(avg)
                w.bias.zero_()

    @staticmethod
    def _shift(z: torch.Tensor, lag: int) -> torch.Tensor:
        """Causal right-shift by ``lag`` along the sequence axis (zero-filled)."""
        if lag <= 0:
            return z
        return F.pad(z, (0, 0, lag, 0))[:, : z.shape[1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale_feats = [x]
        cur = x
        for level in range(self.n_levels):
            prev = self._shift(cur, 1 << level)  # 2**level-ago partner
            pair = torch.cat([prev, cur], dim=-1)  # [B, L, 2D]
            dis = self.disentanglers[level](pair)  # disentangle (identity at init)
            cur = self.isometries[level](dis)  # coarse-grain (average at init)
            scale_feats.append(cur)
        return self.read(torch.cat(scale_feats, dim=-1))


def _quaternion_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two quaternions, batched over the last dim (size 4)."""
    a0, a1, a2, a3 = a.unbind(-1)
    b0, b1, b2, b3 = b.unbind(-1)
    return torch.stack(
        (
            a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
            a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
            a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
            a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0,
        ),
        dim=-1,
    )


def _quaternion_conj(a: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate ``(a0, -a1, -a2, -a3)``."""
    sign = a.new_tensor([1.0, -1.0, -1.0, -1.0])
    return a * sign


def octonion_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Octonion product, batched over the last dim (size 8), via Cayley–Dickson.

    An octonion is a pair of quaternions ``(p, q)``; the product is
    ``(p, q)(r, s) = (p r - conj(s) q,  s p + q conj(r))``. Octonions are a
    NON-ASSOCIATIVE normed division algebra: ``|xy| = |x||y|`` (so multiplying by
    a unit octonion preserves norm — no blow-up), but ``(xy)z != x(yz)`` in
    general — a genuinely new algebraic structure with no softmax analog.
    """
    p, q = x[..., :4], x[..., 4:]
    r, s = y[..., :4], y[..., 4:]
    real = _quaternion_mul(p, r) - _quaternion_mul(_quaternion_conj(s), q)
    imag = _quaternion_mul(s, p) + _quaternion_mul(q, _quaternion_conj(r))
    return torch.cat((real, imag), dim=-1)


class OctonionicMixerLane(nn.Module):
    """Causal non-associative octonionic sequence mixer (NM-9 long-tail exotic).

    Channels are grouped into octonions (blocks of 8). Each token group is mixed
    with a causal power-law-decayed context ``c_t = Σ_{s≤t} ρ**(t-s) x_s`` through
    the OCTONION product — a non-associative normed division algebra. The readout
    uses the left-associated bracketing ``(u · c_t) · x_t`` where ``u`` is a
    learned UNIT octonion (norm-preserving, so the mix cannot blow up); the
    distinct bracketing is where non-associativity carries information a
    commutative/associative averager cannot. There is no score normalization and
    no convex token average, so it is anti-softmax-twin by construction: a
    token-constant input is NOT preserved (``c·x`` of a constant ≠ that constant)
    and the map is degree-2 (non-homogeneous), both far from the softmax basin.

    ``dim`` must be a multiple of 8; the dispatcher falls back to a dense linear
    map otherwise. Finite forward/backward at init (unit twist = identity
    octonion → the mix is ``c_t · x_t``), and non-degenerate (the gated residual
    starts at ``tanh(0.5) ≈ 0.46`` of the octonionic path).
    """

    def __init__(self, dim: int, decay_init: float = 0.9) -> None:
        super().__init__()
        if dim <= 0 or dim % 8 != 0:
            raise ValueError("OctonionicMixerLane requires dim to be a multiple of 8")
        if not 0.0 < decay_init < 1.0:
            raise ValueError("decay_init must be in (0, 1)")
        self.dim = dim
        self.groups = dim // 8
        # Learned twist octonions, one per group; init to the identity octonion
        # e0 = (1, 0, ..., 0) so the mix starts as c_t · x_t.
        twist = torch.zeros(self.groups, 8)
        twist[:, 0] = 1.0
        self.twist = nn.Parameter(twist)
        # Per-group decay ρ = sigmoid(log_decay) ∈ (0, 1).
        logit = torch.logit(torch.tensor(decay_init))
        self.log_decay = nn.Parameter(torch.full((self.groups,), float(logit)))
        self.out_proj = nn.Linear(dim, dim, bias=False)
        # Gated residual: start at tanh(0.5) ≈ 0.46 so the lane is non-degenerate.
        self.mix_gate = nn.Parameter(torch.full((dim,), 0.5))

    def _decayed_context(self, xo: torch.Tensor) -> torch.Tensor:
        """Causal power-law context ``c[b,t,g,:] = Σ_{s≤t} ρ_g**(t-s) x[b,s,g,:]``.

        Delegates to the shared chunked scan (O(L·chunk), not O(L²)); the per-group
        decay is broadcast to the 8 octonion channels of each group.
        """
        b, length, groups, _ = xo.shape
        decay = torch.sigmoid(self.log_decay).clamp(1e-4, 1 - 1e-4)  # [G]
        per_channel_decay = decay.repeat_interleave(8)  # [G*8]
        flat = xo.reshape(b, length, groups * 8)
        context = causal_decay_context(flat, per_channel_decay)
        return context.reshape(b, length, groups, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, length, _ = x.shape
        xo = x.view(b, length, self.groups, 8)
        context = self._decayed_context(xo)
        unit_twist = self.twist / (self.twist.norm(dim=-1, keepdim=True) + 1e-8)
        twist = unit_twist.view(1, 1, self.groups, 8).expand(b, length, self.groups, 8)
        twisted = octonion_mul(twist, context)  # norm-preserving rotation of context
        mixed = octonion_mul(twisted, xo)  # (u · c_t) · x_t — non-associative combine
        mixed = mixed.reshape(b, length, self.dim)
        return x + torch.tanh(self.mix_gate) * self.out_proj(mixed)
