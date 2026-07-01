"""Novel-math invention lanes for component_fab (Tier-1 expansion).

Two genuinely new, non-QKV sequence mixers, both anti-softmax-twin by structure:

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

Both preserve ``[B, L, D]`` shape and produce finite gradients at init.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


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
