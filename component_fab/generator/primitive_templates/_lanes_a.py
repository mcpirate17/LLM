"""Reciprocal / semiring / tropical attention lanes (part A). See _core."""

import torch
from torch import nn
from component_fab.harness.rope import RotaryEmbedding, apply_rope

from ._core import (
    _cumsum_dim1_eager,
    _cummax_dim1_eager,
    _reciprocal_attn_logits,
    _QKVRopeAttentionBase,
    _causal_sparsemax,
    _pick_n_heads,
    _heads_for_head_dim,
    get_causal_bool_mask,
)


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


class ReciprocalRankAttention(_QKVRopeAttentionBase):
    """Causal attention boosted by reciprocal (mutual) content agreement.

    Standard attention asks "how much should query i read key j?". This adds
    the reverse compatibility "how much does j point back at i?" over the same
    causal prefix and multiplies it in (log-space), favouring mutual matches —
    the useful shape for binding/retrieval. The boost scale is ``tanh`` of a
    learnable scalar initialised at 0, so the lane starts as plain softmax
    attention and learns how much reciprocity to use. Ported from the
    ``reciprocal_rank_attention`` synthesis op (AR-gate 1.0, top nano BLiMP).
    """

    def __init__(
        self,
        dim: int,
        causal: bool = True,
        *,
        use_rope: bool = False,
        max_seq_len: int = 1024,
    ) -> None:
        super().__init__(dim, causal=causal, use_rope=use_rope, max_seq_len=max_seq_len)
        self.reciprocal_logit_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        q, k, v = self.q(x), self.k(x), self.v(x)
        if self.rope is not None:
            cos, sin = self.rope(seq_len, device=x.device, dtype=x.dtype)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        raw = torch.einsum("bid,bjd->bij", q, k) * self.scale
        tri = get_causal_bool_mask(seq_len, x.device) if self.causal else None
        boost = torch.tanh(self.reciprocal_logit_scale).to(x.dtype)
        logits = _reciprocal_attn_logits(raw, boost, tri)
        weights = torch.softmax(logits, dim=-1)
        return torch.einsum("bij,bjd->bid", weights, v)


class PhaseLockAttention(_QKVRopeAttentionBase):
    """Causal attention with a phase-synchrony content score.

    Adds ``phase_scale * mean_d cos(tanh(q)_i - tanh(k)_j)`` to the dot-product
    affinity before softmax: keys are favoured when their channel-wise bounded
    "phase" pattern synchronises with the query, a content address distinct from
    dot-product magnitude. ``phase_scale = tanh`` of a learnable scalar (init 0)
    so the lane starts as plain softmax attention. Ported from the
    ``phase_lock_attention`` synthesis op (AR-gate 1.0, top nano BLiMP).
    """

    def __init__(
        self,
        dim: int,
        causal: bool = True,
        *,
        use_rope: bool = False,
        max_seq_len: int = 1024,
    ) -> None:
        super().__init__(dim, causal=causal, use_rope=use_rope, max_seq_len=max_seq_len)
        self.phase_lock_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        q, k, v = self.q(x), self.k(x), self.v(x)
        if self.rope is not None:
            cos, sin = self.rope(seq_len, device=x.device, dtype=x.dtype)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        dot = torch.einsum("bid,bjd->bij", q, k) * self.scale
        # Gemini CLI: Optimized phase via cosine addition formula to avoid O(L^2 D) tensor.
        # cos(qi - kj) = cos(qi)cos(kj) + sin(qi)sin(kj)
        t_q, t_k = torch.tanh(q), torch.tanh(k)
        c_q, c_k = torch.cos(t_q), torch.cos(t_k)
        s_q, s_k = torch.sin(t_q), torch.sin(t_k)
        phase = (
            torch.einsum("bid,bjd->bij", c_q, c_k)
            + torch.einsum("bid,bjd->bij", s_q, s_k)
        ) / float(self.dim)
        if self.causal:
            tri = get_causal_bool_mask(seq_len, x.device)
            dot = dot.masked_fill(tri, float("-inf"))
            phase = phase.masked_fill(tri, 0.0)
        phase_scale = torch.tanh(self.phase_lock_scale).to(x.dtype)
        weights = torch.softmax(dot + phase_scale * phase, dim=-1)
        return torch.einsum("bij,bjd->bid", weights, v)


class ReciprocalPrimaryRefine(nn.Module):
    """Reciprocal-rank attention as a full-strength residual backbone plus a
    small gated side lane.

    The FFW sweep showed equal-weight gating (2-/3-lane) dilutes reciprocal's
    standout nano_induction_nearest (0.44 → 0.29). Here the reciprocal lane runs
    undiluted and the side lane (phase-lock by default) is added through a
    per-channel ``sigmoid`` gate initialised at bias −2 (≈0.12), so the backbone
    dominates at init and the model keeps reciprocal's induction-nearest while
    the side lane contributes LM / ni05 gains. ``side="tropical"`` swaps in the
    max-plus lane instead.
    """

    def __init__(
        self,
        dim: int,
        *,
        side: str = "phase",
        use_rope: bool = True,
        max_seq_len: int = 1024,
    ) -> None:
        super().__init__()
        self.primary = ReciprocalRankAttention(
            dim, use_rope=use_rope, max_seq_len=max_seq_len
        )
        if side == "tropical":
            self.side: nn.Module = TropicalAttention(dim)
        else:
            self.side = PhaseLockAttention(
                dim, use_rope=use_rope, max_seq_len=max_seq_len
            )
        self.gate = nn.Parameter(torch.full((dim,), -2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.primary(x) + torch.sigmoid(self.gate) * self.side(x)


class SparseReciprocalAttention(_QKVRopeAttentionBase):
    """Mutual-nearest-neighbour SPARSE attention.

    Keeps only keys that are sparse matches in BOTH directions: forward =
    sparsemax(QKᵀ) (query i over keys j≤i); reverse = sparsemax((QKᵀ)ᵀ) (token j
    as a query matching key i); weights ∝ forward·reverse, renormalised. Unlike
    reciprocal_rank (a dense additive bias to softmax logits), this changes the
    mixing *structure* — non-convex (sparsemax zeros most keys) AND bidirectional,
    a hard mutual-binding operator. Causal.
    """

    _NEG: float = -1e4

    def __init__(
        self,
        dim: int,
        causal: bool = True,
        *,
        use_rope: bool = True,
        max_seq_len: int = 1024,
    ) -> None:
        super().__init__(dim, causal=causal, use_rope=use_rope, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        S = x.shape[1]
        q, k, v = self.q(x), self.k(x), self.v(x)
        if self.rope is not None:
            cos, sin = self.rope(S, device=x.device, dtype=x.dtype)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        raw = torch.einsum("bid,bjd->bij", q, k) * self.scale
        if self.causal:
            tri = get_causal_bool_mask(S, x.device)
            fwd = _causal_sparsemax(raw.masked_fill(tri, self._NEG))
            rev = _causal_sparsemax(raw.transpose(-2, -1).masked_fill(tri, self._NEG))
        else:
            fwd = _causal_sparsemax(raw)
            rev = _causal_sparsemax(raw.transpose(-2, -1))
        mutual = fwd * rev
        weights = mutual / mutual.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.einsum("bij,bjd->bid", weights, v)


class SemiringReciprocalAttention(_QKVRopeAttentionBase):
    """Reciprocal content addressing + learnable-semiring value pooling.

    Addressing = reciprocal-rank weights (mutual query↔key agreement). Value
    aggregation uses a learned semiring instead of a convex weighted mean:
    ``out_id = (1/γ)·logsumexp_j(log w_ij + γ·v_jd)`` with γ = ``exp(param)``
    (>0, init 1). γ→0 recovers the weighted mean; γ large → soft-max pooling, so
    the readout escapes softmax's convex hull (selection/extremisation). Causal.
    Note: materialises a (B,S,S,D) tensor — fine for nano screening, chunk at scale.
    """

    def __init__(
        self,
        dim: int,
        causal: bool = True,
        *,
        use_rope: bool = True,
        max_seq_len: int = 1024,
    ) -> None:
        super().__init__(dim, causal=causal, use_rope=use_rope, max_seq_len=max_seq_len)
        self.reciprocal_logit_scale = nn.Parameter(torch.zeros(1))
        self.semiring_beta = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        S = x.shape[1]
        q, k, v = self.q(x), self.k(x), self.v(x)
        if self.rope is not None:
            cos, sin = self.rope(S, device=x.device, dtype=x.dtype)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        raw = torch.einsum("bid,bjd->bij", q, k) * self.scale
        tri = get_causal_bool_mask(S, x.device) if self.causal else None
        boost = torch.tanh(self.reciprocal_logit_scale).to(x.dtype)
        logits = _reciprocal_attn_logits(raw, boost, tri)
        gamma = torch.exp(self.semiring_beta).clamp(0.05, 10.0)
        logw = torch.log_softmax(logits, dim=-1).unsqueeze(-1)  # (B,S,S,1)
        z = logw + gamma * v.unsqueeze(1)  # (B,S,S,D)
        if tri is not None:
            # Future logw is -inf → z is -inf; pin to a finite floor so the
            # logsumexp backward stays finite (exp(-1e4) underflows to 0).
            z = z.masked_fill(tri.view(1, S, S, 1), -1e4)
        return torch.logsumexp(z, dim=2) / gamma


class HeteroSemiringReciprocalAttention(_QKVRopeAttentionBase):
    """Heterogeneous-algebra multi-head reciprocal attention.

    Attacks the width-dilution of the single-head ``SemiringReciprocalAttention``
    (whose scalar reciprocity β and scalar semiring γ get averaged-out as the
    model widens) by giving **each head its own learned algebra**:

    - per-head reciprocity ``β_h = tanh(param_h)`` controls how much mutual
      query↔key agreement is folded into that head's addressing (init 0 → plain
      softmax addressing per head);
    - per-head **signed** semiring temperature ``γ_h`` controls value pooling via
      ``out = (1/γ_h)·logsumexp_j(log w_ij + γ_h·v_jd)``. Unlike the single-head
      op (which used ``γ = exp(param) > 0``, reaching only mean↔max), γ_h here is
      a *signed* learnable scalar: ``γ_h < 0`` gives soft-**min** pooling,
      ``γ_h → 0`` the convex mean (softmax attention), ``γ_h > 0`` soft-**max**
      (tropical/winner-take-all). The full softmax-mean↔max↔min spectrum is thus
      available per head, and heads are seeded with a deterministic spread of
      γ (linspace, symmetry-broken) so the model starts as a *heterogeneous*
      bank of algebras rather than collapsing to one.

    The novelty over standard MHA: each head operates in a different learned
    semiring AND a different reciprocity regime — a per-head learned algebra,
    not a per-head learned projection. No output projection (consistent with the
    softmax/semiring/reciprocal lane family) — heads are concatenated and
    cross-head mixing is deferred to the next block's input projection; with
    ``n_heads == 1`` the lane therefore reduces EXACTLY to the single-head
    ``SemiringReciprocalAttention`` (γ init 1.0 = exp(0)). Causal. Materialises a
    per-head ``(B,H,S,S,dh)`` tensor (total ``B·S·S·D`` — same footprint as the
    single-head op; use a smaller batch at large width).
    """

    _GAMMA_EPS: float = 0.05
    _GAMMA_MAX: float = 10.0

    def __init__(
        self,
        dim: int,
        causal: bool = True,
        *,
        target_head_dim: int = 96,
        n_heads: int | None = None,
        use_rope: bool = True,
        max_seq_len: int = 1024,
    ) -> None:
        # Base builds q/k/v Linears; rope/scale are rebuilt per-head below.
        super().__init__(dim, causal=causal, use_rope=False, max_seq_len=max_seq_len)
        # Fix head_dim near the induction-nearest sweet spot (~96) and let n_heads
        # grow with width; an explicit n_heads overrides (for head-dim ablations).
        self.n_heads = (
            _pick_n_heads(dim, preferred=n_heads)
            if n_heads is not None
            else _heads_for_head_dim(dim, target_head_dim)
        )
        self.head_dim = dim // self.n_heads
        self.scale = float(self.head_dim) ** -0.5
        self.rope = (
            RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)
            if use_rope
            else None
        )
        # per-head reciprocity strength (init 0 → tanh 0 → plain softmax addressing)
        self.reciprocal_logit_scale = nn.Parameter(torch.zeros(self.n_heads))
        # per-head SIGNED semiring temperature. >1 head: a symmetry-broken spread
        # across soft-min (<0) / mean (~0) / soft-max (>0) so heads start diverse;
        # 1 head: γ=1.0, the proven single-head SemiringReciprocal init.
        if self.n_heads == 1:
            gamma_init = torch.ones(1)
        else:
            gamma_init = torch.linspace(-1.5, 1.5, self.n_heads)
        self.semiring_gamma = nn.Parameter(gamma_init)

    def _signed_gamma(self) -> torch.Tensor:
        """Clamp |γ| into ``[eps, max]`` keeping sign — avoids the 1/γ blow-up
        at 0 while preserving the soft-min/mean/soft-max regime per head."""
        g = self.semiring_gamma
        mag = g.abs().clamp(self._GAMMA_EPS, self._GAMMA_MAX)
        return torch.where(g < 0, -mag, mag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        H, dh = self.n_heads, self.head_dim
        q = self.q(x).view(B, S, H, dh).transpose(1, 2)  # (B,H,S,dh)
        k = self.k(x).view(B, S, H, dh).transpose(1, 2)
        v = self.v(x).view(B, S, H, dh).transpose(1, 2)
        if self.rope is not None:
            cos, sin = self.rope(S, device=x.device, dtype=x.dtype)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        raw = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale  # (B,H,S,S)
        tri = get_causal_bool_mask(S, x.device) if self.causal else None
        boost = torch.tanh(self.reciprocal_logit_scale).view(1, H, 1, 1).to(x.dtype)
        logits = _reciprocal_attn_logits(raw, boost, tri)
        gamma = self._signed_gamma().view(1, H, 1, 1, 1).to(x.dtype)
        logw = torch.log_softmax(logits, dim=-1).unsqueeze(-1)  # (B,H,S,S,1)
        z = logw + gamma * v.unsqueeze(2)  # (B,H,S,S,dh)
        if tri is not None:
            # Exclude future keys from the pooling EXACTLY: a finite −1e4 makes
            # exp underflow to 0 independent of v_j (a negative soft-min γ_h could
            # otherwise amplify a leak) and keeps the backward finite.
            z = z.masked_fill(tri.view(1, 1, S, S, 1), -1e4)
        pooled = torch.logsumexp(z, dim=3) / gamma.squeeze(3)  # (B,H,S,dh)
        return pooled.transpose(1, 2).reshape(B, S, H * dh)


class AnisotropicSemiringReciprocalAttention(_QKVRopeAttentionBase):
    """Reciprocal addressing + **per-channel** learnable-semiring value pooling.

    A single, full-width attention (the 100M-best variant for induction-nearest:
    the single 576-d head beat 6 head-split heads, 0.115 vs 0.073) whose ONLY
    change from ``SemiringReciprocalAttention`` is that the semiring temperature
    is a learned **vector** ``γ_d`` (one per value channel) instead of a scalar:

        ``out_id = (1/γ_d)·logsumexp_j(log w_ij + γ_d·v_jd)``

    Each value feature is thus pooled under its OWN algebra along the
    mean(γ_d→0)↔max(γ_d large) spectrum — an *anisotropic* semiring readout,
    novel value-aggregation algebra rather than a per-head split. ``γ_d =
    exp(param_d)`` (init 0 → γ_d = 1 ∀d ⇒ identical to the scalar-γ semiring at
    init, so it strictly generalises the proven lane). Reciprocal (mutual q↔k)
    addressing as in the parent. Causal; materialises a ``(B,S,S,D)`` tensor.
    """

    def __init__(
        self,
        dim: int,
        causal: bool = True,
        *,
        use_rope: bool = True,
        max_seq_len: int = 1024,
    ) -> None:
        super().__init__(dim, causal=causal, use_rope=use_rope, max_seq_len=max_seq_len)
        self.reciprocal_logit_scale = nn.Parameter(torch.zeros(1))
        self.semiring_beta = nn.Parameter(torch.zeros(dim))  # per-channel γ_d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        S = x.shape[1]
        q, k, v = self.q(x), self.k(x), self.v(x)
        if self.rope is not None:
            cos, sin = self.rope(S, device=x.device, dtype=x.dtype)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        raw = torch.einsum("bid,bjd->bij", q, k) * self.scale
        tri = get_causal_bool_mask(S, x.device) if self.causal else None
        boost = torch.tanh(self.reciprocal_logit_scale).to(x.dtype)
        logits = _reciprocal_attn_logits(raw, boost, tri)
        gamma = torch.exp(self.semiring_beta).clamp(0.05, 10.0).to(x.dtype)  # (D,)
        logw = torch.log_softmax(logits, dim=-1).unsqueeze(-1)  # (B,S,S,1)
        z = logw + gamma.view(1, 1, 1, -1) * v.unsqueeze(1)  # (B,S,S,D)
        if tri is not None:
            # Exclude future keys exactly; −1e4 underflows to 0 in exp.
            z = z.masked_fill(tri.view(1, S, S, 1), -1e4)
        return torch.logsumexp(z, dim=2) / gamma.view(1, 1, -1)


class FixedRankReciprocalAttention(nn.Module):
    """Reciprocal addressing whose SCORE lives in a fixed-rank subspace.

    Tests the competing hypothesis to head-splitting: the induction-nearest
    advantage may compress at width because the q·k *matching* happens over the
    full (growing) model width. Here Q,K project to a FIXED rank ``r`` (≈ the
    nano sweet spot, 96) regardless of model dim, so the matching subspace is
    width-invariant, while V stays full-width and a SINGLE attention pattern
    mixes the full values — unlike ``HeteroSemiringReciprocalAttention`` which
    used many *competing* head patterns over *partitioned* values (that hurt:
    0.073 vs single-head 0.115 at 100M). Reciprocal (mutual q↔k) addressing as
    in ``ReciprocalRankAttention``; plain convex value mean (isolates the score
    subspace as the only change). At ``r == dim`` it is exactly reciprocal_rank.
    Causal.
    """

    def __init__(
        self,
        dim: int,
        causal: bool = True,
        *,
        rank: int = 96,
        use_rope: bool = True,
        max_seq_len: int = 1024,
    ) -> None:
        super().__init__()
        self.rank = min(rank, dim) if (min(rank, dim) % 2 == 0) else min(rank, dim) - 1
        self.qr = nn.Linear(dim, self.rank, bias=False)
        self.kr = nn.Linear(dim, self.rank, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.scale = float(self.rank) ** -0.5
        self.causal = causal
        self.reciprocal_logit_scale = nn.Parameter(torch.zeros(1))
        self.rope = (
            RotaryEmbedding(self.rank, max_seq_len=max_seq_len) if use_rope else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        S = x.shape[1]
        q, k, v = self.qr(x), self.kr(x), self.v(x)
        if self.rope is not None:
            cos, sin = self.rope(S, device=x.device, dtype=x.dtype)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        raw = torch.einsum("bir,bjr->bij", q, k) * self.scale
        tri = get_causal_bool_mask(S, x.device) if self.causal else None
        boost = torch.tanh(self.reciprocal_logit_scale).to(x.dtype)
        logits = _reciprocal_attn_logits(raw, boost, tri)
        weights = torch.softmax(logits, dim=-1)
        return torch.einsum("bij,bjd->bid", weights, v)


class TemperedTropicalAttention(_QKVRopeAttentionBase):
    """Max-plus attention with a learnable Boltzmann temperature (Track B —
    novel improvement to ``TropicalAttention``).

    Plain ``TropicalAttention`` takes a hard ``max_{j≤i}(scale·q·k + v)`` — a
    non-smooth winner-take-all with zero gradient to all but the argmax key.
    This replaces the hard max with a temperature-controlled log-sum-exp over the
    SAME ``(affinity + value)`` tropical combination:

        ``out_id = (1/β)·logsumexp_{j≤i}( β·(scale·q_i·k_j + v_jd) )``

    with ``β = softplus(param)`` learnable per **head** (head_dim≈96). β→∞ recovers
    the hard tropical max (winner-take-all); β→0 anneals to a log-mean-exp soft
    pooling — so the model *learns where to sit on the hard↔soft max-plus axis*,
    per head, and every key gets gradient. Distinct from
    ``SemiringReciprocalAttention`` (which softmaxes the affinity into convex
    weights FIRST, then semiring-pools values): here addressing and value are
    fused inside one tempered tropical semiring, never leaving max-plus algebra.
    Causal (``-inf`` mask flows through ``logsumexp`` cleanly).
    """

    def __init__(
        self,
        dim: int,
        causal: bool = True,
        *,
        target_head_dim: int = 96,
        use_rope: bool = True,
        max_seq_len: int = 1024,
    ) -> None:
        super().__init__(dim, causal=causal, use_rope=False, max_seq_len=max_seq_len)
        self.n_heads = _heads_for_head_dim(dim, target_head_dim)
        self.head_dim = dim // self.n_heads
        self.scale = float(self.head_dim) ** -0.5
        self.rope = (
            RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)
            if use_rope
            else None
        )
        # per-head inverse temperature, init softplus(0.5413)≈1.0 (mild-soft max)
        self.log_beta = nn.Parameter(torch.full((self.n_heads,), 0.5413))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        H, dh = self.n_heads, self.head_dim
        q = self.q(x).view(B, S, H, dh).transpose(1, 2)  # (B,H,S,dh)
        k = self.k(x).view(B, S, H, dh).transpose(1, 2)
        v = self.v(x).view(B, S, H, dh).transpose(1, 2)
        if self.rope is not None:
            cos, sin = self.rope(S, device=x.device, dtype=x.dtype)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        affinity = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale  # (B,H,S,S)
        tri = get_causal_bool_mask(S, x.device) if self.causal else None
        beta = torch.nn.functional.softplus(self.log_beta).clamp(0.05, 50.0)
        beta = beta.view(1, H, 1, 1, 1).to(x.dtype)
        combined = affinity.unsqueeze(-1) + v.unsqueeze(2)  # (B,H,S,S,dh)
        logits = beta * combined
        if tri is not None:
            # Mask the FINAL post-β logits, not the affinity: a large future v_j
            # could otherwise survive a finite affinity mask and win the max-plus
            # pool. −1e9 underflows to weight 0 independent of v_j and keeps the
            # logsumexp backward finite.
            logits = logits.masked_fill(tri.view(1, 1, S, S, 1), -1e9)
        pooled = torch.logsumexp(logits, dim=3) / beta.squeeze(3)  # (B,H,S,dh)
        return pooled.transpose(1, 2).reshape(B, S, H * dh)


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
        # Gemini CLI: Vectorized tropical recurrence: s[t] = max_{k<=t} (Bx[k] + (t-k)A)
        # s[t] = tA + max_{k<=t} (Bx[k] - kA)
        k = torch.arange(seq_len, device=x.device, dtype=x.dtype).view(1, -1, 1)
        z = Bx - k * self.A.view(1, 1, -1)
        # Use cummax eager to avoid inductor scan issues.
        m = _cummax_dim1_eager(z)
        state = m + k * self.A.view(1, 1, -1)
        return self.C(state) + x


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
        # Gemini CLI: Use complex weights directly for refactored FFT logic.
        weights = torch.complex(
            self.weight_real[:n_freqs], self.weight_imag[:n_freqs]
        ).to(spectrum.dtype)
        out = torch.einsum("fde,bfd->bfe", weights, spectrum)
        return torch.fft.irfft(out, n=seq_len, dim=1)


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


class HyperbolicAttention(_QKVRopeAttentionBase):
    """Causal attention scored by HYPERBOLIC (Lorentz-model) distance instead of
    the Euclidean QK dot product. In negatively-curved space, volume grows
    exponentially with radius, so a tree's worth of children fits "near" their
    parent with little interference — general/near-root tokens sit close to the
    origin, specifics fan outward. Content matching then respects hierarchy a
    flat dot product cannot pack. Curvature ``c = softplus(param)`` is learned, so
    the lane can bend space only where it pays.

    This is the intended replacement for the reciprocal **softmax-twin**: a
    genuinely non-Euclidean addressing geometry, not a cosmetic re-weighting of
    the same flat scores. (Honest caveat: scores still go through a softmax — the
    novelty is the geometry of the score, not avoiding softmax. The nano gate
    asks whether that geometry buys per-parameter capability over reciprocal /
    softmax on induction + binding.)

    Same q/k/v/RoPE machinery as the rest of the attention family (subclasses the
    shared base), so the ONLY difference vs reciprocal is the scoring geometry —
    a controlled, param-matched comparison.
    """

    def __init__(
        self,
        dim: int,
        causal: bool = True,
        *,
        use_rope: bool = True,
        max_seq_len: int = 1024,
    ) -> None:
        super().__init__(dim, causal=causal, use_rope=use_rope, max_seq_len=max_seq_len)
        # c starts at softplus(0)≈0.69 (mild curvature); temp shapes score sharpness.
        self.log_curvature = nn.Parameter(torch.zeros(1))
        self.log_temp = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from ._core import get_causal_mask

        seq_len = x.shape[1]
        q, k, v = self.q(x), self.k(x), self.v(x)
        if self.rope is not None:
            cos, sin = self.rope(seq_len, device=x.device, dtype=x.dtype)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        c = nn.functional.softplus(self.log_curvature) + 1e-4
        # Lorentz lift: time coord t = sqrt(1/c + ||x||^2) puts each point on the
        # hyperboloid <x,x>_L = -1/c. Lorentzian inner product <p,q>_L = -t_p t_q +
        # p·q; geodesic distance = (1/sqrt c)·acosh(-c<p,q>_L). At p=q the argument
        # is exactly 1 (distance 0); it is >=1 everywhere by Cauchy-Schwarz.
        q_t = torch.sqrt(1.0 / c + q.pow(2).sum(-1, keepdim=True))  # [b, L, 1]
        k_t = torch.sqrt(1.0 / c + k.pow(2).sum(-1, keepdim=True))
        spatial = torch.einsum("bid,bjd->bij", q, k)  # [b, L, L]
        time = q_t * k_t.transpose(1, 2)  # [b, L, L]
        arg = (c * (time - spatial)).clamp_min(1.0 + 1e-5)
        dist = torch.acosh(arg) / torch.sqrt(c)
        scores = -dist * nn.functional.softplus(self.log_temp)  # closer => higher
        if self.causal:
            scores = scores + get_causal_mask(seq_len, x.device, x.dtype)
        attn = torch.softmax(scores, dim=-1)
        return torch.einsum("bij,bjd->bid", attn, v)


__all__ = [
    "HyperbolicAttention",
    "TropicalAttention",
    "SparsemaxAttention",
    "ReciprocalRankAttention",
    "PhaseLockAttention",
    "ReciprocalPrimaryRefine",
    "SparseReciprocalAttention",
    "SemiringReciprocalAttention",
    "HeteroSemiringReciprocalAttention",
    "AnisotropicSemiringReciprocalAttention",
    "FixedRankReciprocalAttention",
    "TemperedTropicalAttention",
    "TropicalStateSpace",
    "TopKLinear",
    "FourierBasisLane",
    "FiniteDifferenceCalculusLane",
    "LowRankFactorizedLane",
    "SparseBandedMatrixLane",
]
