"""Stateful content-addressed memory lanes for component_fab.

Split out of ``primitive_templates.py`` (which crossed the god-file limit).
These are the recurrent, content-addressed memory primitives — the
invention-track family whose forward pass is a strictly causal left-to-right
scan over a per-example memory:

- ``CausalFastWeightMemoryLane`` — pure Hebbian ``k ⊗ v`` fast-weight write
  with scalar decay (linear-attention memory).
- ``CausalSlotRouterMemoryLane`` — routing-as-memory over a few slots.
- ``HierarchicalResidualCompressorLane`` — multi-timescale summary stack.
- ``_SurpriseMemoryBase`` / ``TropicalSurpriseMemoryLane`` /
  ``SemiringSurpriseMemoryLane`` / ``PadicSurpriseMemoryLane`` — Titans/TTT-style
  test-time memory whose write is the *surprise* (associative prediction error),
  generalizing the Hebbian lane. ``SemiringSurpriseMemoryLane`` makes the
  family's max-plus retrieval a *learnable* tempered semiring (β slides
  mean↔max). See ``_SurpriseMemoryBase`` for the mechanism.

All preserve ``[B, L, D]`` shape and produce finite gradients at init.
"""

from __future__ import annotations

import torch
from torch import nn

from ..harness.rope import RotaryEmbedding as RotaryEmbedding  # noqa: PLC0414 (re-export keeps autoflake)
from ..harness.rope import apply_rope as apply_rope  # noqa: PLC0414


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
            outputs.append(torch.einsum("bi,bij->bj", q[:, t], memory))
        # One batched projection over [B, L, m] instead of L per-step GEMMs.
        return self.out(torch.stack(outputs, dim=1))


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
            outputs.append(torch.einsum("bs,bsd->bd", route, slots))
        # One batched projection over [B, L, d] instead of L per-step GEMMs.
        return self.out(torch.stack(outputs, dim=1))


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
            outputs.append(torch.cat(summaries, dim=-1))
        # One batched read over [B, L, d*levels] instead of L per-step GEMMs.
        return self.read(torch.stack(outputs, dim=1))


class _SurpriseMemoryBase(nn.Module):
    """Titans/TTT-style test-time memory scaffold (delta-rule, not Hebbian).

    The distinguishing mechanism vs ``CausalFastWeightMemoryLane`` (which
    writes a pure Hebbian ``k ⊗ v`` outer product): here the per-token write
    is the **surprise** — the associative prediction error
    ``e_t = v_t − read(M_{t-1}, k_t)`` — so each step performs one update of
    online gradient descent on the per-token associative loss
    ``½‖M k_t − v_t‖²`` (whose negative gradient is exactly ``e_t k_tᵀ``).
    On top of the delta write we add:

    - a learned **momentum** term ``S_t = μ·S_{t-1} + g_t·(k_t ⊗ e_t)`` over
      the surprise stream (Titans' "past surprise"),
    - a **data-dependent forget gate** ``α_t = σ(W_α x_t)`` decaying the
      memory per key-row (Titans' adaptive weight decay), and
    - a **data-dependent write gate** ``g_t = σ(W_g x_t)`` (momentary
      surprise scaling).

    Subclasses supply ``_read(memory, addr)`` — the retrieval *algebra* — and
    may override ``forward`` for multi-memory layouts. The base ``forward``
    runs a single memory matrix with whatever ``_read`` the subclass defines.
    The scan is a strictly causal left-to-right loop, so the output at
    position ``t`` depends only on tokens ``≤ t``. Output is the projected
    readout (no self-residual; the surrounding ``LaneTestBlock`` adds it).
    """

    def __init__(
        self,
        dim: int,
        memory_dim: int | None = None,
        *,
        use_rope: bool = False,
        max_seq_len: int = 1024,
        compile_step: bool = False,
    ) -> None:
        super().__init__()
        memory_dim = memory_dim or min(dim, 32)
        self.q = nn.Linear(dim, memory_dim, bias=False)
        self.k = nn.Linear(dim, memory_dim, bias=False)
        self.v = nn.Linear(dim, memory_dim, bias=False)
        self.write_gate = nn.Linear(dim, 1)
        self.forget_gate = nn.Linear(dim, memory_dim)
        self.out = nn.Linear(memory_dim, dim, bias=False)
        self.momentum_logit = nn.Parameter(torch.tensor(0.0))
        # Slow default forgetting + active default writing so an UNTRAINED
        # memory still propagates a position-0 key to position -1 (needed to
        # clear the ERF + state-propagation gates before any training runs).
        nn.init.constant_(self.forget_gate.bias, -4.0)  # α ≈ 0.018 / step
        nn.init.constant_(self.write_gate.bias, 1.0)  # g ≈ 0.73 / step
        self.dim = dim
        self.memory_dim = memory_dim
        self._scale = float(memory_dim) ** -0.5
        # Optional RoPE on the ADDRESSING q/k. The delta-rule writes key k_s into
        # memory at step s and reads with query q_t at step t; rotating q,k by
        # absolute position injects the relative phase (t−s) into the retrieval
        # score q_t·k_s, giving the memory an explicit notion of "how far back".
        # A hypothesis worth testing for induction/associative recall — off by
        # default so the family's published behavior + tests are unchanged.
        self.rope = (
            RotaryEmbedding(memory_dim, max_seq_len=max_seq_len)
            if (use_rope and memory_dim % 2 == 0)
            else None
        )
        # The O(T) scan fires ~20 tiny kernels per timestep; the loop is
        # launch-bound (GPU ~15-40% util). Compiling the FIXED-SHAPE per-step
        # _delta_step (NOT the whole forward — tracing the 512-iter Python loop
        # OOM/hangs Inductor) fuses those kernels: measured 2.8x fwd+bwd
        # (25.7k -> 71.3k tok/s, dim576/seq512/b16). Lazy — compiles on first
        # call, after subclasses finish __init__ (e.g. semiring_temp). Off by
        # default (skip compile overhead for the fab's tiny dim16 grades);
        # the 100M lane factory turns it on.
        if compile_step:
            self._delta_step = torch.compile(self._delta_step)  # type: ignore[method-assign]

    @staticmethod
    def _unit(t: torch.Tensor) -> torch.Tensor:
        """L2-normalize the last dim so retrieval scores are bounded."""
        return t / t.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def _addr(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project + bound + (optionally RoPE) the addressing q/k. Returns
        ``(q, k)`` each ``[B, L, memory_dim]``; RoPE rotates per position."""
        q = self._unit(torch.tanh(self.q(x)))
        k = self._unit(torch.tanh(self.k(x)))
        if self.rope is not None:
            cos, sin = self.rope(x.shape[1], device=x.device, dtype=x.dtype)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        return q, k

    def _read(self, memory: torch.Tensor, addr: torch.Tensor) -> torch.Tensor:
        """Tropical (max-plus) retrieval from ``memory`` [B, m, m].

        ``read[b, j] = max_i (memory[b, i, j] + addr[b, i])`` — winner-take-all
        over the key axis. Empirically (n=6 seeds) this is what lets an O(L)
        compressed memory match O(L²) full attention on the 2-hop induction
        circuit; the Euclidean ``Σ_i memory[i, j]·addr_i`` read collapses to
        the random baseline. It is the shared retrieval algebra of the
        surprise-memory family; subclasses vary only the addressing.
        """
        return (memory + addr.unsqueeze(-1)).amax(dim=1)

    def _delta_step(
        self,
        memory: torch.Tensor,
        surprise: torch.Tensor,
        *,
        k_t: torch.Tensor,
        v_t: torch.Tensor,
        q_t: torch.Tensor,
        write: torch.Tensor,
        forget: torch.Tensor,
        momentum: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One causal delta-rule update; returns (memory, surprise, readout)."""
        batch = memory.shape[0]
        prediction = self._read(memory, k_t)
        error = v_t - prediction  # surprise
        delta = torch.einsum("bi,bj->bij", k_t, error) * self._scale
        surprise = momentum * surprise + write.view(batch, 1, 1) * delta
        memory = (1.0 - forget).unsqueeze(-1) * memory + surprise
        return memory, surprise, self._read(memory, q_t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q, k = self._addr(x)
        v = self.v(x)
        write = torch.sigmoid(self.write_gate(x)).squeeze(-1)  # [B, L]
        forget = torch.sigmoid(self.forget_gate(x))  # [B, L, m]
        momentum = torch.sigmoid(self.momentum_logit)
        memory = x.new_zeros(batch_size, self.memory_dim, self.memory_dim)
        surprise = x.new_zeros(batch_size, self.memory_dim, self.memory_dim)
        outputs = []
        for t in range(seq_len):
            memory, surprise, read = self._delta_step(
                memory,
                surprise,
                k_t=k[:, t],
                v_t=v[:, t],
                q_t=q[:, t],
                write=write[:, t],
                forget=forget[:, t],
                momentum=momentum,
            )
            outputs.append(read)
        # One batched projection over [B, L, m] instead of L per-step GEMMs.
        return self.out(torch.stack(outputs, dim=1))


class TropicalSurpriseMemoryLane(_SurpriseMemoryBase):
    """Test-time surprise memory with max-plus (tropical) retrieval.

    Read is the tropical matrix-vector product
    ``read[j] = max_i (M[i, j] + addr_i)`` instead of the Euclidean
    ``Σ_i M[i, j]·addr_i``. Retrieval is therefore winner-take-all: the
    single strongest stored key–value association dominates the readout,
    which sharpens *exact* recall and suppresses cross-key interference —
    the failure mode plain linear-attention memory hits on dense binding /
    induction. Combining max-plus retrieval with the Titans delta write is,
    to our knowledge, an unbuilt corner of the design space.

    Single-scale member of the family: it uses the base max-plus ``_read``
    and base scan unchanged. ``PadicSurpriseMemoryLane`` is the multi-scale
    (ultrametric) generalization built on the same retrieval algebra.
    """


class SemiringSurpriseMemoryLane(_SurpriseMemoryBase):
    """Test-time surprise memory with a LEARNABLE tempered-semiring read.

    The family's retrieval algebra is the only thing this varies, but it varies
    it *learnably*: the read is the tempered log-sum-exp over the key axis

        ``read[b, j] = (1/β)·( logsumexp_i (β·(M[b, i, j] + addr[b, i])) − log m )``

    with ``β = softplus(param) > 0`` a single learned inverse-temperature. This
    is a strict generalization of BOTH existing reads in the family and slides
    smoothly between them:

    - ``β → ∞`` recovers ``TropicalSurpriseMemoryLane``'s max-plus
      ``max_i (M[i, j] + addr_i)`` — the winner-take-all read proven (n=6 seeds)
      to keep O(L) compressed memory at induction 1.00 where the Euclidean read
      collapses to 0.25;
    - ``β → 0`` recovers the arithmetic mean ``mean_i (M[i, j] + addr_i)`` — the
      soft, interference-prone read.

    So instead of *fixing* the retrieval sharpness, the model *learns* where to
    sit on the mean↔max axis from the data, per the SOTA-plan thesis (proven
    Titans/TTT delta-rule write substrate carrying a novel learnable algebra).
    ``β`` is initialised high (≈4) so the lane starts at the proven tropical
    behavior and can only soften if that helps — a safe, capability-preserving
    init. The ``− log m`` keeps the β→0 limit a true mean (no log-cardinality
    blow-up), so the read is well-conditioned across the whole β range.

    Neither subclass of ``_SurpriseMemoryBase`` overrides the write; the novelty
    is entirely in this retrieval algebra.
    """

    def __init__(
        self,
        dim: int,
        memory_dim: int | None = None,
        *,
        use_rope: bool = False,
        max_seq_len: int = 1024,
        compile_step: bool = False,
    ) -> None:
        super().__init__(
            dim,
            memory_dim=memory_dim,
            use_rope=use_rope,
            max_seq_len=max_seq_len,
            compile_step=compile_step,
        )
        # softplus(4.0) ≈ 4.02: start sharp (near the proven tropical max read).
        # compile_step wraps _delta_step lazily (compiles on first forward), so
        # this param exists before the compiled step ever reads it via _read.
        self.semiring_temp = nn.Parameter(torch.tensor(4.0))

    def _read(self, memory: torch.Tensor, addr: torch.Tensor) -> torch.Tensor:
        beta = torch.nn.functional.softplus(self.semiring_temp).clamp(1e-2, 30.0)
        m = memory.shape[1]
        scores = memory + addr.unsqueeze(-1)  # [B, m_keys, m_vals]
        lse = torch.logsumexp(beta * scores, dim=1)  # [B, m_vals]
        log_m = torch.log(
            torch.tensor(float(m), device=memory.device, dtype=memory.dtype)
        )
        return (lse - log_m) / beta


class PadicSurpriseMemoryLane(_SurpriseMemoryBase):
    """Test-time surprise memory over an ultrametric (p-adic) hierarchy.

    Maintains ``n_levels`` memory matrices, each updated by the same Titans
    delta rule, but addressed with a **p-adic block-pooled** key/query. At
    level ``ℓ`` the address is mean-pooled within nested blocks of size
    ``p^(n_levels-1-ℓ)`` and broadcast back, so two keys that agree on their
    coarse p-adic prefix collide (share capacity) at coarse levels but stay
    isolated at the finest level (block size 1). Each level retrieves with
    the family's max-plus read; the readout is a learned gated sum across
    levels. This realizes ultrametric associative recall: the finest level
    behaves like ``TropicalSurpriseMemoryLane`` (so induction recall is
    preserved — n=6 seeds, 1.00 vs 0.25 for a Euclidean-read variant), while
    coarse levels generalize across p-adically-near keys. The delta write
    keeps each level error-driven.
    """

    def __init__(
        self,
        dim: int,
        memory_dim: int | None = None,
        p: int = 2,
        n_levels: int = 3,
    ) -> None:
        memory_dim = memory_dim or min(dim, 16)
        # Clamp depth so the coarsest block size divides memory_dim evenly.
        levels = 1
        while (
            levels < n_levels
            and p ** (levels) <= memory_dim
            and (memory_dim % (p**levels) == 0)
        ):
            levels += 1
        super().__init__(dim, memory_dim=memory_dim)
        self.p = p
        self.n_levels = levels
        self.level_gate = nn.Parameter(torch.ones(levels) / float(levels))

    def _pool(self, addr: torch.Tensor, block: int) -> torch.Tensor:
        """Ultrametric block-mean: collapse addr within nested blocks."""
        if block <= 1:
            return addr
        batch, m = addr.shape
        pooled = addr.view(batch, m // block, block).mean(dim=-1, keepdim=True)
        return pooled.expand(batch, m // block, block).reshape(batch, m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self._unit(torch.tanh(self.q(x)))
        k = self._unit(torch.tanh(self.k(x)))
        v = self.v(x)
        write = torch.sigmoid(self.write_gate(x)).squeeze(-1)
        forget = torch.sigmoid(self.forget_gate(x))
        momentum = torch.sigmoid(self.momentum_logit)
        gates = torch.softmax(self.level_gate, dim=0)
        blocks = [
            self.p ** (self.n_levels - 1 - level) for level in range(self.n_levels)
        ]
        memories = [
            x.new_zeros(batch_size, self.memory_dim, self.memory_dim)
            for _ in range(self.n_levels)
        ]
        surprises = [
            x.new_zeros(batch_size, self.memory_dim, self.memory_dim)
            for _ in range(self.n_levels)
        ]
        outputs = []
        for t in range(seq_len):
            read_sum = x.new_zeros(batch_size, self.memory_dim)
            for level, block in enumerate(blocks):
                memories[level], surprises[level], read = self._delta_step(
                    memories[level],
                    surprises[level],
                    k_t=self._pool(k[:, t], block),
                    v_t=v[:, t],
                    q_t=self._pool(q[:, t], block),
                    write=write[:, t],
                    forget=forget[:, t],
                    momentum=momentum,
                )
                read_sum = read_sum + gates[level] * read
            outputs.append(read_sum)
        # One batched projection over [B, L, m] instead of L per-step GEMMs.
        return self.out(torch.stack(outputs, dim=1))


class DataDependentDecayMemoryLane(nn.Module):
    """SSM-like linear memory with data-dependent decay.

    Unlike the constant decay in CausalFastWeightMemoryLane, this learns a
    data-dependent gate that controls the forgetting rate per position.
    This enables hard state tracking (like Mamba or recurrent models) by
    selectively ignoring or retaining state.
    """

    def __init__(self, dim: int, memory_dim: int | None = None) -> None:
        super().__init__()
        memory_dim = memory_dim or min(dim, 32)
        self.q = nn.Linear(dim, memory_dim, bias=False)
        self.k = nn.Linear(dim, memory_dim, bias=False)
        self.v = nn.Linear(dim, memory_dim, bias=False)
        self.write_gate = nn.Linear(dim, memory_dim)
        self.decay_gate = nn.Linear(dim, memory_dim)
        self.out = nn.Linear(memory_dim, dim, bias=False)

        # Init decay gate bias so that default is slow forgetting
        nn.init.constant_(self.decay_gate.bias, -2.0)

        self.dim = dim
        self.memory_dim = memory_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = torch.tanh(self.q(x))
        k = torch.tanh(self.k(x))
        v = torch.tanh(self.v(x))

        # Gates
        write_strength = torch.sigmoid(self.write_gate(x))
        decay = torch.sigmoid(self.decay_gate(x))

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
            # Elementwise broadcasting of decay and write_strength
            memory = (
                decay[:, t].unsqueeze(-1) * memory
                + write_strength[:, t].unsqueeze(-1) * write
            )
            outputs.append(torch.einsum("bi,bij->bj", q[:, t], memory))
        # One batched projection over [B, L, m] instead of L per-step GEMMs.
        return self.out(torch.stack(outputs, dim=1))


class LegendreSSMLane(nn.Module):
    """UNIMPLEMENTED placeholder — construction fails loud.

    The previous body returned ``x`` (identity), so every cohort that graded
    "legendre" measured the TinyLM scaffold, not a lane (2026-06-11 audit, C1).
    Implement a real Legendre/HiPPO scan before re-enabling.
    """

    def __init__(self, dim: int, state_dim: int = 64) -> None:
        super().__init__()
        raise NotImplementedError(
            "LegendreSSMLane is an unimplemented stub (forward was identity); "
            "grading it would measure the host scaffold. Implement or remove "
            "it from the cohort."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class PowerSemiringMemoryLane(nn.Module):
    """UNIMPLEMENTED placeholder — construction fails loud.

    The previous body was a plain ``nn.Linear`` in costume; any
    "power_semiring" grade measured the scaffold (2026-06-11 audit, C1).
    """

    def __init__(self, dim: int, memory_dim: int = 32) -> None:
        super().__init__()
        raise NotImplementedError(
            "PowerSemiringMemoryLane is an unimplemented stub (forward was a "
            "plain nn.Linear); grading it would measure the host scaffold. "
            "Implement or remove it from the cohort."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class SlotTableMemoryLane(nn.Module):
    """Per-token causal content-addressed slot memory (no temporal pooling).

    Each token softly routes its (key, value) into a slot table; the running per-slot
    means are read content-addressably via softmax(q·slot_key)·slot_val. A one-step
    shift makes the read strictly causal (token i reads only writes from tokens < i).
    This is the no-pooling sibling of UniversalMasterLane — the clean comparison point
    for whether the pooling/key-cache front-end actually helps.
    """

    def __init__(self, dim: int, n_slots: int = 16, memory_dim: int = 64) -> None:
        super().__init__()
        self.q = nn.Linear(dim, memory_dim, bias=False)
        self.k = nn.Linear(dim, memory_dim, bias=False)
        self.v = nn.Linear(dim, memory_dim, bias=False)
        self.write_route = nn.Linear(memory_dim, n_slots)
        self.out = nn.Linear(memory_dim, dim, bias=False)
        self.n_slots = n_slots
        self.memory_dim = memory_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = torch.tanh(self.q(x))
        k = torch.tanh(self.k(x))
        v = torch.tanh(self.v(x))
        route = torch.softmax(self.write_route(k), dim=-1)  # [b, seq, S]
        wk = route.unsqueeze(-1) * k.unsqueeze(2)  # [b, seq, S, m]
        wv = route.unsqueeze(-1) * v.unsqueeze(2)
        denom = route.cumsum(dim=1).clamp_min(1e-6).unsqueeze(-1)  # [b, seq, S, 1]
        slot_key = wk.cumsum(dim=1) / denom  # running per-slot key
        slot_val = wv.cumsum(dim=1) / denom

        # Strict causality: read the slot state BEFORE this token's own write.
        slot_key = torch.cat(
            [slot_key.new_zeros(slot_key[:, :1].shape), slot_key[:, :-1]], dim=1
        )
        slot_val = torch.cat(
            [slot_val.new_zeros(slot_val[:, :1].shape), slot_val[:, :-1]], dim=1
        )

        scores = torch.einsum("blm,blsm->bls", q, slot_key) * (self.memory_dim**-0.5)
        read = torch.einsum("bls,blsm->blm", torch.softmax(scores, dim=-1), slot_val)
        return self.out(read)


class MultiHeadSlotTableMemoryLane(nn.Module):
    """Multi-head causal content-addressed slot memory with selective writes.

    The optional improvements are cumulative and independently ablatable:
    null-write gates suppress non-memory tokens, a causal depthwise composer binds
    local key/value or entity/attribute/value groups, gated delta updates preserve
    slot plasticity, and cosine reads use a learned temperature per head.

    Reads always happen before the current token's write, including in delta mode.
    """

    def __init__(
        self,
        dim: int,
        memory_dim: int = 64,
        n_slots: int = 8,
        n_heads: int = 4,
        *,
        use_null_write: bool = True,
        use_composer: bool = True,
        use_delta_update: bool = True,
        normalize_read: bool = True,
        grouped_router: bool = False,
        use_query_lift: bool = False,
        route_from_input: bool = False,
        bilinear_read: bool = False,
        refine_write_route: bool = False,
        consolidate_slots: bool = False,
        normalize_slot_values: bool = False,
        use_router_prior: bool = False,
        composer_width: int = 3,
    ) -> None:
        super().__init__()
        if min(memory_dim, n_slots, n_heads) <= 0:
            raise ValueError("memory_dim, n_slots, and n_heads must be positive")
        if composer_width <= 0:
            raise ValueError("composer_width must be positive")
        self.n_heads = n_heads
        self.head_dim = max(1, memory_dim // n_heads)
        self.memory_dim = self.head_dim * n_heads
        self.n_slots = n_slots
        self.use_null_write = use_null_write
        self.use_composer = use_composer
        self.use_delta_update = use_delta_update
        self.normalize_read = normalize_read
        self.grouped_router = grouped_router
        self.use_query_lift = use_query_lift
        self.route_from_input = route_from_input
        self.bilinear_read = bilinear_read
        self.refine_write_route = refine_write_route
        self.consolidate_slots = consolidate_slots
        self.normalize_slot_values = normalize_slot_values
        self.use_router_prior = use_router_prior
        self.composer_width = composer_width
        if grouped_router and route_from_input:
            raise ValueError("grouped_router and route_from_input cannot be combined")
        if use_composer:
            self.composer = nn.Conv1d(
                dim,
                dim,
                kernel_size=composer_width,
                groups=dim,
                bias=False,
            )
            with torch.no_grad():
                self.composer.weight.fill_(1.0 / composer_width)
        if use_query_lift:
            self.query_lift = nn.Conv1d(
                dim,
                dim,
                kernel_size=composer_width,
                groups=dim,
                bias=False,
            )
            nn.init.zeros_(self.query_lift.weight)
        self.q = nn.Linear(dim, self.memory_dim, bias=False)
        self.k = nn.Linear(dim, self.memory_dim, bias=False)
        self.v = nn.Linear(dim, self.memory_dim, bias=False)
        if grouped_router:
            self.write_route = nn.ModuleList(
                nn.Linear(self.head_dim, n_slots) for _ in range(n_heads)
            )
        elif route_from_input:
            self.write_route = nn.Linear(dim, n_heads * n_slots)
        else:
            self.write_route = nn.Linear(self.memory_dim, n_heads * n_slots)
        if use_null_write:
            self.write_gate = nn.Linear(dim, n_heads)
            nn.init.constant_(self.write_gate.bias, -1.0)
        if normalize_read:
            self.log_read_scale = nn.Parameter(
                torch.full((n_heads,), 0.5 * torch.log(torch.tensor(self.head_dim)))
            )
        if bilinear_read:
            self.read_metric = nn.Parameter(
                torch.eye(self.head_dim).expand(n_heads, -1, -1).clone()
            )
        if refine_write_route:
            self.content_route_scale = nn.Parameter(torch.zeros(n_heads))
        if consolidate_slots:
            self.consolidation_gate = nn.Parameter(torch.zeros(n_heads))
        self.out = nn.Linear(self.memory_dim, dim, bias=False)
        if use_router_prior:
            self.route_proto = nn.Parameter(
                torch.randn(n_heads, n_slots, self.head_dim) * 0.02
            )
            self.route_proto_beta = nn.Parameter(torch.zeros(()))

    def _compose(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_composer:
            return x
        left_pad = self.composer_width - 1
        composed = nn.functional.pad(x.transpose(1, 2), (left_pad, 0))
        return self.composer(composed).transpose(1, 2)

    def _lift_query(self, x: torch.Tensor, memory_input: torch.Tensor) -> torch.Tensor:
        if not self.use_query_lift:
            return memory_input
        left_pad = self.composer_width - 1
        context = nn.functional.pad(x.transpose(1, 2), (left_pad, 0))
        return memory_input + self.query_lift(context).transpose(1, 2)

    def _prewrite_slot_states(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        write_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_delta_update:
            from research.synthesis.compiler_ops_sequence import (
                _parallel_associative_scan,
            )

            alpha = write_weight.clamp(0.0, 1.0 - 1e-6)
            log_decay = torch.log1p(-alpha).permute(0, 2, 3, 1).contiguous()
            key_write = (alpha.unsqueeze(-1) * k.unsqueeze(3)).permute(0, 2, 3, 4, 1)
            val_write = (alpha.unsqueeze(-1) * v.unsqueeze(3)).permute(0, 2, 3, 4, 1)
            slot_key = _parallel_associative_scan(log_decay, key_write)
            slot_val = _parallel_associative_scan(log_decay, val_write)
            slot_key = slot_key.permute(0, 4, 1, 2, 3)
            slot_val = slot_val.permute(0, 4, 1, 2, 3)
            zeros_key = slot_key.new_zeros(slot_key[:, :1].shape)
            zeros_val = slot_val.new_zeros(slot_val[:, :1].shape)
            return (
                torch.cat([zeros_key, slot_key[:, :-1]], dim=1),
                torch.cat([zeros_val, slot_val[:, :-1]], dim=1),
            )

        weighted_key = write_weight.unsqueeze(-1) * k.unsqueeze(3)
        weighted_val = write_weight.unsqueeze(-1) * v.unsqueeze(3)
        denom = write_weight.cumsum(dim=1).clamp_min(1e-6).unsqueeze(-1)
        slot_key = weighted_key.cumsum(dim=1) / denom
        slot_val = weighted_val.cumsum(dim=1) / denom
        zeros_key = slot_key.new_zeros(slot_key[:, :1].shape)
        zeros_val = slot_val.new_zeros(slot_val[:, :1].shape)
        return (
            torch.cat([zeros_key, slot_key[:, :-1]], dim=1),
            torch.cat([zeros_val, slot_val[:, :-1]], dim=1),
        )

    def _read_slots(
        self,
        q: torch.Tensor,
        slot_key: torch.Tensor,
        slot_val: torch.Tensor,
    ) -> torch.Tensor:
        if self.bilinear_read:
            q = torch.einsum("blhd,hde->blhe", q, self.read_metric)
        if self.normalize_read:
            q = nn.functional.normalize(q, dim=-1)
            slot_key = nn.functional.normalize(slot_key, dim=-1)
            scale = self.log_read_scale.exp().clamp(max=100.0).view(1, 1, -1, 1)
        else:
            scale = self.head_dim**-0.5
        if self.normalize_slot_values:
            slot_val = slot_val * torch.rsqrt(
                slot_val.pow(2).mean(dim=-1, keepdim=True) + 1e-6
            )
        scores = torch.einsum("blhd,blhsd->blhs", q, slot_key) * scale
        weights = torch.softmax(scores, dim=-1)
        return torch.einsum("blhs,blhsd->blhd", weights, slot_val)

    def _refine_route(
        self,
        route_logits: torch.Tensor,
        route: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        if not self.refine_write_route:
            return route
        provisional_key, _ = self._prewrite_slot_states(k, v, route)
        content_scores = torch.einsum(
            "blhd,blhsd->blhs",
            nn.functional.normalize(k, dim=-1),
            nn.functional.normalize(provisional_key, dim=-1),
        )
        scale = self.content_route_scale.view(1, 1, -1, 1)
        return torch.softmax(route_logits + scale * content_scores, dim=-1)

    def _consolidate(
        self,
        slot_key: torch.Tensor,
        slot_val: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.consolidate_slots:
            return slot_key, slot_val
        normalized_key = nn.functional.normalize(slot_key, dim=-1)
        scores = torch.einsum("blhsd,blhtd->blhst", normalized_key, normalized_key) * (
            self.head_dim**-0.5
        )
        weights = torch.softmax(scores, dim=-1)
        key_context = torch.einsum("blhst,blhtd->blhsd", weights, slot_key)
        val_context = torch.einsum("blhst,blhtd->blhsd", weights, slot_val)
        gate = self.consolidation_gate.tanh().view(1, 1, -1, 1, 1)
        return (
            slot_key + gate * (key_context - slot_key),
            slot_val + gate * (val_context - slot_val),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq_len, _ = x.shape
        h, s, hd = self.n_heads, self.n_slots, self.head_dim
        memory_input = self._compose(x)
        query_input = self._lift_query(x, memory_input)
        q = torch.tanh(self.q(query_input)).view(b, seq_len, h, hd)
        k = torch.tanh(self.k(memory_input)).view(b, seq_len, h, hd)
        v = torch.tanh(self.v(memory_input)).view(b, seq_len, h, hd)
        if self.grouped_router:
            route_logits = torch.stack(
                [
                    router(k[:, :, head_index])
                    for head_index, router in enumerate(self.write_route)
                ],
                dim=2,
            )
        elif self.route_from_input:
            route_logits = self.write_route(memory_input).view(b, seq_len, h, s)
        else:
            route_logits = self.write_route(k.reshape(b, seq_len, -1)).view(
                b, seq_len, h, s
            )
        if self.use_router_prior:
            route_bias = torch.einsum("blhd,hsd->blhs", k, self.route_proto)
            route_logits = route_logits + self.route_proto_beta * route_bias
        route = torch.softmax(route_logits, dim=-1)
        gate = None
        if self.use_null_write:
            gate = torch.sigmoid(self.write_gate(memory_input)).unsqueeze(-1)
            route = route * gate
        route = self._refine_route(route_logits, route, k, v)
        if gate is not None and self.refine_write_route:
            route = route * gate
        slot_key, slot_val = self._prewrite_slot_states(k, v, route)
        slot_key, slot_val = self._consolidate(slot_key, slot_val)
        read = self._read_slots(q, slot_key, slot_val)
        return self.out(read.reshape(b, seq_len, self.memory_dim))


class UniversalMasterLane(nn.Module):
    """Temporal-pooling + slotted-memory + selection-head key-cache lane.

    Causal by construction (verified by autograd Jacobian): each token reads only the
    slot state of the *previous completed* pool window, so within-window pooled writes
    never leak to the tokens that read them. Routing is soft (differentiable softmax,
    not argmax) and the read is content-addressed (softmax(q·slot_key)·slot_val), not a
    slot-sum. Supersedes the earlier acausal/frozen-router/blind-read prototype.
    """

    def __init__(
        self,
        dim: int,
        n_slots: int = 16,
        memory_dim: int = 64,
        latch_len: int = 8,
        pool_period: int = 4,
    ) -> None:
        super().__init__()
        self.q = nn.Linear(dim, memory_dim, bias=False)
        self.k = nn.Linear(dim, memory_dim, bias=False)
        self.v = nn.Linear(dim, memory_dim, bias=False)
        self.selection_q = nn.Linear(dim, memory_dim)
        self.write_route = nn.Linear(memory_dim, n_slots)
        self.out = nn.Linear(memory_dim, dim, bias=False)
        self.n_slots = n_slots
        self.memory_dim = memory_dim
        self.latch_len = latch_len
        self.pool_period = pool_period

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq_len, dim = x.shape
        device, dtype = x.device, x.dtype
        p = self.pool_period
        pad_len = (p - (seq_len % p)) % p
        xp = torch.cat([x, x.new_zeros(b, pad_len, dim)], dim=1) if pad_len else x
        n_pool = xp.shape[1] // p
        win = xp.view(b, n_pool, p, dim)

        # Temporal pooling. The full-window mean is safe: a window is only ever read by
        # tokens in strictly later windows (see read-index shift below), so no future leak.
        pooled = win.mean(dim=2)  # [b, n_pool, dim]
        pk = torch.tanh(self.k(pooled))  # [b, n_pool, m]
        pv = torch.tanh(self.v(pooled))  # [b, n_pool, m]

        # Selection-head key-cache: causal window of size latch_len over PAST pooled keys.
        pk_pad = torch.cat(
            [pk.new_zeros(b, self.latch_len - 1, self.memory_dim), pk], dim=1
        )
        l_keys = pk_pad.unfold(1, self.latch_len, 1).permute(
            0, 1, 3, 2
        )  # [b, n_pool, L, m]
        sq = torch.tanh(self.selection_q(win[:, :, -1, :]))  # [b, n_pool, m]
        attn = torch.softmax(torch.einsum("blm,blkm->blk", sq, l_keys), dim=-1)
        latched = torch.einsum("blk,blkm->blm", attn, l_keys)  # [b, n_pool, m]

        # Soft (differentiable) slot routing — replaces the zero-gradient argmax one-hot.
        route = torch.softmax(self.write_route(latched), dim=-1)  # [b, n_pool, S]
        wk = route.unsqueeze(-1) * latched.unsqueeze(2)  # [b, n_pool, S, m]
        wv = route.unsqueeze(-1) * pv.unsqueeze(2)
        denom = route.cumsum(dim=1).clamp_min(1e-6).unsqueeze(-1)  # [b, n_pool, S, 1]
        slot_keys = wk.cumsum(dim=1) / denom  # running per-slot key
        slot_vals = wv.cumsum(dim=1) / denom  # running per-slot value

        # Causal read: token i reads the slot state of the PREVIOUS completed window.
        read_idx = (torch.arange(seq_len, device=device) // p) - 1  # [seq]
        valid = (read_idx >= 0).view(1, seq_len, 1).to(dtype)
        idx = read_idx.clamp_min(0)
        sk = slot_keys[:, idx, :, :]  # [b, seq, S, m]
        sv = slot_vals[:, idx, :, :]
        qt = torch.tanh(self.q(x))  # [b, seq, m]

        # Content-addressed read — replaces the content-blind qt·Σ_slots.
        scores = torch.einsum("blm,blsm->bls", qt, sk) * (self.memory_dim**-0.5)
        read = torch.einsum("bls,blsm->blm", torch.softmax(scores, dim=-1), sv)
        return self.out(read * valid)
