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
  ``PadicSurpriseMemoryLane`` — Titans/TTT-style test-time memory whose write
  is the *surprise* (associative prediction error), generalizing the Hebbian
  lane. See ``_SurpriseMemoryBase`` for the mechanism.

All preserve ``[B, L, D]`` shape and produce finite gradients at init.
"""

from __future__ import annotations

import torch
from torch import nn


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

    def __init__(self, dim: int, memory_dim: int | None = None) -> None:
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

    @staticmethod
    def _unit(t: torch.Tensor) -> torch.Tensor:
        """L2-normalize the last dim so retrieval scores are bounded."""
        return t / t.norm(dim=-1, keepdim=True).clamp_min(1e-6)

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
        q = self._unit(torch.tanh(self.q(x)))
        k = self._unit(torch.tanh(self.k(x)))
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
            outputs.append(self.out(read))
        return torch.stack(outputs, dim=1)


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
            outputs.append(self.out(read_sum))
        return torch.stack(outputs, dim=1)
