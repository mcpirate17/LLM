"""MoR-style learnable recursion router on the surprise-memory substrate.

Phase-1 torch prototype for the design in
``research/notes/mor_native_recursion_router_2026-06-01.md``. It replaces the
native lane's inert fixed-threshold depth gate (non-differentiable, frozen
``lo/hi`` buffers, ``lo=0`` forbids skip → saturates to max depth) with a
**differentiable Mixture-of-Recursions / PonderNet halting router** over the
per-token delta-recursion, optionally conditioned on the surprise signal.

Reuses, rather than reinvents:
- ``SemiringSurpriseMemoryLane`` — the learnable tempered-semiring read + the
  Titans/TTT delta-rule substrate (``_addr``/``_read``/gates).
- the PonderNet halting pattern from
  ``block_templates.RecursiveDepthRouterBlock`` (``p_r = remainder·halt``,
  weighted-sum the per-step candidates, ``remainder *= (1-halt)``).

Differentiability is the whole point: the halt gate ``h_r ∈ (0,1)`` multiplies
each refinement's contribution, so gradient reaches the router even though
"how many steps" is conceptually discrete — exactly what the native threshold
gate lacks. The committed memory/surprise/readout are the ``p_r``-weighted
expectations over per-step candidates, so the causal scan still commits a single
state per token (PonderNet's expected-state commit). Compute-saving hard
early-exit (expert-choice top-β) is Phase 2 (native CUDA port); this prototype
runs all ``max_recursive_steps`` to keep the gradient dense and answer the
capability question first.
"""

from __future__ import annotations

import torch
from torch import nn

from .memory_primitives import SemiringSurpriseMemoryLane


class MoRSemiringSurpriseMemoryLane(SemiringSurpriseMemoryLane):
    """Surprise-memory lane with a learned MoR/PonderNet recursion-depth router.

    Per token, runs up to ``max_recursive_steps`` inner delta-rule refinements of
    the memory; a halting head emits ``h_r = σ(W·feat_r)`` per refinement and
    PonderNet weights ``p_r = (Π_{j<r}(1-h_j))·h_r`` (last step forced to halt)
    form a depth distribution. Committed ``memory``/``surprise``/``readout`` are
    the ``p_r``-weighted expectations; expected depth ``Σ_r p_r·r`` is exposed as
    ``last_ponder_cost`` (× ``ponder_weight``) for an optional compute penalty the
    trainer can add to the loss.

    ``surprise_conditioned=True`` feeds the per-step associative error magnitude
    into the halting head — the novel ingredient (route on prediction error, not
    only the hidden readout). Set False for the ablation that isolates whether the
    surprise input actually earns its place vs a plain readout-only router.
    """

    def __init__(
        self,
        dim: int,
        memory_dim: int | None = None,
        *,
        max_recursive_steps: int = 4,
        surprise_conditioned: bool = True,
        ponder_weight: float = 1e-2,
        use_rope: bool = False,
        max_seq_len: int = 1024,
    ) -> None:
        super().__init__(
            dim,
            memory_dim=memory_dim,
            use_rope=use_rope,
            max_seq_len=max_seq_len,
        )
        if max_recursive_steps < 1:
            raise ValueError("max_recursive_steps must be >= 1")
        self.max_recursive_steps = int(max_recursive_steps)
        self.surprise_conditioned = bool(surprise_conditioned)
        self.ponder_weight = float(ponder_weight)
        halt_in = self.memory_dim + (1 if surprise_conditioned else 0)
        self.halt_head = nn.Linear(halt_in, 1)
        # Start at low halt prob so the router begins by using the full depth
        # (the proven deep behavior); it can only learn to exit earlier if that
        # helps — a capability-preserving init, mirroring the semiring β≈4 init.
        nn.init.zeros_(self.halt_head.weight)
        nn.init.constant_(self.halt_head.bias, -2.0)  # σ(-2) ≈ 0.12
        self.last_ponder_cost: torch.Tensor | None = None

    def _halt(self, read_r: torch.Tensor, err_r: torch.Tensor) -> torch.Tensor:
        """Halting probability [B, 1] for the current refinement step."""
        feat = read_r
        if self.surprise_conditioned:
            e = err_r.norm(dim=-1, keepdim=True)  # prediction-error magnitude
            feat = torch.cat([read_r, e], dim=-1)
        return torch.sigmoid(self.halt_head(feat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q, k = self._addr(x)
        v = self.v(x)
        write = torch.sigmoid(self.write_gate(x)).squeeze(-1)  # [B, L]
        forget = torch.sigmoid(self.forget_gate(x))  # [B, L, m]
        momentum = torch.sigmoid(self.momentum_logit)
        memory = x.new_zeros(batch_size, self.memory_dim, self.memory_dim)
        surprise = x.new_zeros(batch_size, self.memory_dim, self.memory_dim)
        outputs: list[torch.Tensor] = []
        ponder_total = x.new_zeros(())
        steps = self.max_recursive_steps
        for t in range(seq_len):
            k_t, v_t, q_t = k[:, t], v[:, t], q[:, t]
            w_t = write[:, t].view(batch_size, 1, 1)
            decay_base = (1.0 - forget[:, t]).unsqueeze(-1) * memory  # shared base
            s_r = surprise
            mem_r = memory
            remainder = x.new_ones(batch_size, 1)
            mem_acc = x.new_zeros(batch_size, self.memory_dim, self.memory_dim)
            sur_acc = x.new_zeros(batch_size, self.memory_dim, self.memory_dim)
            read_acc = x.new_zeros(batch_size, self.memory_dim)
            depth_acc = x.new_zeros(batch_size, 1)
            for r in range(1, steps + 1):
                err_r = v_t - self._read(mem_r, k_t)  # read-before-write (causal)
                delta_r = torch.einsum("bi,bj->bij", k_t, err_r) * self._scale
                s_r = momentum * s_r + w_t * delta_r
                mem_r = decay_base + s_r
                read_r = self._read(mem_r, q_t)
                halt = self._halt(read_r, err_r)
                if r == steps:
                    halt = torch.ones_like(halt)
                p_r = remainder * halt  # [B, 1]
                p3 = p_r.unsqueeze(-1)
                mem_acc = mem_acc + p3 * mem_r
                sur_acc = sur_acc + p3 * s_r
                read_acc = read_acc + p_r * read_r
                depth_acc = depth_acc + p_r * float(r)
                remainder = remainder * (1.0 - halt)
            memory = mem_acc
            surprise = sur_acc
            outputs.append(read_acc)
            ponder_total = ponder_total + depth_acc.mean()
        self.last_ponder_cost = self.ponder_weight * (ponder_total / seq_len)
        # One batched [B, L, m] projection instead of L per-step GEMMs.
        return self.out(torch.stack(outputs, dim=1))
