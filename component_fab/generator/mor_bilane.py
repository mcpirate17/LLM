"""MoR-routed bilane that resumes the native adaptive bilane's trained weights.

The native ``NativeAdaptiveSemiringBiLaneSurpriseMemoryLane`` decides per-token
recursion depth with a frozen, non-differentiable threshold gate that saturates
to max depth. This module subclasses it and swaps ONLY that gate for a
differentiable Mixture-of-Recursions / PonderNet halting router, by overriding
``lane_a._scan`` with a faithful torch port of the native recursion (same
``_scan_params`` projections, semiring read, delta-rule, balanced surprise,
per-key-row decay) in which the integer ``adaptive_steps`` count is replaced by a
learned soft-halting weighting. Every existing parameter is unchanged, so a
checkpoint trained with the native bilane loads directly (``strict=False``); the
only new parameter is ``lane_a.halt_head``.

Faithfulness: with ``force_max_depth=True`` the router puts all mass on the
deepest step, reproducing the native scan at max depth (validated to rel<1e-4
against the C++/CUDA kernel) — so the port is exact and the router only modulates
how much of each refinement is committed.

This is the Phase-1 *torch* path (no CUDA kernel): correct + differentiable, but
slower per step than the native scan — meant for a short validation resume before
investing in the Phase-2 CUDA port. See
``research/notes/mor_native_recursion_router_2026-06-01.md``.
"""

from __future__ import annotations

import torch
from torch import nn

from .native_surprise_memory import (
    NativeAdaptiveSemiringBiLaneSurpriseMemoryLane,
    NativeAdaptiveSemiringRopeTitansMACSurpriseMemoryLane,
)


class MoRLaneA(NativeAdaptiveSemiringRopeTitansMACSurpriseMemoryLane):
    """Adaptive semiring MAC lane_a with the threshold gate replaced by a learned
    MoR/PonderNet halting router over the surprise-memory delta-recursion."""

    def __init__(self, *args, ponder_weight: float = 1e-2, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Surprise-conditioned halting head: input is [mean|surprise_r|,
        # mean|raw0|] — both prediction-error-driven magnitudes (the same signal
        # the native threshold gate compared, now learned). Zero-init weight +
        # negative bias => starts deep (proven behavior), can only learn to exit.
        self.halt_head = nn.Linear(2, 1)
        nn.init.zeros_(self.halt_head.weight)
        nn.init.constant_(self.halt_head.bias, -2.0)
        self.ponder_weight = float(ponder_weight)
        self.force_max_depth = False
        self.last_ponder_cost: torch.Tensor | None = None
        self.last_mean_depth: float | None = None
        self.last_depth_hist: list[float] | None = None

    @staticmethod
    def _bal(raw: torch.Tensor, balance: torch.Tensor) -> torch.Tensor:
        return raw / (1.0 + balance * raw.abs())

    @staticmethod
    def _semiring_read(
        mem: torch.Tensor, addr: torch.Tensor, beta: torch.Tensor, log_m: torch.Tensor
    ) -> torch.Tensor:
        """read[b, j] = (logsumexp_i β(mem[b,i,j]+addr[b,i]) − log m) / β."""
        scores = mem + addr.unsqueeze(-1)  # [B, m_key, m_val]
        lse = torch.logsumexp(beta * scores, dim=1)  # over key axis -> [B, m_val]
        return (lse - log_m) / beta

    def _scan(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v, write, forget, momentum, beta, balance = self._scan_params(x)
        b, length, m = v.shape
        scale = float(m) ** -0.5
        log_m = torch.log(torch.tensor(float(m), device=x.device, dtype=x.dtype))
        steps = self.max_recursive_steps
        mem = x.new_zeros(b, m, m)
        sur = x.new_zeros(b, m, m)
        outputs: list[torch.Tensor] = []
        ponder_total = x.new_zeros(())
        depth_total = 0.0
        hist = [0.0] * steps  # accumulated halting mass per depth r=1..steps
        for t in range(length):
            q_t, k_t, v_t = q[:, t], k[:, t], v[:, t]
            w_t = write[:, t].view(b, 1, 1)
            read_q = self._semiring_read(mem, q_t, beta, log_m)  # output (pre-write)
            err = v_t - self._semiring_read(mem, k_t, beta, log_m)
            delta = torch.einsum("bi,bj->bij", k_t, err) * scale
            raw0 = momentum * sur + w_t * delta
            raw0_mag = raw0.abs().mean(dim=(1, 2), keepdim=False).unsqueeze(-1)  # [B,1]
            s = self._bal(raw0, balance)
            remainder = x.new_ones(b, 1)
            sur_acc = x.new_zeros(b, m, m)
            depth_acc = x.new_zeros(b, 1)
            for r in range(1, steps + 1):
                if r > 1:
                    s = self._bal(momentum * s + w_t * delta, balance)
                feat = torch.cat(
                    [s.abs().mean(dim=(1, 2), keepdim=False).unsqueeze(-1), raw0_mag],
                    dim=-1,
                )  # [B, 2]
                halt = torch.sigmoid(self.halt_head(feat))  # [B, 1]
                if self.force_max_depth:
                    # all halting mass on the deepest step (always-max ablation /
                    # faithfulness check): never halt early, force-halt at the end.
                    halt = (
                        torch.ones_like(halt) if r == steps else torch.zeros_like(halt)
                    )
                elif r == steps:
                    halt = torch.ones_like(halt)  # last step must commit
                p_r = remainder * halt
                sur_acc = sur_acc + p_r.unsqueeze(-1) * s
                depth_acc = depth_acc + p_r * float(r)
                hist[r - 1] += float(p_r.mean().detach())
                remainder = remainder * (1.0 - halt)
            sur = sur_acc
            decay = (1.0 - forget[:, t]).unsqueeze(-1)  # per key-row i
            mem = decay * mem + sur
            outputs.append(read_q)
            ponder_total = ponder_total + depth_acc.mean()
            depth_total += float(depth_acc.mean().detach())
        self.last_ponder_cost = self.ponder_weight * (ponder_total / length)
        self.last_mean_depth = depth_total / length
        total_mass = sum(hist) or 1.0
        self.last_depth_hist = [round(h / total_mass, 4) for h in hist]
        return torch.stack(outputs, dim=1)  # [B, L, m]


class MoRAdaptiveSemiringBiLaneSurpriseMemoryLane(
    NativeAdaptiveSemiringBiLaneSurpriseMemoryLane
):
    """Native adaptive bilane with lane_a's depth gate replaced by the MoR router.

    Param-compatible with ``NativeAdaptiveSemiringBiLaneSurpriseMemoryLane`` except
    for the added ``lane_a.halt_head`` — so a native-bilane checkpoint resumes with
    ``strict=False`` and only that head fresh.
    """

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
        return MoRLaneA(
            dim,
            memory_dim=memory_dim,
            gate_bias=gate_bias,
            semiring_temp_init=semiring_temp_init,
            recursive_balance_init=recursive_balance_init,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            max_recursive_steps=max_recursive_steps,
        )

    @property
    def last_ponder_cost(self) -> torch.Tensor | None:
        return self.lane_a.last_ponder_cost


def apply_resume_init(model: nn.Module) -> int:
    """Re-apply the deep-start halt init (bias −2, zero weight) to every MoR lane.

    ``_build_tinylm``'s GPT-2 init overwrites ``halt_head`` to ~0.5 halt prob
    (mean depth ~1.9). For resuming a checkpoint trained at deep recursion we want
    the router to *start* near that trained depth so we isolate its learning, not
    a depth-shock. Call after the checkpoint load. Returns the number of lanes hit.
    """
    n = 0
    for mod in model.modules():
        if isinstance(mod, MoRLaneA):
            nn.init.zeros_(mod.halt_head.weight)
            nn.init.constant_(mod.halt_head.bias, -2.0)
            n += 1
    return n


def collect_ponder_cost(model: nn.Module) -> torch.Tensor | None:
    """Sum the per-lane MoR ponder costs across a model (None if there are none)."""
    total = None
    for mod in model.modules():
        if isinstance(mod, MoRLaneA) and mod.last_ponder_cost is not None:
            total = (
                mod.last_ponder_cost if total is None else total + mod.last_ponder_cost
            )
    return total
