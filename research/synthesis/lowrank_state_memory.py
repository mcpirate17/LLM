# pyright: reportPrivateImportUsage=false
"""NM-C13 — low-rank-native state memory (Lever 4: O(r·D) state VRAM).

A causal ``[B, L, D] -> [B, L, D]`` sequence mixer whose recurrent fast-weight
state is rank-r BY CONSTRUCTION: the state exists ONLY as two factor banks

    K ∈ R^{B, r, D}  (slot key factors),   V ∈ R^{B, r, D}  (slot value factors)

representing ``S = Σ_i v_i k_iᵀ`` — no ``D×D`` tensor is ever materialized, at
any point, on any path. State VRAM is O(r·D) instead of the O(D²) every other
fast-weight mixer in the repo carries (``dplr_gated_delta`` and NM-F2 both keep
a full ``B×D×D`` state; their "low-rank" is the UPDATE, not the state). The
compaction claim is structural, not asymptotic hand-waving: the rank of the
represented operator can never exceed r because only r factor pairs exist.

Update law (token t, exclusive read-before-write):

    read_t  = Σ_i v_i ⟨k_i, q_t⟩                      (unnormalized bilinear)
    i*      = argmax_i  1 / (1 + ‖k_i − k_t‖²/γ)      (hard, Lorentzian STE)
    K[i*]  ← (1 − β_t) K[i*] + β_t k_t                (EMA address — bounded)
    V[i*]  ← λ_{i*} V[i*] + β_t v_t                   (decayed accumulation)

Content-addressed hard slot writes: the incoming key selects the slot whose
CURRENT key factor it matches (same key ⟹ same slot — the binding-friendly
addressing the multi-slot-wall work demands), via the validated Lorentzian
bounded-reciprocal STE (NM-C11/C20/F9 convention). The ADDRESS update is
deliberately an EMA, not an accumulation: an accumulated address drifts past
the key it stores (after n same-key writes it sits at ~nβ·k), so repeats
eventually scatter to fresh slots — measured during this lane's own test
design. The EMA converges TO the key, keeping addresses stable while the
VALUE factor accumulates evidence. There is no softmax anywhere: writes are
hard top-1, reads are raw bilinear contractions with signed, unbounded
coefficients (the NM-C12 lesson — any simplex-normalized read measures as a
softmax twin on NM-11).

DISTINCT from: NM-C10 persistent memory (memory-as-PARAMETERS, shared across
sequences; here the state is per-sequence, written by the tokens), NM-F9 CDMA
(code-division superposition in ONE vector; here r explicit factor pairs with
learned content addressing), NM-F2 (full D×D state, rank-r projector UPDATE
law), linear attention (S = Σ v kᵀ grows to rank min(L, D); here rank ≤ r
forever).

Collapse modes and gates:
- Write pile-up — every token writes one slot ⟹ effective rank collapses to 1.
  ``write_balance_loss`` (differentiable, 0 balanced, exactly 1 at full
  pile-up) + ``slot_utilization`` diagnostic.
- Decay runaway — per-slot forget λ ∈ (0, 1) by sigmoid construction; factor
  norms stay bounded for bounded inputs.

Identity-at-init: ReZero ``α = 0``. Cross-token by design ⟹ the NM-11 twin
test carries no pointwise waiver. NM-10-measurable. The scan is a torch
reference implementation (Python loop over L, vectorized over batch and slots
— the NM-F2 precedent); a native scan is the production path if the lane
graduates.
"""

from __future__ import annotations

import torch
from torch import nn

_EPS = 1e-6


def lowrank_state_param_count(dim: int, rank: int) -> int:
    """Exact trainable parameter count.

    Three identity-init lifts + one output lift (``4·D²``) + slot salt
    (``r·D``) + per-slot decay logits (``r``) + write-strength gate (``D+1``)
    + ReZero scale (1).
    """
    _validate(dim, rank)
    return 4 * dim * dim + rank * dim + rank + dim + 2


def _validate(dim: int, rank: int) -> None:
    if dim < 1 or rank < 1:
        raise ValueError(f"need dim>=1 and rank>=1, got {dim=}, {rank=}")
    if rank >= dim:
        raise ValueError(
            f"rank must be < dim (the low-rank state claim); got {rank=} >= {dim=}"
        )


class LowRankStateMemory(nn.Module):
    """NM-C13 — fast-weight memory whose state is rank-r by construction.

    ``forward(x)`` = ``x + α · out_lift(read)`` with exclusive causal reads
    (token t reads the state written by tokens < t). ``scan_memory`` exposes
    the factor banks; ``write_balance_loss`` / ``slot_utilization`` /
    ``represented_rank`` are the gates.
    """

    def __init__(
        self,
        dim: int,
        *,
        rank: int = 8,
        lorentz_gamma: float = 1.0,
    ) -> None:
        super().__init__()
        _validate(dim, rank)
        if lorentz_gamma <= 0:
            raise ValueError(f"lorentz_gamma must be > 0, got {lorentz_gamma}")
        self.dim = int(dim)
        self.rank = int(rank)
        self.lorentz_gamma = float(lorentz_gamma)

        self.key_lift = nn.Linear(dim, dim, bias=False)
        self.query_lift = nn.Linear(dim, dim, bias=False)
        self.value_lift = nn.Linear(dim, dim, bias=False)
        self.out_lift = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            eye = torch.eye(dim)
            self.key_lift.weight.copy_(eye)
            self.query_lift.weight.copy_(eye)
            self.value_lift.weight.copy_(eye)
        # Learned per-slot key salt at UNIT scale: each slot gets a
        # well-separated resting address basin (the C10 bank prior). Tiny
        # salts make empty-slot selection a near-tie for any strong input —
        # one write then flips downstream slot choices chaotically
        # (position-dependent displacement, measured during this lane's own
        # binding-control test design). The EMA bends addresses toward
        # content from these anchors.
        self.slot_salt = nn.Parameter(torch.randn(rank, dim))
        # Per-slot forget: sigmoid ⟹ λ ∈ (0,1) — bounded by construction.
        self.decay_logit = nn.Parameter(torch.full((rank,), 3.0))  # λ ≈ 0.95
        # Write strength β_t = sigmoid(w·x_t + b). Bias init OPEN (β ≈ 0.88):
        # a half-strength write leaves the slot address midway to the key
        # (‖salt − 0.5k‖² ≈ 1.25 at unit scales), where a lucky fresh salt
        # (min over slots ≈ 1.5, fluctuation ±0.35) can steal the query —
        # measured during this lane's binding-control design. Decisive writes
        # put the address at ≈ 1.01, a robust margin; training modulates down.
        self.gate_weight = nn.Parameter(torch.zeros(dim))
        self.gate_bias = nn.Parameter(torch.full((), 2.0))
        # ReZero: exact identity at init.
        self.scale = nn.Parameter(torch.zeros(()))

    @property
    def num_parameters(self) -> int:
        return lowrank_state_param_count(self.dim, self.rank)

    def _slot_affinity(self, slot_keys: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Lorentzian affinity of incoming key ``k (B, D)`` to each slot's
        current salted key factor: ``(B, r)`` in (0, 1]. NON-softmax."""
        salted = slot_keys + self.slot_salt.unsqueeze(0)
        dist2 = (salted - k.unsqueeze(1)).square().mean(dim=-1)
        return 1.0 / (1.0 + dist2 / self.lorentz_gamma)

    def scan_memory(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the causal scan; return ``(reads, K, V, soft_writes)``.

        ``reads (B, L, D)`` are exclusive (token t sees state from < t).
        ``K``/``V`` are the FINAL factor banks ``(B, r, D)`` — the entire
        state; their product is the only operator this memory can represent,
        so rank ≤ r by construction. ``soft_writes (B, L, r)`` is the
        Lorentzian write distribution (backward path + balance gate input).
        """
        if x.ndim != 3:
            raise ValueError(f"x must be (B, L, D), got {tuple(x.shape)}")
        if x.shape[-1] != self.dim:
            raise ValueError(f"last dim must be {self.dim}, got {x.shape[-1]}")
        batch, length, _ = x.shape
        keys = self.key_lift(x)
        queries = self.query_lift(x)
        values = self.value_lift(x)
        beta = torch.sigmoid(x @ self.gate_weight + self.gate_bias)  # (B, L)
        decay = torch.sigmoid(self.decay_logit)  # (r,)

        slot_k = x.new_zeros(batch, self.rank, self.dim)
        slot_v = x.new_zeros(batch, self.rank, self.dim)
        reads: list[torch.Tensor] = []
        softs: list[torch.Tensor] = []
        for t in range(length):
            q_t = queries[:, t]
            # Exclusive bilinear read: Σ_i v_i ⟨k_i, q⟩ — signed, unnormalized.
            coeff = torch.einsum("brd,bd->br", slot_k, q_t)
            reads.append(torch.einsum("br,brd->bd", coeff, slot_v))

            k_t, v_t, b_t = keys[:, t], values[:, t], beta[:, t]
            soft = self._slot_affinity(slot_k, k_t)  # (B, r)
            soft = soft / soft.sum(dim=-1, keepdim=True).clamp_min(_EPS)
            hard = torch.zeros_like(soft)
            hard.scatter_(-1, soft.argmax(dim=-1, keepdim=True), 1.0)
            select = hard + soft - soft.detach()  # (B, r) STE
            softs.append(soft)

            write = (select * b_t.unsqueeze(-1)).unsqueeze(-1)  # (B, r, 1)
            # Address: EMA toward the incoming key (bounded, repeat-stable).
            slot_k = (1.0 - write) * slot_k + write * k_t.unsqueeze(1)
            # Value: decayed accumulation on the selected slot only.
            keep_v = 1.0 - select * (1.0 - decay.unsqueeze(0))  # (B, r)
            slot_v = keep_v.unsqueeze(-1) * slot_v + write * v_t.unsqueeze(1)
        return (
            torch.stack(reads, dim=1),
            slot_k,
            slot_v,
            torch.stack(softs, dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reads, _k, _v, _s = self.scan_memory(x)
        return x + self.scale * self.out_lift(reads)

    def represented_rank(self, x: torch.Tensor) -> int:
        """Numerical rank of the final represented operator ``Σ v_i k_iᵀ`` —
        ≤ rank BY CONSTRUCTION (materialized here for INSPECTION ONLY; the
        forward never builds a D×D tensor)."""
        with torch.no_grad():
            _r, slot_k, slot_v, _s = self.scan_memory(x)
            op = torch.einsum("brd,bre->bde", slot_v.float(), slot_k.float())
            return int(torch.linalg.matrix_rank(op).max().item())

    def slot_utilization(self, x: torch.Tensor) -> float:
        """Fraction of slots that receive at least one hard write: 1.0 = all
        slots used, 1/r = total pile-up (the rank-1 collapse mode)."""
        with torch.no_grad():
            _r, _k, _v, soft = self.scan_memory(x)
            written = soft.argmax(dim=-1)  # (B, L)
            used = torch.zeros(x.shape[0], self.rank, device=x.device)
            used.scatter_(1, written.reshape(x.shape[0], -1), 1.0)
            return float(used.mean(dim=-1).mean().item())

    def write_balance_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable anti-pile-up guard on the HARD write mass: 0 at a
        perfectly balanced write distribution, exactly 1 when ALL writes pile
        onto one slot (state degenerates to rank 1).

        Measured on the STE selection, not the raw Lorentzian soft mass — the
        Lorentzian is polynomial-tailed (the NM-C20 lesson), so soft mass
        stays spread even at total hard pile-up and would under-report the
        collapse; the STE keeps the forward value exact while the soft term
        carries the gradient.
        """
        if self.rank < 2:
            return x.new_zeros(())
        _r, _k, _v, soft = self.scan_memory(x)
        hard = torch.zeros_like(soft)
        hard.scatter_(-1, soft.argmax(dim=-1, keepdim=True), 1.0)
        select = hard + soft - soft.detach()
        p = select.mean(dim=(0, 1))  # (r,) — forward: hard write frequencies
        dev = p - 1.0 / self.rank
        return (dev * dev).sum() * self.rank / (self.rank - 1)
