"""NM-F9 — CDMA slot binding: code-division multiplexed superposition memory.

A causal ``[B, L, D] -> [B, L, D]`` sequence mixer that attacks the field-wide
multi-slot binding wall with the structure spread-spectrum communications solved
decades ago: **bind by spreading, retrieve by despreading**. Each slot owns a fixed
±1 spreading code; a token's value payload is spread over the code's chips and
superposed into ONE running state vector by an exact associative prefix sum (linear
time, causal by construction). Retrieval correlates the state against the addressed
slot's code — despreading gain = ``chips``, cross-slot interference bounded by the
code family's cross-correlation:

    write:  M_t = M_{t-1} + g_t · (v_t ⊗ c_{i_t})          (superposition)
    read:   v̂_t = (M_{t-1} reshaped) · c_{j_t} / chips      (despreading)

with ``v_t ∈ ℝ^{d_v}`` the payload (``d_v = D / chips``), ``c_i ∈ {±1}^{chips}`` the
slot's code, and ``i_t / j_t`` hard top-1 slot addresses from code correlation.

The **binding law itself has zero learned parameters** — the codes are fixed:

  * ``code_family="gold"`` (default): Gold codes from a preferred pair of LFSR
    m-sequences. Pairwise cross-correlation is three-valued and bounded by
    ``t(n) = 1 + 2^⌊(n+2)/2⌋`` — near the Welch lower bound — and the family has
    ``2^n + 1`` codes, i.e. MORE slots than chips with provably bounded crosstalk.
    Codes longer than ``chips`` are truncated (partial-period correlations degrade
    gracefully; the exact bound holds at full length ``chips = 2^n − 1``).
  * ``code_family="hadamard"``: Walsh–Hadamard rows — exactly orthogonal under the
    synchronous despreading used here (zero interference), limited to
    ``n_slots ≤ chips``. The interference-free control for the Gold family.

Capacity is an explicit engineering trade straight from CDMA, not an emergent hope:
more ``chips`` ⟹ more interference suppression (Gold: ≤ t(n)/chips per bound slot)
but a smaller payload ``d_v``. That yields the probe memorization cannot fake — a
predicted **interference curve**: binding accuracy vs number of bound slots must
degrade along the Welch-bound line as ``chips`` sweeps 32/64/128.

Non-QKV by construction and NON-softmax throughout: slot assignment is hard top-1 on
code correlation (straight-through estimator), the write gate is a sigmoid highway
(the validated non-twin form), and there is no normalization across positions or
slots anywhere. State is a single ``D``-vector regardless of sequence length — the
"little effective state" this operator family is built around.

Learned parameters (all outside the binding law): key/query lifts ``D→chips`` for
slot addressing, value compressor ``D→d_v``, output lift ``d_v→D`` (zero-init ⟹
**identity-at-init**), and an O(D) gate. At D=256/chips=32 that is ~21K params/layer
— capability-per-non-embedding-param is the design currency.

Mission adjacency: codex M3X-C1 ``ECCCodewordEmbedding`` uses codes to compress the
*vocab table*; this module uses codes for *state multiplexing* — same mathematics,
different layer (graph-checked 2026-07-01: no prior CDMA/spreading code in repo).
Self-contained on purpose — imports only ``torch`` so it is measurable by
``PhysicsDescriptorProbe`` (NM-10-scorable). Registry wiring deferred per the
NM-C3/C5/C15 convention. Lane: ``tasks/nm_f_operator_families_2026-07-01.md``.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Preferred pairs of primitive polynomials for Gold-code construction, as bitmasks
# (bit e set ⟺ coefficient of x^e is 1, including x^degree and x^0). Classic pairs
# from the Gold / Sarwate–Pursley tables (octal: 45/75, 103/147, 211/217, 1021/1131,
# 2011/3515). No preferred pairs exist for degree ≡ 0 (mod 4), hence no 8.
_PREFERRED_PAIRS: dict[int, tuple[int, int]] = {
    5: (0b100101, 0b111101),
    6: (0b1000011, 0b1100111),
    7: (0b10001001, 0b10001111),
    9: (0b1000010001, 0b1001011001),
    10: (0b10000001001, 0b11101001101),
}


def gold_cross_correlation_bound(degree: int) -> int:
    """``t(n) = 1 + 2^⌊(n+2)/2⌋`` — the three-valued Gold cross-correlation bound."""
    if degree not in _PREFERRED_PAIRS:
        raise ValueError(
            f"no preferred pair for degree {degree}; available: "
            f"{sorted(_PREFERRED_PAIRS)}"
        )
    return 1 + (1 << ((degree + 2) // 2))


def _m_sequence(poly: int, degree: int) -> list[int]:
    """Maximal-length binary sequence (period ``2^degree − 1``) from a Fibonacci
    LFSR with characteristic polynomial ``poly`` (must be primitive)."""
    period = (1 << degree) - 1
    reg = [1] + [0] * (degree - 1)
    taps = [e for e in range(degree) if (poly >> e) & 1]
    seq: list[int] = []
    for _ in range(period):
        seq.append(reg[0])
        new = 0
        for e in taps:
            new ^= reg[e]
        reg = reg[1:] + [new]
    return seq


def _gold_codes(n_slots: int, chips: int) -> Tuple[torch.Tensor, int]:
    """First ``n_slots`` Gold codes as a ±1 tensor ``(n_slots, chips)``.

    Picks the smallest degree with a preferred pair whose period covers ``chips``
    and whose family size (``2^n + 1``) covers ``n_slots``; truncates each code to
    ``chips``. Returns ``(codes, degree)``.
    """
    degree = None
    for n in sorted(_PREFERRED_PAIRS):
        period = (1 << n) - 1
        if period >= chips and period + 2 >= n_slots:
            degree = n
            break
    if degree is None:
        raise ValueError(
            f"no Gold family covers chips={chips}, n_slots={n_slots} "
            f"(max period {(1 << max(_PREFERRED_PAIRS)) - 1})"
        )
    poly_u, poly_v = _PREFERRED_PAIRS[degree]
    u = _m_sequence(poly_u, degree)
    v = _m_sequence(poly_v, degree)
    period = len(u)
    family = [u, v]
    for shift in range(period):
        if len(family) >= n_slots:
            break
        shifted = v[shift:] + v[:shift]
        family.append([a ^ b for a, b in zip(u, shifted)])
    bits = torch.tensor(
        [code[:chips] for code in family[:n_slots]], dtype=torch.float32
    )
    return 1.0 - 2.0 * bits, degree  # 0 -> +1, 1 -> -1


def _hadamard_codes(n_slots: int, chips: int) -> torch.Tensor:
    """First ``n_slots`` rows of the Sylvester–Hadamard matrix of order ``chips``
    (power of two) — exactly orthogonal ±1 codes."""
    if chips & (chips - 1) != 0:
        raise ValueError(f"hadamard chips must be a power of two, got {chips}")
    if n_slots > chips:
        raise ValueError(
            f"hadamard supports at most chips={chips} slots, got {n_slots}"
        )
    h = torch.ones(1, 1)
    while h.shape[0] < chips:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h[:n_slots].contiguous()


def cdma_param_count(dim: int, chips: int) -> int:
    """Trainable params: key+query lifts (``2·chips·D``), value compressor +
    output lift (``2·d_v·D``), gate (``D + 1``). Codes cost zero."""
    if dim < 1 or chips < 1:
        raise ValueError(f"dim and chips must be >= 1, got dim={dim}, chips={chips}")
    if dim % chips != 0:
        raise ValueError(f"chips must divide dim, got dim={dim}, chips={chips}")
    d_v = dim // chips
    return 2 * chips * dim + 2 * d_v * dim + dim + 1


def _hard_top1(logits: torch.Tensor) -> torch.Tensor:
    """Hard one-hot over the last dim (straight-through: forward is the exact
    argmax one-hot, backward passes the gradient to the raw correlations). No
    softmax — the selection is a max, not a normalized exponential."""
    idx = logits.argmax(dim=-1)
    hard = F.one_hot(idx, logits.shape[-1]).to(logits.dtype)
    return hard + logits - logits.detach()


class CDMASlotBinding(nn.Module):
    """Code-division multiplexed slot binding over a single superposed state."""

    def __init__(
        self,
        dim: int,
        *,
        n_slots: int = 8,
        chips: int = 32,
        code_family: str = "gold",
    ) -> None:
        super().__init__()
        if n_slots < 2:
            raise ValueError(f"n_slots must be >= 2, got {n_slots}")
        d_v = dim // chips
        if dim % chips != 0 or d_v < 1:
            raise ValueError(f"chips must divide dim, got dim={dim}, chips={chips}")
        self.d = dim
        self.n_slots = n_slots
        self.chips = chips
        self.d_v = d_v
        self.code_family = code_family
        if code_family == "gold":
            codes, self.degree = _gold_codes(n_slots, chips)
        elif code_family == "hadamard":
            codes = _hadamard_codes(n_slots, chips)
            self.degree = 0
        else:
            raise ValueError(f"unknown code_family {code_family!r}")
        self.register_buffer("codes", codes)  # (S, chips), fixed — never trained

        self.key_lift = nn.Linear(dim, chips, bias=False)
        self.query_lift = nn.Linear(dim, chips, bias=False)
        self.value_compress = nn.Linear(dim, d_v, bias=False)
        # Zero-init output lift ⟹ forward(x) == x at init (identity-at-init).
        self.out_lift = nn.Linear(d_v, dim, bias=False)
        nn.init.zeros_(self.out_lift.weight)
        # Sigmoid-highway write gate (validated non-twin form), O(D) params.
        self.gate_weight = nn.Parameter(torch.zeros(dim))
        self.gate_bias = nn.Parameter(torch.zeros(1))

    @property
    def num_parameters(self) -> int:
        return cdma_param_count(self.d, self.chips)

    def interference_bound(self) -> float:
        """Per-slot despread interference amplitude bound: ``t(n)/chips`` for Gold
        (exact at full period, approximate under truncation), 0 for Hadamard."""
        if self.code_family == "hadamard":
            return 0.0
        return gold_cross_correlation_bound(self.degree) / self.chips

    def _bind_and_despread(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Core binding math: returns ``(v_hat, write_idx, read_idx)`` where
        ``v_hat`` is the despread payload ``(B, L, d_v)`` read from the strictly-past
        superposition (a token never retrieves its own write)."""
        scale = 1.0 / math.sqrt(self.chips)
        write_sel = _hard_top1(self.key_lift(x) @ self.codes.T * scale)  # (B,L,S)
        read_sel = _hard_top1(self.query_lift(x) @ self.codes.T * scale)
        write_code = write_sel @ self.codes  # (B, L, chips)
        read_code = read_sel @ self.codes
        gate = torch.sigmoid(x @ self.gate_weight + self.gate_bias)  # (B, L)
        payload = self.value_compress(x)  # (B, L, d_v)
        spread = (
            gate.unsqueeze(-1).unsqueeze(-1)
            * payload.unsqueeze(-1)
            * write_code.unsqueeze(-2)
        )  # (B, L, d_v, chips)
        # Exact associative prefix sum, exclusive: state of strictly earlier tokens.
        memory = torch.cumsum(spread, dim=1) - spread
        v_hat = (memory * read_code.unsqueeze(-2)).sum(dim=-1) / self.chips
        return v_hat, write_sel.argmax(dim=-1), read_sel.argmax(dim=-1)

    def read_raw(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Despread payload + slot addresses, pre-output-lift (verification only)."""
        return self._bind_and_despread(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, L, D) -> (B, L, D)``: residual + despread-payload lift."""
        v_hat, _, _ = self._bind_and_despread(x)
        return x + self.out_lift(v_hat)
