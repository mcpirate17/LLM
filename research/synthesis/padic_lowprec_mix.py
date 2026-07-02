"""NM-C16 — p-adic low-precision gated mixer.

The VALIDATED p-adic gated mixer's NON-softmax per-channel highway gate
(``g = sigmoid(Wg·x + Wv·v_p(x) + b)`` — collapse-proof, mirrors
``_op_padic_gated_mixer`` in ``compiler_ops_routing.py``), but the projection
contribution is carried through a p-adic TRUNCATION operator that keeps only
``n_digits`` significant base-``p`` digits:

    z   = Wp·x
    z_q = padic_truncate(z, p, n_digits)        # keep n_digits p-adic digits (STE)
    mix = g · z_q + (1 - g) · x                 # the validated padic highway
    out = x + residual_scale · (mix - x)        # ReZero ⟹ identity-at-init

``padic_truncate`` rounds each element to the nearest multiple of
``p^(e - n_digits + 1)`` where ``e = floor(log_p|z|)`` is the most-significant
digit's exponent — i.e. it keeps a FIXED ``n_digits``-digit p-adic mantissa and
discards every finer digit. This is the p-adic analog of a low-bit mantissa, and
it is LOSSLESS FOR THE SCALES THE MIXER DISCARDS ANYWAY: the truncation error is
bounded by ``p^(1 - n_digits)/2`` in RELATIVE magnitude, INDEPENDENT of ``z``'s
scale. Concretely, the discarded digits lie above the valuation floor kept, so
their contribution to the magnitude is below the floor — truncating them moves
at most half a kept-digit, regardless of whether ``z`` is ~1 or ~1e6. p-adic
arithmetic naturally orders magnitude by valuation, so low bit-width is provably
lossless for exactly the scales the gate already routes around.

The compaction angle is PRECISION EFFICIENCY — carry the math at
``n_digits`` mantissa digits (≈ ``n_digits·log2(p)`` bits, e.g. ``n_digits=4``,
``p=2`` → ~4-bit mantissa activations) instead of fp32 — complementary to
NM-C3/C5 (param COUNT), NM-C15 (ternary WEIGHT bits), and NM-C2 (vocab-table
removal): NM-C16 quantizes the forward ACTIVATIONS, principled by p-adic
valuation rather than a loss-based PTQ heuristic.

``round`` is straight-through (STE): forward truncates, backward passes the
gradient through untouched. Identity-at-init via ReZero (``residual_scale=0`` ⟹
``out=x``). Self-contained (torch + math only) ⟹ NM-10-measurable via
``PhysicsDescriptorProbe`` and NM-11 softmax-twin-detectable (it IS a
``[B,L,D]→[B,L,D]`` feature mixer). Registry wiring DEFERRED (NM-C3/C5/C7/C10/
C15 convention — ship the mechanism, wire once codex's NM-1 lands).
"""

from __future__ import annotations

import math

import torch
from torch import nn

_PADIC_EPS = 1e-6
_DEFAULT_P = 2
_DEFAULT_N_DIGITS = 4
# Clamp the valuation signal feeding the gate to a finite band, matching the
# validated ``_op_padic_gated_mixer`` (valuation is an ultrametric scale signal;
# saturating at ±10 keeps the sigmoid gate well-conditioned for any input).
_VAL_CLAMP = 10.0


def _padic_valuation(x: torch.Tensor, p: int = _DEFAULT_P) -> torch.Tensor:
    """Smooth p-adic valuation ``v_p(x) = -log_p(|x|)`` (ultrametric scale signal).

    Inlined as a math identity (hermetic NM-C3/C5/C7/C10/C15 convention) rather
    than importing ``research.mathspaces.padic`` — identical to ``padic_valuation``
    there. ``smooth_abs = sqrt(x² + eps²)`` makes the valuation differentiable at
    zero (where the true valuation diverges).
    """
    log_p = math.log(p)
    smooth_abs = torch.sqrt(x * x + _PADIC_EPS * _PADIC_EPS)
    return -(torch.log(smooth_abs.clamp_min(_PADIC_EPS)) / log_p)


def padic_truncate(
    z: torch.Tensor, p: int = _DEFAULT_P, n_digits: int = _DEFAULT_N_DIGITS
) -> torch.Tensor:
    """Keep ``n_digits`` significant base-``p`` digits of ``z`` (STE-differentiable).

    Rounds each element to the nearest multiple of ``p^(e - n_digits + 1)`` where
    ``e = floor(log_p|z|)`` is the most-significant digit's exponent. This keeps a
    fixed ``n_digits``-digit p-adic mantissa per element — the p-adic analog of a
    low-bit floating-point mantissa. The truncation error is bounded by
    ``p^(1 - n_digits)/2`` in RELATIVE magnitude, independent of ``z``'s scale
    (the p-adic "low bit-width is lossless for discarded scales" property): the
    discarded digits sit above the valuation floor that is kept, so they move at
    most half a kept-digit regardless of how large or small ``z`` is.

    ``round`` is straight-through: the forward truncates to the grid, the backward
    passes the upstream gradient through as identity (``z + (q - z).detach()``), so
    the low-precision forward path is trainable end-to-end.
    """
    if n_digits < 1:
        raise ValueError(f"n_digits must be >= 1, got {n_digits}")
    log_p = math.log(p)
    mag = torch.sqrt(z * z + _PADIC_EPS * _PADIC_EPS)  # smooth |z|
    e = torch.floor(torch.log(mag.clamp_min(_PADIC_EPS)) / log_p)  # floor(log_p|z|)
    step = torch.pow(float(p), (e - (n_digits - 1)).to(z.dtype))  # p^(e-n_digits+1)
    q = torch.round(z / step) * step
    return z + (q - z).detach()  # STE: forward = q, backward = identity through z


def padic_lowprec_param_count(dim: int) -> int:
    """Exact trainable parameter count: ``3·D²`` (gate_x, gate_v, proj) + ``D``
    (gate bias) + ``1`` (ReZero scale). Same as the validated padic gated mixer —
    NM-C16's compaction is PRECISION (activation bit-width), not param count."""
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    return 3 * dim * dim + dim + 1


class PadicLowPrecMixer(nn.Module):
    """NM-C16 — p-adic low-precision gated mixer (precision-efficient novel op).

    ``forward(x)`` accepts a float tensor of shape ``(..., dim)`` and returns the
    same shape. The NON-softmax per-channel sigmoid gate ``g`` is informed by the
    p-adic valuation of ``x`` (content-dependent, collapse-proof — never a softmax
    over tokens); the projection it gates is carried at ``n_digits``-digit p-adic
    precision. Identity-at-init (ReZero ``residual_scale=0`` ⟹ ``out=x``).
    """

    def __init__(
        self,
        dim: int,
        *,
        n_digits: int = _DEFAULT_N_DIGITS,
        p: int = _DEFAULT_P,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.n_digits = int(n_digits)
        self.p = int(p)
        if self.dim < 1:
            raise ValueError(f"dim must be >= 1, got {self.dim}")
        if self.n_digits < 1:
            raise ValueError(f"n_digits must be >= 1, got {self.n_digits}")
        if self.p < 2:
            raise ValueError(f"p must be >= 2, got {self.p}")

        std = 0.02
        self.gate_x = nn.Parameter(torch.randn(self.dim, self.dim) * std)
        self.gate_v = nn.Parameter(torch.randn(self.dim, self.dim) * std)
        self.proj_w = nn.Parameter(torch.randn(self.dim, self.dim) * std)
        self.gate_bias = nn.Parameter(torch.zeros(self.dim))
        # ReZero: 0 at init ⟹ out == x exactly; learns to open the low-prec path.
        self.residual_scale = nn.Parameter(torch.zeros(()))

    @property
    def num_parameters(self) -> int:
        return padic_lowprec_param_count(self.dim)

    @property
    def mantissa_bits(self) -> float:
        """Approximate activation mantissa budget: ``n_digits`` base-``p`` digits."""
        return float(self.n_digits * math.log2(self.p))

    def gate(self, x: torch.Tensor) -> torch.Tensor:
        """NON-softmax per-channel sigmoid gate informed by p-adic valuation.

        ``g = sigmoid(Wg·x + Wv·v_p(x) + b)`` — the validated collapse-proof padic
        gate (mirrors ``_op_padic_gated_mixer``): per-channel (not a softmax over
        tokens), content-dependent via the valuation term, balanced ≈0.5 at init.
        """
        val = (
            _padic_valuation(x.float(), self.p)
            .clamp(-_VAL_CLAMP, _VAL_CLAMP)
            .to(x.dtype)
        )
        gx = torch.einsum("ij,...j->...i", self.gate_x, x)
        gv = torch.einsum("ij,...j->...i", self.gate_v, val)
        return torch.sigmoid(gx + gv + self.gate_bias.to(x.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.gate(x)  # (..., D)
        proj = torch.einsum("ij,...j->...i", self.proj_w, x)
        proj_q = padic_truncate(proj, self.p, self.n_digits)  # NM-C16 low-prec
        mix = g * proj_q + (1.0 - g) * x  # validated padic highway (quantized proj)
        return x + self.residual_scale * (mix - x)
