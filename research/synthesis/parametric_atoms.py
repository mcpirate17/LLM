"""Parametric atoms for the non-mixer architectural slots (P0, sibling of ``parametric_ops``).

``parametric_ops`` makes the token-MIXER a sampled composition of identity-at-init
stages (the ``semiring_beta`` pattern: a learnable knob that vanishes at init so a
sampled op starts as plain softmax attention). This module extends that exact
pattern to the OTHER slots of a transformer block — normalization, basis, and
state/scan — so the search can dial a whole block, not just its mixer, and so the
"named mechanism" (RMSNorm vs LayerNorm, identity vs Fourier, no-state vs EMA vs
long-memory SSM) becomes a *coordinate* the optimizer/generator can travel through
rather than a discrete class a human must pick.

Every atom obeys the same two-part contract, asserted in ``test_parametric_atoms``:

1. **Identity at init** — its blend/gate knob starts at ~0, so the atom is a
   near pass-through (`||f(x) - x|| / ||x||` ~ 1e-4). A sampled stack of atoms is
   therefore stable, finite and gradient-carrying before any training.
2. **Knob steers a physics coordinate** — opening the knob moves a specific
   axis of ``physics_descriptors`` in a known direction (norm lowers
   ``scale_homogeneity``; token-basis lowers ``perm_equivariance``; the causal
   scan lowers ``perm_equivariance`` and lengthens memory). This is what lets the
   discovery loop *aim* at an empty physics niche instead of guessing names.

Grammar/dispatch registration (op_roles, compiler_ops_*) is a later integration
step and is intentionally not done here, mirroring ``parametric_ops``.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

# Identity-at-init: sigmoid(_OFF) ~ 4.5e-5, so a gated atom is a pass-through
# at init to ~1e-4 relative error while staying fully differentiable in the knob.
_OFF = -10.0
_EPS = 1e-6

NORM_AXES = ("channel", "token")
BASIS_AXES = ("channel", "token")


# --------------------------------------------------------------------------- #
# Normalization atom — none <-> RMS <-> LayerNorm, continuously
# --------------------------------------------------------------------------- #
class ParametricNorm(nn.Module):
    """Normalization as a continuous blend, identity at init.

    ``blend`` (sigmoid of a learnable logit, ~0 at init) mixes the input with its
    normalized form; ``center`` (sigmoid, ~0 at init) slides RMS->LayerNorm by
    turning on mean subtraction. At init blend~0 so the atom is a pass-through;
    opening ``blend`` divides out scale and so lowers ``scale_homogeneity``.
    """

    def __init__(self, dim: int, axis: str = "channel") -> None:
        super().__init__()
        if axis not in NORM_AXES:
            raise ValueError(f"unknown norm axis: {axis!r}")
        self.dim = dim
        self.axis = axis
        self.blend_logit = nn.Parameter(torch.tensor(_OFF))
        self.center_logit = nn.Parameter(torch.tensor(_OFF))
        self.gamma = nn.Parameter(torch.ones(dim))

    def _reduce_dim(self) -> int:
        return -1 if self.axis == "channel" else 1

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 3:
            raise ValueError(f"ParametricNorm needs [B, L, D]; got {tuple(x.shape)}")
        d = self._reduce_dim()
        center = torch.sigmoid(self.center_logit)
        mean = x.mean(dim=d, keepdim=True)
        xc = x - center * mean
        rstd = torch.rsqrt(xc.pow(2).mean(dim=d, keepdim=True) + _EPS)
        normalized = xc * rstd * self.gamma
        blend = torch.sigmoid(self.blend_logit)
        return x + blend * (normalized - x)


# --------------------------------------------------------------------------- #
# Basis atom — identity <-> fixed orthonormal basis (DCT), continuously
# --------------------------------------------------------------------------- #
class ParametricBasis(nn.Module):
    """Reversible basis rotation as a continuous blend, identity at init.

    ``mix`` (sigmoid, ~0 at init) blends the input with its projection onto a
    fixed orthonormal (DCT-II) basis along ``axis``. The basis is energy
    preserving, so opening ``mix`` rotates the representation without changing
    gain; on the TOKEN axis it mixes positions by fixed weights and so lowers
    ``perm_equivariance`` (a name-free way to introduce global, content-free
    token structure — the opposite end of the axis from attention).
    """

    def __init__(self, dim: int, axis: str = "channel", n: int | None = None) -> None:
        super().__init__()
        if axis not in BASIS_AXES:
            raise ValueError(f"unknown basis axis: {axis!r}")
        self.dim = dim
        self.axis = axis
        self.mix_logit = nn.Parameter(torch.tensor(_OFF))
        # Basis size is fixed for the channel axis; for the token axis it depends
        # on seq len, so it is built lazily and cached per length.
        self._basis_n = n if axis == "channel" else None
        if axis == "channel":
            self.register_buffer("_basis", _dct_matrix(dim), persistent=False)
        else:
            self._basis = None

    def _token_basis(self, seq_len: int, device, dtype) -> Tensor:
        if self._basis is None or self._basis.shape[0] != seq_len:
            self._basis = _dct_matrix(seq_len).to(device=device, dtype=dtype)
        return self._basis

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 3:
            raise ValueError(f"ParametricBasis needs [B, L, D]; got {tuple(x.shape)}")
        if self.axis == "channel":
            basis = self._basis.to(device=x.device, dtype=x.dtype)
            rotated = torch.matmul(x, basis)  # mix over channel (last) axis
        else:
            basis = self._token_basis(x.shape[1], x.device, x.dtype)
            rotated = torch.einsum("st,btd->bsd", basis, x)  # mix over token axis
        mix = torch.sigmoid(self.mix_logit)
        return x + mix * (rotated - x)


# --------------------------------------------------------------------------- #
# Scan atom — no state <-> EMA <-> long-memory diagonal SSM, continuously
# --------------------------------------------------------------------------- #
class ParametricScan(nn.Module):
    """Causal diagonal state as a continuous blend, identity at init.

    A per-channel causal EMA ``s_t = a*s_{t-1} + (1-a)*x_t`` with learnable decay
    ``a = sigmoid(log_decay)`` (per channel). ``gate`` (sigmoid, ~0 at init)
    blends the scanned signal with the input, so at init it is a pass-through.
    Opening ``gate`` introduces causal, order-dependent state — lowering
    ``perm_equivariance`` — while ``log_decay`` dials effective memory length
    (and pushes ``spectral_radius`` toward 1 / marginal stability).
    """

    def __init__(self, dim: int, decay_init: float = 0.5) -> None:
        super().__init__()
        if not 0.0 < decay_init < 1.0:
            raise ValueError("decay_init must be in (0, 1)")
        self.dim = dim
        self.gate_logit = nn.Parameter(torch.tensor(_OFF))
        self.log_decay = nn.Parameter(
            torch.full((dim,), math.log(decay_init / (1.0 - decay_init)))
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 3:
            raise ValueError(f"ParametricScan needs [B, L, D]; got {tuple(x.shape)}")
        decay = torch.sigmoid(self.log_decay)  # [D]
        state = torch.zeros(x.shape[0], x.shape[2], device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(x.shape[1]):
            state = decay * state + (1.0 - decay) * x[:, t, :]
            outputs.append(state)
        scanned = torch.stack(outputs, dim=1)
        gate = torch.sigmoid(self.gate_logit)
        return x + gate * (scanned - x)


# --------------------------------------------------------------------------- #
# Composition + spec
# --------------------------------------------------------------------------- #
ATOM_KINDS = ("norm", "basis", "scan")


@dataclass(frozen=True)
class AtomSpec:
    """A sampled stack of atoms = one ordered choice of kinds (+ axes).

    The empty stack is a pure pass-through; any stack is a pass-through at init.
    """

    kinds: tuple[str, ...] = ()
    norm_axis: str = "channel"
    basis_axis: str = "channel"

    def __post_init__(self) -> None:
        for kind in self.kinds:
            if kind not in ATOM_KINDS:
                raise ValueError(f"unknown atom kind: {kind!r}")
        if self.norm_axis not in NORM_AXES:
            raise ValueError(f"unknown norm axis: {self.norm_axis!r}")
        if self.basis_axis not in BASIS_AXES:
            raise ValueError(f"unknown basis axis: {self.basis_axis!r}")

    @property
    def key(self) -> str:
        return "+".join(self.kinds) if self.kinds else "identity"


def build_atom(kind: str, dim: int, spec: AtomSpec | None = None) -> nn.Module:
    spec = spec or AtomSpec()
    if kind == "norm":
        return ParametricNorm(dim, axis=spec.norm_axis)
    if kind == "basis":
        return ParametricBasis(dim, axis=spec.basis_axis)
    if kind == "scan":
        return ParametricScan(dim)
    raise ValueError(f"unknown atom kind: {kind!r}")


def build_atom_stack(dim: int, spec: AtomSpec | None = None) -> nn.Sequential:
    """An ordered stack of atoms; pass-through at init regardless of contents."""
    spec = spec or AtomSpec()
    return nn.Sequential(*[build_atom(kind, dim, spec) for kind in spec.kinds])


def enumerate_atom_specs(max_depth: int = 2) -> list[AtomSpec]:
    """Every atom stack up to ``max_depth`` (incl. the empty pass-through)."""
    specs: list[AtomSpec] = [AtomSpec()]
    for depth in range(1, max_depth + 1):
        for kinds in itertools.product(ATOM_KINDS, repeat=depth):
            specs.append(AtomSpec(kinds=kinds))
    return specs


def _dct_matrix(n: int) -> Tensor:
    """Orthonormal DCT-II matrix ``B`` (columns are basis vectors), ``B^T B = I``."""
    k = torch.arange(n, dtype=torch.float32).reshape(n, 1)
    j = torch.arange(n, dtype=torch.float32).reshape(1, n)
    basis = torch.cos(math.pi / n * (j + 0.5) * k)
    basis[0, :] *= 1.0 / math.sqrt(2.0)
    return (basis * math.sqrt(2.0 / n)).T.contiguous()
