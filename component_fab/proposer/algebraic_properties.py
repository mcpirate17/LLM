"""Measured algebraic-property signatures for candidate operators (NM-11).

Every probe here is empirical and name-free: it runs a bounded battery on a
callable ``f: [B, L, D] -> [B, L, D]`` and reports scalar properties in ``[0, 1]``.

The headline output is ``softmax_twin_score`` — a *structural* detector for the
softmax-shaped convex-token-averaging family (softmax / reciprocal-attention /
sparsemax / phase-lock / tropical-as-attention / semiring-as-average). Per the
project mission this signature exists to steer the search AWAY from twins toward
genuinely novel geometry; a high twin score is a pathology to demote, never a
target to hit.

Historically the fab flagged twins with a HAND-DECLARED ``softmax_twin_like``
bool on each catalog variant (``research/tools/dynamic_math_sweep.py``). This
module replaces that guesswork with an operator-level measurement that works on
any built lane.

A convex token-mixer is defined by three structural tells, all measured on
stimuli and AND-combined (geometric mean) so a candidate must exhibit ALL of
them to be flagged:

- partition of unity  — token-constant inputs pass through unchanged (rows of
  the effective mixing sum to 1). ``constant_token_preservation``.
- convex range        — outputs stay inside the per-channel token hull (mixing
  weights are non-negative). ``convex_range_fraction``.
- cross-token mixing  — the output at a position genuinely depends on OTHER
  tokens. ``cross_token_mixing``. This is the tell that a value blends tokens,
  and it is exactly zero for pointwise channel ops (gates / MLPs / norms) — the
  FFN building blocks that must never be mistaken for attention. A *signed*
  token mixer (e.g. a token difference / high-pass) mixes but breaks partition
  of unity, so the AND still spares it; only NON-NEGATIVE, SUM-TO-1, all-to-all
  blending lands in the softmax basin.

Two more properties are reported as novelty features (not twin tells):

- ``idempotence`` — ``f(f(x)) ≈ f(x)`` (projections / normalizations).
- ``additivity``  — ``f(x + y) ≈ f(x) + f(y)`` (the second linearity axiom,
  complementing the physics-descriptor ``scale_homogeneity``'s degree-1
  homogeneity; together they characterize a linear map).

All probes run under ``torch.no_grad`` and require a rank-3 ``[B, L, D]`` tensor;
a mis-shaped operator fails loud rather than being silently scored zero.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import torch
from torch import Tensor

Operator = Callable[[Tensor], Tensor]

_EPS = 1e-9

#: Default cut for :meth:`AlgebraicProperties.is_softmax_twin`. All three convex
#: averaging tells must be individually strong for the geometric mean to clear
#: this, which keeps pointwise and signed-mixing ops on the novel side.
SOFTMAX_TWIN_THRESHOLD = 0.7

ALGEBRAIC_PROPERTY_NAMES: tuple[str, ...] = (
    "constant_token_preservation",
    "convex_range_fraction",
    "cross_token_mixing",
    "softmax_twin_score",
    "idempotence",
    "additivity",
)


def _require_3d(x: Tensor) -> None:
    if x.dim() != 3:
        raise ValueError(
            f"algebraic properties require a [B, L, D] tensor; got shape {tuple(x.shape)}"
        )


def _rel_err(a: Tensor, b: Tensor) -> float:
    """Symmetric relative error, scale-free and bounded below by 0."""
    if a.shape != b.shape:
        raise ValueError(
            f"operator changed tensor shape {tuple(b.shape)} -> {tuple(a.shape)}; "
            "algebraic-property probes require a shape-preserving [B, L, D] op"
        )
    denom = 0.5 * (a.norm() + b.norm()) + _EPS
    return float((a - b).norm() / denom)


def _bounded(rel_err: float) -> float:
    """Map a relative error to ``(0, 1]``: 0 err -> 1.0, large err -> 0."""
    return 1.0 / (1.0 + rel_err)


@torch.no_grad()
def constant_token_preservation(f: Operator, x: Tensor) -> float:
    """Partition-of-unity tell: does a token-constant input pass through unchanged?

    Feeds a stimulus whose tokens are all equal to a per-example vector ``v``. A
    mixer ``y_i = sum_j A_ij x_j`` with rows summing to 1 returns ``v`` exactly
    (score ~1). A pointwise nonlinearity maps ``v -> g(v) != v`` and scores low;
    a signed token mixer maps a constant to something non-constant and scores low.
    """
    _require_3d(x)
    v = x[:, :1, :]
    xc = v.expand(-1, x.shape[1], -1).contiguous()
    return _bounded(_rel_err(f(xc), xc))


@torch.no_grad()
def convex_range_fraction(f: Operator, x: Tensor) -> float:
    """Fraction of output entries inside the per-channel token hull of the input.

    Non-negative token mixing keeps every output inside ``[min_j x_j, max_j x_j]``
    (score ~1). A tolerance scaled to the channel range absorbs float noise.
    """
    _require_3d(x)
    y = f(x)
    lo = x.min(dim=1, keepdim=True).values
    hi = x.max(dim=1, keepdim=True).values
    tol = 1e-4 * (hi - lo).abs() + _EPS
    inside = (y >= lo - tol) & (y <= hi + tol)
    return float(inside.float().mean())


@torch.no_grad()
def cross_token_mixing(f: Operator, x: Tensor, generator=None) -> float:
    """Does the output at a position depend on OTHER tokens? 0 for pointwise ops.

    Probes causally at the LAST position: resamples every earlier token, holds
    the last token fixed, and measures how much the last output moves. A pointwise
    channel op (gate / MLP / norm) leaves it unchanged (0) because output token i
    depends only on input token i; an all-to-all mixer (attention / mean pool /
    convolution) moves it (→1). Probing the last position keeps causal mixers
    (which at position 0 see only themselves) from reading as pointwise.
    """
    _require_3d(x)
    last = x.shape[1] - 1
    if last < 1:
        return 0.0
    y_last = f(x)[:, last, :]
    noise = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
    xp = x.clone()
    xp[:, :last, :] = noise[:, :last, :]
    yp_last = f(xp)[:, last, :]
    return min(1.0, _rel_err(y_last, yp_last))


@torch.no_grad()
def idempotence(f: Operator, x: Tensor) -> float:
    """``f(f(x)) ≈ f(x)``? Projections and normalizations score ~1."""
    _require_3d(x)
    y = f(x)
    return _bounded(_rel_err(f(y), y))


@torch.no_grad()
def additivity(f: Operator, x: Tensor, y: Tensor) -> float:
    """``f(x + y) ≈ f(x) + f(y)``? Linear maps score ~1, nonlinear ones low.

    Content-dependent mixers (softmax weights computed from the input) are NOT
    additive, so this is orthogonal to the twin signature: it is a novelty /
    linearity feature, not a twin tell.
    """
    _require_3d(x)
    _require_3d(y)
    return _bounded(_rel_err(f(x + y), f(x) + f(y)))


def softmax_twin_score(
    const_preservation: float, convex_range: float, mixing: float
) -> float:
    """Geometric mean of the three convex-averaging tells, in ``[0, 1]``.

    Geometric (not arithmetic) mean so that a low score on ANY single tell drags
    the composite down — a candidate must be a non-negative, partition-of-unity,
    all-to-all blend on all three axes to be flagged as a softmax twin.
    """
    product = max(0.0, const_preservation) * max(0.0, convex_range) * max(0.0, mixing)
    return float(product ** (1.0 / 3.0))


@dataclass(frozen=True, slots=True)
class AlgebraicProperties:
    """Measured algebraic-property signature for one operator (seed-averaged)."""

    constant_token_preservation: float
    convex_range_fraction: float
    cross_token_mixing: float
    softmax_twin_score: float
    idempotence: float
    additivity: float
    n_seeds: int

    def is_softmax_twin(self, threshold: float = SOFTMAX_TWIN_THRESHOLD) -> bool:
        """True when the operator behaves like a convex token-averager (softmax-shaped)."""
        return self.softmax_twin_score >= threshold

    def as_metadata(self) -> dict[str, float]:
        """Rounded, JSON-safe fields for ledger metadata (excludes ``n_seeds``)."""
        return {
            name: round(float(getattr(self, name)), 5)
            for name in ALGEBRAIC_PROPERTY_NAMES
        }

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(slots=True)
class AlgebraicPropertyProbe:
    """Measure the algebraic-property signature of an operator on random stimuli.

    Descriptors are averaged over ``n_seeds`` fresh stimuli. Use :meth:`measure`
    for a bare ``[B, L, D] -> [B, L, D]`` callable, or :meth:`measure_model` for a
    model exposing the ``embed(ids)`` + ``_fingerprint_forward_from_embed(emb)``
    contract (the ``SynthesizedModel`` / probe-adapter contract shared with
    ``measured_descriptors``), which probes at the model's own embedding width.
    A bare callable that changes tensor shape fails loud in the probes rather than
    being silently scored zero.
    """

    batch: int = 4
    seq_len: int = 16
    dim: int = 32
    vocab: int = 64
    n_seeds: int = 3
    device: str = "cpu"

    def measure(self, f: Operator) -> AlgebraicProperties:
        """Signature for a bare ``[B, L, D] -> [B, L, D]`` callable."""
        rows: list[dict[str, float]] = []
        for seed in range(self.n_seeds):
            gen = torch.Generator(device=self.device).manual_seed(seed)
            shape = (self.batch, self.seq_len, self.dim)
            x = torch.randn(*shape, device=self.device, generator=gen)
            y = torch.randn(*shape, device=self.device, generator=gen)
            rows.append(self._measure_from_stimuli(f, x, y, gen))
        return self._aggregate(rows)

    def measure_model(
        self, factory: Callable[[int], object]
    ) -> AlgebraicProperties | None:
        """Signature for a model built by ``factory(seed)``.

        ``factory(seed)`` must return a model exposing ``embed`` and
        ``_fingerprint_forward_from_embed``. Stimuli are real embeddings, so the
        probe runs at the model's own width regardless of ``dim``. Seeds that
        raise are skipped (mirrors ``measured_descriptors.descriptors_from_factory``);
        returns ``None`` if no seed could be probed.
        """
        rows: list[dict[str, float]] = []
        for seed in range(self.n_seeds):
            try:
                model = factory(seed)
                gen = torch.Generator(device=self.device).manual_seed(seed)
                shape = (self.batch, self.seq_len)
                ids = torch.randint(
                    0, self.vocab, shape, device=self.device, generator=gen
                )
                ids2 = torch.randint(
                    0, self.vocab, shape, device=self.device, generator=gen
                )
                x = model.embed(ids).detach()
                y = model.embed(ids2).detach()
                f = model._fingerprint_forward_from_embed
                rows.append(self._measure_from_stimuli(f, x, y, gen))
            except Exception:
                continue
        if not rows:
            return None
        return self._aggregate(rows)

    def _aggregate(self, rows: list[dict[str, float]]) -> AlgebraicProperties:
        agg = {
            name: sum(row[name] for row in rows) / len(rows)
            for name in ALGEBRAIC_PROPERTY_NAMES
        }
        return AlgebraicProperties(n_seeds=len(rows), **agg)

    @staticmethod
    def _measure_from_stimuli(
        f: Operator, x: Tensor, y: Tensor, gen: torch.Generator
    ) -> dict[str, float]:
        const = constant_token_preservation(f, x)
        convex = convex_range_fraction(f, x)
        mixing = cross_token_mixing(f, x, generator=gen)
        return {
            "constant_token_preservation": const,
            "convex_range_fraction": convex,
            "cross_token_mixing": mixing,
            "softmax_twin_score": softmax_twin_score(const, convex, mixing),
            "idempotence": idempotence(f, x),
            "additivity": additivity(f, x, y),
        }


def measure_algebraic_properties(
    f: Operator,
    *,
    dim: int = 32,
    seq_len: int = 16,
    batch: int = 4,
    n_seeds: int = 3,
    device: str = "cpu",
) -> AlgebraicProperties:
    """Convenience wrapper: measure ``f``'s algebraic-property signature."""
    probe = AlgebraicPropertyProbe(
        batch=batch, seq_len=seq_len, dim=dim, n_seeds=n_seeds, device=device
    )
    return probe.measure(f)
