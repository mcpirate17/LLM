"""Physics/symmetry behavior descriptors for label-free, name-free discovery.

These characterize WHAT an operator does to its input under the symmetry group of
a sequence model, not what it is called. Two operators with the same physics
fingerprint occupy the same niche; an empty region of this space is a mechanism
"never seen" — that is the definition of novelty the search rewards.

The descriptors are measured empirically on random stimuli, so they apply to any
operator ``f: [B, L, D] -> [B, L, D]`` (a synthesized graph, a hand-written lane,
a normalization, an MLP — anything that preserves the token/channel shape):

- ``perm_equivariance``   — does ``f`` commute with permuting the TOKEN axis?
                            1.0 = set-like / pure channel op; low = position-aware
                            (attention, causal, conv all break token permutation).
- ``shift_equivariance``  — does ``f`` commute with a circular SHIFT of tokens?
                            1.0 = translation-equivariant (conv / SSM-like);
                            low = absolute-position-dependent (learned-pos attn).
- ``scale_homogeneity``   — is ``f`` degree-1 homogeneous, ``f(a x) = a f(x)``?
                            1.0 = linear-ish; low = strongly nonlinear (gates,
                            normalization, saturating activations).
- ``energy_gain``         — mean ``||f(x)|| / ||x||``. <1 dissipative, ~1
                            conservative, >1 expansive (a physics stability axis).
- ``spectral_radius``     — dominant eigenvalue magnitude of the local
                            linearization (finite-difference power iteration).
                            <1 contractive, ~1 marginal, >1 expansive.

Equivariance scores use the bounded map ``1 / (1 + rel_err)`` so 0 (broken) ..
1 (exact) regardless of operator scale. All functions are pure and fail loud on
the wrong rank — a mis-shaped operator must not be silently scored 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from .quality_diversity import BehaviorAxis

Operator = Callable[[Tensor], Tensor]

_EPS = 1e-9

PHYSICS_DESCRIPTOR_NAMES: tuple[str, ...] = (
    "perm_equivariance",
    "shift_equivariance",
    "scale_homogeneity",
    "energy_gain",
    "spectral_radius",
)


def _require_3d(x: Tensor) -> None:
    if x.dim() != 3:
        raise ValueError(
            f"physics descriptors require a [B, L, D] tensor; got shape {tuple(x.shape)}"
        )


def _rel_err(a: Tensor, b: Tensor) -> float:
    """Symmetric relative error, scale-free and bounded below by 0."""
    denom = 0.5 * (a.norm() + b.norm()) + _EPS
    return float((a - b).norm() / denom)


def _equivariance_score(rel_err: float) -> float:
    """Map a relative error to (0, 1]: 0 err -> 1.0, large err -> 0."""
    return 1.0 / (1.0 + rel_err)


@torch.no_grad()
def perm_equivariance(
    f: Operator, x: Tensor, perm: Tensor, *, fx: Tensor | None = None
) -> float:
    """How well ``f`` commutes with permuting the token (sequence) axis.

    ``f(P x)`` vs ``P f(x)``. A pure channel/pointwise op commutes exactly (1.0);
    any op that reads token position or token-token structure breaks it.
    """
    _require_3d(x)
    if perm.shape != (x.shape[1],):
        raise ValueError(
            f"perm must have shape ({x.shape[1]},); got {tuple(perm.shape)}"
        )
    f_of_perm = f(x[:, perm, :])
    base = f(x) if fx is None else fx
    perm_of_f = base[:, perm, :]
    return _equivariance_score(_rel_err(f_of_perm, perm_of_f))


@torch.no_grad()
def shift_equivariance(
    f: Operator, x: Tensor, k: int, *, fx: Tensor | None = None
) -> float:
    """How well ``f`` commutes with a circular shift of tokens by ``k``.

    ``f(shift_k x)`` vs ``shift_k f(x)``. Translation-equivariant operators
    (convolution, SSM with circular boundary) score ~1.0; absolute-position
    operators score low. Circular shift keeps the test boundary-clean.
    """
    _require_3d(x)
    f_of_shift = f(torch.roll(x, shifts=k, dims=1))
    base = f(x) if fx is None else fx
    shift_of_f = torch.roll(base, shifts=k, dims=1)
    return _equivariance_score(_rel_err(f_of_shift, shift_of_f))


@torch.no_grad()
def scale_homogeneity(
    f: Operator, x: Tensor, alpha: float = 2.0, *, fx: Tensor | None = None
) -> float:
    """Degree-1 homogeneity: how close is ``f(alpha x)`` to ``alpha f(x)``?

    Linear maps score 1.0; saturating activations, gates and normalization
    (which divide out scale) score well below 1.0.
    """
    _require_3d(x)
    if alpha == 0.0:
        raise ValueError("alpha must be non-zero")
    base = f(x) if fx is None else fx
    return _equivariance_score(_rel_err(f(alpha * x), alpha * base))


@torch.no_grad()
def energy_gain(f: Operator, x: Tensor, *, fx: Tensor | None = None) -> float:
    """Mean per-example energy ratio ``||f(x)|| / ||x||`` over the batch.

    A physics stability descriptor: <1 dissipative, ~1 conservative, >1 expansive.
    """
    _require_3d(x)
    y = f(x) if fx is None else fx
    xn = x.flatten(1).norm(dim=1)
    yn = y.flatten(1).norm(dim=1)
    return float((yn / (xn + _EPS)).mean())


@torch.no_grad()
def spectral_radius(
    f: Operator,
    x: Tensor,
    *,
    iters: int = 6,
    eps: float = 1e-2,
    generator=None,
    fx: Tensor | None = None,
) -> float:
    """Dominant eigenvalue magnitude of the local linearization of ``f`` at ``x``.

    Finite-difference power iteration: ``J v ~= (f(x + eps v) - f(x)) / eps`` needs
    only forward passes (no autograd graph), so it works on any operator including
    ``no_grad``/native ones. <1 contractive, ~1 marginal, >1 expansive.
    """
    _require_3d(x)
    fx = f(x) if fx is None else fx
    scale = eps * x.norm() / (x.numel() ** 0.5 + _EPS)
    v = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
    v = v / (v.norm() + _EPS)
    sigma = 0.0
    for _ in range(max(1, iters)):
        jv = (f(x + scale * v) - fx) / (scale + _EPS)
        sigma = float(jv.norm())
        v = jv / (jv.norm() + _EPS)
    return sigma


def physics_behavior_axes() -> tuple[BehaviorAxis, ...]:
    """MAP-Elites behavior axes over the physics fingerprint.

    Coarse 3-bin edges so the search first spreads across the qualitatively
    distinct symmetry classes (position-aware vs set-like, translation-equivariant
    vs absolute, linear vs nonlinear, contractive vs expansive) before refining.
    ``energy_gain`` is omitted from the default axes (correlated with
    ``spectral_radius``); add it explicitly when a finer stability split is wanted.
    """
    return (
        # <0.3 strongly position-aware (attention/causal); >0.7 set-like.
        BehaviorAxis("perm_equivariance", (0.3, 0.7)),
        # >0.7 translation-equivariant (conv/SSM); <0.3 absolute-position.
        BehaviorAxis("shift_equivariance", (0.3, 0.7)),
        # >0.85 linear-ish; <0.5 strongly nonlinear (gates/norm/saturation).
        BehaviorAxis("scale_homogeneity", (0.5, 0.85)),
        # <0.9 contractive, [0.9,1.1] marginal, >1.1 expansive.
        BehaviorAxis("spectral_radius", (0.9, 1.1)),
    )


@dataclass(slots=True)
class PhysicsDescriptorProbe:
    """Measure the physics fingerprint of an operator on random stimuli.

    The operator may be supplied directly as a callable, or as a model exposing
    the ``embed(ids)`` + ``_fingerprint_forward_from_embed(emb)`` contract (the
    SynthesizedModel / probe-adapter contract shared with ``measured_descriptors``)
    — in which case ``f = model._fingerprint_forward_from_embed`` and the stimulus
    is ``model.embed(random_ids)``. Descriptors are averaged over ``n_seeds`` fresh
    stimuli; ``None`` is returned only if no seed could be probed.
    """

    batch: int = 4
    seq_len: int = 32
    dim: int = 32
    vocab: int = 64
    n_seeds: int = 3
    device: str = "cpu"

    def describe_operator(self, f: Operator) -> dict[str, float]:
        """Physics descriptors for a bare ``[B, L, D] -> [B, L, D]`` callable."""
        per_seed: list[dict[str, float]] = []
        for seed in range(self.n_seeds):
            gen = torch.Generator(device=self.device).manual_seed(seed)
            x = torch.randn(
                self.batch, self.seq_len, self.dim, device=self.device, generator=gen
            )
            per_seed.append(self._describe_once(f, x, gen))
        return self._mean(per_seed)

    def describe_model(
        self, factory: Callable[[int], object]
    ) -> dict[str, float] | None:
        """Physics descriptors for a model built by ``factory(seed)``.

        ``factory(seed)`` must return a model exposing ``embed`` and
        ``_fingerprint_forward_from_embed``. Seeds that raise are skipped.
        """
        per_seed: list[dict[str, float]] = []
        for seed in range(self.n_seeds):
            try:
                model = factory(seed)
                gen = torch.Generator(device=self.device).manual_seed(seed)
                ids = torch.randint(
                    0,
                    self.vocab,
                    (self.batch, self.seq_len),
                    device=self.device,
                    generator=gen,
                )
                x = model.embed(ids).detach()
                f = model._fingerprint_forward_from_embed
                per_seed.append(self._describe_once(f, x, gen))
            except Exception:
                continue
        if not per_seed:
            return None
        return self._mean(per_seed)

    def _describe_once(
        self, f: Operator, x: Tensor, gen: torch.Generator
    ) -> dict[str, float]:
        perm = torch.randperm(x.shape[1], device=x.device, generator=gen)
        k = int(torch.randint(1, x.shape[1], (1,), device=x.device, generator=gen))
        with torch.no_grad():
            fx = f(x)
        return {
            "perm_equivariance": perm_equivariance(f, x, perm, fx=fx),
            "shift_equivariance": shift_equivariance(f, x, k, fx=fx),
            "scale_homogeneity": scale_homogeneity(f, x, fx=fx),
            "energy_gain": energy_gain(f, x, fx=fx),
            "spectral_radius": spectral_radius(f, x, generator=gen, fx=fx),
        }

    @staticmethod
    def _mean(per_seed: list[dict[str, float]]) -> dict[str, float]:
        return {
            name: float(sum(d[name] for d in per_seed) / len(per_seed))
            for name in PHYSICS_DESCRIPTOR_NAMES
        }
