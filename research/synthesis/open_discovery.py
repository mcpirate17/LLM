"""Open-ended, name-free mechanism discovery loop.

This is the integration that turns the two substrates into a running discoverer:

- a PROGRAM is a sampled composition of parametric atoms (norm/basis/scan, from
  ``parametric_atoms``) wrapped around a parametric mixer (``parametric_ops``),
  with its knobs randomized so it occupies some region of behaviour space;
- its NOVELTY coordinate is the measured physics fingerprint (``physics_descriptors``);
- its FITNESS is the label-free induction/binding capability score
  (``measured_descriptors``) — graded on what the operator DOES, never on a name;
- selection is MAP-Elites (``quality_diversity``): keep the most capable program
  PER physics niche, and bias new samples toward empty niches.

The result is an archive of capable mechanisms spread across the symmetry classes
— each named only after the fact, by the niche it fell into. No mechanism catalog,
no hand-written class per idea: the loop steers the atom knobs into empty regions
of physics space and keeps whatever works there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import Tensor, nn

from research.eval.induction_probe import _RESTRICTED_VOCAB
from research.tools.measured_descriptors import (
    MeasuredDescriptorExtractor,
    capability_score_from_descriptors,
)

from .parametric_atoms import (
    ATOM_KINDS,
    BASIS_AXES,
    NORM_AXES,
    AtomSpec,
    build_atom_stack,
)
from .parametric_ops import (
    ADDRESS_FAMILIES,
    AGGREGATE_FAMILIES,
    SCORE_NORM_FAMILIES,
    StageSpec,
    build_parametric_mix,
)
from .physics_descriptors import PhysicsDescriptorProbe, physics_behavior_axes
from .quality_diversity import Elite, MapElitesArchive

# Substrings that mark a learnable "knob" (vs a weight matrix). Randomizing only
# these moves the program through behaviour space while keeping the projections
# at their seeded init — the same identity-at-init knobs, sampled instead of zero.
_KNOB_MARKERS = ("logit", "beta", "gate", "scale", "decay", "tau")


@dataclass(frozen=True)
class ProgramSpec:
    """A sampled program = atom stack + mixer stage choice + knob spread."""

    atom: AtomSpec
    stage: StageSpec
    knob_scale: float

    @property
    def key(self) -> str:
        return f"{self.atom.key}>>{self.stage.key}@{self.knob_scale:.2f}"


class OperatorModel(nn.Module):
    """Wrap a bare ``[B, L, D] -> [B, L, D]`` operator in the probe contract.

    Exposes ``embed(ids)`` + ``_fingerprint_forward_from_embed(emb)`` so both the
    capability extractor and the physics probe can characterise it unchanged.
    """

    def __init__(self, operator: nn.Module, vocab: int, dim: int) -> None:
        super().__init__()
        self.tok = nn.Embedding(vocab, dim)
        self.operator = operator

    def embed(self, ids: Tensor) -> Tensor:
        return self.tok(ids)

    def _fingerprint_forward_from_embed(self, emb: Tensor) -> Tensor:
        return self.operator(emb)

    def forward(
        self, ids: Tensor
    ) -> Tensor:  # pragma: no cover - probe uses the two above
        return self._fingerprint_forward_from_embed(self.embed(ids))


def _randomize_knobs(module: nn.Module, gen: torch.Generator, scale: float) -> None:
    """Fill every knob parameter with ``N(0, scale)``; leave weight matrices."""
    with torch.no_grad():
        for name, param in module.named_parameters():
            if any(marker in name for marker in _KNOB_MARKERS):
                param.copy_(torch.randn(param.shape, generator=gen) * scale)


def build_program(spec: ProgramSpec, dim: int, seed: int) -> nn.Module:
    """Deterministically build (and knob-randomize) a program for one seed."""
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    atoms = build_atom_stack(dim, spec.atom)
    mixer = build_parametric_mix(dim, spec.stage)
    program = nn.Sequential(atoms, mixer)
    _randomize_knobs(program, gen, spec.knob_scale)
    return program


def sample_spec(
    gen: torch.Generator,
    archive: MapElitesArchive,
    *,
    max_atom_depth: int = 2,
    mutate_prob: float = 0.5,
) -> ProgramSpec:
    """Sample a program spec, biased toward illuminating the archive.

    With probability ``mutate_prob`` and a non-empty archive, mutate an existing
    elite (re-roll its knob spread / perturb its atom stack) — MAP-Elites
    illumination toward neighbouring niches. Otherwise sample fresh.
    """
    elites = archive.elites if archive is not None else []
    if elites and float(torch.rand(1, generator=gen)) < mutate_prob:
        parent = elites[int(torch.randint(len(elites), (1,), generator=gen))]
        base = parent.payload
        if isinstance(base, ProgramSpec):
            return _mutate(base, gen)
    return _fresh(gen, max_atom_depth)


def _choice(seq, gen: torch.Generator):
    return seq[int(torch.randint(len(seq), (1,), generator=gen))]


def _fresh(gen: torch.Generator, max_atom_depth: int) -> ProgramSpec:
    depth = int(torch.randint(max_atom_depth + 1, (1,), generator=gen))
    kinds = tuple(_choice(ATOM_KINDS, gen) for _ in range(depth))
    atom = AtomSpec(
        kinds=kinds,
        norm_axis=_choice(NORM_AXES, gen),
        basis_axis=_choice(BASIS_AXES, gen),
    )
    stage = StageSpec(
        address=_choice(ADDRESS_FAMILIES, gen),
        score_norm=_choice(SCORE_NORM_FAMILIES, gen),
        aggregate=_choice(AGGREGATE_FAMILIES, gen),
    )
    knob_scale = float(0.5 + 2.5 * torch.rand(1, generator=gen))
    return ProgramSpec(atom=atom, stage=stage, knob_scale=knob_scale)


def _mutate(base: ProgramSpec, gen: torch.Generator) -> ProgramSpec:
    # Re-roll the knob spread (the main behaviour driver) and occasionally swap
    # one stage/atom choice, keeping the rest of the parent's structure.
    knob_scale = float(0.5 + 2.5 * torch.rand(1, generator=gen))
    stage = base.stage
    if float(torch.rand(1, generator=gen)) < 0.5:
        stage = StageSpec(
            address=_choice(ADDRESS_FAMILIES, gen),
            score_norm=base.stage.score_norm,
            aggregate=_choice(AGGREGATE_FAMILIES, gen),
        )
    return ProgramSpec(atom=base.atom, stage=stage, knob_scale=knob_scale)


@dataclass
class DiscoveryResult:
    archive: MapElitesArchive
    evaluated: int
    inserted: int

    @property
    def coverage(self) -> float:
        return self.archive.coverage

    def leaderboard(self, top: int = 10) -> list[Elite]:
        return sorted(self.archive.elites, key=lambda e: e.fitness, reverse=True)[:top]


@dataclass
class OpenDiscovery:
    """Drive the sample → fingerprint → grade → archive loop."""

    dim: int = 32
    vocab: int = 64
    n_seeds: int = 2
    device: str = "cpu"
    physics: PhysicsDescriptorProbe = field(init=False)
    capability: MeasuredDescriptorExtractor = field(init=False)

    def __post_init__(self) -> None:
        # The capability probe draws token ids in [1, _RESTRICTED_VOCAB); the
        # embedding must cover that range or the Jacobian probe indexes OOB.
        self.vocab = max(self.vocab, _RESTRICTED_VOCAB)
        self.physics = PhysicsDescriptorProbe(
            dim=self.dim, vocab=self.vocab, n_seeds=self.n_seeds, device=self.device
        )
        self.capability = MeasuredDescriptorExtractor(
            device=self.device, n_seeds=self.n_seeds
        )

    def _factory(self, spec: ProgramSpec) -> Callable[[int], OperatorModel]:
        def make(seed: int) -> OperatorModel:
            op = build_program(spec, self.dim, seed)
            return OperatorModel(op, self.vocab, self.dim).to(self.device).eval()

        return make

    def evaluate(self, spec: ProgramSpec) -> tuple[dict, float] | None:
        """Return (physics descriptors, capability fitness) or None if unprobeable."""
        factory = self._factory(spec)
        phys = self.physics.describe_model(factory)
        if phys is None:
            return None
        capd = self.capability.descriptors_from_factory(factory)
        if capd is None:
            return None
        return phys, float(capability_score_from_descriptors(capd))

    def run(self, iters: int, seed: int = 0) -> DiscoveryResult:
        gen = torch.Generator().manual_seed(seed)
        archive = MapElitesArchive(axes=physics_behavior_axes())
        evaluated = inserted = 0
        for i in range(iters):
            spec = sample_spec(gen, archive)
            graded = self.evaluate(spec)
            if graded is None:
                continue
            phys, fitness = graded
            evaluated += 1
            if archive.add(f"{spec.key}#{i}", phys, fitness, payload=spec):
                inserted += 1
        return DiscoveryResult(archive=archive, evaluated=evaluated, inserted=inserted)
