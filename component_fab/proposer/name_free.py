"""Name-free, physics-driven proposal source for autonomous fab cycles.

The old fab loop mostly proposes by recombining named mechanisms. This module
adds a different track: run small deterministic physics experiments over the
parametric atom/mixer grammar, measure each candidate's behavior descriptors,
and emit the candidates that land closest to useful under-covered physics
regions. The emitted specs are still normal ``ProposalSpec`` objects, so the
existing smoke/capability/paired/scale gates remain the source of truth.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from component_fab.proposer.spec_generator import ProposalSpec, build_spec_from_axes
from component_fab.state.gates import GATE_S05_CAUSALITY_STABILITY
from component_fab.state.ledger import Ledger, LedgerEntry
from research.synthesis.open_discovery import ProgramSpec, build_program
from research.synthesis.parametric_atoms import AtomSpec
from research.synthesis.parametric_ops import StageSpec
from research.synthesis.physics_descriptors import (
    PHYSICS_DESCRIPTOR_NAMES,
    PhysicsDescriptorProbe,
    physics_behavior_axes,
)
from research.synthesis.quality_diversity import MapElitesArchive


@dataclass(frozen=True, slots=True)
class PhysicsExperiment:
    """One targeted physics hypothesis to test with name-free atom programs."""

    name: str
    target: str
    descriptor_target: Mapping[str, float]
    atom_specs: tuple[AtomSpec, ...]
    stage_specs: tuple[StageSpec, ...]
    knob_scales: tuple[float, ...]
    rationale: str
    base_priority: float = 1.0


@dataclass(frozen=True, slots=True)
class ScoredPhysicsProgram:
    """A measured candidate program for one physics experiment."""

    experiment: PhysicsExperiment
    spec: ProgramSpec
    descriptors: Mapping[str, float]
    niche: tuple[int, ...]
    distance: float
    seed: int


_BASELINE_STAGE = StageSpec()
_DEFAULT_EXPERIMENTS: tuple[PhysicsExperiment, ...] = (
    PhysicsExperiment(
        name="long_gap_ordered_memory",
        target="long_gap_ordered_memory",
        descriptor_target={
            "perm_equivariance": 0.25,
            "shift_equivariance": 0.70,
            "scale_homogeneity": 0.75,
            "spectral_radius": 0.95,
        },
        atom_specs=(
            AtomSpec(kinds=("scan",)),
            AtomSpec(kinds=("scan", "basis"), basis_axis="token"),
            AtomSpec(kinds=("basis", "scan"), basis_axis="token"),
            AtomSpec(kinds=("scan", "norm", "basis"), basis_axis="token"),
        ),
        stage_specs=(
            StageSpec("reciprocal", "sharpen", "semiring"),
            StageSpec("cosine", "sharpen", "semiring"),
            StageSpec("dot", "sharpen", "semiring"),
            StageSpec("reciprocal", "softmax", "mean"),
        ),
        knob_scales=(1.25, 2.25, 3.25),
        rationale=(
            "Test causal scan + content/basis transforms for long-gap ordered "
            "state while keeping the local linearization near marginal stability."
        ),
        base_priority=1.25,
    ),
    PhysicsExperiment(
        name="content_addressed_binding",
        target="binding_content_addressed_state",
        descriptor_target={
            "perm_equivariance": 0.35,
            "shift_equivariance": 0.35,
            "scale_homogeneity": 0.55,
            "spectral_radius": 0.95,
        },
        atom_specs=(
            AtomSpec(kinds=("basis", "scan"), basis_axis="token"),
            AtomSpec(kinds=("scan", "basis"), basis_axis="token"),
            AtomSpec(kinds=("norm", "basis", "scan"), basis_axis="token"),
            AtomSpec(kinds=("basis", "norm", "scan"), basis_axis="token"),
        ),
        stage_specs=(
            StageSpec("cosine", "sharpen", "semiring"),
            StageSpec("dot", "sharpen", "semiring"),
            StageSpec("reciprocal", "sharpen", "semiring"),
            StageSpec("cosine", "softmax", "semiring"),
        ),
        knob_scales=(1.0, 2.0, 3.0),
        rationale=(
            "Test basis+scan content addressing for binding failures where "
            "state exists but exact key/value isolation is weak."
        ),
        base_priority=1.2,
    ),
    PhysicsExperiment(
        name="induction_symmetry_break",
        target="induction_symmetry_break",
        descriptor_target={
            "perm_equivariance": 0.45,
            "shift_equivariance": 0.55,
            "scale_homogeneity": 0.75,
            "spectral_radius": 1.00,
        },
        atom_specs=(
            AtomSpec(kinds=("basis",), basis_axis="token"),
            AtomSpec(kinds=("basis", "scan"), basis_axis="token"),
            AtomSpec(kinds=("scan", "basis"), basis_axis="token"),
            AtomSpec(kinds=("norm", "basis"), basis_axis="token"),
        ),
        stage_specs=(
            StageSpec("reciprocal", "sharpen", "mean"),
            StageSpec("cosine", "sharpen", "mean"),
            StageSpec("reciprocal", "sharpen", "semiring"),
            StageSpec("dot", "sharpen", "semiring"),
        ),
        knob_scales=(0.75, 1.75, 2.75),
        rationale=(
            "Search for controlled permutation/shift symmetry breaking, the "
            "descriptor pattern observed in induction-capable programs."
        ),
    ),
    PhysicsExperiment(
        name="stable_nonlinear_route",
        target="stable_nonlinear_route",
        descriptor_target={
            "perm_equivariance": 0.55,
            "shift_equivariance": 0.65,
            "scale_homogeneity": 0.45,
            "spectral_radius": 0.85,
        },
        atom_specs=(
            AtomSpec(kinds=("norm", "scan")),
            AtomSpec(kinds=("scan", "norm")),
            AtomSpec(kinds=("norm", "basis"), basis_axis="token"),
            AtomSpec(kinds=("basis", "norm", "scan"), basis_axis="token"),
        ),
        stage_specs=(
            StageSpec("cosine", "sharpen", "mean"),
            StageSpec("dot", "sharpen", "semiring"),
            StageSpec("cosine", "sharpen", "semiring"),
        ),
        knob_scales=(0.75, 1.5, 2.25),
        rationale=(
            "Probe contractive nonlinear routes that can avoid gate collapse "
            "without falling back to a softmax-twin path."
        ),
    ),
)


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=4).digest(), "big")


def _latest_metadata(entry: LedgerEntry) -> dict[str, Any]:
    return dict(entry.metadata_history[-1]) if entry.metadata_history else {}


def _experiment_priorities(
    experiments: Sequence[PhysicsExperiment], ledger: Ledger
) -> dict[str, float]:
    """Rank experiment families from the ledger's measured failures and wins."""

    priorities = {exp.name: exp.base_priority for exp in experiments}
    for entry in ledger.all_entries():
        metadata = _latest_metadata(entry)
        axes = metadata.get("math_axes") or {}
        target = str(axes.get("op_physics_target") or "")
        ratios = metadata.get("physics_probe_task_ratios") or {}
        best_ratio = max((float(v) for v in ratios.values()), default=0.0)
        if target.startswith("long_gap"):
            priorities["long_gap_ordered_memory"] += 0.15 * max(0.0, best_ratio - 1.0)
        if target in {"binding_content_addressed_state", "broad_kv_content_lookup"}:
            priorities["content_addressed_binding"] += 0.15 * max(0.0, best_ratio - 1.0)

        if (
            bool(metadata.get("range_ran"))
            and int(metadata.get("range_effective_distance") or 0) <= 0
        ):
            priorities["long_gap_ordered_memory"] += 0.25
        if (
            not bool(metadata.get("can_bind"))
            and float(metadata.get("nb_max_accuracy") or 0.0) < 0.62
        ):
            priorities["content_addressed_binding"] += 0.20
        if metadata.get("capability_eliminated_by") == GATE_S05_CAUSALITY_STABILITY:
            priorities["stable_nonlinear_route"] += 0.15
    return priorities


def _descriptor_distance(
    descriptors: Mapping[str, float], target: Mapping[str, float]
) -> float:
    total = 0.0
    n = 0
    for name, expected in target.items():
        if name not in descriptors:
            continue
        total += abs(float(descriptors[name]) - float(expected))
        n += 1
    return total / max(1, n)


def _is_softmax_default(spec: ProgramSpec) -> bool:
    return not spec.atom.kinds and spec.stage == _BASELINE_STAGE


def _coordinate_key(axes: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        "physics",
        str(axes.get("op_physics_seed") or ""),
        str(axes.get("op_physics_knob_scale") or ""),
        str(axes.get("op_physics_atom_kinds") or ""),
        str(axes.get("op_physics_basis_axis") or ""),
        str(axes.get("op_physics_norm_axis") or ""),
        str(axes.get("op_physics_address_family") or ""),
        str(axes.get("op_physics_score_norm_family") or ""),
        str(axes.get("op_physics_aggregate_family") or ""),
    )


def _failed_physics_coordinates(ledger: Ledger) -> set[tuple[str, ...]]:
    failed: set[tuple[str, ...]] = set()
    for entry in ledger.all_entries():
        for metadata in entry.metadata_history:
            axes = metadata.get("math_axes") or {}
            if axes.get("op_search_track") != "physics_atom":
                continue
            if (
                metadata.get("capability_eliminated_by") == GATE_S05_CAUSALITY_STABILITY
                or metadata.get("eliminated_by") == GATE_S05_CAUSALITY_STABILITY
            ):
                failed.add(_coordinate_key(axes))
    return failed


def _candidate_specs(experiment: PhysicsExperiment) -> Iterable[ProgramSpec]:
    for atom in experiment.atom_specs:
        for stage in experiment.stage_specs:
            for scale in experiment.knob_scales:
                spec = ProgramSpec(atom=atom, stage=stage, knob_scale=float(scale))
                if not _is_softmax_default(spec):
                    yield spec


def _score_experiment(
    experiment: PhysicsExperiment,
    *,
    dim: int,
    cycle: int,
    max_candidates: int,
) -> list[ScoredPhysicsProgram]:
    probe = PhysicsDescriptorProbe(dim=dim, vocab=64, n_seeds=1, device="cpu")
    archive = MapElitesArchive(axes=physics_behavior_axes())
    seed_base = _stable_seed("name_free", experiment.name, cycle)
    return sorted(
        _iter_scored_candidates(
            experiment,
            dim=dim,
            max_candidates=max_candidates,
            probe=probe,
            archive=archive,
            seed_base=seed_base,
        ),
        key=lambda row: row.distance,
    )


def _iter_scored_candidates(
    experiment: PhysicsExperiment,
    *,
    dim: int,
    max_candidates: int,
    probe: PhysicsDescriptorProbe,
    archive: MapElitesArchive,
    seed_base: int,
) -> Iterable[ScoredPhysicsProgram]:
    for index, program in enumerate(_candidate_specs(experiment)):
        if index >= max_candidates:
            break
        seed = seed_base + index
        op = build_program(program, dim=dim, seed=seed)
        descriptors = probe.describe_operator(op)
        yield ScoredPhysicsProgram(
            experiment=experiment,
            spec=program,
            descriptors=descriptors,
            niche=archive.niche_for(descriptors),
            distance=_descriptor_distance(descriptors, experiment.descriptor_target),
            seed=seed,
        )


def _axes_for_scored(row: ScoredPhysicsProgram) -> dict[str, Any]:
    kinds = "+".join(row.spec.atom.kinds) if row.spec.atom.kinds else "identity"
    has_scan = "scan" in row.spec.atom.kinds
    axes: dict[str, Any] = {
        "op_search_track": "physics_atom",
        "op_physics_source": "name_free_experiment",
        "op_physics_experiment": row.experiment.name,
        "op_physics_target": row.experiment.target,
        "op_physics_variant": "measured",
        "op_physics_seed": row.seed,
        "op_physics_atom_kinds": kinds,
        "op_physics_norm_axis": row.spec.atom.norm_axis,
        "op_physics_basis_axis": row.spec.atom.basis_axis,
        "op_physics_address_family": row.spec.stage.address,
        "op_physics_score_norm_family": row.spec.stage.score_norm,
        "op_physics_aggregate_family": row.spec.stage.aggregate,
        "op_physics_knob_scale": round(float(row.spec.knob_scale), 4),
        "op_physics_niche": ".".join(str(i) for i in row.niche),
        "op_physics_descriptor_distance": round(row.distance, 5),
        "op_dynamical_has_state": 1 if has_scan else 0,
        "op_dynamical_memory_length_class": "O(L)" if has_scan else "O(L^2)",
        "op_geometric_receptive_field": "global",
        "op_activation_sparsity_pattern": (
            "learned_structured"
            if row.spec.stage.aggregate == "semiring"
            or row.spec.stage.score_norm == "sharpen"
            else "dense"
        ),
        "op_spectral_preferred_basis": (
            "content" if row.spec.atom.basis_axis == "token" else "channel"
        ),
    }
    if row.spec.stage.aggregate == "semiring":
        axes["op_algebraic_space"] = "parametric_semiring"
    else:
        axes["op_algebraic_space"] = f"parametric_{row.spec.stage.address}"
    for name in PHYSICS_DESCRIPTOR_NAMES:
        axes[f"op_physics_descriptor_{name}"] = round(float(row.descriptors[name]), 5)
    for name, value in row.experiment.descriptor_target.items():
        axes[f"op_physics_target_{name}"] = round(float(value), 5)
    return axes


def _spec_from_scored(row: ScoredPhysicsProgram) -> ProposalSpec:
    axes = _axes_for_scored(row)
    niche = axes["op_physics_niche"]
    name = f"nf_{row.experiment.name}_{niche}"
    desc = ", ".join(
        f"{key}={axes[f'op_physics_descriptor_{key}']}"
        for key in PHYSICS_DESCRIPTOR_NAMES
    )
    rationale = (
        f"Name-free physics experiment {row.experiment.name}: "
        f"{row.experiment.rationale} Measured descriptors: {desc}. "
        f"Target-distance={row.distance:.4f}; niche={niche}."
    )
    return build_spec_from_axes(
        name,
        axes,
        witness_ops=("name_free_physics",),
        anchor_axes={},
        notes=(
            "source=name_free_physics",
            f"experiment={row.experiment.name}",
            f"target={row.experiment.target}",
            f"physics_niche={niche}",
            f"descriptor_distance={row.distance:.5f}",
            row.experiment.rationale,
        ),
        fingerprint_dispatched_axes=True,
        rationale=rationale,
    )


def enumerate_name_free_physics_experiments(
    ledger: Ledger,
    *,
    cycle: int = 0,
    dim: int = 32,
    max_specs: int = 12,
    max_candidates_per_experiment: int = 18,
) -> list[ProposalSpec]:
    """Emit descriptor-backed, name-free physics experiments for fab grading.

    This is deliberately not a random proposer. It runs a bounded, deterministic
    experiment sweep, ranks candidates by measured physics distance to the
    hypothesis target, skips coordinates that already failed the hard S0.5 gate,
    and allocates more budget to experiment families implicated by ledger data.
    """

    if max_specs <= 0:
        return []
    probe_dim = max(8, min(32, int(dim)))
    priorities = _experiment_priorities(_DEFAULT_EXPERIMENTS, ledger)
    experiments = sorted(
        _DEFAULT_EXPERIMENTS,
        key=lambda exp: (priorities.get(exp.name, exp.base_priority), exp.name),
        reverse=True,
    )
    failed = _failed_physics_coordinates(ledger)
    specs: list[ProposalSpec] = []
    seen_coords: set[tuple[str, ...]] = set()
    for experiment in experiments:
        if len(specs) >= max_specs:
            break
        scored = _score_experiment(
            experiment,
            dim=probe_dim,
            cycle=cycle,
            max_candidates=max_candidates_per_experiment,
        )
        for row in scored:
            axes = _axes_for_scored(row)
            coord = _coordinate_key(axes)
            if coord in failed or coord in seen_coords:
                continue
            seen_coords.add(coord)
            specs.append(_spec_from_scored(row))
            if len(specs) >= max_specs:
                break
    return specs
