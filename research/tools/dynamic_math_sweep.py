#!/usr/bin/env python
"""CPU-only pre-assembly math sweep for dynamic component candidates.

The dynamic candidate builder already validates an assembled graph. This module
adds the missing *improver* layer: take a compiled ``[B, L, D] -> [B, L, D]``
operator, wrap it in bounded mathematical variants, measure what each variant
does at random init, and select only a non-collapsed variant that improves the
target descriptor profile.

The wrappers here are deliberately lightweight CPU prototypes. They reuse the
existing descriptor systems rather than adding another proxy stack:

- ``PhysicsDescriptorProbe`` for symmetry/stability behavior.
- ``MeasuredDescriptorExtractor`` for position-Jacobian reach/content signals.

Later builder integration can translate a selected ``VariantDescriptor.axes``
map into a concrete graph mutation or registered primitive. Until then, this
module gives the proposer an auditable variant catalog, explicit failure
reasons, and compact learning fields.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from research.eval.induction_probe import _RESTRICTED_VOCAB
from research.synthesis.physics_descriptors import PhysicsDescriptorProbe
from research.tools.measured_descriptors import MeasuredDescriptorExtractor

SCHEMA_VERSION = "dynamic_math_sweep_v1"

Decision = Literal["parent", "selected", "rejected"]
FailureReason = str | None
MeasureFn = Callable[["VariantDescriptor", nn.Module], "DescriptorBundle"]

_EPS = 1e-9


@dataclass(frozen=True, slots=True)
class VariantDescriptor:
    """One candidate math transform in the pre-assembly sweep catalog."""

    variant_id: str
    family: str
    transform: str
    axes: Mapping[str, Any] = field(default_factory=dict)
    blend: float = 0.35
    rationale: str = ""
    softmax_twin_like: bool = False


@dataclass(frozen=True, slots=True)
class DescriptorBundle:
    """Measured descriptors for a buildable variant."""

    physics: Mapping[str, float]
    measured: Mapping[str, float]

    def combined(self) -> dict[str, float]:
        out = {str(k): float(v) for k, v in self.physics.items()}
        out.update({str(k): float(v) for k, v in self.measured.items()})
        return out


@dataclass(slots=True)
class SweepRecord:
    """One parent/variant row suitable for JSONL audit and ledger summaries."""

    run_id: str
    candidate_id: str
    candidate_name: str
    variant: VariantDescriptor
    parent_variant_id: str
    build_passed: bool = False
    validate_passed: bool = False
    compile_passed: bool = False
    physics_descriptors: dict[str, float] = field(default_factory=dict)
    measured_descriptors: dict[str, float] = field(default_factory=dict)
    descriptor_delta_vs_parent: dict[str, float] = field(default_factory=dict)
    math_variant_score: float = 0.0
    decision: Decision = "rejected"
    failure_reason: FailureReason = None
    error: str | None = None

    @property
    def variant_id(self) -> str:
        return self.variant.variant_id

    def combined_descriptors(self) -> dict[str, float]:
        out = dict(self.physics_descriptors)
        out.update(self.measured_descriptors)
        return out

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "candidate_id": self.candidate_id,
            "candidate_name": self.candidate_name,
            "variant_id": self.variant.variant_id,
            "parent_variant_id": self.parent_variant_id,
            "variant_family": self.variant.family,
            "variant_transform": self.variant.transform,
            "variant_axes": dict(self.variant.axes),
            "build_passed": self.build_passed,
            "validate_passed": self.validate_passed,
            "compile_passed": self.compile_passed,
            "physics_descriptors": self.physics_descriptors,
            "measured_descriptors": self.measured_descriptors,
            "descriptor_delta_vs_parent": self.descriptor_delta_vs_parent,
            "math_variant_score": round(float(self.math_variant_score), 6),
            "decision": self.decision,
            "failure_reason": self.failure_reason,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class TargetProfile:
    """Target-specific scoring and hard-fail thresholds."""

    name: str
    reward_weights: Mapping[str, float]
    min_improvement: float = 0.01
    max_causality_violation: float = 0.03
    max_self_dominance: float = 0.985
    min_effective_rank: float = 1.05
    max_spectral_radius: float = 2.5
    max_energy_gain: float = 4.0
    spectral_radius_target: float = 1.0
    spectral_penalty: float = 0.08
    causality_penalty: float = 2.0
    self_dominance_penalty: float = 0.5


def target_profile(name: str = "binding") -> TargetProfile:
    """Return the sweep scoring profile for a candidate target class."""

    key = name.strip().lower()
    if key in {"binding", "content", "retrieval", "slot"}:
        return TargetProfile(
            name="binding",
            reward_weights={
                "long_range_reach": 0.7,
                "content_dependence": 1.0,
                "content_match_gating": 1.2,
                "effective_rank": 0.06,
            },
        )
    if key in {"long_memory", "ar", "induction", "long_gap"}:
        return TargetProfile(
            name="long_memory",
            reward_weights={
                "long_range_reach": 1.4,
                "content_dependence": 0.4,
                "effective_rank": 0.04,
            },
            min_improvement=0.008,
        )
    if key in {"compression", "bottleneck"}:
        return TargetProfile(
            name="compression",
            reward_weights={
                "effective_rank": 0.10,
                "content_dependence": 0.5,
                "long_range_reach": 0.4,
            },
            min_improvement=0.005,
        )
    return TargetProfile(
        name=key or "generic",
        reward_weights={
            "long_range_reach": 0.8,
            "content_dependence": 0.8,
            "content_match_gating": 0.6,
            "effective_rank": 0.04,
        },
    )


PARENT_VARIANT = VariantDescriptor(
    variant_id="parent",
    family="parent",
    transform="identity",
    axes={},
    blend=0.0,
    rationale="Unmodified compiled candidate.",
)

DEFAULT_VARIANT_CATALOG: tuple[VariantDescriptor, ...] = (
    PARENT_VARIANT,
    VariantDescriptor(
        variant_id="algebraic_reciprocal_cauchy_read",
        family="algebraic",
        transform="reciprocal_cauchy_read",
        axes={
            "op_math_variant_family": "algebraic",
            "op_physics_address_family": "reciprocal_cauchy",
            "op_physics_aggregate_family": "inverse_distance_mean",
        },
        rationale="Inverse-distance causal read; non-softmax score geometry.",
    ),
    VariantDescriptor(
        variant_id="algebraic_tropical_prefix_max",
        family="algebraic",
        transform="tropical_prefix_max",
        axes={
            "op_math_variant_family": "algebraic",
            "op_algebraic_space": "tropical",
            "op_physics_aggregate_family": "max_plus_prefix",
        },
        blend=0.25,
        rationale="Max-plus causal prefix state; probes tropical read structure.",
    ),
    VariantDescriptor(
        variant_id="spectral_dct_token_rotation",
        family="spectral_trig",
        transform="dct_token_rotation",
        axes={
            "op_math_family": "spectral_graph",
            "op_spectral_graph_operator": "dct",
        },
        blend=0.30,
        rationale="Token-axis DCT rotation; explicit spectral basis variant.",
    ),
    VariantDescriptor(
        variant_id="calculus_causal_gradient",
        family="calculus_dynamical",
        transform="causal_gradient",
        axes={
            "op_math_family": "calculus",
            "op_calculus_operator": "causal_gradient",
        },
        blend=0.25,
        rationale="Causal finite-difference signal mixed into the parent.",
    ),
    VariantDescriptor(
        variant_id="dynamical_causal_integral",
        family="calculus_dynamical",
        transform="causal_running_integral",
        axes={
            "op_math_family": "calculus",
            "op_calculus_operator": "causal_running_integral",
            "op_dynamical_has_state": 1,
        },
        blend=0.35,
        rationale="Causal running integral; cheap long-memory probe.",
    ),
    VariantDescriptor(
        variant_id="kernel_positive_cosine",
        family="kernel",
        transform="positive_cosine_kernel_read",
        axes={
            "op_math_family": "kernel_methods",
            "op_kernel_feature_map": "positive_cosine",
        },
        rationale="Positive cosine kernel read without row-softmax.",
    ),
    VariantDescriptor(
        variant_id="graph_causal_path_diffusion",
        family="graph_diffusion",
        transform="causal_path_diffusion",
        axes={
            "op_math_family": "graph_diffusion",
            "op_graph_topology": "causal_path_laplacian",
        },
        blend=0.30,
        rationale="Causal path diffusion against the previous token state.",
    ),
)


class MathVariantWrapper(nn.Module):
    """Apply one CPU math prototype around an existing compiled operator."""

    def __init__(self, parent: nn.Module, descriptor: VariantDescriptor) -> None:
        super().__init__()
        if not 0.0 <= descriptor.blend <= 1.0:
            raise ValueError(
                f"variant blend must be in [0, 1]; got {descriptor.blend}"
            )
        self.parent = parent
        self.descriptor = descriptor

    def forward(self, x: Tensor) -> Tensor:
        y = self.parent(x)
        if self.descriptor.transform == "identity":
            return y
        transformed = _apply_transform(y, self.descriptor.transform)
        return y + float(self.descriptor.blend) * (transformed - y)


class _OperatorProbeModel(nn.Module):
    """Probe adapter for ``MeasuredDescriptorExtractor``."""

    def __init__(self, operator: nn.Module, vocab: int, dim: int, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.tok = nn.Embedding(vocab, dim)
        self.operator = operator

    def embed(self, ids: Tensor) -> Tensor:
        return self.tok(ids)

    def _fingerprint_forward_from_embed(self, emb: Tensor) -> Tensor:
        return self.operator(emb)


def default_variant_catalog() -> tuple[VariantDescriptor, ...]:
    """Return the built-in bounded math variant catalog."""

    return DEFAULT_VARIANT_CATALOG


def build_variant_operator(parent: nn.Module, variant: VariantDescriptor) -> nn.Module:
    """Build a CPU prototype for ``variant`` around ``parent``.

    The parent is deep-copied so variants do not share stateful buffers such as
    cached transport plans or routing telemetry.
    """

    parent_copy = copy.deepcopy(parent).to("cpu").eval()
    if variant.transform == "identity":
        return parent_copy
    _ensure_supported_transform(variant.transform)
    return MathVariantWrapper(parent_copy, variant).eval()


def measure_operator_descriptors(
    operator: nn.Module,
    *,
    dim: int,
    device: str = "cpu",
    vocab: int = 128,
    physics_batch: int = 2,
    physics_seq_len: int = 16,
    physics_n_seeds: int = 1,
    measured_batch: int = 8,
    measured_gap: int = 5,
    measured_n_seeds: int = 1,
) -> DescriptorBundle:
    """Measure existing physics + position-Jacobian descriptors on CPU."""

    op = operator.to(device).eval()
    vocab = max(int(vocab), int(_RESTRICTED_VOCAB))
    physics = PhysicsDescriptorProbe(
        batch=physics_batch,
        seq_len=physics_seq_len,
        dim=dim,
        vocab=vocab,
        n_seeds=physics_n_seeds,
        device=device,
    ).describe_operator(op)
    measured = MeasuredDescriptorExtractor(
        device=device,
        n_seeds=measured_n_seeds,
        gap=measured_gap,
        batch=measured_batch,
    ).descriptors_from_factory(
        lambda seed: _OperatorProbeModel(
            copy.deepcopy(op).to(device).eval(), vocab, dim, seed
        ).to(device)
    )
    if measured is None:
        raise RuntimeError("MeasuredDescriptorExtractor returned no valid seeds")
    return DescriptorBundle(physics=physics, measured=measured)


def run_dynamic_math_sweep(
    parent: nn.Module,
    *,
    candidate_id: str,
    candidate_name: str,
    dim: int,
    run_id: str,
    target: str | TargetProfile = "binding",
    catalog: Sequence[VariantDescriptor] = DEFAULT_VARIANT_CATALOG,
    measure_fn: MeasureFn | None = None,
    device: str = "cpu",
) -> list[SweepRecord]:
    """Measure parent + variants and mark the selected non-collapsed winner."""

    if not catalog:
        raise ValueError("catalog must include at least the parent variant")
    if catalog[0].variant_id != PARENT_VARIANT.variant_id:
        catalog = (PARENT_VARIANT, *tuple(catalog))

    profile = target if isinstance(target, TargetProfile) else target_profile(target)
    records: list[SweepRecord] = []
    parent_record: SweepRecord | None = None
    for variant in catalog:
        record = SweepRecord(
            run_id=run_id,
            candidate_id=candidate_id,
            candidate_name=candidate_name,
            variant=variant,
            parent_variant_id=PARENT_VARIANT.variant_id,
        )
        try:
            operator = build_variant_operator(parent, variant)
            record.build_passed = True
            record.validate_passed = True
            record.compile_passed = True
            bundle = (
                measure_fn(variant, operator)
                if measure_fn is not None
                else measure_operator_descriptors(operator, dim=dim, device=device)
            )
            record.physics_descriptors = _rounded(bundle.physics)
            record.measured_descriptors = _rounded(bundle.measured)
            if not _all_finite(record.combined_descriptors()):
                record.failure_reason = "nonfinite_descriptor"
        except Exception as exc:
            record.failure_reason = _failure_reason_for_stage(record)
            record.error = f"{type(exc).__name__}: {exc}"

        records.append(record)
        if variant.variant_id == PARENT_VARIANT.variant_id:
            parent_record = record
            if record.failure_reason is None:
                record.decision = "parent"

    if parent_record is None or parent_record.failure_reason is not None:
        return records

    finalize_sweep_decisions(records, profile=profile, parent=parent_record)
    return records


def finalize_sweep_decisions(
    records: Sequence[SweepRecord],
    *,
    profile: TargetProfile,
    parent: SweepRecord | None = None,
) -> SweepRecord:
    """Compute deltas/scores/failure reasons and return the selected record.

    If no variant clears ``profile.min_improvement``, the parent remains the
    selected baseline row.
    """

    if not records:
        raise ValueError("records must be non-empty")
    parent_record = parent or records[0]
    parent_record.decision = "parent"
    parent_desc = parent_record.combined_descriptors()

    best: SweepRecord = parent_record
    best_score = 0.0
    for record in records:
        if record is parent_record:
            record.descriptor_delta_vs_parent = {
                key: 0.0 for key in sorted(parent_desc)
            }
            record.math_variant_score = 0.0
            continue
        if record.failure_reason is None:
            record.descriptor_delta_vs_parent = _descriptor_delta(
                record.combined_descriptors(), parent_desc
            )
            record.math_variant_score = score_variant(record, profile)
            record.failure_reason = hard_failure_reason(record, profile)
        if record.failure_reason is None:
            if record.math_variant_score < profile.min_improvement:
                record.failure_reason = "no_target_improvement"
            elif record.math_variant_score > best_score:
                best = record
                best_score = record.math_variant_score

    for record in records:
        if record is parent_record:
            record.decision = "parent"
        elif record is best:
            record.decision = "selected"
        else:
            record.decision = "rejected"
    return best


def score_variant(record: SweepRecord, profile: TargetProfile) -> float:
    """Target-aware scalar score from descriptor deltas."""

    delta = record.descriptor_delta_vs_parent
    score = sum(
        float(weight) * float(delta.get(name, 0.0))
        for name, weight in profile.reward_weights.items()
    )
    desc = record.combined_descriptors()
    causality = max(0.0, float(desc.get("causality_violation", 0.0)))
    self_dom = max(0.0, float(desc.get("self_dominance", 0.0)))
    spectral = float(desc.get("spectral_radius", profile.spectral_radius_target))
    score -= profile.causality_penalty * causality
    score -= profile.self_dominance_penalty * self_dom
    score -= profile.spectral_penalty * abs(spectral - profile.spectral_radius_target)
    return float(score)


def hard_failure_reason(
    record: SweepRecord, profile: TargetProfile
) -> FailureReason:
    """Return a hard rejection reason for a measured variant, if any."""

    if record.variant.softmax_twin_like:
        return "softmax_twin_regression"
    desc = record.combined_descriptors()
    if not _all_finite(desc):
        return "nonfinite_descriptor"
    causality = float(desc.get("causality_violation", 0.0))
    if causality > profile.max_causality_violation:
        return "causality_violation"
    spectral = float(desc.get("spectral_radius", 1.0))
    if spectral <= 0.0 or spectral > profile.max_spectral_radius:
        return "spectral_instability"
    energy = float(desc.get("energy_gain", 1.0))
    if energy > profile.max_energy_gain:
        return "energy_blowup"
    effective_rank = float(desc.get("effective_rank", profile.min_effective_rank))
    if effective_rank < profile.min_effective_rank:
        return "rank_collapse"
    self_dom = float(desc.get("self_dominance", 0.0))
    if self_dom > profile.max_self_dominance:
        return "self_dominance_collapse"
    return None


def selected_summary(records: Sequence[SweepRecord]) -> dict[str, Any]:
    """Compact fields for candidate artifacts / ledger metadata."""

    selected = next((row for row in records if row.decision == "selected"), records[0])
    return {
        "math_sweep_version": SCHEMA_VERSION,
        "math_variant_selected": selected.decision == "selected",
        "math_variant_family": selected.variant.family,
        "math_variant_transform": selected.variant.transform,
        "math_variant_axes": dict(selected.variant.axes),
        "math_variant_score": round(float(selected.math_variant_score), 6),
        "math_variant_failure_reason": selected.failure_reason,
        "math_variant_delta_long_range_reach": selected.descriptor_delta_vs_parent.get(
            "long_range_reach", 0.0
        ),
        "math_variant_delta_content_dependence": (
            selected.descriptor_delta_vs_parent.get("content_dependence", 0.0)
        ),
        "math_variant_delta_content_match_gating": (
            selected.descriptor_delta_vs_parent.get("content_match_gating", 0.0)
        ),
        "math_variant_delta_effective_rank": selected.descriptor_delta_vs_parent.get(
            "effective_rank", 0.0
        ),
        "math_variant_delta_causality_violation": (
            selected.descriptor_delta_vs_parent.get("causality_violation", 0.0)
        ),
        "math_sweep_variant_count": len(records),
    }


def write_sweep_jsonl(records: Iterable[SweepRecord], path: Path | str) -> None:
    """Write detailed sweep audit rows."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_json(), sort_keys=True) + "\n")


def _failure_reason_for_stage(record: SweepRecord) -> str:
    if not record.build_passed:
        return "variant_build_failed"
    if not record.validate_passed:
        return "validate_failed"
    if not record.compile_passed:
        return "compile_failed"
    return "measured_descriptor_error"


def _rounded(values: Mapping[str, float]) -> dict[str, float]:
    return {str(key): round(float(value), 6) for key, value in values.items()}


def _descriptor_delta(
    descriptors: Mapping[str, float], parent: Mapping[str, float]
) -> dict[str, float]:
    keys = set(descriptors) | set(parent)
    return {
        key: round(float(descriptors.get(key, 0.0)) - float(parent.get(key, 0.0)), 6)
        for key in sorted(keys)
    }


def _all_finite(values: Mapping[str, float]) -> bool:
    return all(math.isfinite(float(value)) for value in values.values())


def _ensure_supported_transform(transform: str) -> None:
    supported = {variant.transform for variant in DEFAULT_VARIANT_CATALOG}
    if transform not in supported:
        raise ValueError(f"unsupported math variant transform: {transform!r}")


def _apply_transform(y: Tensor, transform: str) -> Tensor:
    if transform == "reciprocal_cauchy_read":
        return _reciprocal_cauchy_read(y)
    if transform == "tropical_prefix_max":
        return torch.cummax(y, dim=1).values
    if transform == "dct_token_rotation":
        return _dct_token_rotation(y)
    if transform == "causal_gradient":
        prev = F.pad(y[:, :-1, :], (0, 0, 1, 0))
        return y + (y - prev)
    if transform == "causal_running_integral":
        denom = torch.arange(1, y.shape[1] + 1, device=y.device, dtype=y.dtype)
        return y.cumsum(dim=1) / denom.view(1, -1, 1)
    if transform == "positive_cosine_kernel_read":
        return _positive_cosine_kernel_read(y)
    if transform == "causal_path_diffusion":
        prev = torch.cat([y[:, :1, :], y[:, :-1, :]], dim=1)
        return y + 0.5 * (prev - y)
    raise ValueError(f"unsupported math variant transform: {transform!r}")


def _causal_mask(seq_len: int, device: torch.device) -> Tensor:
    return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), 1)


def _reciprocal_cauchy_read(y: Tensor) -> Tensor:
    dist = torch.cdist(y.float(), y.float(), p=2).pow(2)
    weights = 1.0 / (1.0 + dist)
    weights = weights.masked_fill(_causal_mask(y.shape[1], y.device), 0.0)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + _EPS)
    return torch.bmm(weights.to(y.dtype), y)


def _positive_cosine_kernel_read(y: Tensor) -> Tensor:
    normalized = F.normalize(y.float(), dim=-1)
    weights = 0.5 * (torch.bmm(normalized, normalized.transpose(1, 2)) + 1.0)
    weights = weights.clamp_min(0.0)
    weights = weights.masked_fill(_causal_mask(y.shape[1], y.device), 0.0)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + _EPS)
    return torch.bmm(weights.to(y.dtype), y)


def _dct_token_rotation(y: Tensor) -> Tensor:
    basis = _dct_matrix(y.shape[1], y.device, y.dtype)
    return torch.einsum("st,btd->bsd", basis, y)


def _dct_matrix(n: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    k = torch.arange(n, device=device, dtype=dtype).reshape(n, 1)
    j = torch.arange(n, device=device, dtype=dtype).reshape(1, n)
    basis = torch.cos(math.pi / n * (j + 0.5) * k)
    basis[0, :] *= 1.0 / math.sqrt(2.0)
    return (basis * math.sqrt(2.0 / n)).T.contiguous()
