"""Archive-guided generation: empty behavior niches → exploration targets (M4).

Closes the diversity loop (``diversity_generator_charter_2026-06-03.md`` M4).
The MAP-Elites archive (``quality_diversity.MapElitesArchive``) tells us which
regions of the measured-descriptor behavior space are *empty*; this module maps
those empty niches back to the grammar ops that would produce candidates landing
in them, and emits a ``GrammarConfig`` that boosts exactly those ops — so the
next generation explores the diversity the population currently lacks instead of
piling onto the global maximum.

The one curated input is ``_OP_BEHAVIOR_SIGNATURE`` — each op's characteristic
*behavior corner*, one coarse level (LOW/MID/HIGH) per behavior axis. Grounded in
the descriptor semantics documented on ``quality_diversity._DEFAULT_AXES``
(long_range_reach: local↔routes-back; content_dependence: fixed-routing↔
attention-class; content_match_gating: diffuse↔hard content-gated copy) and the
charter's winnable-niche op lists. Curated, like ``research_priors`` — not fit to
labels. A niche is "winnable" only if some op's signature is within ``radius``
bins of it; niches no op can produce stay empty (never chase impossible behavior
combos). Every target op is filtered against ``PRIMITIVE_REGISTRY`` so we never
emit a non-existent exploration target.

This module stays inside ``research.synthesis`` — it does not import
``component_fab``. The fab-side nudge (boost ``research_priors`` families whose
mapped ops hit ``target_ops``) belongs to the component_fab consumer of this
guidance, not to the grammar library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .quality_diversity import BehaviorAxis, MapElitesArchive

# Coarse behavior levels, axis-name keyed so the table is robust to axis order /
# custom archives. Mapped onto a concrete axis by ``_level_to_bin``.
LOW, MID, HIGH = 0, 1, 2

# Op -> {axis_name: level}. Ops are the fab-invention vocabulary
# (``_motifs_fab.fab_invention_ops``) plus the core fixed-routing / local family
# needed to cover the low-content-dependence niches.
_OP_BEHAVIOR_SIGNATURE: dict[str, dict[str, int]] = {
    # ── attention-class binders: route far, content-dependent, hard gating ──
    "tropical_attention": {
        "long_range_reach": HIGH,
        "content_dependence": HIGH,
        "content_match_gating": HIGH,
    },
    "ultrametric_attention": {
        "long_range_reach": HIGH,
        "content_dependence": HIGH,
        "content_match_gating": HIGH,
    },
    "reciprocal": {
        "long_range_reach": HIGH,
        "content_dependence": HIGH,
        "content_match_gating": HIGH,
    },
    "learnable_semiring_attention": {
        "long_range_reach": HIGH,
        "content_dependence": HIGH,
        "content_match_gating": HIGH,
    },
    "stdp_attention": {
        "long_range_reach": HIGH,
        "content_dependence": HIGH,
        "content_match_gating": HIGH,
    },
    "sparsemax_attention": {
        "long_range_reach": HIGH,
        "content_dependence": HIGH,
        "content_match_gating": HIGH,
    },
    "softmax_attention": {
        "long_range_reach": HIGH,
        "content_dependence": HIGH,
        "content_match_gating": MID,
    },
    # ── explicit slot / addressable KV memory: addressed hard copy ──
    "product_key_memory": {
        "long_range_reach": HIGH,
        "content_dependence": HIGH,
        "content_match_gating": HIGH,
    },
    "role_slot_attention": {
        "long_range_reach": HIGH,
        "content_dependence": HIGH,
        "content_match_gating": HIGH,
    },
    "associative_memory": {
        "long_range_reach": HIGH,
        "content_dependence": HIGH,
        "content_match_gating": HIGH,
    },
    # ── delta-rule / gated recurrence: far reach, gated (mid content), soft gating ──
    "gated_delta": {
        "long_range_reach": HIGH,
        "content_dependence": MID,
        "content_match_gating": MID,
    },
    "dplr_gated_delta": {
        "long_range_reach": HIGH,
        "content_dependence": MID,
        "content_match_gating": MID,
    },
    "gated_linear_attention": {
        "long_range_reach": HIGH,
        "content_dependence": MID,
        "content_match_gating": MID,
    },
    "retention_mix": {
        "long_range_reach": HIGH,
        "content_dependence": MID,
        "content_match_gating": LOW,
    },
    "linear_attention": {
        "long_range_reach": HIGH,
        "content_dependence": MID,
        "content_match_gating": LOW,
    },
    # ── SSM / long fixed-routing: far reach, content-independent, no gating ──
    "selective_scan": {
        "long_range_reach": HIGH,
        "content_dependence": MID,
        "content_match_gating": LOW,
    },
    "state_space": {
        "long_range_reach": HIGH,
        "content_dependence": LOW,
        "content_match_gating": LOW,
    },
    "long_conv_hyena": {
        "long_range_reach": HIGH,
        "content_dependence": LOW,
        "content_match_gating": LOW,
    },
    # ── local / short-range: near reach, no gating ──
    "local_window_attn": {
        "long_range_reach": MID,
        "content_dependence": HIGH,
        "content_match_gating": MID,
    },
    "conv1d_seq": {
        "long_range_reach": LOW,
        "content_dependence": LOW,
        "content_match_gating": LOW,
    },
    "wavelet_packet_mix": {
        "long_range_reach": MID,
        "content_dependence": LOW,
        "content_match_gating": LOW,
    },
}


@dataclass(frozen=True, slots=True)
class ArchiveGuidance:
    """Archive-derived exploration directive for the next generation."""

    target_ops: frozenset[str]
    boost_factor: float
    coverage: float
    reachable_empty: int
    unreachable_empty: int
    per_op_demand: Mapping[str, int]


def _level_to_bin(level: int, n_bins: int) -> int:
    """Map a coarse LOW/MID/HIGH level onto a concrete axis with ``n_bins`` bins."""

    if level <= LOW:
        return 0
    if level >= HIGH:
        return n_bins - 1
    return (n_bins - 1) // 2


def _op_distance(
    niche: Sequence[int],
    axes: Sequence[BehaviorAxis],
    signature: Mapping[str, int],
) -> int:
    """L1 distance from a niche to an op's signature (axes absent from the
    signature do not constrain it)."""

    return sum(
        abs(int(niche[i]) - _level_to_bin(signature[axis.name], axis.n_bins))
        for i, axis in enumerate(axes)
        if axis.name in signature
    )


def _registered_ops() -> set[str]:
    """Signature ops that actually exist in the primitive registry (fail-safe)."""

    from .primitives import PRIMITIVE_REGISTRY

    return {op for op in _OP_BEHAVIOR_SIGNATURE if op in PRIMITIVE_REGISTRY}


def archive_guidance(
    archive: MapElitesArchive,
    *,
    radius: int = 1,
    underfilled_below: float | None = None,
    base_boost: float = 4.0,
    max_boost: float = 12.0,
    max_target_ops: int | None = None,
) -> ArchiveGuidance:
    """Map an archive's empty/under-filled niches to grammar exploration targets.

    Args:
        radius: max L1 niche distance for an op to count as able to fill a niche.
            Niches with no op within ``radius`` are unreachable and left empty.
        underfilled_below: also target niches whose elite fitness is below this
            (under-exploited niches), not just empty ones. ``None`` = empty only.
        base_boost / max_boost: exploration boost floor and ceiling; the boost
            scales up with the archive's coverage gap (emptier → more exploration).
        max_target_ops: cap the op set to the highest-demand ops (``None`` = all).
    """

    axes = archive.axes
    ops = _registered_ops()

    targets: list[tuple[int, ...]] = list(archive.empty_niches())
    if underfilled_below is not None:
        targets.extend(e.niche for e in archive.elites if e.fitness < underfilled_below)

    per_op_demand: dict[str, int] = {op: 0 for op in ops}
    reachable = 0
    for niche in targets:
        matched = [
            op
            for op in ops
            if _op_distance(niche, axes, _OP_BEHAVIOR_SIGNATURE[op]) <= radius
        ]
        if not matched:
            continue
        reachable += 1
        for op in matched:
            per_op_demand[op] += 1

    demanded = {op: c for op, c in per_op_demand.items() if c > 0}
    ranked = sorted(demanded, key=lambda op: (-demanded[op], op))
    if max_target_ops is not None:
        ranked = ranked[:max_target_ops]

    coverage = archive.coverage()
    return ArchiveGuidance(
        target_ops=frozenset(ranked),
        boost_factor=min(max_boost, base_boost * (2.0 - coverage)),
        coverage=coverage,
        reachable_empty=reachable,
        unreachable_empty=len(targets) - reachable,
        per_op_demand=demanded,
    )


def exploration_config_from_archive(
    archive: MapElitesArchive,
    *,
    model_dim: int = 256,
    **guidance_kwargs: object,
):
    """Archive → a ready ``GrammarConfig`` that boosts the empty niches.

    Returns ``(config, guidance)``. ``config`` is ``None`` when no reachable empty
    niche remains (fully covered, or only unreachable holes) — the caller should
    keep its base grammar rather than a no-op exploration config.
    """

    from .grammar import GrammarConfig

    guidance = archive_guidance(archive, **guidance_kwargs)  # type: ignore[arg-type]
    if not guidance.target_ops:
        return None, guidance
    config = GrammarConfig.exploration(
        target_ops=guidance.target_ops,
        model_dim=model_dim,
        boost_factor=guidance.boost_factor,
    )
    return config, guidance


__all__ = (
    "ArchiveGuidance",
    "archive_guidance",
    "exploration_config_from_archive",
)
