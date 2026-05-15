"""Adaptive anchor selection — feed promoted fab components back as anchors.

The static improver always uses the top-5 underperforming-novel ops from
the scoper. That's a fixed search space. To compound gains across the
autonomous loop, promoted fab components should rejoin the anchor pool
so cross-anchor variants can hybridize fab-discovered math with corpus
math.

Also bounds the variant pool per cycle so a growing anchor pool doesn't
explode combinatorially: cross-anchor pairs get sampled with novelty
bias (pairs not yet present in the ledger preferred).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Sequence

from ..proposer.spec_generator import ProposalSpec
from ..state.ledger import Ledger, PROMOTION_PROMOTED
from .axis_variants import (
    DEFAULT_AXIS_VARIANT_TEMPLATES,
    DEFAULT_META_DB,
    AnchorAxes,
    AxisVariant,
    anchor_axes_for_op,
    spec_for_variant,
)
from .cross_anchor import _hybrid_spec, is_hosting_anchor


@dataclass(frozen=True, slots=True)
class AdaptiveAnchorPool:
    corpus_anchors: tuple[AnchorAxes, ...]
    fab_anchors: tuple[AnchorAxes, ...]

    @property
    def all_anchors(self) -> tuple[AnchorAxes, ...]:
        return self.corpus_anchors + self.fab_anchors


def _promoted_fab_components_as_anchors(ledger: Ledger) -> list[AnchorAxes]:
    """Each promoted ledger entry can act as an anchor — its axes come from the spec name parse-out.

    We don't have the original math_axes on the LedgerEntry, so we synthesize
    a minimal anchor by inferring the math axes from the synthesis_kind +
    category labels. This is intentionally lossy; the cross-anchor function
    needs only ``axes`` keys to merge.
    """
    out: list[AnchorAxes] = []
    for entry in ledger.all_entries():
        if entry.promotion_status != PROMOTION_PROMOTED:
            continue
        # Default axes — placeholders. Cross-anchor merges donor axes into host,
        # so what matters is that the host (corpus anchor) keeps its algebra.
        axes = {
            "op_algebraic_space": "fab_promoted",
            "op_dynamical_has_state": 1,
            "op_dynamical_memory_length_class": "O(L)",
            "op_activation_sparsity_pattern": "learned_structured",
            "op_geometric_receptive_field": "global",
            "op_spectral_preferred_basis": "content",
        }
        pass_rate = sum(s for s in entry.composite_history[-2:]) / max(
            1, len(entry.composite_history[-2:])
        )
        out.append(
            AnchorAxes(
                op_name=entry.name,
                axes=axes,
                eval_count=len(entry.composite_history),
                pass_rate=pass_rate,
            )
        )
    return out


def build_anchor_pool(
    corpus_anchor_names: Sequence[str],
    ledger: Ledger,
    *,
    use_promoted_as_anchors: bool = True,
    db_path: Path | str = DEFAULT_META_DB,
) -> AdaptiveAnchorPool:
    corpus: list[AnchorAxes] = []
    for name in corpus_anchor_names:
        anchor = anchor_axes_for_op(name, db_path=db_path)
        if anchor is not None:
            corpus.append(anchor)
    fab = _promoted_fab_components_as_anchors(ledger) if use_promoted_as_anchors else []
    return AdaptiveAnchorPool(
        corpus_anchors=tuple(corpus),
        fab_anchors=tuple(fab),
    )


def _failed_axis_deltas(ledger: Ledger, min_failures: int = 3) -> frozenset[str]:
    """Find axis-delta names that have been rejected ``>= min_failures`` times."""
    rejections: dict[str, int] = {}
    for entry in ledger.all_entries():
        if entry.promotion_status != "rejected":
            continue
        # Extract the delta name from "improve_<anchor>_<delta>" — best-effort.
        parts = entry.name.split("_")
        if len(parts) < 3 or parts[0] != "improve":
            continue
        delta = "_".join(parts[2:])
        rejections[delta] = rejections.get(delta, 0) + 1
    return frozenset(k for k, v in rejections.items() if v >= min_failures)


def adaptive_axis_variants(
    anchor_pool: AdaptiveAnchorPool,
    ledger: Ledger,
    *,
    variants: Sequence[AxisVariant] = DEFAULT_AXIS_VARIANT_TEMPLATES,
) -> list[ProposalSpec]:
    """Axis variants of corpus anchors, deprioritizing repeatedly-failed deltas."""
    failed = _failed_axis_deltas(ledger)
    keep = [v for v in variants if v.delta_name not in failed]
    out: list[ProposalSpec] = []
    for anchor in anchor_pool.corpus_anchors:
        for variant in keep:
            out.append(spec_for_variant(anchor, variant))
    return out


def adaptive_cross_anchor_variants(
    anchor_pool: AdaptiveAnchorPool,
    ledger: Ledger,
    *,
    max_pairs: int = 30,
    seed: int = 0,
) -> list[ProposalSpec]:
    """Sample up to ``max_pairs`` cross-anchor hybrids with novelty bias."""
    anchors = anchor_pool.all_anchors
    if len(anchors) < 2:
        return []
    seen_pair_keys: set[frozenset[str]] = set()
    for entry in ledger.all_entries():
        if entry.name.startswith("cross_"):
            parts = entry.name.removeprefix("cross_").split("_x_")
            if len(parts) == 2:
                seen_pair_keys.add(frozenset(parts))

    all_pairs = list(combinations(anchors, 2))
    rng = random.Random(seed)
    rng.shuffle(all_pairs)

    novel: list[tuple[AnchorAxes, AnchorAxes]] = []
    seen: list[tuple[AnchorAxes, AnchorAxes]] = []
    for a, b in all_pairs:
        key = frozenset({a.op_name, b.op_name})
        if key in seen_pair_keys:
            seen.append((a, b))
        else:
            novel.append((a, b))

    selected = novel + seen
    selected = selected[: max(0, max_pairs)]
    out: list[ProposalSpec] = []
    for a, b in selected:
        if is_hosting_anchor(a):
            out.append(_hybrid_spec(a, b))
        if is_hosting_anchor(b):
            out.append(_hybrid_spec(b, a))
    return out
