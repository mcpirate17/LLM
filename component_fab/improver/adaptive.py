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

import json
import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Sequence

from ..proposer.spec_generator import ProposalSpec
from ..state.ledger import (
    Ledger,
    PROMOTION_PROMOTED,
    iter_jsonl_records,
    iter_rotated_jsonl_paths,
)
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


_PROPOSALS_AXES_CACHE: dict[str, dict] | None = None


def _load_proposal_axes_index() -> dict[str, dict]:
    """Build {proposal_id: math_axes} index across all proposals.jsonl* files.

    Promoted ledger entries lose their math_axes (LedgerEntry doesn't store
    them). We rebuild from the catalog jsonl + rotated copies so fab
    anchors keep their actual axes (e.g. ``op_block_template=gated_parallel``)
    instead of generic placeholders. Cached after first load.
    """

    global _PROPOSALS_AXES_CACHE
    if _PROPOSALS_AXES_CACHE is not None:
        return _PROPOSALS_AXES_CACHE
    catalog = Path(__file__).resolve().parents[2] / "component_fab" / "catalog"
    out: dict[str, dict] = {}
    if not catalog.exists():
        _PROPOSALS_AXES_CACHE = out
        return out
    for p in iter_rotated_jsonl_paths(catalog / "proposals.jsonl"):
        for row in iter_jsonl_records(p):
            pid = row.get("proposal_id")
            axes = row.get("math_axes")
            if pid and isinstance(axes, dict):
                out[str(pid)] = dict(axes)
    _PROPOSALS_AXES_CACHE = out
    return out


_SAVED_WINNERS_PATH = (
    Path(__file__).resolve().parents[2]
    / "component_fab"
    / "catalog"
    / "saved_winners.json"
)


def _load_saved_winners() -> list[AnchorAxes]:
    """User-pinned winning specs that survive ledger rotation and reset.

    Read from ``component_fab/catalog/saved_winners.json``. Each entry
    carries the full ``math_axes`` of a notable architecture so hybrid
    compositions can always use it as an anchor, even if the ledger
    forgot the original promotion. Set per-spec via the saved_winners
    JSON file; never auto-populated.

    A missing file is fine (no winners pinned); a corrupt file RAISES —
    silently dropping user-pinned winners defeats the file's purpose.
    """

    if not _SAVED_WINNERS_PATH.exists():
        return []
    data = json.loads(_SAVED_WINNERS_PATH.read_text(encoding="utf-8"))
    out: list[AnchorAxes] = []
    for row in data.get("winners", []):
        axes = row.get("math_axes")
        if not isinstance(axes, dict):
            continue
        out.append(
            AnchorAxes(
                op_name=str(
                    row.get("name") or row.get("proposal_id") or "saved_winner"
                ),
                axes=dict(axes),
                eval_count=int(row.get("best_blimp_train_steps") or 1),
                pass_rate=float(row.get("best_blimp_observed") or 0.5),
            )
        )
    return out


def _promoted_fab_components_as_anchors(ledger: Ledger) -> list[AnchorAxes]:
    """Each promoted ledger entry becomes an anchor with its ACTUAL math_axes.

    Lookup goes proposal_id → math_axes via the cached proposals.jsonl index
    (including rotated files). Falls back to a generic placeholder if the
    spec's axes aren't recoverable (rare; happens for legacy entries
    from before the catalog was being persisted). Also includes any
    saved_winners.json entries so user-pinned architectures survive
    rotation/reset.
    """
    axes_index = _load_proposal_axes_index()
    fallback = {
        "op_algebraic_space": "fab_promoted",
        "op_dynamical_has_state": 1,
        "op_dynamical_memory_length_class": "O(L)",
        "op_activation_sparsity_pattern": "learned_structured",
        "op_geometric_receptive_field": "global",
        "op_spectral_preferred_basis": "content",
    }
    out: list[AnchorAxes] = []
    for entry in ledger.all_entries():
        if entry.promotion_status != PROMOTION_PROMOTED:
            continue
        axes = axes_index.get(entry.proposal_id) or fallback
        pass_rate = entry.mean_composite(2)
        out.append(
            AnchorAxes(
                op_name=entry.name,
                axes=dict(axes),
                eval_count=len(entry.composite_history),
                pass_rate=pass_rate,
            )
        )
    # Force-include user-pinned saved winners (dedup by op_name).
    seen_names = {a.op_name for a in out}
    for saved in _load_saved_winners():
        if saved.op_name not in seen_names:
            out.append(saved)
            seen_names.add(saved.op_name)
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
    max_fab_anchors: int = 3,
) -> list[ProposalSpec]:
    """Axis variants of corpus anchors PLUS top-K promoted fab anchors.

    Day-3 finding: corpus_anchors-only enumeration kept the search locked
    on the original 5 underperforming anchors. Day-3 winners like
    block_gated_parallel never had axis variants applied to them. This
    function now expands to top-``max_fab_anchors`` promoted fab
    components by pass_rate, allowing e.g. ``improve_block_gated_parallel
    _route_top_k_moe`` compositions. Capped to keep combinatorics bounded.
    """
    failed = _failed_axis_deltas(ledger)
    keep = [v for v in variants if v.delta_name not in failed]
    out: list[ProposalSpec] = []
    for anchor in anchor_pool.corpus_anchors:
        for variant in keep:
            out.append(spec_for_variant(anchor, variant))
    top_fab = _diverse_fab_anchors(anchor_pool.fab_anchors, max_fab_anchors)
    for anchor in top_fab:
        for variant in keep:
            out.append(spec_for_variant(anchor, variant))
    return out


def _fab_anchor_category(anchor: AnchorAxes) -> str:
    """Bucket a fab anchor by architectural pattern from its op_name.

    Order matters: block_ and route_ override the more generic substrings.
    Used by ``_diverse_fab_anchors`` to ensure architectural breadth in
    the seed pool rather than concentrating on whichever pattern happens
    to dominate composite_score.
    """
    name = anchor.op_name
    if "block_" in name:
        return "block"
    if "route_" in name:
        return "routing"
    if any(k in name for k in ("knob_fisher", "knob_chebyshev", "knob_tucker")):
        return "phase2_knob"
    if "space_quaternion" in name or "space_hyperbolic" in name:
        return "novel_algebra"
    if name.startswith("compose_"):
        return "compose_classic"
    if name.startswith(("cross_", "hybrid_")):
        return "cross_classic"
    return "classic"


def _diverse_fab_anchors(
    anchors: tuple[AnchorAxes, ...], max_total: int
) -> list[AnchorAxes]:
    """Pick top-by-pass_rate from each architectural bucket, then fill any
    remaining slots greedily by pass_rate. Ensures e.g. block_gated_parallel
    and route_top_k_moe both make it into the variant-expansion pool
    even when composite-classic anchors numerically dominate.
    """
    buckets: dict[str, list[AnchorAxes]] = {}
    for a in anchors:
        buckets.setdefault(_fab_anchor_category(a), []).append(a)
    for bucket_list in buckets.values():
        bucket_list.sort(key=lambda a: a.pass_rate, reverse=True)
    picked: list[AnchorAxes] = []
    seen: set[str] = set()
    for bucket_list in buckets.values():
        if bucket_list:
            cand = bucket_list[0]
            if cand.op_name not in seen:
                picked.append(cand)
                seen.add(cand.op_name)
    # Fill the rest greedily by pass_rate from anywhere.
    remaining = sorted(
        (a for a in anchors if a.op_name not in seen),
        key=lambda a: a.pass_rate,
        reverse=True,
    )
    for a in remaining:
        if len(picked) >= max_total:
            break
        picked.append(a)
        seen.add(a.op_name)
    return picked[:max_total]


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
    anchor_names = frozenset(a.op_name for a in anchors)
    seen_pair_keys: set[frozenset[str]] = set()
    for entry in ledger.all_entries():
        key = _pair_key_from_hybrid_name(entry.name, anchor_names)
        if key is not None:
            seen_pair_keys.add(key)

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


def _split_known_anchor_pair(
    body: str, delimiter: str, anchor_names: frozenset[str]
) -> frozenset[str] | None:
    for left in anchor_names:
        prefix = left + delimiter
        if body.startswith(prefix):
            right = body[len(prefix) :]
            if right in anchor_names:
                return frozenset((left, right))
    if body.count(delimiter) == 1:
        left, right = body.split(delimiter)
        if left and right:
            return frozenset((left, right))
    return None


def _pair_key_from_hybrid_name(
    name: str, anchor_names: frozenset[str]
) -> frozenset[str] | None:
    """Extract an unordered anchor-pair key from current/legacy hybrid names.

    Anchor names are not guaranteed to avoid delimiters such as ``_plus_`` or
    ``_x_``. Prefer matching against the known anchor-name vocabulary before
    falling back to the old one-delimiter split for legacy records.
    """
    if name.startswith("hybrid_"):
        return _split_known_anchor_pair(
            name.removeprefix("hybrid_"), "_plus_", anchor_names
        )
    if name.startswith("cross_"):
        return _split_known_anchor_pair(
            name.removeprefix("cross_"), "_x_", anchor_names
        )
    return None
