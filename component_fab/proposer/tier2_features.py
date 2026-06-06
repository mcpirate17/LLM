"""Shared feature extraction for the Tier-2 value predictor.

Single source of truth for the feature vector used both to TRAIN the predictor
(``tools.train_tier2_predictor``) and to CONSUME it at scoring time
(``state.tier2_predictor``). Keeping it here guarantees the two stay in lock-step
— a train/serve feature skew would silently corrupt predictions.

Features = measured position-Jacobian descriptors + op-multiset size + key axis
indicators, computed from a candidate's ``math_axes``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from component_fab.proposer.capability_screen import fab_op_multiset
from component_fab.proposer.measured_screen import measured_screen_for_spec
from component_fab.proposer.spec_generator import (
    ProposalSpec,
    category_from_axes,
    synthesis_kind_for_axes,
)

DESCRIPTORS: tuple[str, ...] = (
    "long_range_reach",
    "content_dependence",
    "content_match_gating",
    "causality_violation",
    "measured_lipschitz",
    "effective_rank",
    "nonlinearity",
    "self_dominance",
)
FEATURE_NAMES: tuple[str, ...] = (
    *DESCRIPTORS,
    "op_count",
    "n_distinct_ops",
    "has_state",
    "memory_o_l",
    "global_receptive",
)


def spec_from_axes(pid: str, name: str, axes: dict[str, Any]) -> ProposalSpec:
    return ProposalSpec(
        proposal_id=pid,
        name=name or pid,
        category=category_from_axes(axes),
        synthesis_kind=str(
            axes.get("synthesis_kind") or synthesis_kind_for_axes(axes, axes)
        ),
        math_axes=dict(axes),
        anchor_witness_op="",
        anchor_witnesses_all=(),
        declared_property_row=dict(axes),
        predicted_lift=0.5,
        rationale="",
    )


def features_for_spec(spec: ProposalSpec, *, extractor: Any) -> list[float] | None:
    """Build the predictor feature vector for ``spec``.

    Returns ``None`` when the measured screen is unavailable or any feature is
    non-finite — callers must treat that as "cannot predict", never as zeros.
    """
    axes = dict(spec.math_axes)
    ms = measured_screen_for_spec(spec, extractor=extractor)
    if not ms.available or ms.descriptors is None:
        return None
    d = ms.descriptors
    ops = fab_op_multiset(spec)
    feat = [float(d.get(k, 0.0)) for k in DESCRIPTORS]
    feat += [float(len(ops)), float(len(set(ops)))]
    feat += [
        float(axes.get("op_dynamical_has_state") or 0),
        1.0 if axes.get("op_dynamical_memory_length_class") == "O(L)" else 0.0,
        1.0
        if axes.get("op_geometric_receptive_field") in ("global", "hybrid_local_global")
        else 0.0,
    ]
    return feat if all(np.isfinite(feat)) else None


def features_for_row(row: dict[str, Any], extractor: Any) -> list[float] | None:
    """Training convenience: build features from a stored label row."""
    axes = dict(row.get("math_axes") or {})
    spec = spec_from_axes(row.get("proposal_id", ""), row.get("name", ""), axes)
    return features_for_spec(spec, extractor=extractor)
