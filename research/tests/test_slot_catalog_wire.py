"""Tests for the slot_property_catalog → _pick_compatible_motif wire-in.

Proves that the declared catalog narrowing actually intersects the
caller's class tuple in ``_pick_compatible_motif``, and that the safety
fallback (empty intersection → keep original classes) holds.
"""

from __future__ import annotations

import random
from unittest.mock import patch

import pytest

from research.synthesis._template_helpers import (
    MOTIF_CLASS_ATTENTION,
    MOTIF_CLASS_CONV,
    MOTIF_CLASS_FFN,
    MOTIF_CLASS_MOE,
    _pick_compatible_motif,
)
from research.synthesis.graph import ComputationGraph


pytestmark = [pytest.mark.unit]


def _make_graph_with_template(template: str, slot_idx: int) -> ComputationGraph:
    g = ComputationGraph(model_dim=128)
    g.add_input()
    g.metadata["_active_template"] = template
    g.metadata["_active_template_slot_counter"] = slot_idx
    g.metadata["_active_template_instance"] = 0
    return g


def test_declared_catalog_narrows_class_tuple() -> None:
    """When the declared catalog says only CONV fits, FFN/MOE/ATTENTION are pruned."""
    g = _make_graph_with_template("tpl_test_narrow", 0)
    classes_in = (
        MOTIF_CLASS_FFN,
        MOTIF_CLASS_MOE,
        MOTIF_CLASS_ATTENTION,
        MOTIF_CLASS_CONV,
    )

    # Catalog says only conv_core qualifies. Should narrow to (CONV,).
    with patch(
        "research.synthesis._template_helpers.slot_classes_for",
        return_value=(MOTIF_CLASS_CONV,),
    ):
        _pick_compatible_motif(
            g, node_id=0, rng=random.Random(0), motif_class_or_classes=classes_in
        )

    usage = g.metadata.get("template_slot_usage") or []
    assert usage, "expected slot usage to be recorded"
    recorded = usage[-1]
    # The narrowed class list should reflect the catalog intersection.
    assert recorded["slot_classes"] == [MOTIF_CLASS_CONV], (
        f"declared catalog narrowing did not apply: {recorded['slot_classes']}"
    )


def test_declared_catalog_empty_falls_back_to_original() -> None:
    """When catalog has no entry (returns ()), original tuple is preserved."""
    g = _make_graph_with_template("tpl_unknown_to_catalog", 0)
    classes_in = (MOTIF_CLASS_FFN, MOTIF_CLASS_MOE)

    with patch(
        "research.synthesis._template_helpers.slot_classes_for",
        return_value=(),
    ):
        _pick_compatible_motif(
            g, node_id=0, rng=random.Random(0), motif_class_or_classes=classes_in
        )

    recorded = g.metadata.get("template_slot_usage", [])[-1]
    assert set(recorded["slot_classes"]) == set(classes_in), (
        "empty declared catalog should leave classes untouched"
    )


def test_declared_catalog_empty_intersection_falls_back() -> None:
    """When catalog narrows to a class that's NOT in the caller's tuple,
    the wire-in must NOT narrow to () — it falls back to the caller's
    tuple to avoid emptying the candidate pool.
    """
    g = _make_graph_with_template("tpl_disjoint", 0)
    classes_in = (MOTIF_CLASS_FFN, MOTIF_CLASS_MOE)

    with patch(
        "research.synthesis._template_helpers.slot_classes_for",
        return_value=("some_class_we_dont_pass",),
    ):
        _pick_compatible_motif(
            g, node_id=0, rng=random.Random(0), motif_class_or_classes=classes_in
        )

    recorded = g.metadata.get("template_slot_usage", [])[-1]
    assert set(recorded["slot_classes"]) == set(classes_in), (
        "disjoint intersection should fall back, not empty the slot"
    )


def test_single_class_slot_not_narrowed() -> None:
    """Single-class slots (e.g. MOTIF_CLASS_NORM only) are not touched."""
    g = _make_graph_with_template("tpl_single_class", 0)

    # Spy that the catalog wasn't even consulted for single-class slots.
    with patch(
        "research.synthesis._template_helpers.slot_classes_for",
        side_effect=AssertionError("should not be called for single-class slots"),
    ):
        _pick_compatible_motif(
            g,
            node_id=0,
            rng=random.Random(0),
            motif_class_or_classes=MOTIF_CLASS_FFN,
        )

    recorded = g.metadata.get("template_slot_usage", [])[-1]
    assert recorded["slot_classes"] == [MOTIF_CLASS_FFN]


def test_missing_template_metadata_skips_narrowing() -> None:
    """When _active_template isn't set on the graph, narrowing is skipped."""
    g = ComputationGraph(model_dim=128)
    g.add_input()
    # NOTE: no _active_template — caller is invoking the picker outside a template.

    classes_in = (MOTIF_CLASS_FFN, MOTIF_CLASS_MOE)
    with patch(
        "research.synthesis._template_helpers.slot_classes_for",
        side_effect=AssertionError("should not be called without _active_template"),
    ):
        _pick_compatible_motif(
            g,
            node_id=0,
            rng=random.Random(0),
            motif_class_or_classes=classes_in,
        )
    # No assertion error from the spy → narrowing block skipped, as intended.
