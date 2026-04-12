from __future__ import annotations

import pytest

from research.tools.multiscale_catalogue import (
    assert_no_duplicate_logical_candidates,
    build_multiscale_registry,
)


def test_multiscale_catalogue_reports_known_duplicate_manifest_id():
    registry = build_multiscale_registry()
    duplicates = {row["manifest_id"]: row for row in registry["duplicate_manifest_ids"]}
    assert "rwkv_channel" in duplicates


def test_multiscale_catalogue_reduces_medium_and_hard_candidates_canonically():
    registry = build_multiscale_registry()
    medium = registry["medium_candidates"]
    hard = registry["hard_candidates"]

    assert len({row["canonical_name"] for row in medium}) == 11
    assert len({row["canonical_name"] for row in hard}) == 9
    assert any(
        row["slot_ref"] == "routing/adaptive_lane_mixer"
        and row["canonical_name"] == "difficulty_blend_3way"
        for row in medium
    )
    assert any(
        row["slot_ref"] == "routing/mixed_recursion_gate"
        and row["canonical_name"] == "score_depth_blend"
        for row in hard
    )


def test_multiscale_catalogue_duplicate_logical_candidate_guard_raises():
    with pytest.raises(ValueError):
        assert_no_duplicate_logical_candidates(
            [
                {
                    "slot_ref": "routing/compression_mixture_experts",
                    "canonical_name": "dual_compression_blend",
                },
                {
                    "slot_ref": "routing/dual_compression_blend",
                    "canonical_name": "dual_compression_blend",
                },
            ],
            "hard",
        )
