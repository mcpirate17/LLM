"""Unit tests for the range-audit tool's reconstruction-fidelity logic."""

from __future__ import annotations

from component_fab.tools.run_range_audit import classify, reconstruction_fidelity


def test_classify_thresholds() -> None:
    assert classify(256) == "FULL-RANGE"
    assert classify(128) == "FULL-RANGE"
    assert classify(64) == "mid-range"
    assert classify(8) == "short"
    assert classify(0) == "no-bind"


def test_fidelity_low_for_block_without_slots() -> None:
    # A block template with no stored slot composition rebuilds with default
    # lanes -> low fidelity (the cross_clifford winner failure mode).
    fid, reason = reconstruction_fidelity(
        {"op_algebraic_space": "tropical", "op_block_template": "gated_parallel"}
    )
    assert fid == "low"
    assert "slot" in reason


def test_fidelity_ok_for_lane_and_specified_block() -> None:
    # A plain lane spec is fully captured by math_axes.
    assert reconstruction_fidelity({"op_algebraic_space": "tropical"})[0] == "ok"
    # A block that DOES carry its slot composition rebuilds faithfully.
    assert (
        reconstruction_fidelity(
            {
                "op_block_template": "top_ar_block",
                "op_block_slot_b": "local_window_attn",
            }
        )[0]
        == "ok"
    )
