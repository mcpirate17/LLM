"""Tests for curated external research priors + affinity scoring."""

from __future__ import annotations

from typing import Any

from component_fab.proposer.research_priors import (
    load_research_priors,
    prior_affinity_for_spec,
    to_catalog_rows,
)
from component_fab.proposer.spec_generator import ProposalSpec

_CATALOG_COLUMNS = {
    "external_family",
    "mapped_ops_json",
    "mapped_templates_json",
    "expected_strength",
    "expected_risk",
    "hardware_note",
    "tags_json",
    "confidence",
    "source_ref",
}


def _spec(axes: dict[str, Any], pid: str = "cand_x") -> ProposalSpec:
    return ProposalSpec(
        proposal_id=pid,
        name="cand",
        category="lane",
        synthesis_kind="novel_hybrid",
        math_axes=axes,
        anchor_witness_op="",
        anchor_witnesses_all=(),
        declared_property_row=dict(axes),
        predicted_lift=0.5,
        rationale="test",
    )


def test_priors_are_well_formed() -> None:
    priors = load_research_priors()
    assert len(priors) == 6
    families = {p.family for p in priors}
    assert len(families) == 6  # unique family names
    for prior in priors:
        assert 0.0 < prior.confidence <= 1.0
        assert prior.source_url.startswith("http")
        assert prior.validation_tasks
        assert prior.summary and prior.hardware_note


def test_affinity_matches_long_context_memory_family() -> None:
    spec = _spec(
        {
            "op_dynamical_has_state": 1,
            "op_dynamical_memory_length_class": "O(L)",
            "op_geometric_receptive_field": "global",
        }
    )
    affinity = prior_affinity_for_spec(spec)
    assert affinity.family == "chunked_attention_gated_fifo_memory"
    # 3 of 3 axis signals matched, but the prior also suggests a block template
    # the spec does not set, so the realized affinity is partial (3/4).
    assert affinity.affinity == 0.75
    assert "long_gap_recall" in affinity.validation_tasks
    assert affinity.reasons


def test_affinity_is_zero_for_unmatched_spec() -> None:
    spec = _spec(
        {
            "op_dynamical_has_state": 0,
            "op_dynamical_memory_length_class": "O(1)",
            "op_geometric_receptive_field": "local",
            "op_activation_sparsity_pattern": "dense",
        }
    )
    affinity = prior_affinity_for_spec(spec)
    assert affinity.affinity == 0.0
    assert affinity.family == "unknown"


def test_template_and_knob_signals_count() -> None:
    spec = _spec(
        {
            "op_routing_kind": "top_k_moe",
            "op_activation_sparsity_pattern": "top_k",
            "op_block_template": "sparse_moe_block",
        }
    )
    affinity = prior_affinity_for_spec(spec)
    # ReSSFormer or mixture-of-memory both plausibly match; affinity must be > 0.
    assert affinity.affinity > 0.0
    assert any("block_template" in r for r in affinity.reasons)


def test_catalog_rows_match_schema() -> None:
    rows = to_catalog_rows()
    assert len(rows) == 6
    for row in rows:
        assert set(row.keys()) == _CATALOG_COLUMNS
        assert isinstance(row["confidence"], float)
        # validation tasks are carried in tags_json since the table has no column
        assert "validation:" in row["tags_json"]
