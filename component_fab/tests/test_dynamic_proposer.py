from __future__ import annotations

from pathlib import Path

import torch

from component_fab.generator.code_generator import generate_module_from_spec
from component_fab.proposer.dynamic import (
    collect_dynamic_evidence_cases,
    enumerate_dynamic_proposals,
    spec_from_ledger_entry,
)
from component_fab.state.ledger import Ledger, PROMOTION_PROMOTED, PROMOTION_REJECTED
from component_fab.proposer.enumeration import enumerate_cycle_specs


def _base_axes() -> dict:
    return {
        "op_algebraic_space": "tropical",
        "op_spectral_preferred_basis": "identity",
        "op_dynamical_memory_length_class": "O(1)",
        "op_dynamical_has_state": 0,
        "op_activation_sparsity_pattern": "dense",
        "op_geometric_receptive_field": "local",
        "synthesis_kind": "novel_hybrid",
    }


def _seed_range_blind_ledger(tmp_path: Path) -> Ledger:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.record_grade(
        proposal_id="range_blind_case_0000000000",
        name="range_blind_case",
        category="lane",
        synthesis_kind="novel_hybrid",
        cycle=1,
        composite_score=0.42,
        smoke_pass=True,
        learned_signal=False,
        metadata={
            "math_axes": _base_axes(),
            "eliminated_by": None,
            "can_bind": False,
            "erf_density": 0.02,
            "nb_max_accuracy": 0.55,
            "range_ran": True,
            "range_effective_distance": 0,
        },
    )
    return ledger


def test_dynamic_proposer_repairs_range_and_binding_axes(tmp_path: Path) -> None:
    ledger = _seed_range_blind_ledger(tmp_path)

    cases = collect_dynamic_evidence_cases(ledger)
    assert cases
    assert "range_blind" in cases[0].weaknesses

    specs = enumerate_dynamic_proposals(
        [],
        ledger,
        max_specs=8,
        include_anchor_fallback=False,
    )
    assert specs
    assert any(spec.name.startswith("dynamic_range_blind_case") for spec in specs)

    repaired = [spec for spec in specs if "extend_receptive_state" in spec.name][0]
    assert repaired.math_axes["op_dynamical_has_state"] == 1
    assert repaired.math_axes["op_dynamical_memory_length_class"] == "O(L)"
    assert repaired.math_axes["op_geometric_receptive_field"] == "global"
    assert repaired.math_axes["op_spectral_preferred_basis"] == "content"


def test_dynamic_specs_are_buildable_modules(tmp_path: Path) -> None:
    ledger = _seed_range_blind_ledger(tmp_path)
    spec = enumerate_dynamic_proposals(
        [],
        ledger,
        max_specs=1,
        include_anchor_fallback=False,
    )[0]

    module = generate_module_from_spec(spec, dim=16)
    x = torch.randn(2, 7, 16)
    y = module(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_dynamic_generation_skips_terminal_repairs(tmp_path: Path) -> None:
    ledger = _seed_range_blind_ledger(tmp_path)
    first = enumerate_dynamic_proposals(
        [],
        ledger,
        max_specs=1,
        include_anchor_fallback=False,
    )[0]
    ledger.record_grade(
        proposal_id=first.proposal_id,
        name=first.name,
        category=first.category,
        synthesis_kind=first.synthesis_kind,
        cycle=2,
        composite_score=0.0,
        smoke_pass=False,
        learned_signal=False,
        metadata={"math_axes": first.math_axes},
    )
    ledger.record_promotion(first.proposal_id, PROMOTION_REJECTED)

    fresh = enumerate_dynamic_proposals(
        [],
        ledger,
        max_specs=1,
        include_anchor_fallback=False,
    )

    assert fresh
    assert fresh[0].proposal_id != first.proposal_id


def test_dynamic_generation_prefers_unseen_repairs_over_pending(
    tmp_path: Path,
) -> None:
    ledger = _seed_range_blind_ledger(tmp_path)
    pending = enumerate_dynamic_proposals(
        [],
        ledger,
        max_specs=1,
        include_anchor_fallback=False,
    )[0]
    ledger.record_grade(
        proposal_id=pending.proposal_id,
        name=pending.name,
        category=pending.category,
        synthesis_kind=pending.synthesis_kind,
        cycle=2,
        composite_score=0.2,
        smoke_pass=True,
        learned_signal=False,
        metadata={"math_axes": pending.math_axes},
    )

    fresh = enumerate_dynamic_proposals(
        [],
        ledger,
        max_specs=1,
        include_anchor_fallback=False,
    )

    assert fresh
    assert fresh[0].proposal_id != pending.proposal_id
    assert not ledger.has_seen(fresh[0].proposal_id)


def test_autonomous_cycle_includes_dynamic_specs_from_ledger(tmp_path: Path) -> None:
    ledger = _seed_range_blind_ledger(tmp_path)

    specs = enumerate_cycle_specs(
        ledger,
        [],
        cycle=1,
        use_promoted_as_anchors=False,
        max_cross_pairs=0,
        max_knob_specs=0,
        max_dynamic_specs=4,
    )

    assert specs
    assert any(spec.name.startswith("dynamic_range_blind_case") for spec in specs)


def test_ledger_entry_reconstructs_dynamic_spec_by_exact_axes(tmp_path: Path) -> None:
    ledger = _seed_range_blind_ledger(tmp_path)
    spec = enumerate_dynamic_proposals(
        [],
        ledger,
        max_specs=1,
        include_anchor_fallback=False,
    )[0]
    ledger.record_grade(
        proposal_id=spec.proposal_id,
        name=spec.name,
        category=spec.category,
        synthesis_kind=spec.synthesis_kind,
        cycle=2,
        composite_score=0.7,
        smoke_pass=True,
        learned_signal=True,
        metadata={"math_axes": spec.math_axes},
    )
    ledger.record_promotion(spec.proposal_id, PROMOTION_PROMOTED)

    rebuilt = spec_from_ledger_entry(ledger.entries[spec.proposal_id])
    assert rebuilt is not None
    assert rebuilt.proposal_id == spec.proposal_id
    assert rebuilt.math_axes == spec.math_axes


# --- characterization: _repairs_for_case table refactor (behaviour-preserving) ---

from component_fab.proposer.dynamic import (  # noqa: E402
    DynamicEvidenceCase,
    _repairs_for_case,
)
from component_fab.proposer.tier2_feedback import (  # noqa: E402
    WEAK_FAIL_COMPOSITIONAL,
    WEAK_FAIL_LONG_GAP,
    WEAK_REJECTED,
)


def _case(*weaknesses: str, axes: dict | None = None) -> DynamicEvidenceCase:
    base = axes or {}
    return DynamicEvidenceCase(
        source_id="t",
        root_source_id="t",
        name="t",
        base_axes=dict(base),
        anchor_axes=dict(base),
        score=0.5,
        weaknesses=tuple(weaknesses),
    )


def test_repairs_no_weakness_yields_only_fallback():
    repairs = _repairs_for_case(_case(), {})
    assert [r.name for r in repairs] == ["feedback_depth_router"]


def test_repairs_long_gap_fires_two_rules_in_order():
    repairs = _repairs_for_case(_case(WEAK_FAIL_LONG_GAP), {})
    assert [r.name for r in repairs[:2]] == [
        "extend_receptive_state",
        "repair_long_gap_memory",
    ]
    assert len(repairs) > 2
    assert repairs[0].delta["op_search_track"] == "physics_atom"
    assert repairs[1].delta["op_search_track"] == "physics_atom"
    assert repairs[1].delta["op_physics_atom_kinds"] == "scan+basis"
    assert repairs[1].delta["op_physics_aggregate_family"] == "semiring"
    assert repairs[1].delta["op_physics_target"] == "long_gap_recursive_memory"
    variants = [r for r in repairs if r.delta.get("op_physics_variant")]
    assert {r.delta["op_physics_variant"] for r in variants} >= {
        "physv01",
        "physv02",
        "physv03",
    }
    assert any(
        str(r.delta["op_physics_variant"]).startswith("physod") for r in variants
    )
    assert {r.delta["op_physics_seed"] for r in variants} >= {1, 2, 3}
    assert any(r.delta["op_physics_address_family"] == "dot" for r in variants)
    assert any(r.delta["op_physics_aggregate_family"] == "mean" for r in variants)
    open_discovery_variants = [
        r for r in variants if str(r.delta["op_physics_variant"]).startswith("physod")
    ]
    assert all(
        "scan" in r.delta["op_physics_atom_kinds"] for r in open_discovery_variants
    )


def test_repairs_rejected_only_when_no_prior():
    # alone -> fires
    assert [r.name for r in _repairs_for_case(_case(WEAK_REJECTED), {})] == [
        "rejected_to_memory_lookup"
    ]
    # with an earlier match -> suppressed
    names = [
        r.name
        for r in _repairs_for_case(_case(WEAK_FAIL_COMPOSITIONAL, WEAK_REJECTED), {})
    ]
    assert names == ["repair_compositional_tensor"]


def test_repairs_dynamic_delta_mines_value_pool():
    pool = {
        "op_activation_sparsity_pattern": ["mined_sparse"],
        "op_routing_kind": ["mined_route"],
    }
    repairs = _repairs_for_case(_case("weak_nano_bind"), pool)
    assert repairs[0].name == "bind_sparse_content"
    assert len(repairs) > 1
    delta = repairs[0].delta
    assert delta["op_search_track"] == "physics_atom"
    assert delta["op_physics_atom_kinds"] == "scan+basis"
    assert delta["op_physics_address_family"] == "cosine"
    assert delta["op_physics_aggregate_family"] == "semiring"
    assert delta["op_dynamical_has_state"] == 1
    assert delta["op_dynamical_memory_length_class"] == "O(L)"
    assert delta["op_activation_sparsity_pattern"] == "mined_sparse"
    assert delta["op_spectral_preferred_basis"] == "content"
    variants = [r.delta for r in repairs if r.delta.get("op_physics_variant")]
    assert variants
    assert any(str(v["op_physics_variant"]).startswith("physod") for v in variants)
    assert {v["op_physics_address_family"] for v in variants} >= {
        "dot",
        "reciprocal",
    }
    assert {v["op_physics_seed"] for v in variants} >= {1, 2, 3}


def test_physics_repairs_strip_named_composition_axes() -> None:
    repairs = _repairs_for_case(
        _case(
            WEAK_FAIL_LONG_GAP,
            axes={
                **_base_axes(),
                "op_algebraic_space": "tropical",
                "op_math_family": "composite",
                "op_math_knobs": "sparse_matrix_banded+kernel_random_features",
                "op_sparse_matrix_pattern": "causal_banded",
                "op_kernel_feature_map": "positive_random_features",
                "op_graph_topology": "causal_path_laplacian",
                "op_activation_sparsity_pattern": "top_k",
                "op_block_template": "gated_parallel",
                "op_block_slot_b": "fisher_attention",
                "op_routing_kind": "hash",
                "op_max_depth": 8,
                "op_top_k": 2,
            },
        ),
        {},
    )
    from component_fab.proposer.dynamic import _spec_from_case_and_repair

    spec = _spec_from_case_and_repair(
        _case(
            WEAK_FAIL_LONG_GAP,
            axes={
                **_base_axes(),
                "op_algebraic_space": "tropical",
                "op_math_family": "composite",
                "op_math_knobs": "sparse_matrix_banded+kernel_random_features",
                "op_sparse_matrix_pattern": "causal_banded",
                "op_kernel_feature_map": "positive_random_features",
                "op_graph_topology": "causal_path_laplacian",
                "op_activation_sparsity_pattern": "top_k",
                "op_block_template": "gated_parallel",
                "op_block_slot_b": "fisher_attention",
                "op_routing_kind": "hash",
                "op_max_depth": 8,
                "op_top_k": 2,
            },
        ),
        repairs[0],
    )

    assert spec.math_axes["op_search_track"] == "physics_atom"
    assert spec.category == "lane"
    assert "op_algebraic_space" not in spec.math_axes
    assert "op_math_family" not in spec.math_axes
    assert "op_math_knobs" not in spec.math_axes
    assert "op_sparse_matrix_pattern" not in spec.math_axes
    assert "op_kernel_feature_map" not in spec.math_axes
    assert "op_graph_topology" not in spec.math_axes
    assert "op_activation_sparsity_pattern" not in spec.math_axes
    assert "op_block_template" not in spec.math_axes
    assert "op_block_slot_b" not in spec.math_axes
    assert "op_routing_kind" not in spec.math_axes
    assert "op_max_depth" not in spec.math_axes
    assert "op_top_k" not in spec.math_axes


def test_repairs_static_delta_not_shared_by_reference():
    a = _repairs_for_case(_case(WEAK_FAIL_COMPOSITIONAL), {})[0]
    b = _repairs_for_case(_case(WEAK_FAIL_COMPOSITIONAL), {})[0]
    a.delta["op_math_knobs"] = "MUTATED"
    assert b.delta["op_math_knobs"] == "tensor_tucker"


def test_collect_dynamic_cases_skips_over_recursed_dynamic_bases(
    tmp_path: Path,
) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.record_grade(
        proposal_id="dynamic_dynamic_case_0000000000",
        name="dynamic_dynamic_case",
        category="lane",
        synthesis_kind="novel_hybrid",
        cycle=1,
        composite_score=0.42,
        smoke_pass=True,
        learned_signal=False,
        metadata={
            "math_axes": _base_axes(),
            "can_bind": False,
            "erf_density": 0.02,
            "nb_max_accuracy": 0.55,
        },
    )

    assert collect_dynamic_evidence_cases(ledger) == []
