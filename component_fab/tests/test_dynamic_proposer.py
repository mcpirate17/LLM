from __future__ import annotations

from pathlib import Path

import pytest
import torch

from component_fab.generator.code_generator import generate_module_from_spec
from component_fab.proposer.dynamic import (
    collect_dynamic_evidence_cases,
    enumerate_dynamic_proposals,
    spec_from_ledger_entry,
)
from component_fab.state.ledger import Ledger, PROMOTION_PROMOTED, PROMOTION_REJECTED
from component_fab.improver.axis_variants import DEFAULT_META_DB
from component_fab.proposer.enumeration import (
    enumerate_data_route_variants,
    enumerate_cycle_specs,
    enumerate_training_regime_variants,
)
from component_fab.improver.axis_variants import AnchorAxes
from component_fab.tests.conftest import base_dynamic_axes
from research.synthesis.training_regime_grammar import (
    AXIS_TRAIN_REGIME,
    AXIS_TRAIN_STAGES,
)


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
            "math_axes": base_dynamic_axes(),
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


def test_autonomous_cycle_includes_loss_monster_pair_axis_variants(
    tmp_path: Path,
) -> None:
    if not DEFAULT_META_DB.exists():
        pytest.skip("meta_analysis.db not present")
    ledger = Ledger(tmp_path / "ledger.jsonl")

    specs = enumerate_cycle_specs(
        ledger,
        ["tropical_attention"],
        cycle=1,
        include_frontier=False,
        include_nas=False,
        max_cross_pairs=0,
        max_knob_specs=0,
        max_dynamic_specs=0,
    )

    assert any(
        spec.math_axes.get("op_block_template") == "loss_monster_paired"
        for spec in specs
    )


def test_training_regime_variants_add_explicit_train_axes() -> None:
    if not DEFAULT_META_DB.exists():
        pytest.skip("meta_analysis.db not present")

    specs = enumerate_training_regime_variants(
        ["tropical_attention"],
        max_specs=2,
    )

    assert len(specs) == 2
    regimes = {spec.math_axes[AXIS_TRAIN_REGIME] for spec in specs}
    assert regimes == {"embed_warm_then_all", "body_warm_then_all"}
    assert all(AXIS_TRAIN_STAGES in spec.math_axes for spec in specs)
    assert all(spec.anchor_witness_op == "tropical_attention" for spec in specs)
    assert all("op_algebraic_space" in spec.math_axes for spec in specs)


def test_autonomous_cycle_can_include_training_regime_specs(
    tmp_path: Path,
) -> None:
    if not DEFAULT_META_DB.exists():
        pytest.skip("meta_analysis.db not present")
    ledger = Ledger(tmp_path / "ledger.jsonl")

    specs = enumerate_cycle_specs(
        ledger,
        ["tropical_attention"],
        cycle=1,
        include_frontier=False,
        include_nas=False,
        max_cross_pairs=0,
        max_knob_specs=0,
        max_dynamic_specs=0,
        max_training_specs=1,
    )

    training_specs = [spec for spec in specs if spec.math_axes.get(AXIS_TRAIN_REGIME)]
    assert len(training_specs) == 1
    assert training_specs[0].math_axes[AXIS_TRAIN_REGIME] == "embed_warm_then_all"


def test_data_route_variants_emit_adjustable_fold_axes(monkeypatch) -> None:
    def fake_anchor_axes_for_op(name: str) -> AnchorAxes:
        return AnchorAxes(
            op_name=name,
            axes={
                "op_algebraic_space": "tropical",
                "op_dynamical_has_state": 0,
            },
            eval_count=4,
            pass_rate=0.5,
        )

    monkeypatch.setattr(
        "component_fab.proposer.enumeration.anchor_axes_for_op",
        fake_anchor_axes_for_op,
    )

    specs = enumerate_data_route_variants(["carrier"], max_specs=32)
    by_name = {spec.name: spec for spec in specs}

    vertical = next(
        spec for spec in specs if "data_fold16_vertical_alternate" in spec.name
    )
    sparse = next(spec for spec in specs if "data_fold16_sparse_vertical" in spec.name)
    intermittent = next(
        spec for spec in specs if "data_fold16_intermittent_horizontal" in spec.name
    )
    half = next(spec for spec in specs if "data_fold16_vertical_half" in spec.name)

    assert any(name.startswith("route_carrier_data_") for name in by_name)
    assert vertical.math_axes["op_seq_fold"] == 16
    assert vertical.math_axes["op_seq_fold_orientation"] == "vertical"
    assert vertical.math_axes["op_seq_fold_direction"] == "alternate"
    assert sparse.math_axes["op_seq_fold_pattern"] == "sparse"
    assert intermittent.math_axes["op_seq_fold_pattern"] == "intermittent"
    assert intermittent.math_axes["op_seq_fold_orientation"] == "horizontal"
    assert half.math_axes["op_seq_fold_fraction"] == 0.5
    assert all(
        "data_route_routes_tokens_into_mixer_sooner" in spec.notes for spec in specs
    )


def test_autonomous_cycle_can_opt_into_data_route_specs(
    monkeypatch, tmp_path: Path
) -> None:
    def fake_anchor_axes_for_op(name: str) -> AnchorAxes:
        return AnchorAxes(
            op_name=name,
            axes={
                "op_algebraic_space": "tropical",
                "op_dynamical_has_state": 0,
            },
            eval_count=4,
            pass_rate=0.5,
        )

    monkeypatch.setattr(
        "component_fab.proposer.enumeration.anchor_axes_for_op",
        fake_anchor_axes_for_op,
    )
    ledger = Ledger(tmp_path / "ledger.jsonl")

    disabled = enumerate_cycle_specs(
        ledger,
        ["carrier"],
        cycle=1,
        include_static_variants=False,
        include_frontier=False,
        include_nas=False,
        include_training_regimes=False,
        include_data_routes=False,
        include_name_free_physics=False,
        max_knob_specs=0,
        max_dynamic_specs=0,
    )
    enabled = enumerate_cycle_specs(
        ledger,
        ["carrier"],
        cycle=1,
        include_static_variants=False,
        include_frontier=False,
        include_nas=False,
        include_training_regimes=False,
        include_data_routes=True,
        max_data_route_specs=2,
        include_name_free_physics=False,
        max_knob_specs=0,
        max_dynamic_specs=0,
    )

    assert not any("source=data_route_axis" in spec.notes for spec in disabled)
    assert len(enabled) == 2
    assert all("source=data_route_axis" in spec.notes for spec in enabled)


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


def test_repairs_loss_monster_unpaired_pairs_with_long_range_carrier() -> None:
    repairs = _repairs_for_case(_case("loss_monster_unpaired"), {})
    assert repairs[0].name == "pair_loss_monster_with_carrier"
    delta = repairs[0].delta
    assert delta["op_block_template"] == "loss_monster_paired"
    assert delta["op_partner_kind"] == "hyper_mor"
    assert delta["op_block_slot_loss"] == "routed_bottleneck"
    assert delta["op_candidate_role"] == "loss_specialist_pair"
    assert delta["op_loss_specialist_partner_op"] == "hyper_mor_b_145m"


def test_collect_dynamic_cases_labels_loss_floor_reasoning(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.record_grade(
        proposal_id="loss_monster",
        name="loss_monster",
        category="lane",
        synthesis_kind="novel_hybrid",
        cycle=1,
        composite_score=0.42,
        smoke_pass=True,
        learned_signal=False,
        metadata={
            "math_axes": {
                **base_dynamic_axes(),
                "op_candidate_role": "loss_specialist",
            },
            "screening_loss_ratio": 0.03,
            "can_bind": False,
        },
    )

    case = collect_dynamic_evidence_cases(ledger)[0]

    assert "loss_monster_unpaired" in case.weaknesses
    assert "strong_loss_floor_reasoning" in case.weaknesses


def test_physics_repairs_strip_named_composition_axes() -> None:
    repairs = _repairs_for_case(
        _case(
            WEAK_FAIL_LONG_GAP,
            axes={
                **base_dynamic_axes(),
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
                **base_dynamic_axes(),
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
            "math_axes": base_dynamic_axes(),
            "can_bind": False,
            "erf_density": 0.02,
            "nb_max_accuracy": 0.55,
        },
    )

    assert collect_dynamic_evidence_cases(ledger) == []


# --- WS-1: compression is actively searched (weakness -> repair loop) ---

from component_fab.proposer.dynamic import _compression_is_weak  # noqa: E402


def _seed_compression_weak_ledger(tmp_path: Path) -> Ledger:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.record_grade(
        proposal_id="compress_cand_0000000000",
        name="compress_cand",
        category="lane",
        synthesis_kind="novel_hybrid",
        cycle=1,
        composite_score=0.4,
        smoke_pass=True,
        learned_signal=True,
        metadata={
            "math_axes": base_dynamic_axes(),
            "can_bind": True,
            "compression_declared": True,
            "compression_effective_rank_ratio": 0.18,
            "compression_reconstruct_mse": 0.9,
        },
    )
    return ledger


def test_compression_weakness_requires_declared_and_a_real_metric() -> None:
    # Not declared -> never weak, even with a bad metric value present.
    assert not _compression_is_weak({"compression_effective_rank_ratio": 0.1})
    # Declared but no measured metric -> not weak (never fabricate a weakness).
    assert not _compression_is_weak({"compression_declared": True})
    # Declared + under-utilized latent budget -> weak.
    assert _compression_is_weak(
        {"compression_declared": True, "compression_effective_rank_ratio": 0.2}
    )
    # Declared + high reconstruction error -> weak.
    assert _compression_is_weak(
        {"compression_declared": True, "compression_reconstruct_mse": 0.9}
    )
    # Declared + healthy compression -> not weak.
    assert not _compression_is_weak(
        {
            "compression_declared": True,
            "compression_effective_rank_ratio": 0.95,
            "compression_reconstruct_mse": 0.01,
        }
    )


def test_repairs_weak_compression_targets_content_bottleneck() -> None:
    repairs = _repairs_for_case(_case("weak_compression"), {})
    assert repairs[0].name == "compress_content_bottleneck"
    delta = repairs[0].delta
    assert delta["op_search_track"] == "physics_atom"
    assert delta["op_physics_target"] == "compression_bottleneck_state"
    assert delta["op_physics_aggregate_family"] == "semiring"
    assert delta["op_dynamical_has_state"] == 1
    assert delta["op_spectral_preferred_basis"] == "content"


def test_dynamic_proposer_repairs_weak_compression(tmp_path: Path) -> None:
    ledger = _seed_compression_weak_ledger(tmp_path)

    cases = collect_dynamic_evidence_cases(ledger)
    assert cases
    assert "weak_compression" in cases[0].weaknesses

    specs = enumerate_dynamic_proposals(
        [],
        ledger,
        max_specs=8,
        include_anchor_fallback=False,
    )
    repaired = [s for s in specs if "compress_content_bottleneck" in s.name]
    assert repaired
    spec = repaired[0]
    assert spec.math_axes["op_search_track"] == "physics_atom"
    assert spec.math_axes["op_physics_target"] == "compression_bottleneck_state"
    assert spec.math_axes["op_spectral_preferred_basis"] == "content"
    assert spec.math_axes["op_dynamical_has_state"] == 1


def test_weak_compression_repair_spec_is_buildable(tmp_path: Path) -> None:
    ledger = _seed_compression_weak_ledger(tmp_path)
    spec = [
        s
        for s in enumerate_dynamic_proposals(
            [], ledger, max_specs=8, include_anchor_fallback=False
        )
        if "compress_content_bottleneck" in s.name
    ][0]

    module = generate_module_from_spec(spec, dim=16)
    x = torch.randn(2, 7, 16)
    y = module(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


# --- score-norm spectrum repair: escape measured softmax-basin collapse ---


def _score_norm_softmax_axes(score_norm: str = "softmax") -> dict:
    return {
        **base_dynamic_axes(),
        "op_search_track": "physics_atom",
        "op_physics_atom_kinds": "scan+basis",
        "op_physics_basis_axis": "token",
        "op_physics_address_family": "dot",
        "op_physics_score_norm_family": score_norm,
        "op_physics_aggregate_family": "mean",
        "op_physics_knob_scale": 1.0,
        "op_physics_target": "softmax_basin_source",
    }


def _seed_score_norm_softmax_basin_ledger(tmp_path: Path) -> Ledger:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.record_grade(
        proposal_id="softmax_basin_cand_0000000000",
        name="softmax_basin_cand",
        category="lane",
        synthesis_kind="novel_hybrid",
        cycle=1,
        composite_score=0.4,
        smoke_pass=True,
        learned_signal=True,
        metadata={
            "math_axes": _score_norm_softmax_axes(),
            "can_bind": True,
            "softmax_twin_score": 0.93,
        },
    )
    return ledger


def test_collect_dynamic_cases_labels_score_norm_softmax_basin(
    tmp_path: Path,
) -> None:
    ledger = _seed_score_norm_softmax_basin_ledger(tmp_path)

    cases = collect_dynamic_evidence_cases(ledger)

    assert cases
    assert cases[0].weaknesses == ("score_norm_softmax_basin",)


def test_collect_dynamic_cases_uses_math_sweep_twin_failure_reason(
    tmp_path: Path,
) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.record_grade(
        proposal_id="sweep_twin_cand_0000000000",
        name="sweep_twin_cand",
        category="lane",
        synthesis_kind="novel_hybrid",
        cycle=1,
        composite_score=0.3,
        smoke_pass=True,
        learned_signal=False,
        metadata={
            "math_axes": _score_norm_softmax_axes("sharpen"),
            "can_bind": True,
            "math_variant_failure_reason": "softmax_twin_regression",
        },
    )

    cases = collect_dynamic_evidence_cases(ledger)

    assert cases
    assert "score_norm_softmax_basin" in cases[0].weaknesses


def test_repairs_score_norm_basin_span_non_softmax_spectrum() -> None:
    repairs = _repairs_for_case(
        _case(
            "score_norm_softmax_basin",
            axes=_score_norm_softmax_axes(),
        ),
        {},
    )

    assert repairs[0].name == "repair_score_norm_spectrum"
    spectrum_repairs = [
        repair
        for repair in repairs
        if repair.delta.get("op_physics_target") == "score_norm_spectrum_escape"
    ]
    score_norms = {
        str(repair.delta["op_physics_score_norm_family"])
        for repair in spectrum_repairs
    }
    assert score_norms >= {"tsallis_q", "renyi", "entmax_alpha"}
    assert score_norms.isdisjoint({"softmax", "sharpen"})
    assert {
        repair.delta["op_physics_aggregate_family"] for repair in spectrum_repairs
    } == {"semiring"}


def test_dynamic_proposer_repairs_score_norm_basin_with_buildable_spectrum(
    tmp_path: Path,
) -> None:
    ledger = _seed_score_norm_softmax_basin_ledger(tmp_path)

    specs = enumerate_dynamic_proposals(
        [],
        ledger,
        max_specs=8,
        include_anchor_fallback=False,
    )

    spectrum_specs = [
        spec
        for spec in specs
        if spec.math_axes.get("op_physics_target") == "score_norm_spectrum_escape"
    ]
    score_norms = {
        str(spec.math_axes["op_physics_score_norm_family"])
        for spec in spectrum_specs
    }
    assert score_norms >= {"tsallis_q", "renyi", "entmax_alpha"}
    assert score_norms.isdisjoint({"softmax", "sharpen"})

    for score_norm in ("tsallis_q", "renyi", "entmax_alpha"):
        spec = next(
            spec
            for spec in spectrum_specs
            if spec.math_axes["op_physics_score_norm_family"] == score_norm
        )
        module = generate_module_from_spec(spec, dim=16)
        x = torch.randn(2, 7, 16)
        y = module(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()
