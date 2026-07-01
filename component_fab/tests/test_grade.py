"""Tests for the shared candidate grading chain."""

from __future__ import annotations

from types import SimpleNamespace

from torch import nn

from component_fab.proposer.spec_generator import ProposalSpec
from component_fab.state.gates import GATE_NANO_BIND, GATE_S05_CAUSALITY_STABILITY
from component_fab.state.ledger import Ledger, PROMOTION_REJECTED
from component_fab.tests.conftest import make_spec
from component_fab.runner.grading import metadata_for_grade as _metadata_for_grade
from component_fab.runner.selection import (
    physics_s05_prescreen_specs as _physics_s05_prescreen_specs,
    select_active_specs as _select_active_specs,
)
from component_fab.validator.capability import CapabilityScorecard
from component_fab.validator.grade import grade_candidate
from component_fab.validator.in_context import InContextScorecard
from component_fab.validator.solo import SoloScorecard


def _capability(eliminated_by: str | None) -> CapabilityScorecard:
    return CapabilityScorecard(
        proposal_id="p",
        name="p",
        s05_passed=True,
        s05_stability_passed=True,
        s05_causality_passed=True,
        s05_max_first_half_drift=0.0,
        erf_passed=True,
        erf_density=0.1,
        erf_density_entropy=0.0,
        erf_decay_slope=0.0,
        nb_passed=eliminated_by is None,
        nb_max_accuracy=0.2,
        nb_rejected_persistent_zero=False,
        ind_ran=False,
        ind_max_accuracy=0.0,
        ind_final_accuracy=0.0,
        ind_above_baseline=False,
        can_bind=False,
        binds_per_probe={},
        relative_recall_per_probe={},
        eliminated_by=eliminated_by,
    )


def _solo(spec: ProposalSpec) -> SoloScorecard:
    return SoloScorecard(
        proposal_id=spec.proposal_id,
        name=spec.name,
        category=spec.category,
        synthesis_kind=spec.synthesis_kind,
        math_axes=dict(spec.math_axes),
        smoke={
            "forward_passed": True,
            "backward_passed": True,
            "output_finite": True,
            "param_grad_finite": True,
        },
        metrics={},
        property_cross_check={},
        promoted=False,
    )


def _probe(spec: ProposalSpec) -> InContextScorecard:
    return InContextScorecard(
        proposal_id=spec.proposal_id,
        name=spec.name,
        category=spec.category,
        per_task={},
        aggregate_loss_ratio=1.0,
        mean_loss_ratio=1.0,
        learned_signal=False,
    )


def test_physics_repair_soft_escapes_nano_bind_gate(monkeypatch) -> None:
    spec = make_spec(
        {"op_search_track": "physics_atom", "op_physics_target": "long_gap_memory"},
        "physics",
    )
    monkeypatch.setattr(
        "component_fab.validator.grade.generate_module_from_spec",
        lambda _spec, dim: nn.Identity(),
    )
    monkeypatch.setattr(
        "component_fab.validator.grade.validate_capabilities",
        lambda *args, **kwargs: _capability(GATE_NANO_BIND),
    )
    monkeypatch.setattr(
        "component_fab.validator.grade.validate_solo",
        lambda spec, *_, **__: _solo(spec),
    )
    monkeypatch.setattr(
        "component_fab.validator.grade.validate_in_context",
        lambda spec, *_, **__: _probe(spec),
    )

    bundle = grade_candidate(spec, dim=8, seq_len=8, n_steps=1)

    assert bundle.eliminated_by is None
    assert bundle.capability["eliminated_by"] == GATE_NANO_BIND
    assert bundle.solo is not None
    assert bundle.in_context is not None


def test_non_physics_candidate_still_halts_on_nano_bind_gate(monkeypatch) -> None:
    spec = make_spec({"op_algebraic_space": "tropical"}, "nonphysics")
    monkeypatch.setattr(
        "component_fab.validator.grade.generate_module_from_spec",
        lambda _spec, dim: nn.Identity(),
    )
    monkeypatch.setattr(
        "component_fab.validator.grade.validate_capabilities",
        lambda *args, **kwargs: _capability(GATE_NANO_BIND),
    )

    bundle = grade_candidate(spec, dim=8, seq_len=8, n_steps=1)

    assert bundle.eliminated_by == GATE_NANO_BIND
    assert bundle.solo is None
    assert bundle.in_context is None


def test_physics_probe_task_ratios_persist_in_grade_metadata() -> None:
    spec = make_spec(
        {"op_search_track": "physics_atom", "op_physics_target": "long_gap_memory"},
        "physics",
    )
    metadata = _metadata_for_grade(
        spec,
        {"eliminated_by": GATE_NANO_BIND, "erf_density": 0.1, "nb_max_accuracy": 0.25},
        None,
        {
            "aggregate_loss_ratio": 1.18,
            "mean_loss_ratio": 1.08,
            "notes": ("physics_probe_tasks=shifted_copy", "physics_probe_lr=0.003"),
            "per_task": {
                "shifted_copy": {
                    "loss_ratio_initial_over_final": 1.18,
                    "trained_successfully": True,
                },
                "running_mean": {
                    "loss_ratio_initial_over_final": 0.9,
                    "trained_successfully": False,
                },
            },
        },
    )

    assert metadata["physics_probe_aggregate_loss_ratio"] == 1.18
    assert metadata["physics_probe_mean_loss_ratio"] == 1.08
    assert metadata["physics_probe_task_ratios"] == {"shifted_copy": 1.18}
    assert metadata["physics_probe_notes"] == [
        "physics_probe_tasks=shifted_copy",
        "physics_probe_lr=0.003",
    ]


def test_math_sweep_axes_persist_as_compact_grade_metadata() -> None:
    spec = make_spec(
        {
            "op_math_variant_family": "algebraic",
            "op_math_variant_transform": "reciprocal_cauchy_read",
            "op_math_variant_target": "binding",
            "op_math_variant_score": 0.42,
            "op_math_variant_delta_long_range_reach": 0.15,
            "op_math_variant_delta_content_match_gating": 0.08,
            "op_math_variant_artifact_ref": "research/reports/sweep.jsonl",
            "math_sweep_passed": True,
            "math_sweep_version": "dynamic_math_sweep_v1",
        },
        "sweep_spec",
    )

    metadata = _metadata_for_grade(
        spec,
        {"can_bind": True, "erf_density": 0.1},
        None,
    )

    assert metadata["math_sweep_passed"] is True
    assert metadata["math_sweep_version"] == "dynamic_math_sweep_v1"
    assert metadata["math_variant_family"] == "algebraic"
    assert metadata["math_variant_transform"] == "reciprocal_cauchy_read"
    assert metadata["math_variant_target"] == "binding"
    assert metadata["math_variant_score"] == 0.42
    assert metadata["math_variant_delta_long_range_reach"] == 0.15
    assert metadata["math_variant_delta_content_match_gating"] == 0.08
    assert metadata["math_variant_target_improved"] is True
    assert metadata["math_variant_artifact_ref"] == "research/reports/sweep.jsonl"


def test_selection_preskips_known_s05_physics_coordinate(tmp_path, monkeypatch) -> None:
    bad_axes = {
        "op_search_track": "physics_atom",
        "op_physics_target": "long_gap_recursive_memory",
        "op_physics_variant": "physv02",
        "op_physics_seed": 2,
        "op_physics_knob_scale": 2.5875,
        "op_physics_atom_kinds": "basis+scan",
        "op_physics_basis_axis": "token",
        "op_physics_norm_axis": "channel",
        "op_physics_address_family": "reciprocal",
        "op_physics_score_norm_family": "softmax",
        "op_physics_aggregate_family": "mean",
    }
    good_axes = {
        **bad_axes,
        "op_physics_variant": "physod03",
        "op_physics_seed": 103,
        "op_physics_knob_scale": 1.9798,
        "op_physics_address_family": "dot",
        "op_physics_score_norm_family": "sharpen",
        "op_physics_aggregate_family": "semiring",
    }
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.record_grade(
        proposal_id="prior_bad",
        name="prior_bad",
        category="lane",
        synthesis_kind="novel_hybrid",
        cycle=0,
        composite_score=0.0,
        smoke_pass=False,
        learned_signal=False,
        metadata={
            "math_axes": bad_axes,
            "capability_eliminated_by": GATE_S05_CAUSALITY_STABILITY,
        },
    )
    bad = make_spec(bad_axes, "new_bad", name="dynamic_bad")
    good = make_spec(good_axes, "new_good", name="dynamic_good")
    monkeypatch.setattr(
        "component_fab.runner.selection.physics_s05_prescreen_specs",
        lambda specs, ledger, *, cycle, dim, seq_len: (list(specs), 0),
    )

    (
        active,
        _nas,
        _buckets,
        n_new_selected,
        n_new_available,
        n_terminal_skipped,
        n_physics_s05_skipped,
        n_physics_s05_prescreen_failed,
    ) = _select_active_specs(
        [bad, good],
        ledger,
        cycle=1,
        dim=8,
        seq_len=8,
        selection="legacy",
        acquisition_beta=1.0,
        use_nas_screen=False,
        use_quality_order=False,
        max_graded_per_cycle=0,
        tier2_feedback_by_id={},
    )

    assert [spec.proposal_id for spec in active] == [good.proposal_id]
    assert n_new_selected == 1
    assert n_new_available == 1
    assert n_terminal_skipped == 0
    assert n_physics_s05_skipped == 1
    assert n_physics_s05_prescreen_failed == 0


def test_physics_s05_prescreen_records_new_hard_failure(tmp_path, monkeypatch) -> None:
    spec = make_spec(
        {
            "op_search_track": "physics_atom",
            "op_physics_target": "long_gap_memory",
            "op_physics_atom_kinds": "basis+scan",
        },
        "physics",
    )
    ledger = Ledger(tmp_path / "ledger.jsonl")
    monkeypatch.setattr(
        "component_fab.runner.selection.generate_module_from_spec",
        lambda _spec, dim: nn.Identity(),
    )
    monkeypatch.setattr(
        "component_fab.runner.selection.causality_stability_gate",
        lambda *args, **kwargs: SimpleNamespace(
            passed=False,
            max_first_half_drift=0.25,
            notes=("future drift",),
        ),
    )

    safe, failed = _physics_s05_prescreen_specs(
        [spec],
        ledger,
        cycle=3,
        dim=8,
        seq_len=8,
    )

    assert safe == []
    assert failed == 1
    entry = ledger.entries[spec.proposal_id]
    assert entry.promotion_status == PROMOTION_REJECTED
    metadata = entry.metadata_history[-1]
    assert metadata["physics_s05_prescreen_failed"] is True
    assert metadata["capability_eliminated_by"] == GATE_S05_CAUSALITY_STABILITY
