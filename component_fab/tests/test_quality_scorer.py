"""Tests for the fused proposal-quality scorer + budget bucketing."""

from __future__ import annotations

from typing import Any

from component_fab.proposer.nas_screen import NasScreenResult
from component_fab.proposer.quality import (
    BUCKET_EXPLOIT,
    BUCKET_EXPLORATION,
    BUCKET_REPAIR,
    SIGNATURE_DYNAMIC_LEDGER_REPAIR,
    allocate_budget_buckets,
    bucket_counts,
    physics_s05_failure_count_for_spec,
    score_quality,
    score_specs_quality,
)
from component_fab.proposer.spec_generator import ProposalSpec
from component_fab.state.ledger import LedgerEntry  # noqa: F401  (used in fixtures)
from component_fab.proposer.tier2_feedback import (
    Tier2Feedback,
    WEAK_NARROW_DISTRACTOR_ONLY,
    WEAK_FAIL_LONG_GAP,
)
from component_fab.state.gates import GATE_S05_CAUSALITY_STABILITY
from component_fab.tests.conftest import make_spec


def _spec(
    pid: str, axes: dict[str, Any] | None = None, name: str = "cand"
) -> ProposalSpec:
    return make_spec(axes or {}, pid, name=name, category="lane")


def _tier2(
    pid: str, *, passed: bool, signatures: tuple[str, ...], mean_delta: float
) -> Tier2Feedback:
    return Tier2Feedback(
        proposal_id=pid,
        name="cand",
        pass_count=4 if passed else 1,
        n_tasks=6,
        tier2_passed=passed,
        tier2_passed_niche=passed,
        mean_delta=mean_delta,
        wins=(),
        failures=(),
        signatures=signatures,
        task_results=(),
    )


def _nas(
    pid: str,
    *,
    gate_pass: bool = True,
    rank: float = 1.0,
    raw: dict[str, float] | None = None,
) -> NasScreenResult:
    return NasScreenResult(
        proposal_id=pid,
        available=True,
        gate_pass=gate_pass,
        downstream_gate_pass=gate_pass,
        rank_score=rank,
        source="test",
        raw=raw,
    )


def test_tier2_survivor_outranks_distractor_only() -> None:
    survivor_spec = _spec("survivor")
    distractor_spec = _spec("distractor")
    survivor = score_quality(
        survivor_spec,
        tier2=_tier2("survivor", passed=True, signatures=(), mean_delta=0.05),
        nas=_nas("survivor"),
    )
    distractor = score_quality(
        distractor_spec,
        tier2=_tier2(
            "distractor",
            passed=False,
            signatures=(WEAK_NARROW_DISTRACTOR_ONLY,),
            mean_delta=-0.02,
        ),
        nas=_nas("distractor"),
    )
    assert survivor.quality_score > distractor.quality_score
    assert survivor.bucket == BUCKET_EXPLOIT
    assert survivor.tier2_win_probability >= 0.7
    assert distractor.risk_score > survivor.risk_score
    assert distractor.bucket == BUCKET_REPAIR


def test_repair_bucket_from_failure_signature() -> None:
    score = score_quality(
        _spec("repairme"),
        tier2=_tier2(
            "repairme", passed=False, signatures=(WEAK_FAIL_LONG_GAP,), mean_delta=0.0
        ),
        nas=_nas("repairme"),
    )
    assert score.bucket == BUCKET_REPAIR
    assert WEAK_FAIL_LONG_GAP in score.repair_signatures
    assert (
        "long_gap" in score.why_beats_tier2
        or WEAK_FAIL_LONG_GAP in score.why_beats_tier2
    )


def test_dynamic_proposal_without_tier2_gets_repair_bucket() -> None:
    score = score_quality(
        _spec(
            "dynamic_repair",
            name="dynamic_source_extend_receptive_state_weak_nano_bind",
        )
    )

    assert score.bucket == BUCKET_REPAIR
    assert SIGNATURE_DYNAMIC_LEDGER_REPAIR in score.repair_signatures


def test_unseen_dynamic_physics_variant_outranks_seen_base_repair() -> None:
    axes = {
        "op_search_track": "physics_atom",
        "op_physics_atom_kinds": "scan+basis",
        "op_physics_address_family": "reciprocal",
        "op_physics_score_norm_family": "sharpen",
        "op_physics_aggregate_family": "semiring",
    }
    base = score_quality(
        _spec(
            "dynamic_base",
            axes,
            name="dynamic_source_extend_receptive_state_weak_nano_bind",
        ),
        entry=LedgerEntry(
            proposal_id="dynamic_base",
            name="dynamic_source_extend_receptive_state_weak_nano_bind",
            category="lane",
            synthesis_kind="novel_hybrid",
            composite_history=[0.6],
        ),
    )
    variant = score_quality(
        _spec(
            "dynamic_variant",
            {**axes, "op_physics_variant": "physv01"},
            name="dynamic_source_extend_receptive_state_physv01_weak_nano_bind",
        )
    )

    assert variant.bucket == BUCKET_REPAIR
    assert variant.quality_score > base.quality_score
    assert any("physics variant" in r for r in variant.evidence_reasons)


def test_dynamic_physics_long_gap_uses_measured_target_alignment() -> None:
    axes = {
        "op_search_track": "physics_atom",
        "op_physics_target": "long_gap_recursive_memory",
    }
    weak = score_quality(
        _spec("dynamic_weak", axes, name="dynamic_source_repair_long_gap_memory"),
        nas=_nas(
            "dynamic_weak",
            raw={
                "long_range_reach": 0.005,
                "causality_violation": 0.45,
                "content_dependence": 0.0,
                "content_match_gating": 0.0,
            },
        ),
    )
    aligned = score_quality(
        _spec("dynamic_aligned", axes, name="dynamic_source_repair_long_gap_memory"),
        nas=_nas(
            "dynamic_aligned",
            raw={
                "long_range_reach": 0.05,
                "causality_violation": 0.05,
                "content_dependence": 0.0,
                "content_match_gating": 0.0,
            },
        ),
    )

    assert aligned.quality_score > weak.quality_score
    assert any("long-gap repair" in r for r in aligned.evidence_reasons)


def test_dynamic_physics_binding_uses_measured_target_alignment() -> None:
    axes = {
        "op_search_track": "physics_atom",
        "op_physics_target": "binding_content_addressed_state",
    }
    weak = score_quality(
        _spec("dynamic_weak_bind", axes, name="dynamic_source_bind_sparse_content"),
        nas=_nas(
            "dynamic_weak_bind",
            raw={
                "long_range_reach": 0.005,
                "causality_violation": 0.4,
                "content_dependence": 0.05,
                "content_match_gating": 0.05,
            },
        ),
    )
    aligned = score_quality(
        _spec("dynamic_aligned_bind", axes, name="dynamic_source_bind_sparse_content"),
        nas=_nas(
            "dynamic_aligned_bind",
            raw={
                "long_range_reach": 0.02,
                "causality_violation": 0.05,
                "content_dependence": 0.7,
                "content_match_gating": 0.5,
            },
        ),
    )

    assert aligned.quality_score > weak.quality_score
    assert any("binding repair" in r for r in aligned.evidence_reasons)


def test_dynamic_physics_quality_uses_target_task_learning_ratios() -> None:
    axes = {
        "op_search_track": "physics_atom",
        "op_physics_target": "long_gap_recursive_memory",
    }
    weak_entry = LedgerEntry(
        proposal_id="dynamic_weak_task",
        name="dynamic_source_repair_long_gap_memory",
        category="lane",
        synthesis_kind="novel_hybrid",
        composite_history=[0.4],
        metadata_history=[
            {
                "physics_probe_task_ratios": {
                    "shifted_copy": 1.01,
                    "copy_from_uniform_past": 1.0,
                    "causal_induction": 1.0,
                    "running_parity": 1.0,
                }
            }
        ],
    )
    strong_entry = LedgerEntry(
        proposal_id="dynamic_strong_task",
        name="dynamic_source_repair_long_gap_memory",
        category="lane",
        synthesis_kind="novel_hybrid",
        composite_history=[0.4],
        metadata_history=[
            {
                "physics_probe_task_ratios": {
                    "shifted_copy": 1.06,
                    "copy_from_uniform_past": 1.18,
                    "causal_induction": 1.04,
                    "running_parity": 1.02,
                }
            }
        ],
    )

    weak = score_quality(_spec("dynamic_weak_task", axes), entry=weak_entry)
    strong = score_quality(_spec("dynamic_strong_task", axes), entry=strong_entry)

    assert strong.quality_score > weak.quality_score
    assert any("physics task learning" in r for r in strong.evidence_reasons)


def test_dynamic_physics_quality_inherits_task_ratios_from_matching_coordinate() -> (
    None
):
    axes = {
        "op_search_track": "physics_atom",
        "op_physics_target": "long_gap_recursive_memory",
        "op_physics_atom_kinds": "basis+scan",
        "op_physics_basis_axis": "token",
        "op_physics_norm_axis": "channel",
        "op_physics_address_family": "reciprocal",
        "op_physics_score_norm_family": "sharpen",
        "op_physics_aggregate_family": "mean",
        "op_physics_variant": "physod01",
    }
    prior = LedgerEntry(
        proposal_id="prior_physod01",
        name="dynamic_prior",
        category="lane",
        synthesis_kind="novel_hybrid",
        metadata_history=[
            {
                "math_axes": axes,
                "physics_probe_task_ratios": {
                    "copy_from_uniform_past": 1.2,
                    "causal_induction": 1.1,
                },
            }
        ],
    )
    candidate = _spec("dynamic_candidate", axes, name="dynamic_candidate")
    plain = list(score_specs_quality([candidate]).values())[0]
    inherited = list(
        score_specs_quality(
            [candidate], entries_by_id={prior.proposal_id: prior}
        ).values()
    )[0]

    assert inherited.quality_score > plain.quality_score
    assert any("physics task learning" in r for r in inherited.evidence_reasons)


def test_dynamic_physics_quality_penalizes_repeated_s05_coordinate() -> None:
    bad_axes = {
        "op_search_track": "physics_atom",
        "op_physics_target": "long_gap_recursive_memory",
        "op_physics_atom_kinds": "basis+scan",
        "op_physics_basis_axis": "token",
        "op_physics_norm_axis": "channel",
        "op_physics_address_family": "reciprocal",
        "op_physics_score_norm_family": "softmax",
        "op_physics_aggregate_family": "mean",
        "op_physics_variant": "physv02",
    }
    good_axes = {
        **bad_axes,
        "op_physics_address_family": "dot",
        "op_physics_score_norm_family": "sharpen",
        "op_physics_variant": "physod01",
    }
    failed = LedgerEntry(
        proposal_id="failed_physv02",
        name="dynamic_failed",
        category="lane",
        synthesis_kind="novel_hybrid",
        metadata_history=[
            {
                "math_axes": bad_axes,
                "capability_eliminated_by": GATE_S05_CAUSALITY_STABILITY,
            }
        ],
    )
    bad = _spec("dynamic_bad", bad_axes, name="dynamic_bad")
    good = _spec("dynamic_good", good_axes, name="dynamic_good")
    scores = score_specs_quality(
        [bad, good], entries_by_id={failed.proposal_id: failed}
    )

    assert (
        scores[good.proposal_id].quality_score > scores[bad.proposal_id].quality_score
    )
    assert any("S0.5 failures" in r for r in scores[bad.proposal_id].evidence_reasons)


def test_physics_s05_memory_keys_dispatched_program_not_repair_label() -> None:
    failed_axes = {
        "op_search_track": "physics_atom",
        "op_physics_target": "long_gap_recursive_memory",
        "op_physics_variant": "physod01",
        "op_physics_seed": 101,
        "op_physics_knob_scale": 2.0,
        "op_physics_atom_kinds": "basis+scan",
        "op_physics_basis_axis": "token",
        "op_physics_norm_axis": "channel",
        "op_physics_address_family": "reciprocal",
        "op_physics_score_norm_family": "sharpen",
        "op_physics_aggregate_family": "mean",
    }
    relabeled_axes = {
        **failed_axes,
        "op_physics_target": "long_gap_ordered_memory",
        "op_physics_variant": "physod99",
    }
    different_program_axes = {**relabeled_axes, "op_physics_seed": 102}
    failed = LedgerEntry(
        proposal_id="failed_physics",
        name="dynamic_failed",
        category="lane",
        synthesis_kind="novel_hybrid",
        metadata_history=[
            {
                "math_axes": failed_axes,
                "capability_eliminated_by": GATE_S05_CAUSALITY_STABILITY,
            }
        ],
    )

    assert (
        physics_s05_failure_count_for_spec(
            _spec("relabeled", relabeled_axes), {failed.proposal_id: failed}
        )
        == 1
    )
    assert (
        physics_s05_failure_count_for_spec(
            _spec("different", different_program_axes), {failed.proposal_id: failed}
        )
        == 0
    )


def test_no_tier2_high_prior_goes_to_exploration() -> None:
    spec = _spec(
        "novel",
        {
            "op_dynamical_has_state": 1,
            "op_dynamical_memory_length_class": "O(L)",
            "op_geometric_receptive_field": "global",
        },
    )
    score = score_quality(spec, nas=_nas("novel"))
    assert score.has_tier2_evidence is False
    assert score.bucket == BUCKET_EXPLORATION
    assert score.prior_affinity > 0.0
    assert any("estimated" in r for r in score.evidence_reasons)


def test_high_internal_composite_distractor_stays_out_of_exploit() -> None:
    """A high cheap-composite distractor-only false positive must not be exploit."""

    entry = LedgerEntry(
        proposal_id="high_composite",
        name="cand",
        category="lane",
        synthesis_kind="novel_hybrid",
        composite_history=[0.9],
    )
    score = score_quality(
        _spec("high_composite"),
        tier2=_tier2(
            "high_composite",
            passed=False,
            signatures=(WEAK_NARROW_DISTRACTOR_ONLY,),
            mean_delta=0.01,
        ),
        nas=_nas("high_composite"),
        entry=entry,
    )
    assert score.bucket == BUCKET_REPAIR


def test_confirmed_tier2_pass_beats_nas_proxy_gate() -> None:
    """A confirmed Tier-2 pass stays exploit even when the NAS proxy gate fails."""

    score = score_quality(
        _spec("confirmed"),
        tier2=_tier2("confirmed", passed=True, signatures=(), mean_delta=0.05),
        nas=_nas("confirmed", gate_pass=False),
    )
    assert score.bucket == BUCKET_EXPLOIT


def test_measured_nonbinder_raises_risk_and_caps_win() -> None:
    """A measured non-binder (won't route info backward) is genuine risk."""

    score = score_quality(_spec("nonbinder"), nas=_nas("nonbinder", gate_pass=False))
    assert score.risk_score >= 0.4
    assert any("won't bind" in r for r in score.evidence_reasons)
    # pre-Tier-2 win estimate is capped for a non-binder
    binder = score_quality(_spec("binder"), nas=_nas("binder", gate_pass=True))
    assert binder.tier2_win_probability > score.tier2_win_probability


def test_verdict_only_passes_measured_baseline_beaters() -> None:
    from component_fab.proposer.measured_screen import REASON_UNSTABLE
    from component_fab.proposer.quality import (
        VERDICT_BEATS_BASELINE,
        VERDICT_LOSES_TO_BASELINE,
        VERDICT_REJECT_NON_BINDER,
        VERDICT_REJECT_UNSTABLE,
        VERDICT_UNPROVEN,
    )

    # measured to beat baseline → the only PASS
    beats = score_quality(
        _spec("beats"),
        tier2=_tier2("beats", passed=True, signatures=(), mean_delta=0.05),
    )
    assert beats.verdict == VERDICT_BEATS_BASELINE
    assert beats.passes_hard_filter is True

    # measured but loses to baseline → reject
    loses = score_quality(
        _spec("loses"),
        tier2=_tier2("loses", passed=False, signatures=(), mean_delta=-0.02),
    )
    assert loses.verdict == VERDICT_LOSES_TO_BASELINE
    assert loses.passes_hard_filter is False

    # no Tier-2 evidence → UNPROVEN, never a pass ("ok" is not enough)
    unproven = score_quality(_spec("unproven"), nas=_nas("unproven"))
    assert unproven.verdict == VERDICT_UNPROVEN
    assert unproven.passes_hard_filter is False

    # NaN/unstable → reject_unstable
    unstable = NasScreenResult(
        proposal_id="u",
        available=True,
        gate_pass=False,
        downstream_gate_pass=False,
        rank_score=0.0,
        source="measured_descriptors",
        reason=REASON_UNSTABLE,
    )
    su = score_quality(_spec("u"), nas=unstable)
    assert su.verdict == VERDICT_REJECT_UNSTABLE
    assert su.passes_hard_filter is False

    # non-binder → reject_non_binder
    nb = score_quality(_spec("nb"), nas=_nas("nb", gate_pass=False))
    assert nb.verdict == VERDICT_REJECT_NON_BINDER


def test_measured_binding_drives_win_probability_without_tier2() -> None:
    """The measured nb_max_accuracy probe (not NAS) estimates win-prob pre-Tier-2."""

    strong = LedgerEntry(
        proposal_id="strong_bind",
        name="cand",
        category="lane",
        synthesis_kind="novel_hybrid",
        composite_history=[0.3],
        metadata_history=[{"nb_max_accuracy": 0.9, "can_bind": True}],
    )
    weak = LedgerEntry(
        proposal_id="weak_bind",
        name="cand",
        category="lane",
        synthesis_kind="novel_hybrid",
        composite_history=[0.3],
        metadata_history=[{"nb_max_accuracy": 0.05, "can_bind": False}],
    )
    s_strong = score_quality(_spec("strong_bind"), entry=strong)
    s_weak = score_quality(_spec("weak_bind"), entry=weak)
    assert s_strong.tier2_win_probability > s_weak.tier2_win_probability
    assert s_strong.quality_score > s_weak.quality_score
    assert any("nb=" in r for r in s_strong.evidence_reasons)


def test_allocate_budget_buckets_respects_total_and_split() -> None:
    specs = [_spec(f"p{i}") for i in range(20)]
    nas_by_id = {s.proposal_id: _nas(s.proposal_id) for s in specs}
    scores = score_specs_quality(specs, nas_by_id=nas_by_id)
    queue = allocate_budget_buckets(list(scores.values()), total=10)
    assert len(queue) == 10
    # descending quality order
    qs = [s.quality_score for s in queue]
    assert qs == sorted(qs, reverse=True)


def test_allocate_returns_all_when_budget_exceeds_supply() -> None:
    specs = [_spec(f"p{i}") for i in range(3)]
    scores = score_specs_quality(specs)
    queue = allocate_budget_buckets(list(scores.values()), total=99)
    assert len(queue) == 3


def test_bucket_counts_sums() -> None:
    specs = [_spec(f"p{i}") for i in range(5)]
    scores = list(score_specs_quality(specs).values())
    counts = bucket_counts(scores)
    assert sum(counts.values()) == 5
    assert set(counts) >= {BUCKET_EXPLOIT, BUCKET_REPAIR, BUCKET_EXPLORATION}
