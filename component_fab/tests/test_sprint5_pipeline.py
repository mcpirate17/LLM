"""Sprint-5 tests: ledger, probe tasks, cross-anchor, promotion."""

from __future__ import annotations

from pathlib import Path

from component_fab.tests.conftest import make_candidate_spec

import pytest
import torch
from torch import nn

from component_fab.harness.probe_tasks import DEFAULT_PROBE_TASKS
from component_fab.improver.cross_anchor import enumerate_cross_anchor_variants
from component_fab.policies.promotion import (
    DEFAULT_PROMOTION_RULES,
    PROMOTION_PENDING,
    PROMOTION_PROMOTED,
    PROMOTION_REJECTED,
    PromotionRules,
    apply_decisions,
    decide_promotion,
    decide_promotions_for_ledger,
)
from component_fab.proposer.property_miner import DEFAULT_META_DB
from component_fab.state.ledger import Ledger, LedgerEntry
from component_fab.validator.in_context import (
    physics_probe_lr_for_spec,
    physics_probe_steps_for_spec,
    physics_probe_tasks_for_spec,
    validate_in_context,
)


# ---------- Ledger ----------


def test_ledger_persists_grades_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    ledger.record_grade(
        proposal_id="p1",
        name="p1",
        category="lane",
        synthesis_kind="semiring_swap",
        cycle=1,
        composite_score=0.5,
        smoke_pass=True,
        learned_signal=False,
    )
    ledger.record_grade(
        proposal_id="p1",
        name="p1",
        category="lane",
        synthesis_kind="semiring_swap",
        cycle=2,
        composite_score=0.7,
        smoke_pass=True,
        learned_signal=True,
    )
    reborn = Ledger(path)
    entry = reborn.entries["p1"]
    assert entry.composite_history == [0.5, 0.7]
    assert entry.cycles_seen == [1, 2]
    assert entry.smoke_pass_count == 2
    assert entry.learned_signal_count == 1
    assert entry.metadata_history == [{}, {}]


def test_ledger_records_metadata_history(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    metadata = {
        "math_knobs": ["kernel_random_features"],
        "eliminated_by": None,
        "can_bind": True,
        "erf_density": 0.2,
        "nb_max_accuracy": 0.5,
    }
    ledger.record_grade(
        proposal_id="p1",
        name="compose_anchor_kernel_random_features",
        category="lane",
        synthesis_kind="semiring_swap",
        cycle=1,
        composite_score=0.6,
        smoke_pass=True,
        learned_signal=False,
        metadata=metadata,
    )

    reborn = Ledger(path)
    assert reborn.entries["p1"].metadata_history == [metadata]


def test_ledger_has_seen_dedups(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    assert not ledger.has_seen("p1")
    ledger.record_grade(
        proposal_id="p1",
        name="p1",
        category="lane",
        synthesis_kind="x",
        cycle=1,
        composite_score=0.5,
        smoke_pass=True,
        learned_signal=False,
    )
    assert ledger.has_seen("p1")


# ---------- Probe tasks ----------


def test_default_probe_tasks_cover_difficulty_spectrum() -> None:
    names = {t.name for t in DEFAULT_PROBE_TASKS}
    assert "running_mean" in names
    assert "shifted_copy" in names
    difficulties = {t.difficulty for t in DEFAULT_PROBE_TASKS}
    assert "easy" in difficulties
    assert "hard" in difficulties


def test_probe_tasks_preserve_shape() -> None:
    x = torch.randn(2, 8, 4)
    for task in DEFAULT_PROBE_TASKS:
        y = task.target_fn(x)
        assert y.shape == x.shape, f"task {task.name} broke shape"


def test_in_context_validator_runs_full_suite() -> None:
    spec = make_candidate_spec({"op_algebraic_space": "euclidean"})
    lane = nn.Linear(16, 16)
    card = validate_in_context(spec, lane, dim=16, seq_len=16, n_steps=30)
    assert set(card.per_task) == {t.name for t in DEFAULT_PROBE_TASKS}
    for task_result in card.per_task.values():
        assert "loss_ratio_initial_over_final" in task_result


def test_physics_in_context_probe_focuses_long_gap_tasks() -> None:
    spec = make_candidate_spec(
        {
            "op_search_track": "physics_atom",
            "op_physics_target": "long_gap_recursive_memory",
        }
    )
    tasks = physics_probe_tasks_for_spec(spec)

    assert {task.name for task in tasks} == {
        "shifted_copy",
        "copy_from_uniform_past",
        "causal_induction",
        "running_parity",
    }
    assert physics_probe_steps_for_spec(spec, 30) == 80
    assert physics_probe_lr_for_spec(spec, 1e-3) == pytest.approx(3e-3)


# ---------- Cross-anchor ----------


def test_cross_anchor_variants_pairs_compatible_anchors() -> None:
    if not DEFAULT_META_DB.exists():
        pytest.skip("meta_analysis.db not present")
    anchors = ["tropical_attention", "clifford_attention"]
    specs = enumerate_cross_anchor_variants(anchors)
    # Two anchors → C(2,2) * 2 orderings = 2 specs
    assert len(specs) == 2
    names = {s.name for s in specs}
    assert any("hybrid_tropical_attention_plus_clifford_attention" in n for n in names)


def test_cross_anchor_skips_non_hosting_algebras() -> None:
    # Euclidean isn't a hosting algebra; should produce no cross variants.
    specs = enumerate_cross_anchor_variants(["softmax_attention", "rmsnorm"])
    assert specs == []


def test_cross_anchor_excludes_per_position_hosts() -> None:
    """padic and spiking primitives don't mix across positions and would
    collapse to the 1/seq_len ERF floor whenever they host. They may still
    enter as donors via a mixing host, but never on their own.
    """
    if not DEFAULT_META_DB.exists():
        pytest.skip("meta_analysis.db not present")
    # padic_gate (padic algebra) + spike_rate_code (spiking algebra) — both
    # were previously valid hosts; neither should host now.
    specs = enumerate_cross_anchor_variants(["padic_gate", "spike_rate_code"])
    assert specs == []
    # A mixing host paired with a per-position donor: one direction only.
    specs = enumerate_cross_anchor_variants(["tropical_attention", "padic_gate"])
    assert len(specs) == 1
    assert specs[0].name.startswith("hybrid_tropical_attention_plus_padic_gate")


# ---------- Promotion ----------


def _entry(
    composite_history: list[float],
    *,
    smoke_pass: int | None = None,
    learned: int | None = None,
    status: str = PROMOTION_PENDING,
    metadata: dict | None = None,
) -> LedgerEntry:
    if smoke_pass is None:
        smoke_pass = len(composite_history)
    if learned is None:
        learned = len(composite_history)
    return LedgerEntry(
        proposal_id="p1",
        name="p1",
        category="lane",
        synthesis_kind="x",
        composite_history=composite_history,
        cycles_seen=list(range(1, len(composite_history) + 1)),
        metadata_history=[dict(metadata or {}) for _ in composite_history],
        smoke_pass_count=smoke_pass,
        learned_signal_count=learned,
        promotion_status=status,
    )


def test_decide_promotion_promotes_on_streak() -> None:
    entry = _entry(
        [0.7, 0.75],
        metadata={"paired_delta_ci_excludes_zero": True, "paired_delta_ci_low": 0.05},
    )
    decision = decide_promotion(entry, DEFAULT_PROMOTION_RULES)
    assert decision.decision == PROMOTION_PROMOTED


def test_decide_promotion_rejects_after_n_low_cycles() -> None:
    rules = PromotionRules(reject_after_n_cycles=3, reject_max_composite=0.3)
    entry = _entry([0.1, 0.2, 0.25])
    decision = decide_promotion(entry, rules)
    assert decision.decision == PROMOTION_REJECTED


def test_decide_promotion_pending_for_short_history() -> None:
    entry = _entry([0.7])
    decision = decide_promotion(entry, DEFAULT_PROMOTION_RULES)
    assert decision.decision == PROMOTION_PENDING


def test_decide_promotion_default_allows_no_learned_signal() -> None:
    entry = _entry(
        [0.7, 0.75],
        learned=0,
        metadata={"paired_delta_ci_excludes_zero": True, "paired_delta_ci_low": 0.05},
    )
    decision = decide_promotion(entry, DEFAULT_PROMOTION_RULES)
    assert decision.decision == PROMOTION_PROMOTED


def test_decide_promotion_requires_learned_signal_when_set() -> None:
    rules = PromotionRules(promote_require_learned_signal=True)
    entry = _entry([0.7, 0.75], learned=0)
    decision = decide_promotion(entry, rules)
    assert decision.decision == PROMOTION_PENDING


def test_apply_decisions_updates_ledger(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    for cycle, score in enumerate([0.7, 0.75], start=1):
        ledger.record_grade(
            proposal_id="p1",
            name="p1",
            category="lane",
            synthesis_kind="x",
            cycle=cycle,
            composite_score=score,
            smoke_pass=True,
            learned_signal=True,
            metadata={
                "paired_delta_ci_excludes_zero": True,
                "paired_delta_ci_low": 0.05,
            },
        )
    decisions = decide_promotions_for_ledger(ledger)
    counts = apply_decisions(ledger, decisions)
    assert counts[PROMOTION_PROMOTED] == 1
    assert ledger.entries["p1"].promotion_status == PROMOTION_PROMOTED
