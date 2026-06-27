"""Tests for WS-4 (Pareto/niche promotion) + WS-5 (behavioral novelty)."""

from __future__ import annotations

from pathlib import Path

from component_fab.metrics.behavior_fingerprint import (
    DEFAULT_CLONE_EPS,
    Normalizer,
    behavior_fingerprint,
    fingerprint_from_metadata,
    is_clone,
    novelty_distance,
)
from component_fab.improver.ranking import (
    non_dominated_sort,
    objective_vector,
    pareto_front_indices,
)
from component_fab.policies.promotion import (
    PROMOTION_PENDING,
    PROMOTION_PROMOTED,
    PROMOTION_REJECTED,
    PromotionRules,
    decide_promotion,
)
from component_fab.runner.grading import finalize_survivors
from component_fab.runner.niche import annotate_niche_metadata
from component_fab.state.ledger import Ledger


# --------------------------------------------------------------------------- #
# WS-5: behavioral fingerprint + novelty
# --------------------------------------------------------------------------- #
def test_fingerprint_reads_scorecards():
    probe = {"aggregate_loss_ratio": 100.0, "per_task": {}}
    cap = {"erf_density": 0.3, "nb_max_accuracy": 0.8, "ind_max_accuracy": 0.5}
    fp = behavior_fingerprint(probe, cap)
    assert fp["nb_max_accuracy"] == 0.8
    assert fp["ind_max_accuracy"] == 0.5
    assert fp["learning"] > 0.0  # log10(100)/2 = 1.0


def test_identical_behavior_is_clone_regardless_of_axes():
    # Same behavioral fingerprint, different axes -> behavioral clone.
    fp_a = behavior_fingerprint(
        {"aggregate_loss_ratio": 50.0},
        {"erf_density": 0.25, "nb_max_accuracy": 0.7},
    )
    catalog = [fp_a]
    fp_b = dict(fp_a)  # identical behavior, imagine different math_axes
    dist = novelty_distance(fp_b, catalog)
    assert dist == 0.0
    assert is_clone(dist)


def test_distinct_behavior_not_clone():
    fp_a = behavior_fingerprint(
        {"aggregate_loss_ratio": 2.0}, {"nb_max_accuracy": 0.1}
    )
    fp_b = behavior_fingerprint(
        {"aggregate_loss_ratio": 100.0},
        {"nb_max_accuracy": 0.95, "ind_max_accuracy": 0.9},
    )
    fp_c = behavior_fingerprint({"aggregate_loss_ratio": 1.0}, {"erf_density": 0.01})
    catalog = [fp_a, fp_b]
    dist = novelty_distance(
        fp_c, catalog, normalizer=Normalizer.fit([fp_a, fp_b, fp_c])
    )
    assert not is_clone(dist, clone_eps=DEFAULT_CLONE_EPS)


def test_empty_catalog_is_infinitely_novel():
    fp = behavior_fingerprint({"aggregate_loss_ratio": 5.0}, {})
    assert novelty_distance(fp, []) == float("inf")


def test_fingerprint_from_metadata_coarse_fallback():
    fp = fingerprint_from_metadata(
        {"erf_density": 0.4, "nb_max_accuracy": 0.6, "can_bind": True}
    )
    assert fp["erf_density"] == 0.4
    assert fp["binding"] == 1.0  # can_bind -> binding 1.0


# --------------------------------------------------------------------------- #
# WS-4: objective vector + non-dominated sort
# --------------------------------------------------------------------------- #
def test_specialist_and_generalist_share_front_zero():
    # specialist: high binding, ~no learning. generalist: balanced. Neither
    # dominates the other -> both on the first Pareto front.
    specialist = objective_vector(
        {"aggregate_loss_ratio": 1.0},
        {
            "ind_max_accuracy": 0.0,
            "binds_per_probe": {"p": True},
            "relative_recall_per_probe": {"p": 1.0},
        },
    )
    generalist = objective_vector(
        {"aggregate_loss_ratio": 30.0},
        {
            "ind_max_accuracy": 0.4,
            "binds_per_probe": {"p": True},
            "relative_recall_per_probe": {"p": 0.4},
        },
    )
    fronts = non_dominated_sort([specialist, generalist])
    assert fronts == [0, 0]
    assert set(pareto_front_indices([specialist, generalist])) == {0, 1}


def test_dominated_candidate_on_later_front():
    strong = objective_vector(
        {"aggregate_loss_ratio": 100.0},
        {
            "ind_max_accuracy": 0.9,
            "binds_per_probe": {"p": True},
            "relative_recall_per_probe": {"p": 0.9},
        },
    )
    weak = objective_vector(
        {"aggregate_loss_ratio": 1.0}, {"ind_max_accuracy": 0.0}
    )
    fronts = non_dominated_sort([strong, weak])
    assert fronts[0] == 0 and fronts[1] == 1


def test_efficiency_objective_prefers_smaller_model():
    big = objective_vector({"aggregate_loss_ratio": 10.0}, {}, param_count=100_000)
    small = objective_vector({"aggregate_loss_ratio": 10.0}, {}, param_count=1_000)
    # identical except params -> smaller dominates -> small front 0, big front 1
    fronts = non_dominated_sort([big, small])
    assert fronts == [1, 0]


# --------------------------------------------------------------------------- #
# WS-4 acceptance: specialist + generalist both promoted in the same cycle
# --------------------------------------------------------------------------- #
def _front_streak_entry(
    tmp_path: Path,
    pid: str,
    *,
    on_front: bool,
    cycles: int = 2,
    composite: float = 0.3,
    include_pareto_vector: bool = True,
):
    # composite default 0.3 = below the 0.6 scalar bar (so only Pareto can
    # promote) but above the 0.20 reject floor (so it isn't auto-rejected).
    ledger = Ledger(tmp_path / f"{pid}.jsonl")
    metadata = {
        "on_pareto_front": on_front,
        "paired_delta_ci_excludes_zero": True,
        "paired_delta_ci_low": 0.05,
    }
    if include_pareto_vector:
        metadata["pareto_objective_vector"] = {"binding": 1.0, "learning": 0.1}
    for cycle in range(1, cycles + 1):
        ledger.record_grade(
            proposal_id=pid,
            name=pid,
            category="lane",
            synthesis_kind="novel_hybrid",
            cycle=cycle,
            composite_score=composite,
            smoke_pass=True,
            learned_signal=False,
            metadata=metadata,
        )
    return ledger.entries[pid]


def test_pareto_promotes_specialist_and_generalist_same_cycle(tmp_path: Path):
    rules = PromotionRules(promote_by_pareto=True)
    specialist = _front_streak_entry(tmp_path, "spec", on_front=True)
    generalist = _front_streak_entry(tmp_path, "gen", on_front=True)
    assert decide_promotion(specialist, rules).decision == PROMOTION_PROMOTED
    assert decide_promotion(generalist, rules).decision == PROMOTION_PROMOTED
    assert "niche specialist" in decide_promotion(specialist, rules).reason


def test_pareto_promotion_requires_objective_vector(tmp_path: Path):
    rules = PromotionRules(promote_by_pareto=True)
    entry = _front_streak_entry(
        tmp_path, "missing", on_front=True, include_pareto_vector=False
    )
    decision = decide_promotion(entry, rules)
    assert decision.decision == PROMOTION_PENDING
    assert "pareto_objective_vector" in decision.reason


def test_pareto_off_by_default_keeps_low_composite_pending(tmp_path: Path):
    # Same sub-bar entry, default rules (pareto off) -> not promoted on composite.
    entry = _front_streak_entry(tmp_path, "x", on_front=True)
    assert decide_promotion(entry, PromotionRules()).decision == PROMOTION_PENDING


def _survivor(pid: str, *, agg_loss: float, cap: dict) -> dict:
    return {
        "proposal_id": pid,
        "name": pid,
        "category": "lane",
        "synthesis_kind": "novel_hybrid",
        "composite_score": 0.3,
        "smoke_pass": True,
        "learned_signal": False,
        "probe": {"aggregate_loss_ratio": agg_loss, "per_task": {}},
        "capability": cap,
        "metadata": {"math_axes": {}},
    }


def test_annotate_and_finalize_wires_niche_metadata(tmp_path: Path):
    survivors = [
        # specialist: strong binder, ~no learning
        _survivor(
            "spec",
            agg_loss=1.0,
            cap={
                "binds_per_probe": {"p": True},
                "relative_recall_per_probe": {"p": 1.0},
                "ind_max_accuracy": 0.0,
            },
        ),
        # generalist: strong learner + induction, no binding
        _survivor("gen", agg_loss=50.0, cap={"ind_max_accuracy": 0.6}),
    ]
    ledger = Ledger(tmp_path / "l.jsonl")
    annotate_niche_metadata(survivors, ledger)
    for surv in survivors:
        md = surv["metadata"]
        assert "behavior_fingerprint" in md
        assert "novelty_distance" in md
        assert "pareto_objective_vector" in md
        assert "on_pareto_front" in md
    # neither dominates the other -> both on the front
    assert all(s["metadata"]["on_pareto_front"] for s in survivors)

    finalize_survivors(survivors, ledger, cycle=1, niche_promotion=True)
    entry = ledger.entries["spec"]
    assert entry.metadata_history[-1]["on_pareto_front"] is True


def test_front_member_shielded_from_reject(tmp_path: Path):
    rules = PromotionRules(promote_by_pareto=True)
    # 4 sub-reject-floor cycles (composite 0.1 <= 0.20) would normally reject;
    # being on the front shields it (and the pareto streak promotes it).
    on = _front_streak_entry(
        tmp_path, "on", on_front=True, cycles=4, composite=0.1
    )
    assert decide_promotion(on, rules).decision == PROMOTION_PROMOTED
    off = _front_streak_entry(
        tmp_path, "off", on_front=False, cycles=4, composite=0.1
    )
    assert decide_promotion(off, rules).decision == PROMOTION_REJECTED
