"""Tests for the deep-probe tier (selection + frontier-beater promotion).

No training is launched: ``run_deep_probe`` takes an injected ``cohort_runner``
so the Tier-2 engine is faked and the orchestration logic is exercised alone.
"""

from __future__ import annotations

from pathlib import Path

from component_fab.improver.deep_probe import (
    DeepProbeCandidate,
    run_deep_probe,
    select_top_k,
)
from component_fab.state.ledger import (
    PROMOTION_PROMOTED,
    Ledger,
)


def _ledger(tmp_path: Path) -> Ledger:
    return Ledger(tmp_path / "ledger.jsonl")


def _grade(
    ledger: Ledger, pid: str, scores: list[float], *, smoke: bool = True
) -> None:
    for cycle, score in enumerate(scores):
        ledger.record_grade(
            pid,
            name=pid,
            category="lane",
            synthesis_kind="test",
            cycle=cycle,
            composite_score=score,
            smoke_pass=smoke,
            learned_signal=False,
        )


def test_select_top_k_ranks_by_recent_mean(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    _grade(ledger, "a", [0.1, 0.9, 0.9])  # recent mean (window 2) = 0.9
    _grade(ledger, "b", [0.9, 0.5, 0.5])  # recent mean (window 2) = 0.5
    _grade(ledger, "c", [0.2, 0.2, 0.2])  # recent mean (window 2) = 0.2

    top = select_top_k(ledger, k=2, window=2)
    assert [c.proposal_id for c in top] == ["a", "b"]
    assert top[0].mean_composite == 0.9


def test_select_top_k_drops_no_smoke_and_no_history(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    _grade(ledger, "good", [0.6, 0.6])
    _grade(ledger, "nosmoke", [0.95, 0.95], smoke=False)  # never passed smoke

    ids = {c.proposal_id for c in select_top_k(ledger, k=10)}
    assert ids == {"good"}


def test_select_top_k_filters_by_status(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    _grade(ledger, "p", [0.7, 0.7])
    _grade(ledger, "q", [0.8, 0.8])
    ledger.record_promotion("p", PROMOTION_PROMOTED)

    only_promoted = select_top_k(ledger, k=10, statuses=frozenset({PROMOTION_PROMOTED}))
    assert [c.proposal_id for c in only_promoted] == ["p"]


def _fake_cohort(passing_ids: set[str], deltas: dict[str, float]):
    """A run_cohort stand-in: ``passing_ids`` beat frontier; deltas drive ordering."""

    def runner(proposal_ids, **_kwargs):
        results = {}
        for pid in proposal_ids:
            delta = deltas.get(pid, 0.0)
            results[pid] = {
                "status": "ok",
                "name": pid,
                "per_task": {
                    "compositional_binding": {
                        "delta": delta,
                        "beats": pid in passing_ids,
                    },
                    "long_gap_recall": {"delta": delta, "beats": pid in passing_ids},
                },
                "pass_count": 2 if pid in passing_ids else 0,
                "n_tasks": 2,
                "tier2_passed": pid in passing_ids,
            }
        return {"results": results, "survivors": sorted(passing_ids)}

    return runner


def test_run_deep_probe_promotes_only_frontier_beaters(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    _grade(ledger, "winner", [0.5, 0.5])
    _grade(ledger, "loser", [0.6, 0.6])  # higher nano score, still selected

    report = run_deep_probe(
        ledger,
        top_k=5,
        promote=True,
        cohort_runner=_fake_cohort({"winner"}, {"winner": 0.05, "loser": -0.02}),
    )

    assert report["n_selected"] == 2
    assert report["n_beats_frontier"] == 1
    assert report["promoted"] == ["winner"]
    # The frontier-beater is promoted even though its nano composite was LOWER.
    assert ledger.entries["winner"].promotion_status == PROMOTION_PROMOTED
    # The loser is left untouched (above-random is a signal, not a reject).
    assert ledger.entries["loser"].promotion_status != PROMOTION_PROMOTED
    # Outcomes are sorted beats-first.
    assert report["outcomes"][0]["proposal_id"] == "winner"


def test_run_deep_probe_dry_run_does_not_mutate_ledger(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    _grade(ledger, "winner", [0.5, 0.5])

    report = run_deep_probe(
        ledger,
        top_k=5,
        promote=False,
        cohort_runner=_fake_cohort({"winner"}, {"winner": 0.05}),
    )

    assert report["n_beats_frontier"] == 1
    assert report["n_promoted"] == 0
    assert ledger.entries["winner"].promotion_status != PROMOTION_PROMOTED


def test_run_deep_probe_empty_selection(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    report = run_deep_probe(ledger, top_k=5, cohort_runner=_fake_cohort(set(), {}))
    assert report["n_selected"] == 0
    assert report["outcomes"] == []
    assert report["baseline_names"]  # frontier names still reported


def test_run_deep_probe_handles_failed_cohort_row(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    _grade(ledger, "boom", [0.7, 0.7])

    def runner(proposal_ids, **_kwargs):
        return {"results": {"boom": {"status": "failed: kaboom"}}}

    report = run_deep_probe(ledger, top_k=5, promote=True, cohort_runner=runner)
    assert report["n_beats_frontier"] == 0
    assert report["outcomes"][0]["status"] == "failed: kaboom"
    assert report["n_promoted"] == 0


def test_candidate_dataclass_is_frozen() -> None:
    cand = DeepProbeCandidate("x", "x", 0.5, 2, "pending")
    try:
        cand.mean_composite = 0.9  # type: ignore[misc]
    except AttributeError:
        return
    raise AssertionError("DeepProbeCandidate should be frozen")
