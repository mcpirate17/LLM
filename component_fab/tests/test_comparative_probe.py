"""Tests for the label-free comparative binding probe (measure, not predict)."""

from __future__ import annotations

from component_fab.proposer.comparative_probe import ComparativeProbe, _unavailable


def _probe(margin: float, beats: int, n: int) -> ComparativeProbe:
    return ComparativeProbe(
        proposal_id="p",
        available=True,
        baseline_name="softmax_attention",
        margin_mean=margin,
        beats_count=beats,
        n_tasks=n,
    )


def test_beats_baseline_requires_positive_margin_and_majority() -> None:
    assert _probe(0.1, 2, 2).beats_baseline is True  # net positive + 2/2
    assert _probe(0.1, 1, 2).beats_baseline is False  # not a majority (1/2)
    assert _probe(-0.01, 2, 2).beats_baseline is False  # negative net margin
    assert _probe(0.05, 2, 3).beats_baseline is True  # 2*2>3 majority + positive
    assert _probe(0.05, 1, 3).beats_baseline is False  # 1/3 not a majority


def test_unavailable_fails_closed_on_beats() -> None:
    cp = _unavailable("p", "softmax_attention", "boom")
    assert cp.available is False
    assert cp.beats_baseline is False  # no measurement => cannot claim a win
    assert cp.reason == "boom"
