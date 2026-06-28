"""Tests for opt-in orthogonality-aware re-grade selection.

Counters the composite-selection pathology: genuinely-novel candidates (high
orthogonality, mid composite) get starved out of the grading budget by
high-composite recombinations, so they never re-grade / accumulate paired-CI.
``--regrade-top-orthogonality K`` force-includes them. Default 0 = unchanged.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from component_fab.state.ledger import Ledger
from component_fab.runner.selection import (
    inject_novelty_regrades as _inject_novelty_regrades,
    top_orthogonality_pending as _top_orthogonality_pending,
)


def _spec(pid: str) -> SimpleNamespace:
    # _top_orthogonality_pending / _inject_novelty_regrades only read proposal_id.
    return SimpleNamespace(proposal_id=pid)


def _record(ledger: Ledger, pid: str, *, on_front: bool, orths: list[float]) -> None:
    for cycle, orth in enumerate(orths, start=1):
        ledger.record_grade(
            proposal_id=pid,
            name=pid,
            category="lane",
            synthesis_kind="invent",
            cycle=cycle,
            composite_score=0.6,
            smoke_pass=True,
            learned_signal=False,
            metadata={"on_pareto_front": on_front, "orthogonality_radius": orth},
        )


def test_top_orthogonality_uses_peak_and_requires_front(tmp_path: Path):
    ledger = Ledger(tmp_path / "l.jsonl")
    # novel: high peak at first sighting, decayed to 0 latest (it's now in catalog).
    _record(ledger, "novel", on_front=True, orths=[42.0, 0.0])
    _record(ledger, "recomb", on_front=True, orths=[1.0, 1.0])
    # high orthogonality but NOT on the front -> excluded.
    _record(ledger, "offfront", on_front=False, orths=[50.0])
    pool = [_spec("novel"), _spec("recomb"), _spec("offfront")]
    top = _top_orthogonality_pending(pool, ledger, 2)
    ids = [s.proposal_id for s in top]
    assert ids[0] == "novel"  # peak 42 wins despite latest 0 (decay)
    assert "offfront" not in ids  # off-front excluded regardless of orthogonality


def test_inject_prepends_novel_within_budget(tmp_path: Path):
    ledger = Ledger(tmp_path / "l.jsonl")
    _record(ledger, "novel", on_front=True, orths=[42.0, 0.0])
    pool = [_spec("novel")]
    # composite-ordered budget of 3 that excludes the novel candidate
    active = [_spec("a"), _spec("b"), _spec("c")]
    out = _inject_novelty_regrades(active, pool, ledger, k=1, budget=3)
    ids = [s.proposal_id for s in out]
    assert ids[0] == "novel"  # prepended
    assert len(out) == 3 and "c" not in ids  # trimmed back to budget


def test_inject_off_is_noop(tmp_path: Path):
    ledger = Ledger(tmp_path / "l.jsonl")
    active = [_spec("a"), _spec("b")]
    # k=0 must return the same list object, unchanged (default behavior).
    assert _inject_novelty_regrades(active, [], ledger, k=0, budget=0) is active


def test_inject_skips_already_selected(tmp_path: Path):
    ledger = Ledger(tmp_path / "l.jsonl")
    _record(ledger, "novel", on_front=True, orths=[42.0])
    pool = [_spec("novel")]
    active = [_spec("novel"), _spec("a")]  # novel already in the graded set
    out = _inject_novelty_regrades(active, pool, ledger, k=1, budget=2)
    assert [s.proposal_id for s in out] == ["novel", "a"]  # no duplicate, unchanged
