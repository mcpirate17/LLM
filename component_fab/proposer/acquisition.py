"""Acquisition-driven proposal selection (WS-3).

Replaces the static ``max_cross_pairs`` / ``max_knob_specs`` enumeration caps with
a budget filled by the candidates the surrogate rates most promising. Scores every
enumerated spec with the surrogate's (median, upper-quantile) composite estimate and
ranks by an Upper Confidence Bound — ``median + beta * (upper - median)`` — so the
loop spends grading budget on optimistic-but-uncertain candidates (explore) and
high-mean ones (exploit), instead of a hand-tuned marginal-lift cap.

Pure ranking over already-enumerated ``ProposalSpec`` objects; it does not generate
candidates. Falls back to identity ordering (no surrogate) so callers degrade safely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from ..state.surrogate import MeanFieldApproximant, features_for_spec

if TYPE_CHECKING:
    from .spec_generator import ProposalSpec


@dataclass(frozen=True, slots=True)
class ScoredSpec:
    spec: "ProposalSpec"
    median: float
    upper: float
    ucb: float


def score_specs(
    specs: Sequence["ProposalSpec"],
    surrogate: MeanFieldApproximant,
    *,
    beta: float = 1.0,
) -> list[ScoredSpec]:
    """UCB-score each spec; higher ucb = grade sooner. Descending order."""
    scored: list[ScoredSpec] = []
    for spec in specs:
        median, upper = surrogate.predict(features_for_spec(spec))
        ucb = median + beta * max(0.0, upper - median)
        scored.append(ScoredSpec(spec=spec, median=median, upper=upper, ucb=ucb))
    scored.sort(key=lambda s: s.ucb, reverse=True)
    return scored


def select_by_acquisition(
    specs: Sequence["ProposalSpec"],
    surrogate: MeanFieldApproximant | None,
    *,
    budget: int,
    beta: float = 1.0,
) -> list["ProposalSpec"]:
    """Top-``budget`` specs by UCB. No surrogate or budget<=0 → identity prefix.

    A budget of 0 means "no cap" → return all specs unchanged.
    """
    if budget <= 0:
        return list(specs)
    if surrogate is None:
        return list(specs)[:budget]
    return [s.spec for s in score_specs(specs, surrogate, beta=beta)[:budget]]
