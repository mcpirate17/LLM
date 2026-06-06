"""Deep-probe tier: a longer-training Tier-2-vs-frontier bake-off on the top-K
nano survivors, used as the promotion criterion the saturated nano composite
cannot provide.

Why this tier exists
--------------------
Nano grading (``run_autonomous`` at dim=32, short in-context probes) compresses
the *discriminating* subscores — ``binding`` and ``learning`` — toward the floor
for **every** candidate, including the curated frontier cores. Binding capability
only separates after far more training steps than nano grading runs (~thousands,
not ~hundreds). So ranking by the nano composite cannot tell a genuine
frontier-beater apart from a plausible-looking non-learner, and a 200-step Tier-2
cohort still ties baselines (``fab_tier2_dynamic_top2`` → 0 survivors at 200
steps: candidate == baseline on ``compositional_binding``).

This tier closes the gap: it selects candidates by **relative** nano-composite
rank (never the absolute 0.60 promote bar, which saturates), trains each for many
more steps against the real GPT-2 / Mamba / Mamba2 frontier baselines, and reads
off which ones actually *beat frontier* on the niche-survival binding rule. That
verdict — not the nano composite — is the honest promotion signal for "this
component beats a known-good model".

Compute safety
--------------
The Tier-2 micro-models are tiny (~6K params) and the binding suite
(``harder_binding_tasks.run_harder_binding_suite``) is CPU-only — it has no
``.cuda()`` path — so this tier never competes with a live GPU training run. It
is opt-in (it never auto-fires inside the autonomous cycle) and ``promote``
defaults to off, so a bare run is a dry-run report.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import Any

from component_fab.state.ledger import (
    PROMOTION_PROMOTED,
    Ledger,
)

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DeepProbeCandidate:
    """A ledger entry selected for the deep probe, with its nano-rank score."""

    proposal_id: str
    name: str
    mean_composite: float
    n_cycles: int
    promotion_status: str


@dataclass(frozen=True, slots=True)
class DeepProbeOutcome:
    """Result of training one candidate to depth against the frontier baselines."""

    proposal_id: str
    name: str
    beats_frontier: bool
    pass_count: int
    n_tasks: int
    mean_delta_vs_frontier: float
    status: str  # "ok" | "spec_not_found" | "failed: ..."


def _recent_mean(history: list[float], window: int) -> float:
    if not history:
        return 0.0
    recent = history[-window:] if window > 0 else history
    return sum(recent) / len(recent)


def select_top_k(
    ledger: Ledger,
    *,
    k: int,
    window: int = 2,
    statuses: frozenset[str] | None = None,
    min_smoke_pass: int = 1,
) -> list[DeepProbeCandidate]:
    """Top-``k`` ledger entries by RELATIVE recent-mean nano composite.

    The deep probe selects purely on relative rank — never the absolute 0.60
    promote bar, which saturates at nano scale — so strong candidates flow to the
    deeper bake-off even when their nano composite is compressed. ``statuses``,
    when given, keeps only entries with a ``promotion_status`` in the set (e.g.
    ``{"promoted"}`` to re-examine the nano-promoted cohort, or ``None`` for any).
    ``min_smoke_pass`` drops entries that never passed a forward/backward smoke
    check (degenerate / eliminated-at-gate) and entries with no graded history.
    """
    cands: list[DeepProbeCandidate] = []
    for entry in ledger.all_entries():
        if statuses is not None and entry.promotion_status not in statuses:
            continue
        if entry.smoke_pass_count < min_smoke_pass:
            continue
        if not entry.composite_history:
            continue
        cands.append(
            DeepProbeCandidate(
                proposal_id=entry.proposal_id,
                name=entry.name,
                mean_composite=_recent_mean(entry.composite_history, window),
                n_cycles=len(entry.composite_history),
                promotion_status=entry.promotion_status,
            )
        )
    cands.sort(key=lambda c: c.mean_composite, reverse=True)
    return cands[: max(0, k)]


def _mean_delta(per_task: dict[str, Any]) -> float:
    """Mean ``candidate_acc - best_baseline_acc`` across the cohort's tasks."""
    deltas = [
        float(value.get("delta") or 0.0)
        for value in per_task.values()
        if isinstance(value, dict)
    ]
    return sum(deltas) / len(deltas) if deltas else 0.0


def _one_outcome(pid: str, fallback_name: str, res: dict[str, Any]) -> DeepProbeOutcome:
    """Reduce one ``run_cohort`` result row to a ``DeepProbeOutcome``."""
    status = str(res.get("status") or "missing")
    if status != "ok":
        return DeepProbeOutcome(
            proposal_id=pid,
            name=fallback_name,
            beats_frontier=False,
            pass_count=0,
            n_tasks=0,
            mean_delta_vs_frontier=0.0,
            status=status,
        )
    per_task = res.get("per_task") or {}
    return DeepProbeOutcome(
        proposal_id=pid,
        name=str(res.get("name") or fallback_name),
        beats_frontier=bool(res.get("tier2_passed")),
        pass_count=int(res.get("pass_count") or 0),
        n_tasks=int(res.get("n_tasks") or len(per_task)),
        mean_delta_vs_frontier=_mean_delta(per_task),
        status="ok",
    )


def _empty_report(
    n_train_steps: int, dim: int, n_blocks: int, baseline_names: tuple[str, ...]
) -> dict[str, Any]:
    return {
        "n_selected": 0,
        "n_beats_frontier": 0,
        "n_promoted": 0,
        "promoted": [],
        "n_train_steps": n_train_steps,
        "dim": dim,
        "n_blocks": n_blocks,
        "baseline_names": list(baseline_names),
        "selected": [],
        "outcomes": [],
        "cohort": {},
    }


def run_deep_probe(
    ledger: Ledger,
    *,
    top_k: int,
    n_train_steps: int = 2000,
    dim: int = 64,
    n_blocks: int = 2,
    seed: int = 0,
    seed_count: int = 1,
    window: int = 2,
    statuses: frozenset[str] | None = None,
    promote: bool = False,
    quiet: bool = False,
    cohort_runner: Any = None,
) -> dict[str, Any]:
    """Select the top-``top_k`` nano survivors and train them to depth vs frontier.

    Reuses ``run_tier2_binding_cohort.run_cohort`` (the Tier-2 engine) with the
    GPT-2 / Mamba / Mamba2 frontier baselines and a high ``n_train_steps``, so the
    binding signal has room to separate. A candidate that passes the cohort's
    niche-survival rule beat the *best of frontier* on the key binding tasks — the
    honest "beats a known-good model" verdict. When ``promote`` is set, those
    survivors are recorded as promoted in the ledger; losers are left untouched
    (above-random is a signal, not a reject — never auto-reject a deep-probe miss).

    ``cohort_runner`` is injected only by tests; production passes ``None`` and the
    real ``run_cohort`` is used.
    """
    baseline_names = _frontier_baseline_names()
    candidates = select_top_k(ledger, k=top_k, window=window, statuses=statuses)
    if not candidates:
        return _empty_report(n_train_steps, dim, n_blocks, baseline_names)

    if cohort_runner is None:
        from research.tools.run_tier2_binding_cohort import run_cohort as cohort_runner

    proposal_ids = [c.proposal_id for c in candidates]
    by_id = {c.proposal_id: c for c in candidates}
    if not quiet:
        _LOG.info(
            "deep_probe: %d candidates × %d frontier baselines × %d steps (dim=%d)",
            len(proposal_ids),
            len(baseline_names),
            n_train_steps,
            dim,
        )
    cohort = cohort_runner(
        proposal_ids,
        dim=dim,
        n_blocks=n_blocks,
        n_train_steps=n_train_steps,
        seed=seed,
        seed_count=seed_count,
        baseline_names=baseline_names,
        quiet=quiet,
    )

    results = cohort.get("results", {})
    outcomes = [
        _one_outcome(pid, by_id[pid].name, results.get(pid, {})) for pid in proposal_ids
    ]
    promoted: list[str] = []
    if promote:
        for outcome in outcomes:
            if outcome.beats_frontier:
                ledger.record_promotion(outcome.proposal_id, PROMOTION_PROMOTED)
                promoted.append(outcome.proposal_id)

    outcomes.sort(
        key=lambda o: (o.beats_frontier, o.mean_delta_vs_frontier), reverse=True
    )
    return {
        "n_selected": len(candidates),
        "n_beats_frontier": sum(1 for o in outcomes if o.beats_frontier),
        "n_promoted": len(promoted),
        "promoted": promoted,
        "n_train_steps": n_train_steps,
        "dim": dim,
        "n_blocks": n_blocks,
        "baseline_names": list(baseline_names),
        "selected": [dataclasses.asdict(c) for c in candidates],
        "outcomes": [dataclasses.asdict(o) for o in outcomes],
        "cohort": cohort,
    }


def _frontier_baseline_names() -> tuple[str, ...]:
    from component_fab.harness.tiny_lm import FRONTIER_BASELINE_NAMES

    return FRONTIER_BASELINE_NAMES
