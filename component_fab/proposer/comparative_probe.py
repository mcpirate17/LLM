"""Label-free comparative binding probe — MEASURE 'beats baseline', don't predict it.

Training a model on past Tier-2 outcomes to predict new ones is reading the answer
key: it can only recognize architectures resembling labeled ones, so it is blind to
the novel winner the search exists to find (the label-fit GBM scored the STDP winner
0.026; the label-free measured probe put it at the 99.8th pctile). And a 2026-06-03
audit showed NO init-time signal — label-free capability_score included — ranks
'beats baseline' within a homogeneous architecture family.

So 'beats baseline' is MEASURED here, not predicted: run the candidate AND a
baseline through the same short synthetic binding probe and compare. No historical
labels, no fitting — works on any never-seen architecture. It is a cheap mini-Tier-2
(fewer tasks/steps), used as a screen ahead of the full cohort.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from component_fab.generator.code_generator import generate_module_from_spec
from component_fab.proposer.spec_generator import ProposalSpec

logger = logging.getLogger(__name__)

# The two niche tasks that discriminate real binders (long-gap memory +
# compositional structure); a candidate must beat the baseline on these to matter.
DEFAULT_PROBE_TASKS: tuple[str, ...] = ("long_gap_recall", "compositional_binding")


@dataclass(frozen=True, slots=True)
class ComparativeProbe:
    """Measured candidate-vs-baseline binding margin (label-free)."""

    proposal_id: str
    available: bool
    baseline_name: str
    margin_mean: float
    beats_count: int
    n_tasks: int
    per_task: dict[str, dict[str, float]] = field(default_factory=dict)
    reason: str = ""

    @property
    def beats_baseline(self) -> bool:
        """Beat the baseline on net (positive mean margin) AND on >half the tasks."""

        return self.margin_mean > 0.0 and self.beats_count * 2 > self.n_tasks


def comparative_binding_screen(
    spec: ProposalSpec,
    *,
    baseline_name: str = "softmax_attention",
    dim: int = 32,
    n_blocks: int = 2,
    n_train_steps: int = 60,
    task_names: Sequence[str] = DEFAULT_PROBE_TASKS,
    seed: int = 0,
) -> ComparativeProbe:
    """Measure how much ``spec`` beats ``baseline_name`` on a cheap binding probe."""

    try:
        from component_fab.harness.harder_binding_tasks import (
            default_hard_binding_tasks,
            run_harder_binding_suite,
        )

        wanted = set(task_names)
        tasks = tuple(
            t for t in default_hard_binding_tasks(seed=seed) if t.name in wanted
        )
        if not tasks:
            return _unavailable(spec.proposal_id, baseline_name, "no matching tasks")

        def candidate_factory(d: int, _spec: ProposalSpec = spec) -> Any:
            return generate_module_from_spec(_spec, dim=d)

        suite = run_harder_binding_suite(
            candidate_factory,
            spec.name,
            tasks=tasks,
            dim=dim,
            n_blocks=n_blocks,
            n_train_steps=n_train_steps,
            baseline_names=(baseline_name,),
            seed=seed,
        )
        per_task: dict[str, dict[str, float]] = {}
        margins: list[float] = []
        beats = 0
        for task_name, rows in suite.items():
            if not rows:
                continue
            cand = rows[0].eval_accuracy
            base = max((r.eval_accuracy for r in rows[1:]), default=0.0)
            margin = float(cand - base)
            per_task[task_name] = {
                "candidate_acc": float(cand),
                "baseline_acc": float(base),
                "margin": margin,
            }
            margins.append(margin)
            beats += int(margin > 0.0)
        if not margins:
            return _unavailable(
                spec.proposal_id, baseline_name, "probe produced no tasks"
            )
        return ComparativeProbe(
            proposal_id=spec.proposal_id,
            available=True,
            baseline_name=baseline_name,
            margin_mean=sum(margins) / len(margins),
            beats_count=beats,
            n_tasks=len(margins),
            per_task=per_task,
        )
    except Exception as exc:  # noqa: BLE001 - screen is best-effort, fail open
        logger.warning(
            "comparative probe unavailable for %s: %s", spec.proposal_id, exc
        )
        return _unavailable(spec.proposal_id, baseline_name, str(exc))


def _unavailable(proposal_id: str, baseline_name: str, reason: str) -> ComparativeProbe:
    return ComparativeProbe(
        proposal_id=proposal_id,
        available=False,
        baseline_name=baseline_name,
        margin_mean=0.0,
        beats_count=0,
        n_tasks=0,
        reason=reason,
    )
