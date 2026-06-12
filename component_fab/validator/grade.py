"""Shared candidate-grading chain: generate -> capability gates -> solo -> probe.

``run_autonomous``, ``run_invention`` and ``run_fidelity`` each hand-rolled the
same ``generate_module_from_spec -> validate_capabilities -> validate_solo ->
validate_in_context`` sequence with slightly different switches. This module
owns the chain once; ``GradeBundle`` carries the raw scorecards and each
caller keeps its own score assembly.

The autonomous loop's behavior is preserved bit-identically: same validator
call order, same arguments, scorecard persistence in the same position.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from torch import nn

from component_fab.generator.code_generator import (
    generate_module,
    generate_module_from_spec,
)
from component_fab.proposer.spec_generator import ProposalSpec
from component_fab.validator.capability import (
    capability_scorecard_to_dict,
    validate_capabilities,
)
from component_fab.validator.in_context import InContextScorecard, validate_in_context
from component_fab.validator.solo import (
    SoloScorecard,
    append_scorecard,
    validate_solo,
)


@dataclass(frozen=True, slots=True)
class GradeBundle:
    """Raw outputs of one grading pass; score assembly stays with the caller."""

    capability: dict[str, Any]
    eliminated_by: str | None
    solo: SoloScorecard | None
    in_context: InContextScorecard | None


def eliminated_solo_scorecard(spec: ProposalSpec, eliminated_by: str) -> SoloScorecard:
    """Placeholder solo card for a gate-eliminated spec (solo/probe skipped)."""
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
            "eliminated_by": eliminated_by,
        },
        metrics={"skipped": f"eliminated_by_{eliminated_by}"},
        property_cross_check={},
        promoted=False,
    )


def grade_candidate(
    spec: ProposalSpec,
    *,
    dim: int,
    seq_len: int,
    n_steps: int,
    run_range_probe: bool = False,
    range_train_steps: int = 300,
    run_solo: bool = True,
    persist_solo_scorecard: bool = False,
    run_in_context: bool = True,
    in_context_requires_promotion: bool = True,
    halt_on_elimination: bool = True,
) -> GradeBundle:
    """Run the shared validator chain for one spec.

    - ``persist_solo_scorecard``: append the solo card to the proposals
      catalog (autonomous-loop behavior).
    - ``in_context_requires_promotion``: only probe solo-promoted modules.
    - ``halt_on_elimination``: skip solo + probe when a capability gate
      eliminates the spec (the fidelity ladder grades through regardless).

    Raises ``UndispatchableSpecError`` from the generator unchanged.
    """
    module = generate_module_from_spec(spec, dim=dim)
    capability = validate_capabilities(
        spec,
        module,
        dim=dim,
        seq_len=seq_len,
        run_range_probe=run_range_probe,
        range_train_steps=range_train_steps,
    )
    capability_dict = capability_scorecard_to_dict(capability)
    if capability.eliminated_by is not None and halt_on_elimination:
        return GradeBundle(
            capability=capability_dict,
            eliminated_by=capability.eliminated_by,
            solo=None,
            in_context=None,
        )

    solo: SoloScorecard | None = None
    if run_solo:
        solo = validate_solo(spec, module, dim=dim, seq_len=seq_len)
        if persist_solo_scorecard:
            append_scorecard(solo)

    in_context: InContextScorecard | None = None
    probe_allowed = (not in_context_requires_promotion) or bool(solo and solo.promoted)
    if run_in_context and probe_allowed:
        in_context = validate_in_context(
            spec,
            module,
            dim=dim,
            seq_len=seq_len,
            n_steps=n_steps,
        )
    return GradeBundle(
        capability=capability_dict,
        eliminated_by=capability.eliminated_by,
        solo=solo,
        in_context=in_context,
    )


def factory_from_spec(
    spec: ProposalSpec, *, top_k_frac: float = 0.25
) -> Callable[[int], nn.Module]:
    """Lane factory producing a fresh module from this spec at any dim.

    The fab dispatcher reads ``spec.math_axes`` only, so that is all we pass.
    """
    axes = dict(spec.math_axes)

    def factory(dim: int) -> nn.Module:
        return generate_module(axes, dim=dim, top_k_frac=top_k_frac)

    return factory
