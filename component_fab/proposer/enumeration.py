"""Single shared per-cycle spec enumeration.

``tools/run_autonomous._all_specs_for_cycle`` and the drifted mirror in
``tools/run_lm_probe._all_specs`` both lived off-module; this is the one
real enumerator. The autonomous loop uses the defaults it always had
(frontier cores + NAS topologies included, static variants unless the
promoted-anchor pool is active); ``run_lm_probe`` opts into its narrower
historical view (static + adaptive + ledger specs, no frontier/NAS).
"""

from __future__ import annotations

from typing import Mapping, Sequence

from component_fab.improver.adaptive import (
    adaptive_axis_variants,
    adaptive_cross_anchor_variants,
    build_anchor_pool,
)
from component_fab.improver.axis_variants import (
    anchor_axes_for_op,
    enumerate_axis_variants,
)
from component_fab.improver.cross_anchor import (
    enumerate_cross_anchor_variants,
    enumerate_frontier_core_specs,
    enumerate_frontier_hybrids,
)
from component_fab.improver.math_knob_catalog import (
    enumerate_adaptive_math_knob_compositions,
)
from component_fab.proposer.dynamic import (
    enumerate_dynamic_proposals,
    specs_from_ledger_entries,
)
from component_fab.proposer.name_free import enumerate_name_free_physics_experiments
from component_fab.proposer.nas_bridge import nas_graph_specs
from component_fab.proposer.spec_generator import (
    ProposalSpec,
    build_spec_from_axes,
    dedupe_specs_by_axes,
)
from component_fab.proposer.tier2_feedback import Tier2Feedback
from component_fab.state.ledger import Ledger
from research.synthesis.training_regime_grammar import (
    TrainingRegimeSpec,
    implemented_training_regimes,
    training_regime_to_axes,
)

DEFAULT_TRAINING_REGIME_NAMES = ("embed_warm_then_all", "body_warm_then_all")


def _training_regime_rationale(regime: TrainingRegimeSpec) -> str:
    stages = " -> ".join(
        f"{stage.target}:{stage.steps}@{stage.lr_scale:g}x" for stage in regime.stages
    )
    return (
        f"Train the same architecture under staged regime {regime.name} "
        f"({stages}) and grade capability at matched budget."
    )


def enumerate_training_regime_variants(
    anchor_op_names: Sequence[str],
    *,
    regime_names: Sequence[str] = DEFAULT_TRAINING_REGIME_NAMES,
    max_specs: int = 24,
) -> list[ProposalSpec]:
    """Attach staged-training genotypes to existing anchor specs.

    These specs do not change the generated module. They make the training
    regime an explicit proposal axis so a run ledger can compare embedding or
    body warmup against the same architecture trained normally.
    """

    if max_specs <= 0:
        return []
    regimes = implemented_training_regimes()
    out: list[ProposalSpec] = []
    for regime_name in regime_names:
        if regime_name not in regimes:
            raise ValueError(
                f"unknown training regime {regime_name!r}; valid={list(regimes)}"
            )
    for name in anchor_op_names:
        anchor = anchor_axes_for_op(name)
        if anchor is None:
            continue
        for regime_name in regime_names:
            regime = regimes[regime_name]
            axes = {**anchor.axes, **training_regime_to_axes(regime)}
            rationale = _training_regime_rationale(regime)
            out.append(
                build_spec_from_axes(
                    f"train_{anchor.op_name}_{regime.name}",
                    axes,
                    witness_ops=(anchor.op_name,),
                    anchor_axes=anchor.axes,
                    notes=(
                        f"anchor={anchor.op_name} "
                        f"(pass_rate={anchor.pass_rate:.2f} on "
                        f"{anchor.eval_count} evals)",
                        rationale,
                    ),
                    fingerprint_dispatched_axes=True,
                    rationale=rationale,
                )
            )
            if len(out) >= max_specs:
                return out
    return out


def enumerate_cycle_specs(
    ledger: Ledger,
    anchors: Sequence[str],
    *,
    cycle: int = 0,
    dim: int = 32,
    use_promoted_as_anchors: bool = False,
    include_static_variants: bool | None = None,
    include_frontier: bool = True,
    include_nas: bool = True,
    include_ledger_specs: bool = False,
    max_cross_pairs: int = 30,
    max_knob_specs: int = 48,
    max_dynamic_specs: int = 32,
    max_nas_specs: int = 6,
    max_name_free_specs: int = 12,
    max_training_specs: int = 24,
    include_training_regimes: bool = True,
    include_name_free_physics: bool = True,
    nas_archive_guided: bool = False,
    tier2_feedback_by_id: Mapping[str, Tier2Feedback] | None = None,
) -> list[ProposalSpec]:
    """Enumerate every spec a cycle considers, deduped by merged axes.

    ``include_static_variants`` defaults to ``not use_promoted_as_anchors``
    (the autonomous loop's either/or); pass ``True`` alongside the promoted
    pool to get both families (the lm-probe re-enumeration view).
    """
    if include_static_variants is None:
        include_static_variants = not use_promoted_as_anchors
    anchor_list = list(anchors)

    knob_specs = enumerate_adaptive_math_knob_compositions(
        anchor_list,
        ledger,
        max_specs=max_knob_specs,
    )
    dynamic_specs = enumerate_dynamic_proposals(
        anchor_list,
        ledger,
        max_specs=max_dynamic_specs,
        tier2_feedback_by_id=tier2_feedback_by_id,
    )
    name_free_specs: list[ProposalSpec] = []
    if include_name_free_physics:
        name_free_specs = enumerate_name_free_physics_experiments(
            ledger,
            cycle=cycle,
            dim=dim,
            max_specs=max_name_free_specs,
        )
    # "Frontier + delta": grade the proven binder cores standalone, and graft
    # each novel anchor's mechanism (state/memory/sparsity) onto every core.
    # This is the only path that starts from a frontier-tied binder, which is a
    # prerequisite for matching/beating current architectures.
    frontier_specs: list[ProposalSpec] = []
    if include_frontier:
        frontier_specs = enumerate_frontier_core_specs() + enumerate_frontier_hybrids(
            anchor_list
        )
    # Novel NAS topologies: genuinely new op-DAG structures (split/fuse/route/
    # recurse) that fab's fixed templates cannot express, compiled into gradeable
    # lanes. seed varies by cycle so each cycle samples different structures.
    nas_specs: list[ProposalSpec] = []
    if include_nas:
        nas_specs = nas_graph_specs(
            n_fresh=max_nas_specs,
            dim=dim,
            seed=cycle,
            archive_guided=nas_archive_guided,
        )
    training_specs: list[ProposalSpec] = []
    if include_training_regimes:
        training_specs = enumerate_training_regime_variants(
            anchor_list,
            max_specs=max_training_specs,
        )

    static_axis_specs: list[ProposalSpec] = []
    static_cross_specs: list[ProposalSpec] = []
    if include_static_variants:
        static_axis_specs = enumerate_axis_variants(anchor_list)
        static_cross_specs = enumerate_cross_anchor_variants(anchor_list)

    adaptive_axis_specs: list[ProposalSpec] = []
    adaptive_cross_specs: list[ProposalSpec] = []
    if use_promoted_as_anchors:
        anchor_pool = build_anchor_pool(
            anchor_list,
            ledger,
            use_promoted_as_anchors=True,
        )
        adaptive_axis_specs = adaptive_axis_variants(anchor_pool, ledger)
        adaptive_cross_specs = adaptive_cross_anchor_variants(
            anchor_pool,
            ledger,
            max_pairs=max_cross_pairs,
            seed=cycle,
        )

    ledger_specs = specs_from_ledger_entries(ledger) if include_ledger_specs else []

    return dedupe_specs_by_axes(
        static_axis_specs
        + static_cross_specs
        + adaptive_axis_specs
        + adaptive_cross_specs
        + knob_specs
        + dynamic_specs
        + name_free_specs
        + frontier_specs
        + nas_specs
        + training_specs
        + ledger_specs
    )
