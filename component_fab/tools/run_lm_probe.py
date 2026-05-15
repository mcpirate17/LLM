"""CLI: score a candidate fab lane against standard mixers on harder
discrete-binding probes.

Reconstructs the candidate's ``ProposalSpec`` by re-enumerating proposals
under the current anchor set (axes are not persisted in the ledger).
Then trains a ``TinyLM`` wrapper of the candidate AND each baseline
mixer at identical params/steps/optimizer on each task in
``default_hard_binding_tasks``, and reports a comparison table.

Usage:
    python -m component_fab.tools.run_lm_probe --proposal-id <id>
    python -m component_fab.tools.run_lm_probe --top-n-unique 7
    python -m component_fab.tools.run_lm_probe --proposal-id <id> \\
        --n-train-steps 1000 --seed 1 --output /tmp/probe.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from torch import nn

from component_fab.generator.code_generator import generate_module
from component_fab.harness.harder_binding_tasks import (
    HardBindingResult,
    HardBindingTask,
    default_hard_binding_tasks,
    run_harder_binding_suite,
)
from component_fab.harness.tiny_lm import DEFAULT_BASELINE_NAMES
from component_fab.improver.adaptive import (
    adaptive_axis_variants,
    adaptive_cross_anchor_variants,
    build_anchor_pool,
)
from component_fab.improver.axis_variants import enumerate_axis_variants
from component_fab.improver.cross_anchor import enumerate_cross_anchor_variants
from component_fab.improver.math_knob_catalog import (
    enumerate_adaptive_math_knob_compositions,
)
from component_fab.proposer.spec_generator import (
    ProposalSpec,
    axes_fingerprint,
    dedupe_specs_by_axes,
)
from component_fab.state.ledger import (
    DEFAULT_LEDGER_PATH,
    PROMOTION_PROMOTED,
    Ledger,
)


_DEFAULT_ANCHOR_OPS: tuple[str, ...] = (
    "tropical_attention",
    "tropical_router",
    "tropical_gate",
    "clifford_attention",
    "padic_gate",
    "spike_rate_code",
    "grade_mix",
    "ultrametric_attention",
)


def _all_specs(anchors: tuple[str, ...], ledger: Ledger) -> list[ProposalSpec]:
    """Re-enumerate every spec the autonomous loop would consider this cycle.

    Mirrors ``tools/run_autonomous._all_specs_for_cycle`` minus the cycle
    seed (deterministic enough for proposal_id matching since the digest
    is axes-keyed, not seed-keyed).
    """
    knob_specs = enumerate_adaptive_math_knob_compositions(
        list(anchors), ledger, max_specs=128
    )
    anchor_pool = build_anchor_pool(list(anchors), ledger, use_promoted_as_anchors=True)
    axis_specs = adaptive_axis_variants(anchor_pool, ledger)
    cross_specs = adaptive_cross_anchor_variants(
        anchor_pool, ledger, max_pairs=80, seed=0
    )
    static_axis_specs = enumerate_axis_variants(list(anchors))
    static_cross_specs = enumerate_cross_anchor_variants(list(anchors))
    return dedupe_specs_by_axes(
        static_axis_specs + static_cross_specs + axis_specs + cross_specs + knob_specs
    )


def _factory_from_spec(spec: ProposalSpec) -> Callable[[int], nn.Module]:
    """Return a lane factory that produces a fresh module from this spec.

    The fab dispatcher reads ``spec.math_axes`` only, so we pass that.
    """
    axes = dict(spec.math_axes)

    def factory(dim: int) -> nn.Module:
        return generate_module(axes, dim=dim, top_k_frac=0.25)

    return factory


def _resolve_spec_by_proposal_id(
    proposal_id: str, anchors: tuple[str, ...], ledger: Ledger
) -> ProposalSpec | None:
    specs = _all_specs(anchors, ledger)
    by_id = {s.proposal_id: s for s in specs}
    if proposal_id in by_id:
        return by_id[proposal_id]
    # Prefix match — useful when user passes a short id.
    candidates = [s for s in specs if s.proposal_id.startswith(proposal_id)]
    if len(candidates) == 1:
        return candidates[0]
    return None


def _top_unique_promoted_specs(
    n: int, anchors: tuple[str, ...], ledger: Ledger
) -> list[ProposalSpec]:
    """Top-N axes-unique promoted entries, sorted by mean composite score."""
    specs = _all_specs(anchors, ledger)
    by_fp: dict[str, ProposalSpec] = {axes_fingerprint(s.math_axes): s for s in specs}
    scored: list[tuple[float, str, ProposalSpec]] = []
    for fp, spec in by_fp.items():
        entry = ledger.entries.get(spec.proposal_id)
        if entry is None or entry.promotion_status != PROMOTION_PROMOTED:
            continue
        mean = (
            sum(entry.composite_history) / len(entry.composite_history)
            if entry.composite_history
            else 0.0
        )
        scored.append((mean, fp, spec))
    scored.sort(key=lambda r: -r[0])
    return [spec for _, _, spec in scored[:n]]


# ---------- Reporting ----------


def _format_result_table(results: dict[str, list[HardBindingResult]]) -> str:
    lines = [
        "task                       mixer                eval_acc  train_acc  chance  conv  n_params"
    ]
    lines.append("-" * 100)
    for task_name, rows in results.items():
        for r in rows:
            lines.append(
                f"{task_name:<26} {r.mixer_label:<20} "
                f"{r.eval_accuracy:8.3f}  {r.train_accuracy_final:9.3f}  "
                f"{r.chance_accuracy:6.3f}  {('Y' if r.converged else 'N'):>4}  "
                f"{r.n_params:>8d}"
            )
        lines.append("")
    return "\n".join(lines)


def _results_to_json(
    candidate_label: str, results: dict[str, list[HardBindingResult]]
) -> dict:
    return {
        "candidate": candidate_label,
        "tasks": {
            task: [asdict(row) for row in rows] for task, rows in results.items()
        },
    }


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a fab lane on harder binding")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--proposal-id", help="exact or prefix proposal_id from the ledger"
    )
    grp.add_argument(
        "--top-n-unique",
        type=int,
        help="run the top-N axes-unique promoted lanes",
    )
    parser.add_argument(
        "--anchors",
        nargs="*",
        default=list(_DEFAULT_ANCHOR_OPS),
        help="corpus anchor op names to re-enumerate against",
    )
    parser.add_argument(
        "--baseline-names",
        nargs="*",
        default=list(DEFAULT_BASELINE_NAMES),
    )
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER_PATH))
    parser.add_argument("--n-train-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--n-blocks", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", help="write JSON report to this path")
    return parser.parse_args(argv)


def _resolve_targets(args: argparse.Namespace, ledger: Ledger) -> list[ProposalSpec]:
    anchors = tuple(args.anchors)
    if args.proposal_id:
        spec = _resolve_spec_by_proposal_id(args.proposal_id, anchors, ledger)
        if spec is None:
            print(
                f"proposal_id '{args.proposal_id}' not found in re-enumeration",
                file=sys.stderr,
            )
            return []
        return [spec]
    return _top_unique_promoted_specs(int(args.top_n_unique), anchors, ledger)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    ledger = Ledger(args.ledger, include_rotated=True)
    targets = _resolve_targets(args, ledger)
    if not targets:
        print("no targets to run", file=sys.stderr)
        return 2

    tasks: tuple[HardBindingTask, ...] = default_hard_binding_tasks(seed=args.seed)
    out_payloads: list[dict] = []
    for spec in targets:
        candidate_factory = _factory_from_spec(spec)
        results = run_harder_binding_suite(
            candidate_factory,
            candidate_label=spec.name,
            tasks=tasks,
            baseline_names=tuple(args.baseline_names),
            dim=args.dim,
            n_blocks=args.n_blocks,
            n_train_steps=args.n_train_steps,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        print(f"\n=== {spec.name} (proposal_id={spec.proposal_id}) ===")
        print(_format_result_table(results))
        out_payloads.append(_results_to_json(spec.name, results))

    if args.output:
        Path(args.output).write_text(
            json.dumps(out_payloads, indent=2), encoding="utf-8"
        )
        print(f"\nwrote: {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
