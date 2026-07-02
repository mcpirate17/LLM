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

from component_fab.harness.harder_binding_tasks import (
    HardBindingResult,
    HardBindingTask,
    default_hard_binding_tasks,
    run_harder_binding_suite,
)
from component_fab.harness.tiny_lm import DEFAULT_BASELINE_NAMES
from component_fab.proposer.dynamic import spec_from_ledger_entry
from component_fab.proposer.enumeration import enumerate_cycle_specs
from component_fab.proposer.spec_generator import (
    ProposalSpec,
    axes_fingerprint,
)
from component_fab.state.ledger import (
    PROMOTION_PROMOTED,
    Ledger,
    resolve_proposal_id,
)
from component_fab.tools._cli import add_common_args, open_ledger
from component_fab.validator.grade import factory_from_spec


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
    """Re-enumerate the lm-probe view of the cycle spec space.

    Narrower than the autonomous loop on purpose: no frontier cores or NAS
    topologies (they are not reconstructable targets here), cycle seed 0
    (deterministic enough for proposal_id matching since the digest is
    axes-keyed, not seed-keyed), and ledger-persisted specs included.
    """
    return enumerate_cycle_specs(
        ledger,
        list(anchors),
        cycle=0,
        use_promoted_as_anchors=True,
        include_static_variants=True,
        include_frontier=False,
        include_nas=False,
        include_ledger_specs=True,
        max_cross_pairs=80,
        max_knob_specs=128,
        max_dynamic_specs=128,
        include_data_routes=True,
        max_data_route_specs=128,
    )


def _resolve_spec_by_proposal_id(
    proposal_id: str, anchors: tuple[str, ...], ledger: Ledger
) -> ProposalSpec | None:
    try:
        spec = spec_from_ledger_entry(resolve_proposal_id(ledger, proposal_id))
        if spec is not None:
            return spec
    except ValueError:
        pass  # not in the ledger (or ambiguous) — fall back to re-enumeration
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
    parser.add_argument("--n-train-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--n-blocks", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    add_common_args(parser, output_help="write JSON report to this path")
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
    ledger = open_ledger(args)
    targets = _resolve_targets(args, ledger)
    if not targets:
        print("no targets to run", file=sys.stderr)
        return 2

    tasks: tuple[HardBindingTask, ...] = default_hard_binding_tasks(seed=args.seed)
    out_payloads: list[dict] = []
    for spec in targets:
        candidate_factory = factory_from_spec(spec)
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
