"""CLI: invention-track fab loop.

The CLI owns argument parsing and report output. Grading, optional TinyLM binding,
ledger metadata assembly, and promotion policy live in
``component_fab.runner.invention``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from component_fab.inventor.mechanism_catalog import (
    enumerate_invention_specs,
    invention_gate_reasons,
)
from component_fab.proposer.spec_generator import spec_to_json
from component_fab.runner.invention import (
    apply_invention_promotions,
    grade_invention,
    record_invention_result,
)
from component_fab.state.ledger import DEFAULT_LEDGER_PATH
from component_fab.tools._cli import add_common_args, open_ledger, write_report

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_INVENTION_LEDGER = DEFAULT_LEDGER_PATH.with_name("invention_ledger.jsonl")
DEFAULT_REPORT = _REPO / "component_fab" / "catalog" / "invention_run_latest.json"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="component_fab invention loop")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-specs", type=int, default=4)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--probe-steps", type=int, default=60)
    parser.add_argument("--skip-in-context", action="store_true")
    parser.add_argument("--run-lm-binding", action="store_true")
    parser.add_argument("--binding-task-limit", type=int, default=2)
    parser.add_argument("--binding-steps", type=int, default=150)
    parser.add_argument("--binding-batch-size", type=int, default=16)
    parser.add_argument("--binding-dim", type=int, default=32)
    parser.add_argument("--binding-blocks", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-range-probe",
        dest="run_range_probe",
        action="store_false",
        help="Skip the distance-resolved sparse/long-range binding probe.",
    )
    parser.add_argument(
        "--range-train-steps",
        type=int,
        default=300,
        help="Train steps for the range probe.",
    )
    parser.add_argument(
        "--veto-range-blind",
        action="store_true",
        help="Block promotion of specs below --min-range-distance.",
    )
    parser.add_argument("--min-range-distance", type=int, default=1)
    add_common_args(
        parser,
        ledger_default=DEFAULT_INVENTION_LEDGER,
        output_default=DEFAULT_REPORT,
    )
    return parser.parse_args(argv)


def _dry_run_payload(active, blocked) -> dict:
    return {
        "active": [spec_to_json(spec) for spec in active],
        "blocked": [
            {"spec": spec_to_json(spec), "reasons": list(reasons)}
            for spec, reasons in blocked
        ],
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    specs = enumerate_invention_specs()[: max(0, args.max_specs)]
    gated = [(spec, invention_gate_reasons(spec)) for spec in specs]
    blocked = [(spec, reasons) for spec, reasons in gated if reasons]
    active = [spec for spec, reasons in gated if not reasons]
    if args.dry_run:
        print(json.dumps(_dry_run_payload(active, blocked), indent=2))
        return 0

    ledger = open_ledger(args)
    results = []
    for index, spec in enumerate(active, start=1):
        result = grade_invention(
            spec,
            dim=args.dim,
            seq_len=args.seq_len,
            probe_steps=args.probe_steps,
            skip_in_context=args.skip_in_context,
            run_lm_binding=args.run_lm_binding,
            binding_task_limit=args.binding_task_limit,
            binding_steps=args.binding_steps,
            binding_batch_size=args.binding_batch_size,
            binding_dim=args.binding_dim,
            binding_blocks=args.binding_blocks,
            seed=args.seed,
            run_range_probe=args.run_range_probe,
            range_train_steps=args.range_train_steps,
        )
        record_invention_result(ledger, result, cycle=index)
        results.append(result)

    promotion_counts = apply_invention_promotions(
        ledger,
        veto_range_blind=args.veto_range_blind,
        min_range_distance=args.min_range_distance,
    )
    write_report(
        {
            "track": "invention",
            "n_active": len(active),
            "n_blocked": len(blocked),
            "promotion_counts": promotion_counts,
            "results": results,
        },
        default_dir=DEFAULT_REPORT.parent,
        prefix="invention_run",
        output=args.output,
    )
    for result in results:
        spec = result["spec"]
        print(
            f"{spec['name']:<42} {result['status']:<10} "
            f"score={float(result.get('score') or 0.0):.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
