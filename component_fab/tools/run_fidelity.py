"""CLI: grade ledger candidates at two fidelity rungs and report the gap (WS-7).

For each candidate (reconstructed from the ledger), re-grade through the SAME
validator path at R0 (dim32/seq32/60 steps) and R1 (dim128/seq256/500 steps),
extract a shared metric set, append both to ``catalog/fidelity_scores.jsonl``, and
write ``catalog/fidelity_report.json`` with the per-metric R0->R1 Spearman. A metric
whose nano rank does not track R1 is flagged for weight demotion.

Compute-heavy (R1 trains real probes at dim128/500 steps), so default
``--max-candidates`` is small; run it repeatedly to accumulate the >= 30 paired
candidates the WS-7 acceptance wants. Un-dispatchable specs are skipped explicitly.

Usage:
    python -m component_fab.tools.run_fidelity [--max-candidates N] [--r1-steps N]
        [--store PATH] [--out PATH] [--quiet]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict

from component_fab.generator.code_generator import UndispatchableSpecError
from component_fab.improver.ranking import binding_subscore, learning_subscore
from component_fab.proposer.dynamic import spec_from_ledger_entry
from component_fab.state.fidelity import (
    DEFAULT_OUTPUT_PATH,
    DEFAULT_STORE_PATH,
    RungScore,
    append_rung_scores,
    compute_fidelity_report,
    read_rung_scores,
    write_fidelity_report,
)
from component_fab.state.ledger import Ledger
from component_fab.tools._cli import open_ledger
from component_fab.validator.grade import grade_candidate

LEDGER_PATH = DEFAULT_STORE_PATH.parent / "ledger.jsonl"

# Rung settings: R0 = current nano; R1 = dim128/seq256/500 steps.
_R0 = {"dim": 32, "seq_len": 32, "n_steps": 60}
_R1 = {"dim": 128, "seq_len": 256, "n_steps": 500}
_PRODUCED_RUNGS = ("R0", "R1")


def _grade_metrics(spec, *, dim: int, seq_len: int, n_steps: int) -> dict[str, float]:
    """Shared metric set from one validator pass at the given fidelity."""
    bundle = grade_candidate(
        spec,
        dim=dim,
        seq_len=seq_len,
        n_steps=n_steps,
        run_solo=False,
        in_context_requires_promotion=False,
        halt_on_elimination=False,
    )
    cap = bundle.capability
    probe = asdict(bundle.in_context)
    return {
        "binding": binding_subscore(cap),
        "learning": learning_subscore(probe),
        "induction": float(cap.get("ind_max_accuracy") or 0.0),
        "nb_max_accuracy": float(cap.get("nb_max_accuracy") or 0.0),
        "erf_density": float(cap.get("erf_density") or 0.0),
    }


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="component_fab fidelity ladder")
    parser.add_argument("--ledger", default=str(LEDGER_PATH), type=str)
    parser.add_argument("--store", default=str(DEFAULT_STORE_PATH), type=str)
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_PATH), type=str)
    parser.add_argument("--max-candidates", default=8, type=int)
    parser.add_argument("--r1-steps", default=_R1["n_steps"], type=int)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def _existing_rungs(records: list[RungScore]) -> dict[str, set[str]]:
    seen: dict[str, set[str]] = {}
    for record in records:
        seen.setdefault(record.proposal_id, set()).add(record.rung)
    return seen


def _candidate_specs(
    ledger: Ledger,
    limit: int,
    *,
    existing_rungs: dict[str, set[str]] | None = None,
) -> list:
    entries = sorted(
        ledger.entries.values(),
        key=lambda e: e.composite_history[-1] if e.composite_history else 0.0,
        reverse=True,
    )
    specs = []
    existing_rungs = existing_rungs or {}
    for entry in entries:
        if set(_PRODUCED_RUNGS).issubset(existing_rungs.get(entry.proposal_id, set())):
            continue
        spec = spec_from_ledger_entry(entry)
        if spec is not None:
            specs.append(spec)
        if len(specs) >= limit:
            break
    return specs


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    ledger = open_ledger(args)
    existing_records = read_rung_scores(args.store)
    existing = _existing_rungs(existing_records)
    r1 = {**_R1, "n_steps": args.r1_steps}
    specs = _candidate_specs(ledger, args.max_candidates, existing_rungs=existing)
    new_scores: list[RungScore] = []
    graded = skipped = 0
    for spec in specs:
        try:
            candidate_scores: list[RungScore] = []
            if "R0" not in existing.get(spec.proposal_id, set()):
                r0_metrics = _grade_metrics(spec, **_R0)
                candidate_scores.append(RungScore(spec.proposal_id, "R0", r0_metrics))
            if "R1" not in existing.get(spec.proposal_id, set()):
                r1_metrics = _grade_metrics(spec, **r1)
                candidate_scores.append(RungScore(spec.proposal_id, "R1", r1_metrics))
        except UndispatchableSpecError:
            skipped += 1
            continue
        new_scores.extend(candidate_scores)
        graded += 1
        if not args.quiet:
            scored = sorted(score.rung for score in candidate_scores)
            print(f"graded {spec.proposal_id}: rungs={scored}")
    if new_scores:
        append_rung_scores(new_scores, args.store)
    report = compute_fidelity_report(read_rung_scores(args.store))
    out = write_fidelity_report(report, args.out)
    if not args.quiet:
        print(
            f"\ngraded {graded} candidates ({skipped} unbuildable skipped); "
            f"{report.n_candidates_paired} paired in store."
        )
        for finding in report.findings:
            print(f"  - {finding}")
        print(f"wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
