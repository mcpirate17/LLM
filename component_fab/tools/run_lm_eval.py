"""CLI: Level-1 wikitext + BLiMP evaluation for fab lanes.

Builds a pre-norm Transformer (TinyLM with FFN enabled) around the
candidate fab mixer, trains on wikitext-103 BPE, then scores BLiMP.
Same training / params / steps across candidate and baselines — only
the mixer differs.

Reuses research/'s wikitext data prep and BLiMP scorer. Training loop
stays in fab (plain Adam).

Usage:
    python -m component_fab.tools.run_lm_eval --proposal-id <id>
    python -m component_fab.tools.run_lm_eval --top-n-unique 3
    python -m component_fab.tools.run_lm_eval --proposal-id <id> \\
        --n-train-steps 1000 --output /tmp/lm_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from torch import nn

from component_fab.harness.lm_eval import LMEvalResult, evaluate_lm
from component_fab.harness.tiny_lm import (
    DEFAULT_BASELINE_NAMES,
    lane_factory_for_baseline,
)
from component_fab.proposer.spec_generator import ProposalSpec
from component_fab.state.ledger import DEFAULT_LEDGER_PATH, Ledger
from component_fab.tools.run_lm_probe import (
    _DEFAULT_ANCHOR_OPS,
    _factory_from_spec,
    _resolve_spec_by_proposal_id,
    _top_unique_promoted_specs,
)


def _format_row(r: LMEvalResult) -> str:
    return (
        f"{r.mixer_label[:40]:<40} "
        f"n_params={r.n_params:>9,d}  "
        f"wt_ppl_pre={r.wikitext.pre_train_ppl:>10.1f}  "
        f"wt_ppl_post={r.wikitext.post_train_ppl:>10.1f}  "
        f"loss={r.wikitext.initial_loss:.2f}->{r.wikitext.final_loss:.2f}  "
        f"blimp={r.blimp_overall_accuracy:.4f}  "
        f"blimp_status={r.blimp_status}"
    )


def _print_per_subtask_diff(
    candidate: LMEvalResult, baselines: list[LMEvalResult]
) -> None:
    """Per-subtask candidate-vs-best-baseline delta. Highlights where the
    candidate wins or loses on specific linguistic categories."""
    if not candidate.blimp_by_subtask:
        return
    print(f"\n-- BLiMP subtask deltas: {candidate.mixer_label} vs best-baseline --")
    print(f"{'subtask':<48} {'cand':>7} {'best_bl':>9}  best_baseline")
    for subtask, cand_acc in sorted(candidate.blimp_by_subtask.items()):
        bl_pairs = [
            (b.mixer_label, b.blimp_by_subtask.get(subtask, 0.0)) for b in baselines
        ]
        if not bl_pairs:
            continue
        best = max(bl_pairs, key=lambda p: p[1])
        marker = " ✓" if cand_acc > best[1] else ("  =" if cand_acc == best[1] else "")
        print(f"{subtask:<48} {cand_acc:>7.3f} {best[1]:>9.3f}  {best[0]}{marker}")


def _run_one_target(
    candidate_factory: Callable[[int], nn.Module],
    candidate_label: str,
    baseline_names: tuple[str, ...],
    args: argparse.Namespace,
) -> tuple[LMEvalResult, list[LMEvalResult]]:
    cand = evaluate_lm(
        candidate_factory,
        mixer_label=candidate_label,
        dim=args.dim,
        n_blocks=args.n_blocks,
        n_train_steps=args.n_train_steps,
        learning_rate=args.learning_rate,
        blimp_n_per_subtask=args.blimp_n_per_subtask,
        blimp_max_seq_len=args.blimp_max_seq_len,
        device=args.device,
        seed=args.seed,
        max_seq_len=args.seq_len,
    )
    baselines: list[LMEvalResult] = []
    for name in baseline_names:
        baselines.append(
            evaluate_lm(
                lane_factory_for_baseline(name),
                mixer_label=name,
                dim=args.dim,
                n_blocks=args.n_blocks,
                n_train_steps=args.n_train_steps,
                learning_rate=args.learning_rate,
                blimp_n_per_subtask=args.blimp_n_per_subtask,
                blimp_max_seq_len=args.blimp_max_seq_len,
                device=args.device,
                seed=args.seed,
                max_seq_len=args.seq_len,
            )
        )
    return cand, baselines


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Level-1 wikitext + BLiMP eval for fab lanes"
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--proposal-id", help="exact or prefix proposal_id")
    grp.add_argument(
        "--top-n-unique", type=int, help="evaluate the top-N axes-unique promoted lanes"
    )
    parser.add_argument(
        "--anchors",
        nargs="*",
        default=list(_DEFAULT_ANCHOR_OPS),
    )
    parser.add_argument(
        "--baseline-names",
        nargs="*",
        default=list(DEFAULT_BASELINE_NAMES),
    )
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER_PATH))
    parser.add_argument("--n-train-steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--n-blocks", type=int, default=2)
    parser.add_argument("--blimp-n-per-subtask", type=int, default=50)
    parser.add_argument("--blimp-max-seq-len", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", help="write JSON report")
    parser.add_argument(
        "--per-subtask-diff",
        action="store_true",
        help="print per-BLiMP-subtask candidate-vs-best-baseline diff",
    )
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

    payloads: list[dict] = []
    for spec in targets:
        candidate_factory = _factory_from_spec(spec)
        cand, baselines = _run_one_target(
            candidate_factory, spec.name, tuple(args.baseline_names), args
        )
        print(f"\n=== {spec.name} (proposal_id={spec.proposal_id}) ===")
        print(_format_row(cand))
        for b in baselines:
            print(_format_row(b))
        if args.per_subtask_diff:
            _print_per_subtask_diff(cand, baselines)
        payloads.append(
            {
                "proposal_id": spec.proposal_id,
                "candidate": asdict(cand),
                "baselines": [asdict(b) for b in baselines],
            }
        )

    if args.output:
        Path(args.output).write_text(json.dumps(payloads, indent=2), encoding="utf-8")
        print(f"\nwrote: {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
