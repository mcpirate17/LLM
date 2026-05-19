"""BLiMP cohort runner — Phase D of the daily fab loop.

Takes a list of fab ``proposal_id``s that survived Tier-2 binding and
trains each as a ``TinyLM`` on wikitext-103 BPE for 500 steps, then
scores BLiMP (67 subtasks, log-likelihood minimal pairs). Compares
against ``softmax_attention`` and ``causal_conv`` baselines at the same
hyperparameters.

Reuses ``component_fab.harness.lm_eval.evaluate_lm`` — the only thing
the orchestrator does on top is iterate proposal_ids, build factories,
collect per-spec ``LMEvalResult`` rows, and compute ``delta_vs_softmax``
for the report.

Standalone usage:

    python -m research.tools.run_blimp_cohort \
        --proposal-ids id1,id2 \
        --output cohort_blimp.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from component_fab.generator.code_generator import generate_module_from_spec
from component_fab.harness.lm_eval import LMEvalResult, evaluate_lm
from component_fab.harness.tiny_lm import lane_factory_for_baseline
from component_fab.proposer.spec_generator import ProposalSpec

from research.tools.run_tier2_binding_cohort import _load_proposals_by_id


def _evaluate_one(
    spec: ProposalSpec,
    *,
    n_train_steps: int,
    dim: int,
    n_blocks: int,
    blimp_n_per_subtask: int,
    device: str,
) -> LMEvalResult:
    def factory(d: int, _spec: ProposalSpec = spec) -> torch.nn.Module:
        return generate_module_from_spec(_spec, dim=d)

    return evaluate_lm(
        factory,
        mixer_label=spec.name,
        dim=dim,
        n_blocks=n_blocks,
        n_train_steps=n_train_steps,
        blimp_n_per_subtask=blimp_n_per_subtask,
        device=device,
    )


def _baseline_result(
    name: str,
    *,
    n_train_steps: int,
    dim: int,
    n_blocks: int,
    blimp_n_per_subtask: int,
    device: str,
) -> LMEvalResult:
    return evaluate_lm(
        lane_factory_for_baseline(name),
        mixer_label=name,
        dim=dim,
        n_blocks=n_blocks,
        n_train_steps=n_train_steps,
        blimp_n_per_subtask=blimp_n_per_subtask,
        device=device,
    )


def run_cohort(
    proposal_ids: list[str],
    *,
    baseline_names: tuple[str, ...] = ("softmax_attention", "causal_conv"),
    dim: int = 64,
    n_blocks: int = 2,
    n_train_steps: int = 500,
    blimp_n_per_subtask: int = 25,
    device: str | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    """Train+score each spec, return summary with delta vs softmax baseline."""
    specs_by_id = _load_proposals_by_id()
    started = time.monotonic()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if not quiet:
        print(f"BLiMP cohort: device={device} train_steps={n_train_steps}")

    baselines: dict[str, LMEvalResult] = {}
    for name in baseline_names:
        t0 = time.monotonic()
        if not quiet:
            print(f"baseline {name} ...")
        baselines[name] = _baseline_result(
            name,
            n_train_steps=n_train_steps,
            dim=dim,
            n_blocks=n_blocks,
            blimp_n_per_subtask=blimp_n_per_subtask,
            device=device,
        )
        if not quiet:
            print(
                f"  blimp={baselines[name].blimp_overall_accuracy:.4f} "
                f"ppl={baselines[name].wikitext.post_train_ppl:.1f} "
                f"elapsed={time.monotonic() - t0:.0f}s"
            )

    softmax_blimp = baselines.get("softmax_attention")
    softmax_score = (
        float(softmax_blimp.blimp_overall_accuracy) if softmax_blimp else 0.0
    )

    results: dict[str, Any] = {}
    for index, pid in enumerate(proposal_ids):
        spec = specs_by_id.get(pid)
        if spec is None:
            results[pid] = {"status": "spec_not_found"}
            if not quiet:
                print(f"[{index + 1}/{len(proposal_ids)}] {pid} not in catalog")
            continue
        if not quiet:
            print(
                f"[{index + 1}/{len(proposal_ids)}] {pid} ({spec.name[:50]}) "
                f"train+BLiMP ..."
            )
        t0 = time.monotonic()
        try:
            lm = _evaluate_one(
                spec,
                n_train_steps=n_train_steps,
                dim=dim,
                n_blocks=n_blocks,
                blimp_n_per_subtask=blimp_n_per_subtask,
                device=device,
            )
        except Exception as exc:  # noqa: BLE001
            results[pid] = {"status": f"failed: {exc}"}
            if not quiet:
                print(f"    FAILED: {exc}")
            continue
        delta = float(lm.blimp_overall_accuracy) - softmax_score
        elapsed = time.monotonic() - t0
        results[pid] = {
            "status": "ok",
            "name": spec.name,
            "category": spec.category,
            "synthesis_kind": spec.synthesis_kind,
            "math_axes": dict(spec.math_axes),
            "blimp_overall_accuracy": float(lm.blimp_overall_accuracy),
            "blimp_by_subtask": dict(lm.blimp_by_subtask),
            "blimp_status": lm.blimp_status,
            "wikitext_post_ppl": float(lm.wikitext.post_train_ppl),
            "wikitext_pre_ppl": float(lm.wikitext.pre_train_ppl),
            "n_params": int(lm.n_params),
            "delta_vs_softmax_blimp": round(delta, 4),
            "beats_softmax_blimp": bool(delta > 0.0),
            "elapsed_s": round(elapsed, 1),
        }
        if not quiet:
            print(
                f"    blimp={lm.blimp_overall_accuracy:.4f} "
                f"(softmax_baseline={softmax_score:.4f}, delta={delta:+.4f}) "
                f"ppl={lm.wikitext.post_train_ppl:.1f} elapsed={elapsed:.0f}s"
            )
    return {
        "n_evaluated": len(proposal_ids),
        "baselines": {name: asdict(r) for name, r in baselines.items()},
        "softmax_baseline_blimp": softmax_score,
        "results": results,
        "best_candidate_blimp": max(
            (
                r.get("blimp_overall_accuracy", 0.0)
                for r in results.values()
                if r.get("status") == "ok"
            ),
            default=0.0,
        ),
        "n_beat_softmax": sum(
            1 for r in results.values() if r.get("beats_softmax_blimp")
        ),
        "elapsed_total_s": round(time.monotonic() - started, 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal-ids", required=True, type=str)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dim", default=64, type=int)
    parser.add_argument("--n-blocks", default=2, type=int)
    parser.add_argument("--n-train-steps", default=500, type=int)
    parser.add_argument("--blimp-n-per-subtask", default=25, type=int)
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    pids = [p.strip() for p in args.proposal_ids.split(",") if p.strip()]
    summary = run_cohort(
        pids,
        dim=args.dim,
        n_blocks=args.n_blocks,
        n_train_steps=args.n_train_steps,
        blimp_n_per_subtask=args.blimp_n_per_subtask,
        device=args.device,
        quiet=args.quiet,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    if not args.quiet:
        print(
            f"\nBLiMP cohort complete: best={summary['best_candidate_blimp']:.4f} "
            f"beat-softmax={summary['n_beat_softmax']}/{summary['n_evaluated']} "
            f"output: {args.output}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
