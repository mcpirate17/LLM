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
    seed: int,
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
        seed=seed,
    )


def _baseline_result(
    name: str,
    *,
    n_train_steps: int,
    dim: int,
    n_blocks: int,
    blimp_n_per_subtask: int,
    device: str,
    seed: int,
) -> LMEvalResult:
    return evaluate_lm(
        lane_factory_for_baseline(name),
        mixer_label=name,
        dim=dim,
        n_blocks=n_blocks,
        n_train_steps=n_train_steps,
        blimp_n_per_subtask=blimp_n_per_subtask,
        device=device,
        seed=seed,
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _aggregate_lm_results(rows: list[LMEvalResult]) -> dict[str, Any]:
    """Average repeated-seed LM evals into the cohort JSON shape."""
    if not rows:
        return {"status": "missing", "seed_count": 0}
    subtasks = sorted({key for row in rows for key in row.blimp_by_subtask})
    return {
        "status": "ok",
        "mixer_label": rows[0].mixer_label,
        "n_params": int(rows[-1].n_params),
        "wikitext": {
            "initial_loss": _mean([float(row.wikitext.initial_loss) for row in rows]),
            "final_loss": _mean([float(row.wikitext.final_loss) for row in rows]),
            "pre_train_ppl": _mean([float(row.wikitext.pre_train_ppl) for row in rows]),
            "post_train_ppl": _mean(
                [float(row.wikitext.post_train_ppl) for row in rows]
            ),
            "n_steps": int(rows[-1].wikitext.n_steps),
            "converged": all(row.wikitext.converged for row in rows),
        },
        "blimp_overall_accuracy": _mean(
            [float(row.blimp_overall_accuracy) for row in rows]
        ),
        "blimp_by_subtask": {
            subtask: _mean(
                [float(row.blimp_by_subtask.get(subtask, 0.0)) for row in rows]
            )
            for subtask in subtasks
        },
        "blimp_status": (
            "ok" if all(row.blimp_status == "ok" for row in rows) else "mixed"
        ),
        "seed_count": len(rows),
        "per_seed": [asdict(row) for row in rows],
    }


def _run_baselines(
    baseline_names: tuple[str, ...],
    *,
    seed: int,
    seed_count: int,
    n_train_steps: int,
    dim: int,
    n_blocks: int,
    blimp_n_per_subtask: int,
    device: str,
    quiet: bool,
) -> dict[str, dict[str, Any]]:
    """Evaluate every baseline across all seeds and return aggregated results."""
    baselines: dict[str, dict[str, Any]] = {}
    for name in baseline_names:
        t0 = time.monotonic()
        if not quiet:
            print(f"baseline {name} ...")
        baseline_rows = []
        for offset in range(seed_count):
            run_seed = int(seed) + offset
            baseline_rows.append(
                _baseline_result(
                    name,
                    n_train_steps=n_train_steps,
                    dim=dim,
                    n_blocks=n_blocks,
                    blimp_n_per_subtask=blimp_n_per_subtask,
                    device=device,
                    seed=run_seed,
                )
            )
        baselines[name] = _aggregate_lm_results(baseline_rows)
        if not quiet:
            print(
                f"  blimp={baselines[name]['blimp_overall_accuracy']:.4f} "
                f"ppl={baselines[name]['wikitext']['post_train_ppl']:.1f} "
                f"elapsed={time.monotonic() - t0:.0f}s"
            )
    return baselines


def _run_one_proposal(
    pid: str,
    index: int,
    total: int,
    specs_by_id: dict[str, Any],
    *,
    seed: int,
    seed_count: int,
    n_train_steps: int,
    dim: int,
    n_blocks: int,
    blimp_n_per_subtask: int,
    device: str,
    softmax_score: float,
    quiet: bool,
) -> dict[str, Any]:
    """Evaluate a single proposal across all seeds; return the result entry."""

    spec = specs_by_id.get(pid)
    if spec is None:
        if not quiet:
            print(f"[{index + 1}/{total}] {pid} not in catalog")
        return {"status": "spec_not_found"}
    if not quiet:
        print(f"[{index + 1}/{total}] {pid} ({spec.name[:50]}) train+BLiMP ...")
    t0 = time.monotonic()
    try:
        lm_rows = []
        for offset in range(seed_count):
            run_seed = int(seed) + offset
            lm_rows.append(
                _evaluate_one(
                    spec,
                    n_train_steps=n_train_steps,
                    dim=dim,
                    n_blocks=n_blocks,
                    blimp_n_per_subtask=blimp_n_per_subtask,
                    device=device,
                    seed=run_seed,
                )
            )
    except Exception as exc:  # noqa: BLE001
        if not quiet:
            print(f"    FAILED: {exc}")
        return {"status": f"failed: {exc}"}
    aggregate = _aggregate_lm_results(lm_rows)
    delta = float(aggregate["blimp_overall_accuracy"]) - softmax_score
    elapsed = time.monotonic() - t0
    if not quiet:
        print(
            f"    blimp={aggregate['blimp_overall_accuracy']:.4f} "
            f"(softmax_baseline={softmax_score:.4f}, delta={delta:+.4f}) "
            f"ppl={aggregate['wikitext']['post_train_ppl']:.1f} "
            f"elapsed={elapsed:.0f}s"
        )
    return {
        "status": "ok",
        "name": spec.name,
        "category": spec.category,
        "synthesis_kind": spec.synthesis_kind,
        "math_axes": dict(spec.math_axes),
        "blimp_overall_accuracy": float(aggregate["blimp_overall_accuracy"]),
        "blimp_by_subtask": dict(aggregate["blimp_by_subtask"]),
        "blimp_status": str(aggregate["blimp_status"]),
        "wikitext_post_ppl": float(aggregate["wikitext"]["post_train_ppl"]),
        "wikitext_pre_ppl": float(aggregate["wikitext"]["pre_train_ppl"]),
        "n_params": int(aggregate["n_params"]),
        "delta_vs_softmax_blimp": round(delta, 4),
        "beats_softmax_blimp": bool(delta > 0.0),
        "seed_count": seed_count,
        "per_seed": aggregate["per_seed"],
        "elapsed_s": round(elapsed, 1),
    }


def run_cohort(
    proposal_ids: list[str],
    *,
    baseline_names: tuple[str, ...] = ("softmax_attention", "causal_conv"),
    dim: int = 64,
    n_blocks: int = 2,
    n_train_steps: int = 500,
    blimp_n_per_subtask: int = 25,
    device: str | None = None,
    seed: int = 0,
    seed_count: int = 1,
    quiet: bool = False,
) -> dict[str, Any]:
    """Train+score each spec, return summary with delta vs softmax baseline."""
    specs_by_id = _load_proposals_by_id()
    started = time.monotonic()
    seed_count = max(1, int(seed_count))
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if not quiet:
        print(
            f"BLiMP cohort: device={device} train_steps={n_train_steps} "
            f"seed_count={seed_count}"
        )

    baselines = _run_baselines(
        baseline_names,
        seed=seed,
        seed_count=seed_count,
        n_train_steps=n_train_steps,
        dim=dim,
        n_blocks=n_blocks,
        blimp_n_per_subtask=blimp_n_per_subtask,
        device=device,
        quiet=quiet,
    )
    softmax_score = float(
        (baselines.get("softmax_attention") or {}).get("blimp_overall_accuracy") or 0.0
    )

    results: dict[str, Any] = {}
    for index, pid in enumerate(proposal_ids):
        results[pid] = _run_one_proposal(
            pid,
            index,
            len(proposal_ids),
            specs_by_id,
            seed=seed,
            seed_count=seed_count,
            n_train_steps=n_train_steps,
            dim=dim,
            n_blocks=n_blocks,
            blimp_n_per_subtask=blimp_n_per_subtask,
            device=device,
            softmax_score=softmax_score,
            quiet=quiet,
        )

    return {
        "n_evaluated": len(proposal_ids),
        "baselines": baselines,
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
        "seed": int(seed),
        "seed_count": seed_count,
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
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--seed-count", default=1, type=int)
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
        seed=args.seed,
        seed_count=args.seed_count,
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
