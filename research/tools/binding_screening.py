"""Binding-tests-v2 screening tool.

CLI:
    # Validate on known baselines:
    python -m research.tools.binding_screening --baselines --output validation.json

    # Score specific proposal_ids:
    python -m research.tools.binding_screening --proposal-ids id1,id2 --output rank.json

    # Score top-K promoted from ledger:
    python -m research.tools.binding_screening --top-k 20 --output rank.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from torch import nn

from component_fab.generator.code_generator import generate_module_from_spec
from component_fab.generator.primitive_templates import (
    LinearStateSpaceLane,
    MultiscaleWaveletLane,
    TropicalAttention,
)
from component_fab.harness.binding_tests_v2 import (
    run_all_binding_tests_v2,
)
from component_fab.harness.tiny_lm import lane_factory_for_baseline
from component_fab.state.ledger import Ledger, PROMOTION_PROMOTED

from research.tools.run_tier2_binding_cohort import _load_proposals_by_id


BASELINE_FACTORIES: dict[str, "callable"] = {
    "softmax_attention": lane_factory_for_baseline("softmax_attention"),
    "causal_conv": lane_factory_for_baseline("causal_conv"),
    "tropical_attention": TropicalAttention,
    "linear_state_space": LinearStateSpaceLane,
    "multiscale_wavelet": MultiscaleWaveletLane,
}


def _run_one(
    label: str,
    factory: "callable",
    n_train_steps: int,
    seed: int,
    quiet: bool,
) -> dict:
    t0 = time.monotonic()
    try:
        result = run_all_binding_tests_v2(
            factory, label, n_train_steps=n_train_steps, seed=seed
        )
    except Exception as exc:  # noqa: BLE001
        if not quiet:
            print(f"  {label}: FAILED — {exc}")
        return {"label": label, "status": f"failed: {exc}"}
    elapsed = time.monotonic() - t0
    row = asdict(result)
    row["status"] = "ok"
    row["wall_clock_s"] = round(elapsed, 1)
    if not quiet:
        print(
            f"  {label[:50]:50s} mti={result.multi_token_induction_acc:.3f} "
            f"sc={result.selective_copy_acc:.3f} "
            f"vd={result.variable_delay_acc_mean:.3f} "
            f"npi={result.npi_synthetic_acc:.3f} "
            f"overall={result.overall_score:.3f} t={elapsed:.1f}s"
        )
    return row


def run_baselines(n_train_steps: int, seed: int, quiet: bool) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name, factory in BASELINE_FACTORIES.items():
        if not quiet:
            print(f"baseline {name}:")
        out[name] = _run_one(name, factory, n_train_steps, seed, quiet)
    return out


def run_proposals(
    proposal_ids: list[str],
    n_train_steps: int,
    seed: int,
    quiet: bool,
) -> dict[str, dict]:
    specs_by_id = _load_proposals_by_id()
    out: dict[str, dict] = {}
    for index, pid in enumerate(proposal_ids):
        spec = specs_by_id.get(pid)
        if spec is None:
            if not quiet:
                print(f"  [{index + 1}/{len(proposal_ids)}] {pid[:60]} NOT in catalog")
            out[pid] = {"status": "spec_not_found"}
            continue

        def factory(d: int, _spec=spec) -> nn.Module:
            return generate_module_from_spec(_spec, dim=d)

        if not quiet:
            print(f"[{index + 1}/{len(proposal_ids)}] {spec.name[:55]}")
        out[pid] = _run_one(spec.name, factory, n_train_steps, seed, quiet)
        out[pid]["proposal_id"] = pid
    return out


def _pick_top_k_promoted(top_k: int) -> list[str]:
    ledger = Ledger(
        Path(__file__).resolve().parents[2]
        / "component_fab"
        / "catalog"
        / "ledger.jsonl",
        include_rotated=True,
    )
    promoted = [
        e for e in ledger.all_entries() if e.promotion_status == PROMOTION_PROMOTED
    ]
    promoted.sort(key=lambda e: max(e.composite_history or [0.0]), reverse=True)
    return [e.proposal_id for e in promoted[:top_k]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--proposal-ids", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n-train-steps", default=120, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    started = time.monotonic()
    output_payload: dict = {
        "n_train_steps": args.n_train_steps,
        "seed": args.seed,
        "started_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    if args.baselines:
        output_payload["baselines"] = run_baselines(
            args.n_train_steps, args.seed, args.quiet
        )

    if args.proposal_ids:
        pids = [p.strip() for p in args.proposal_ids.split(",") if p.strip()]
        output_payload["proposals"] = run_proposals(
            pids, args.n_train_steps, args.seed, args.quiet
        )
    elif args.top_k:
        pids = _pick_top_k_promoted(args.top_k)
        output_payload["proposals"] = run_proposals(
            pids, args.n_train_steps, args.seed, args.quiet
        )

    output_payload["elapsed_total_s"] = round(time.monotonic() - started, 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output_payload, indent=2, default=str), encoding="utf-8"
    )
    if not args.quiet:
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
