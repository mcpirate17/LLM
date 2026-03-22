#!/usr/bin/env python3
"""Backfill scaling comparison scores for top leaderboard entries.

Trains GPT-2 reference models at multiple layer counts on the same data
as the candidates, builds a local scaling curve, and scores parameter
efficiency.  Targets top percentile only — enough to teach Aria what
"good" looks like vs "mediocre".

Usage:
    python -m research.tools.backfill_scaling [--db PATH] [--device DEVICE]
        [--top-pct 10] [--n-steps 500] [--dry-run]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


def backfill(
    db_path: str,
    device: str = "cuda",
    families: str = "gpt2",
    n_steps: int = 500,
    top_pct: int = 100,
    dry_run: bool = False,
    verbose: bool = True,
):
    """Score top leaderboard entries against GPT-2 reference scaling curves."""
    from ..eval.scaling_reference import ScalingReferenceManager
    from ..scientist.notebook import LabNotebook

    nb = LabNotebook(db_path)
    cache_path = str(Path(db_path).parent / "scaling_reference_cache.db")
    scaling_mgr = ScalingReferenceManager(cache_path=cache_path)
    family_list = [f.strip() for f in families.split(",") if f.strip()]

    # Get ALL validation/breakthrough entries, sorted by loss_ratio (best first)
    all_entries = nb.get_leaderboard(limit=500)
    scoreable = []
    for entry in all_entries:
        tier = entry.get("tier", "screening")
        if tier not in ("validation", "breakthrough"):
            continue
        rid = entry.get("result_id")
        if not rid:
            continue
        detail = nb.get_program_detail(rid)
        if not detail:
            continue
        final_loss = detail.get("final_loss")
        param_count = detail.get("param_count") or detail.get("graph_n_params_estimate")
        if not final_loss or not param_count:
            continue
        scoreable.append(
            {
                "entry": entry,
                "detail": detail,
                "final_loss": final_loss,
                "param_count": param_count,
                "loss_ratio": detail.get("loss_ratio", 1.0),
                "flops_forward": detail.get("flops_forward", 0) or (param_count * 2),
            }
        )

    # Sort by loss_ratio (best convergence first) and take top N%
    scoreable.sort(key=lambda x: x["loss_ratio"])
    n_target = max(1, int(len(scoreable) * top_pct / 100))
    candidates = scoreable[:n_target]

    # Skip entries that already have scaling data
    candidates = [
        c for c in candidates if c["entry"].get("scaling_param_efficiency") is None
    ]

    random_chance = math.log(32000)

    if verbose:
        print("Scaling comparison backfill")
        print(f"  DB: {db_path}")
        print(f"  Device: {device}")
        print(f"  Reference families: {family_list}")
        print(f"  Reference training: {n_steps} steps")
        print(
            f"  Target: top {top_pct}% of {len(scoreable)} scoreable entries "
            f"→ {n_target} entries, {len(candidates)} need scoring"
        )
        print(f"  Random chance loss: {random_chance:.2f}")
        print()

    if not candidates:
        print("Nothing to backfill.")
        nb.close()
        return

    if dry_run:
        print("DRY RUN — candidates to score:")
        for c in candidates:
            rid = c["entry"]["result_id"][:8]
            print(
                f"  {rid} loss={c['final_loss']:.4f} ratio={c['loss_ratio']:.4f} "
                f"params={c['param_count']:,}"
            )
        nb.close()
        return

    # --- Train references once (cached for all candidates) ---
    if verbose:
        print("Training reference models (cached after first run)...")
    # Pre-warm the scaling curve so we see progress
    curve = scaling_mgr.build_local_scaling_curve(
        family_list[0],
        d_model=256,
        n_steps=n_steps,
        seq_len=128,
        vocab_size=32000,
        batch_size=4,
        lr=3e-4,
        device=device,
        data_fn=None,
        data_tag="random",
    )
    if verbose:
        if curve.A > 0:
            print(
                f"  Curve: L(N) = {curve.A:.2f} * N^(-{curve.alpha:.4f})  R²={curve.fit_r2:.3f}"
            )
            print(f"  Points: {len(curve.points)}")
            for pt in curve.points:
                print(f"    {pt.param_count:>12,d} params → loss {pt.loss:.4f}")
        else:
            print(f"  WARNING: Could not fit curve ({len(curve.points)} usable points)")
            if len(curve.points) == 0:
                print("  All reference models failed to learn below random chance.")
                print(
                    f"  Try --n-steps > {n_steps} or use real data (not random tokens)."
                )
                nb.close()
                return
        print()

    # --- Score each candidate ---
    results = {"processed": 0, "pass_3x": 0, "fail": 0, "error": 0}

    for i, c in enumerate(candidates):
        entry = c["entry"]
        rid = entry["result_id"]
        eid = entry["entry_id"]

        try:
            sr = scaling_mgr.compare_candidate(
                candidate_loss=c["final_loss"],
                candidate_params=c["param_count"],
                candidate_flops=c["flops_forward"],
                d_model=256,
                n_steps=n_steps,
                seq_len=128,
                vocab_size=32000,
                batch_size=4,
                lr=3e-4,
                device=device,
                families=family_list,
                param_efficiency_target=3.0,
                flop_ceiling=2.0,
            )

            nb.set_external_benchmarks(rid, sr.to_dict())
            nb.promote_to_tier(
                entry_id=eid,
                tier=entry.get("tier", "validation"),
                scaling_param_efficiency=sr.best_param_efficiency,
                scaling_flop_efficiency=sr.flop_efficiency,
                scaling_gate_passed=sr.scaling_gate_passed,
                scaling_best_family=sr.best_param_efficiency_family,
                scaling_confidence=sr.confidence,
            )

            results["processed"] += 1
            passed = sr.scaling_gate_passed
            if passed:
                results["pass_3x"] += 1
            else:
                results["fail"] += 1

            if verbose:
                status = "PASS 3x" if passed else "FAIL"
                print(
                    f"  [{i + 1}/{len(candidates)}] {rid[:8]}: "
                    f"loss={c['final_loss']:.4f} "
                    f"param_eff={sr.best_param_efficiency:.2f}x "
                    f"flop_eff={sr.flop_efficiency:.2f}x "
                    f"[{status}]"
                )

        except Exception as e:
            results["error"] += 1
            if verbose:
                print(f"  [{i + 1}/{len(candidates)}] {rid[:8]}: ERROR {e}")

    nb.close()

    if verbose:
        print()
        print(f"Done. Scored {results['processed']}/{len(candidates)} entries:")
        print(f"  Pass 3x efficiency gate: {results['pass_3x']}")
        print(f"  Fail: {results['fail']}")
        print(f"  Errors: {results['error']}")


def main():
    parser = argparse.ArgumentParser(
        description="Backfill scaling comparison for top leaderboard entries"
    )
    parser.add_argument(
        "--db", default="research/lab_notebook.db", help="Path to lab_notebook.db"
    )
    parser.add_argument(
        "--device", default="cuda", help="Device for reference training (default: cuda)"
    )
    parser.add_argument(
        "--families", default="gpt2", help="Reference families (default: gpt2)"
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=500,
        help="Training steps for reference models (default: 500)",
    )
    parser.add_argument(
        "--top-pct",
        type=int,
        default=100,
        help="Score top N%% of validation entries (default: 100)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be scored"
    )
    args = parser.parse_args()

    if not Path(args.db).exists():
        print(f"ERROR: Database not found: {args.db}", file=sys.stderr)
        sys.exit(1)

    backfill(
        args.db,
        device=args.device,
        families=args.families,
        n_steps=args.n_steps,
        top_pct=args.top_pct,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
