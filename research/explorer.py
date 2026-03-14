"""
Architecture Explorer — Main Orchestrator

Pipeline: roll dice → build model → Stage 0 smoke test → Stage 1 micro-train → log results

Usage:
    # Generate and evaluate 20 random architectures
    source /home/tim/venvs/llm/bin/activate && python -m research.explorer --n 20

    # Fix one dimension and explore the rest
    source /home/tim/venvs/llm/bin/activate && python -m research.explorer --n 10 --fix token_mixing=compressed_attention

    # Only run Stage 0 (fast screening)
    source /home/tim/venvs/llm/bin/activate && python -m research.explorer --n 50 --stage0_only

    # Mutate the best architectures from previous runs
    source /home/tim/venvs/llm/bin/activate && python -m research.explorer --mutate --n 10

    # Show leaderboard
    source /home/tim/venvs/llm/bin/activate && python -m research.explorer --leaderboard

    # Analyze which choices work best
    source /home/tim/venvs/llm/bin/activate && python -m research.explorer --analyze
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from .morphological_box import (
    ArchSpec, DIMENSIONS, DIMENSION_NAMES,
    batch_roll, roll, mutate, crossover, describe_spec, is_valid_spec,
)
from .arch_builder import BuildConfig, build_model
from .evaluator import stage0_smoke_test, stage1_micro_train, Stage0Result, Stage1Result
from .database import ExperimentDB
from .defaults import VOCAB_SIZE


DEFAULT_DB = Path("research/experiments.db")

# Small config for exploration (keep experiments fast)
EXPLORE_CONFIG = BuildConfig(
    dim=256,
    n_heads=8,
    n_kv_heads=4,
    n_layers=6,
    vocab_size=VOCAB_SIZE,
    max_seq_len=256,
    mlp_ratio=3.0,
)


def _parse_fixed(fixed_strs: List[str]) -> Dict[str, str]:
    """Parse --fix dim=option strings."""
    fixed = {}
    for s in fixed_strs:
        if "=" not in s:
            print(f"Warning: ignoring malformed --fix '{s}' (expected dim=option)")
            continue
        dim, opt = s.split("=", 1)
        if dim not in DIMENSION_NAMES:
            print(f"Warning: unknown dimension '{dim}'. Valid: {DIMENSION_NAMES}")
            continue
        fixed[dim] = opt
    return fixed


def run_exploration(
    n: int,
    db: ExperimentDB,
    config: BuildConfig,
    fixed: Optional[Dict[str, str]] = None,
    stage0_only: bool = False,
    device: str = "cuda",
    generation: int = 0,
    base_seed: Optional[int] = None,
) -> None:
    """Generate N random architectures and evaluate them."""
    print(f"\n{'='*60}")
    print(f"Rolling {n} random architectures (generation {generation})")
    if fixed:
        print(f"Fixed dimensions: {fixed}")
    print(f"{'='*60}\n")

    specs = batch_roll(n, generation=generation, fixed=fixed, base_seed=base_seed)

    for i, spec in enumerate(specs):
        print(f"\n[{i+1}/{n}] {spec.short_name} (id={spec.id})")
        print(f"  Choices: {json.dumps(spec.choices, indent=None)}")

        # Save spec
        db.save_spec(spec)

        # Stage 0
        if db.has_stage0(spec.id):
            print(f"  Stage 0: already evaluated, skipping")
            s0 = db.get_stage0(spec.id)
            s0_passed = bool(s0["passed"])
        else:
            print(f"  Stage 0: smoke test...", end=" ", flush=True)
            t0 = time.time()
            s0_result = stage0_smoke_test(spec, config, device=device)
            elapsed = time.time() - t0
            db.save_stage0(s0_result)
            s0_passed = s0_result.passed

            if s0_passed:
                print(f"PASS ({elapsed:.1f}s, {s0_result.param_count/1e6:.1f}M params, "
                      f"fwd={s0_result.forward_time_ms:.0f}ms, "
                      f"bwd={s0_result.backward_time_ms:.0f}ms, "
                      f"mem={s0_result.peak_memory_mb:.0f}MB)")
            else:
                print(f"FAIL ({elapsed:.1f}s)")
                print(f"  Error: {s0_result.error}")
                continue

        if stage0_only:
            continue

        # Stage 1
        if not s0_passed:
            continue

        if db.has_stage1(spec.id):
            print(f"  Stage 1: already evaluated, skipping")
            continue

        print(f"  Stage 1: micro-training...", end=" ", flush=True)
        t0 = time.time()
        s1_result = stage1_micro_train(spec, config, device=device, n_steps=500, seq_len=128)
        elapsed = time.time() - t0
        db.save_stage1(s1_result)

        if s1_result.passed:
            print(f"PASS ({elapsed:.0f}s)")
            print(f"    loss: {s1_result.initial_loss:.3f} → {s1_result.final_loss:.3f} "
                  f"(ratio={s1_result.loss_ratio:.3f})")
            print(f"    throughput: {s1_result.throughput_tok_s:.0f} tok/s, "
                  f"mem: {s1_result.peak_memory_mb:.0f}MB")
        else:
            print(f"FAIL ({elapsed:.0f}s)")
            if s1_result.error:
                print(f"    Error: {s1_result.error}")
            else:
                print(f"    loss: {s1_result.initial_loss:.3f} → {s1_result.final_loss:.3f} "
                      f"(ratio={s1_result.loss_ratio:.3f}) — didn't converge")


def run_mutation(
    n: int,
    db: ExperimentDB,
    config: BuildConfig,
    n_mutations: int = 2,
    device: str = "cuda",
) -> None:
    """Mutate the top architectures and evaluate."""
    top = db.top_architectures(n=5)
    if not top:
        print("No successful Stage 1 architectures to mutate. Run exploration first.")
        return

    print(f"\nMutating top {len(top)} architectures, {n} mutations each")

    # Determine next generation
    gen = max(t.get("generation", 0) for t in top) + 1

    all_mutants = []
    for parent in top:
        parent_spec = db.reconstruct_spec(parent["spec_id"])
        if parent_spec is None:
            continue

        print(f"\nParent: {parent['short_name']} (loss_ratio={parent['loss_ratio']:.3f})")
        for j in range(n):
            try:
                child = mutate(parent_spec, n_mutations=n_mutations, generation=gen)
                all_mutants.append(child)
            except RuntimeError:
                pass

    if not all_mutants:
        print("No valid mutations generated.")
        return

    print(f"\nGenerated {len(all_mutants)} mutant architectures")

    # Evaluate mutants
    for i, spec in enumerate(all_mutants):
        print(f"\n[{i+1}/{len(all_mutants)}] {spec.short_name} (mutated from {spec.parent_id})")
        db.save_spec(spec)

        # Stage 0
        print(f"  Stage 0: smoke test...", end=" ", flush=True)
        s0 = stage0_smoke_test(spec, config, device=device)
        db.save_stage0(s0)
        if not s0.passed:
            print(f"FAIL — {s0.error}")
            continue
        print(f"PASS ({s0.param_count/1e6:.1f}M params)")

        # Stage 1
        print(f"  Stage 1: micro-training...", end=" ", flush=True)
        s1 = stage1_micro_train(spec, config, device=device, n_steps=500)
        db.save_stage1(s1)
        if s1.passed:
            print(f"PASS (loss_ratio={s1.loss_ratio:.3f})")
        else:
            print(f"FAIL (loss_ratio={s1.loss_ratio:.3f})")


def show_leaderboard(db: ExperimentDB, n: int = 20) -> None:
    """Show top architectures."""
    counts = db.count_experiments()
    print(f"\nExperiment Summary:")
    print(f"  Total specs: {counts['total_specs']}")
    print(f"  Stage 0: {counts['stage0_passed']}/{counts['stage0_evaluated']} passed")
    print(f"  Stage 1: {counts['stage1_passed']}/{counts['stage1_evaluated']} passed")

    top = db.top_architectures(n)
    if not top:
        print("\nNo Stage 1 winners yet.")
        return

    print(f"\nTop {len(top)} Architectures (by loss ratio):")
    print(f"{'Rank':>4} {'ID':>12} {'Loss Ratio':>10} {'Final Loss':>10} "
          f"{'Tok/s':>8} {'Params':>8} {'Gen':>4}  Choices")
    print("-" * 100)
    for i, arch in enumerate(top):
        choices_short = {k: v for k, v in arch["choices"].items()
                        if v not in ("dense_float", "dense_matrix", "rmsnorm_pre",
                                     "swiglu_mlp", "sequential", "uniform", "rope")}
        print(f"{i+1:>4} {arch['spec_id']:>12} {arch['loss_ratio']:>10.4f} "
              f"{arch['final_loss']:>10.4f} {arch['throughput_tok_s']:>8.0f} "
              f"{arch['param_count']/1e6:>7.1f}M {arch.get('generation', 0):>4}  "
              f"{json.dumps(choices_short)}")


def show_analysis(db: ExperimentDB) -> None:
    """Analyze which choices are best."""
    rates = db.choice_success_rates()
    if not rates:
        print("No Stage 1 data yet. Run some experiments first.")
        return

    print(f"\nChoice Success Rates (Stage 1 pass rate):")
    print(f"{'Dimension':>25} {'Option':>30} {'Pass Rate':>10} {'Count':>6}")
    print("-" * 75)

    for dim_name in DIMENSION_NAMES:
        if dim_name not in rates:
            continue
        # Sort by pass rate descending
        sorted_opts = sorted(rates[dim_name].items(), key=lambda x: -x[1])
        for opt_name, rate in sorted_opts:
            print(f"{dim_name:>25} {opt_name:>30} {rate:>10.1%}")
        print()


def main():
    p = argparse.ArgumentParser(description="Architecture Explorer")

    # Mode
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--explore", action="store_true", default=True,
                      help="Generate and evaluate random architectures (default)")
    mode.add_argument("--mutate", action="store_true",
                      help="Mutate top architectures")
    mode.add_argument("--leaderboard", action="store_true",
                      help="Show top architectures")
    mode.add_argument("--analyze", action="store_true",
                      help="Analyze choice success rates")
    mode.add_argument("--describe", type=str, metavar="SPEC_ID",
                      help="Describe a specific architecture")

    # Generation params
    p.add_argument("--n", type=int, default=10, help="Number of architectures to generate")
    p.add_argument("--fix", nargs="*", default=[], metavar="DIM=OPT",
                   help="Fix dimensions (e.g., --fix token_mixing=linear_attention)")
    p.add_argument("--stage0_only", action="store_true",
                   help="Only run Stage 0 (no training)")
    p.add_argument("--seed", type=int, default=None, help="Random seed")

    # Model config
    p.add_argument("--dim", type=int, default=256, help="Model dimension")
    p.add_argument("--n_layers", type=int, default=6, help="Number of layers")
    p.add_argument("--seq_len", type=int, default=256, help="Max sequence length")

    # Eval config
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--stage1_steps", type=int, default=500,
                   help="Training steps for Stage 1")

    # Database
    p.add_argument("--db", type=str, default=str(DEFAULT_DB),
                   help="Path to experiment database")

    args = p.parse_args()

    config = BuildConfig(
        dim=args.dim,
        n_heads=max(1, args.dim // 32),
        n_kv_heads=max(1, args.dim // 64),
        n_layers=args.n_layers,
        max_seq_len=args.seq_len,
    )

    with ExperimentDB(args.db) as db:
        if args.leaderboard:
            show_leaderboard(db, n=args.n)
        elif args.analyze:
            show_analysis(db)
        elif args.describe:
            spec = db.reconstruct_spec(args.describe)
            if spec:
                print(describe_spec(spec))
                s0 = db.get_stage0(args.describe)
                if s0:
                    print(f"\nStage 0: {'PASS' if s0['passed'] else 'FAIL'}")
                    if s0.get("param_count"):
                        print(f"  Params: {s0['param_count']/1e6:.1f}M")
                s1 = db.get_stage1(args.describe)
                if s1:
                    print(f"\nStage 1: {'PASS' if s1['passed'] else 'FAIL'}")
                    print(f"  Loss: {s1['initial_loss']:.3f} → {s1['final_loss']:.3f} "
                          f"(ratio={s1['loss_ratio']:.3f})")
            else:
                print(f"Spec '{args.describe}' not found.")
        elif args.mutate:
            run_mutation(args.n, db, config, device=args.device)
        else:
            fixed = _parse_fixed(args.fix)
            run_exploration(
                args.n, db, config,
                fixed=fixed,
                stage0_only=args.stage0_only,
                device=args.device,
                base_seed=args.seed,
            )

        # Show summary at end
        if not args.leaderboard and not args.analyze and not args.describe:
            print()
            show_leaderboard(db, n=10)


if __name__ == "__main__":
    main()
