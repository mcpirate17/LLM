"""
Research module entry point.
Allow running as: python -m research

Modes:
    python -m research                          # Original morphological exploration
    python -m research --mode=synthesize --n 100  # Program synthesis
    python -m research --mode=continuous         # Continuous AI scientist mode
    python -m research --mode=dashboard          # Start the web dashboard
    python -m research --mode=evolve --n 20     # Evolutionary search
    python -m research --mode=routing-benchmark  # Track C routing strategy benchmark
    python -m research --resume <experiment_id>  # Resume interrupted experiment
"""

import argparse
import math
import os
import sys
import time

# Load .env from project root if present
_env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
if os.path.isfile(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

from research.defaults import (
    MODEL_DIM,
    N_LAYERS,
    DASHBOARD_PORT,
    LAB_NOTEBOOK_DB,
)


def main():
    parser = argparse.ArgumentParser(description="HYDRA Architecture Explorer")
    parser.add_argument(
        "--mode",
        default="synthesize",
        choices=[
            "synthesize",
            "continuous",
            "dashboard",
            "evolve",
            "routing-benchmark",
            "register-references",
        ],
        help="Operation mode",
    )
    parser.add_argument(
        "--n", type=int, default=10, help="Number of programs/architectures"
    )
    parser.add_argument("--dim", type=int, default=MODEL_DIM, help="Model dimension")
    parser.add_argument(
        "--n_layers", type=int, default=N_LAYERS, help="Number of layers"
    )
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument(
        "--port", type=int, default=DASHBOARD_PORT, help="Dashboard port"
    )
    parser.add_argument("--db", type=str, default=LAB_NOTEBOOK_DB)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--math-spaces",
        action="store_true",
        default=True,
        help="Enable math space primitives",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume an interrupted experiment by ID",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="all",
        choices=["gpt2", "mamba", "rag", "rwkv", "all"],
        help="Reference architecture selector for register-references mode",
    )
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="In register-references mode, skip investigation/validation",
    )

    args = parser.parse_args()

    # --resume takes priority over --mode
    if args.resume:
        _run_resume(args)
        return

    if args.mode == "synthesize":
        _run_synthesis(args)

    elif args.mode == "continuous":
        _run_continuous(args)

    elif args.mode == "dashboard":
        _run_dashboard(args)

    elif args.mode == "evolve":
        _run_evolution(args)

    elif args.mode == "routing-benchmark":
        _run_routing_benchmark(args)

    elif args.mode == "register-references":
        _run_register_references(args)


def _run_synthesis(args):
    """Run program synthesis experiment."""
    from research.scientist.runner import ExperimentRunner, RunConfig

    config = RunConfig(
        n_programs=args.n,
        model_dim=args.dim,
        n_layers=args.n_layers,
        device=args.device,
    )

    runner = ExperimentRunner(args.db)
    exp_id = runner.start_experiment(config)
    print(f"Started synthesis experiment: {exp_id}")
    _wait_for_completion(runner)


def _run_continuous(args):
    """Run continuous AI scientist mode."""
    from research.scientist.runner import ExperimentRunner, RunConfig

    config = RunConfig(
        n_programs=args.n,
        model_dim=args.dim,
        n_layers=args.n_layers,
        device=args.device,
        continuous=True,
        max_experiments=100,
    )

    runner = ExperimentRunner(args.db)
    exp_id = runner.start_continuous(config)
    print(f"Started continuous session: {exp_id}")
    _wait_for_completion(runner)


def _run_dashboard(args):
    """Start the web dashboard."""
    from research.scientist.api import run_server

    run_server(notebook_path=args.db, port=args.port, debug=False)


def _run_evolution(args):
    """Run evolutionary search."""
    from research.scientist.runner import ExperimentRunner, RunConfig

    config = RunConfig(
        model_dim=args.dim,
        n_layers=args.n_layers,
        device=args.device,
        population_size=args.n,
        n_generations=10,
    )

    runner = ExperimentRunner(args.db)
    exp_id = runner.start_evolution(config)
    print(f"Started evolution experiment: {exp_id}")
    _wait_for_completion(runner)


def _run_routing_benchmark(args):
    """Run fixed-budget routing benchmark harness (Track C)."""
    from research.scientist.runner import ExperimentRunner, RunConfig

    config = RunConfig(
        n_programs=max(1, args.n),
        model_dim=args.dim,
        n_layers=args.n_layers,
        device=args.device,
        stage1_steps=5,
        stage1_batch_size=2,
    )

    runner = ExperimentRunner(args.db)
    result = runner.run_routing_benchmark(config)

    if not result.get("available"):
        print("Routing benchmark unavailable")
        if result.get("reason"):
            print(f"Reason: {result['reason']}")
        return

    print("Routing benchmark complete.")
    print(f"Seeds: {result.get('seed_set', [])}")
    for point in result.get("points", []):
        mode = point.get("routing_mode")
        vloss = point.get("validation_loss")
        tps = point.get("tokens_per_sec")
        etc = point.get("effective_token_compute")
        stab = point.get("routing_stability")
        vloss_str = (
            f"{vloss:.4f}"
            if isinstance(vloss, (int, float)) and not math.isnan(vloss)
            else "n/a"
        )
        tps_str = f"{tps:.1f}" if isinstance(tps, (int, float)) else "n/a"
        etc_str = f"{etc:.1f}" if isinstance(etc, (int, float)) else "n/a"
        stab_str = f"{stab:.3f}" if isinstance(stab, (int, float)) else "n/a"
        print(
            f"- {mode}: val_loss={vloss_str} tokens/s={tps_str} "
            f"effective_compute={etc_str} stability={stab_str}"
        )


def _run_register_references(args):
    """Register and pin reference architectures in the leaderboard."""
    from research.tools.register_references import register_references

    result = register_references(
        db_path=args.db,
        arch=args.arch,
        device=args.device,
        include_pipeline=not args.skip_pipeline,
    )
    print("Registered references:")
    for key, item in result.items():
        print(
            f"- {key}: result={item.get('result_id')} "
            f"tier={item.get('tier')} loss_ratio={item.get('loss_ratio')}"
        )


def _run_resume(args):
    """Resume an interrupted experiment from checkpoint."""
    from research.scientist.runner import ExperimentRunner

    runner = ExperimentRunner(args.db)
    try:
        exp_id = runner.start_resume(args.resume)
        print(f"Resuming experiment: {exp_id}")
        _wait_for_completion(runner)
    except ValueError as e:
        print(f"Cannot resume: {e}", file=sys.stderr)
        sys.exit(1)


def _wait_for_completion(runner, poll_seconds: float = 1.0):
    """Wait for a background runner job to finish and print lightweight progress."""
    try:
        while runner.is_running:
            progress = runner.progress
            status = progress.status
            current = progress.current_program
            total = progress.total_programs
            gen = progress.current_generation
            gen_total = progress.total_generations
            if total > 0:
                print(
                    f"status={status} program={current}/{total}", end="\r", flush=True
                )
            elif gen_total > 0:
                print(
                    f"status={status} generation={gen}/{gen_total}",
                    end="\r",
                    flush=True,
                )
            else:
                print(f"status={status}", end="\r", flush=True)
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        print("\nStopping run...")
        runner.stop()
        while runner.is_running:
            time.sleep(0.2)

    final = runner.progress
    # Flush async writes to DB
    nb = getattr(runner, "notebook", None)
    if nb and hasattr(nb, "flush_writes"):
        nb.flush_writes()
    print("\nRun finished.")
    print(f"Final status: {final.status}")
    if final.error:
        print(f"Error: {final.error}")


main()
