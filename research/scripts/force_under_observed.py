"""Force-generate graphs containing under-observed components.

Discovers ops with <N observations from the lab notebook DB,
generates graphs with boosted weights for those ops, and runs
them through compile → forward → optional rapid screening.

Usage:
    python -m research.scripts.force_under_observed --threshold 20 --n-graphs 100
    python -m research.scripts.force_under_observed --threshold 20 --n-graphs 50 --rapid-screen
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, FrozenSet

import torch

from research.synthesis.grammar import GrammarConfig, generate_layer_graph
from research.synthesis.compiler import compile_model
from research.tools.explore_under_observed import (
    discover_targets as discover_under_observed,
)


def run_exploration(
    target_ops: FrozenSet[str],
    n_graphs: int = 100,
    boost_factor: float = 8.0,
    run_rapid_screen: bool = False,
    model_dim: int = 256,
    vocab_size: int = 1000,
) -> Dict:
    """Generate graphs targeting under-observed ops, compile, and forward."""
    config = GrammarConfig.exploration(
        target_ops, model_dim=model_dim, boost_factor=boost_factor
    )

    stats = {
        "generated": 0,
        "gen_errors": 0,
        "compile_ok": Counter(),
        "compile_fail": Counter(),
        "forward_ok": Counter(),
        "forward_fail": Counter(),
        "forward_nan": Counter(),
        "rapid_screen_ok": Counter(),
        "rapid_screen_fail": Counter(),
    }

    start = time.monotonic()
    seed = 0
    while stats["generated"] < n_graphs:
        seed += 1
        if seed > n_graphs * 5:
            break

        try:
            g = generate_layer_graph(config, seed=seed)
        except (ValueError, Exception):
            stats["gen_errors"] += 1
            continue

        ops_used = {
            n.op_name for n in g.nodes.values() if n.op_name not in ("input", "output")
        }
        hits = ops_used & target_ops
        stats["generated"] += 1

        if not hits:
            continue

        # Compile
        try:
            model = compile_model([g], vocab_size=vocab_size)
            for op in hits:
                stats["compile_ok"][op] += 1
        except Exception:
            for op in hits:
                stats["compile_fail"][op] += 1
            continue

        # Forward
        try:
            x = torch.randint(0, vocab_size, (2, 32))
            with torch.no_grad():
                out = model(x)
            has_nan = torch.isnan(out).any().item()
            has_inf = torch.isinf(out).any().item()
            for op in hits:
                if has_nan or has_inf:
                    stats["forward_nan"][op] += 1
                else:
                    stats["forward_ok"][op] += 1
        except Exception:
            for op in hits:
                stats["forward_fail"][op] += 1
            continue

        # Optional rapid screening
        if run_rapid_screen and not (has_nan or has_inf):
            try:
                model.train()
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                initial_loss = None
                for step in range(20):
                    optimizer.zero_grad()
                    out = model(x)
                    loss = torch.nn.functional.cross_entropy(
                        out.view(-1, out.size(-1)),
                        x.view(-1),
                    )
                    if initial_loss is None:
                        initial_loss = loss.item()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                final_loss = loss.item()
                learned = final_loss < initial_loss * 0.95
                for op in hits:
                    if learned:
                        stats["rapid_screen_ok"][op] += 1
                    else:
                        stats["rapid_screen_fail"][op] += 1
            except Exception:
                for op in hits:
                    stats["rapid_screen_fail"][op] += 1

    stats["elapsed_s"] = round(time.monotonic() - start, 1)
    stats["seeds_tried"] = seed
    return stats


def print_report(
    target_counts: Dict[str, int], stats: Dict, out_path: str | None = None
) -> None:
    """Print and optionally write a coverage report."""
    lines = []
    lines.append("=" * 80)
    lines.append("Under-Observed Component Forced Exploration Report")
    lines.append("=" * 80)
    lines.append(
        f"Generated {stats['generated']} graphs in {stats['elapsed_s']}s "
        f"({stats['seeds_tried']} seeds, {stats['gen_errors']} gen errors)"
    )
    lines.append("")

    header = (
        f"{'Op':30s} {'Old':>4s} {'Compiled':>8s} {'Forward':>8s} "
        f"{'NaN':>4s} {'Screen':>8s}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for op in sorted(target_counts.keys()):
        old = target_counts[op]
        c = stats["compile_ok"].get(op, 0)
        cf = stats["compile_fail"].get(op, 0)
        f = stats["forward_ok"].get(op, 0)
        fn = stats["forward_nan"].get(op, 0)
        ff = stats["forward_fail"].get(op, 0)
        s = stats["rapid_screen_ok"].get(op, 0)
        sf = stats["rapid_screen_fail"].get(op, 0)

        compiled = f"{c}/{c + cf}" if (c + cf) > 0 else "-"
        forwarded = f"{f}/{f + fn + ff}" if (f + fn + ff) > 0 else "-"
        nan_s = str(fn) if fn > 0 else "-"
        screened = f"{s}/{s + sf}" if (s + sf) > 0 else "-"

        lines.append(
            f"{op:30s} {old:4d} {compiled:>8s} {forwarded:>8s} "
            f"{nan_s:>4s} {screened:>8s}"
        )

    never = set(target_counts.keys()) - (
        set(stats["compile_ok"].keys()) | set(stats["compile_fail"].keys())
    )
    if never:
        lines.append(f"\nNot generated ({len(never)}): {sorted(never)}")

    report = "\n".join(lines)
    print(report)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(report)
        # Also write JSON
        json_path = out_path.replace(".md", ".json")
        json_data = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "target_counts": target_counts,
            "stats": {
                "generated": stats["generated"],
                "elapsed_s": stats["elapsed_s"],
                "compile_ok": dict(stats["compile_ok"]),
                "forward_ok": dict(stats["forward_ok"]),
                "forward_nan": dict(stats["forward_nan"]),
                "rapid_screen_ok": dict(stats["rapid_screen_ok"]),
            },
        }
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"\nReport written to {out_path} and {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Force-generate graphs for under-observed components"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=20,
        help="Observation count threshold (default: 20)",
    )
    parser.add_argument(
        "--n-graphs",
        type=int,
        default=100,
        help="Number of graphs to generate (default: 100)",
    )
    parser.add_argument(
        "--boost",
        type=float,
        default=8.0,
        help="Weight boost factor for target ops (default: 8.0)",
    )
    parser.add_argument(
        "--rapid-screen",
        action="store_true",
        help="Run 20-step rapid screening after forward pass",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "lab_notebook.db"),
        help="Path to lab_notebook.db",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=str(
            Path(__file__).resolve().parent.parent.parent
            / "artifacts"
            / "under_observed_component_coverage_report.md"
        ),
        help="Output report path",
    )
    args = parser.parse_args()

    # Discover targets
    print(f"Querying {args.db} for ops with < {args.threshold} observations...")
    target_counts = discover_under_observed(args.db, args.threshold)
    target_ops = frozenset(target_counts.keys())
    print(f"Found {len(target_ops)} under-observed ops")

    # Run exploration
    print(
        f"Generating {args.n_graphs} graphs with {args.boost}x boost "
        f"(rapid_screen={args.rapid_screen})..."
    )
    stats = run_exploration(
        target_ops,
        n_graphs=args.n_graphs,
        boost_factor=args.boost,
        run_rapid_screen=args.rapid_screen,
    )

    # Report
    print_report(target_counts, stats, args.report)


if __name__ == "__main__":
    main()
