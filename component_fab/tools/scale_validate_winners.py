"""Scale-validation of promoted fab winners vs frontier baselines.

The autonomous loop promotes at nano (dim=32, ~100 steps) by beating the
single-head softmax causal-attention baseline with a paired-CI margin. This
tool re-runs the SAME paired comparison (``run_paired_probe``) at a larger
width + many more steps + more seeds, and against BOTH the single-head baseline
(continuity with the promotion) and multi-head GPT-2 (a stronger frontier), to
test whether each win survives scaling. Nano wins do not always hold — this is
the honest gate before claiming a softmax-beating component.

CPU-only (reuses the paired probe's ``short_training_probe``). Results are
written incrementally per (winner, anchor) so a wall-clock timeout still yields
the partial table. Run::

    python -m component_fab.tools.scale_validate_winners \
        --dim 96 --steps 1500 --seeds 5 \
        --winners invent_semiring_surprise_memory invent_causal_slot_router_memory
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from functools import partial
from pathlib import Path
from typing import Any

from ..generator.code_generator import generate_module
from ..harness.tiny_lm import lane_factory_for_baseline
from ..validator.paired import run_paired_probe
from ..validator.transplant import _default_baseline_factory

_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_WINNERS = (
    "invent_semiring_surprise_memory",
    "invent_causal_slot_router_memory",
    "invent_semiring_surprise_memory_rope",
)
# (label, factory(dim)->nn.Module): the promotion baseline + a stronger frontier.
_ANCHORS = {
    "causal_attention_1h": _default_baseline_factory,
    "gpt2_multihead": lane_factory_for_baseline("gpt2"),
}


def _axes_for_winners(winners: set[str], ledger_glob: str) -> dict[str, dict[str, Any]]:
    """Latest stored ``math_axes`` per winner name, across ledger rotations."""
    axes: dict[str, dict[str, Any]] = {}
    for f in sorted(glob.glob(ledger_glob)):
        for line in Path(f).read_text().splitlines():
            if '"event": "grade"' not in line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("name") in winners:
                ax = (rec.get("metadata") or {}).get("math_axes")
                if ax:
                    axes[rec["name"]] = ax
    return axes


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--winners", nargs="*", default=list(_DEFAULT_WINNERS))
    p.add_argument("--dim", type=int, default=96)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument(
        "--ledger-glob",
        default=str(_REPO / "component_fab" / "catalog" / "ledger.jsonl*"),
    )
    p.add_argument(
        "--output",
        default=str(_REPO / "research" / "reports" / "scale_validate_winners.json"),
    )
    args = p.parse_args()

    axes = _axes_for_winners(set(args.winners), args.ledger_glob)
    missing = [w for w in args.winners if w not in axes]
    if missing:
        print(f"WARNING: no ledger axes for {missing}")
    seeds = tuple(range(args.seeds))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "config": {
            "dim": args.dim,
            "steps": args.steps,
            "seeds": args.seeds,
            "seq_len": args.seq_len,
            "anchors": list(_ANCHORS),
        },
        "results": [],
    }
    print(
        f"scale-validating {len(axes)} winners @ dim{args.dim}/{args.steps}steps/"
        f"{args.seeds}seeds vs {list(_ANCHORS)}"
    )
    for name, ax in axes.items():
        for anchor_label, anchor_factory in _ANCHORS.items():
            t0 = time.time()
            ci = run_paired_probe(
                partial(generate_module, ax, dim=args.dim),
                partial(anchor_factory, args.dim),
                seeds=seeds,
                dim=args.dim,
                seq_len=args.seq_len,
                n_steps=args.steps,
                anchor_cache_key=("scale_anchor", anchor_label),
            )
            row = {
                "winner": name,
                "anchor": anchor_label,
                "beats_frontier": bool(ci.excludes_zero),
                "delta_mean": round(ci.mean, 5),
                "ci_low": round(ci.ci_low, 5),
                "ci_high": round(ci.ci_high, 5),
                "seconds": round(time.time() - t0, 1),
            }
            report["results"].append(row)
            # Incremental write so a timeout still leaves the partial table.
            out_path.write_text(json.dumps(report, indent=2))
            verdict = "BEATS" if row["beats_frontier"] else "no"
            print(
                f"  {name[:38]:38s} vs {anchor_label:18s} "
                f"{verdict:5s} ci_low={row['ci_low']:+.4f} ({row['seconds']}s)"
            )

    beats = [r for r in report["results"] if r["beats_frontier"]]
    print(
        f"\nDONE: {len(beats)}/{len(report['results'])} (winner,anchor) pairs beat frontier"
    )
    print(f"report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
