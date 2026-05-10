"""CLI: dump ranked stable pair compositions the grammar has never assembled.

The first generative wedge of the dynamic-design roadmap. Reads
``meta_analysis.db::op_pair_profile_catalog`` and ``runs.db::program_graph_pairs``
and emits the set difference — pairs the profiler measured as healthy but
that have never appeared in a real program.

Output is advisory. Candidates do not auto-promote into the motif catalog.
A human (or a follow-up registrar with holdout gating) decides which become
template slot fills.

Usage:
    python -m research.tools.propose_untapped_pairs --limit 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.defaults import RUNS_DB
from research.meta_analysis.pair_proposer import propose_untapped_pairs


_DEFAULT_META_DB = "research/meta_analysis.db"
_DEFAULT_OUTPUT = "research/notes/untapped_pair_proposals.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-db", default=_DEFAULT_META_DB)
    parser.add_argument("--runs-db", default=RUNS_DB)
    parser.add_argument("--composition", default="sequential")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--min-output-std", type=float, default=1e-4)
    parser.add_argument("--output", default=_DEFAULT_OUTPUT)
    parser.add_argument(
        "--no-ar-binding-overlay",
        action="store_true",
        help="skip advisory AR/binding overlay annotation",
    )
    args = parser.parse_args()

    candidates = propose_untapped_pairs(
        args.meta_db,
        args.runs_db,
        composition=args.composition,
        limit=args.limit,
        min_output_std=args.min_output_std,
        include_ar_binding_overlay=not args.no_ar_binding_overlay,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "composition": args.composition,
        "min_output_std": args.min_output_std,
        "ar_binding_overlay_included": not args.no_ar_binding_overlay,
        "count": len(candidates),
        "candidates": candidates,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {len(candidates)} untapped pair candidates to {out_path}")
    for c in candidates[:10]:
        print(
            f"  {c['signature']:<55s}  "
            f"std={c['output_std']:.4f}  "
            f"grad={c['grad_norm']:.3f}  "
            f"score={c['stability_score']:.3f}"
        )


if __name__ == "__main__":
    main()
