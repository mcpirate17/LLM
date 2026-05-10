"""CLI: promote mined subgraph chains to a persistent template candidate registry.

Reads ``research/reports/mined_novel_chain_proposals.json`` (output of
``mine_template_subpatterns_v2.py``), filters by support/lift/pass-rate,
dedupes against the live ``TEMPLATES`` registry, and writes a structured
candidate file under ``research/notes/`` for review and follow-on wiring.

Usage:
    python -m research.tools.promote_mined_templates --top-k 25
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from research.meta_analysis.template_promoter import (
    promote_mined_chains,
    write_promotion_registry,
)


_DEFAULT_REPORT = "research/reports/mined_novel_chain_proposals.json"
_DEFAULT_OUTPUT = "research/notes/promoted_template_candidates.json"
_DEFAULT_META_DB = "research/meta_analysis.db"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=_DEFAULT_REPORT)
    parser.add_argument("--output", default=_DEFAULT_OUTPUT)
    parser.add_argument("--min-n-total", type=int, default=5)
    parser.add_argument("--min-lift", type=float, default=1.20)
    parser.add_argument("--min-pass-rate", type=float, default=0.30)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--include-rare", action="store_true")
    parser.add_argument("--meta-db", default=_DEFAULT_META_DB)
    parser.add_argument(
        "--no-ar-binding-overlay",
        action="store_true",
        help="skip advisory AR/binding overlay annotation",
    )
    args = parser.parse_args()

    # Import here so the CLI doesn't pay the synthesis import cost when the
    # mining report isn't present.
    from research.synthesis.templates import TEMPLATES

    report_path = Path(args.report)
    if not report_path.exists():
        print(
            f"no mining report at {report_path}; "
            f"run `python -m research.tools.mine_template_subpatterns_v2` first"
        )
        return

    candidates = promote_mined_chains(
        report_path,
        existing_template_names=list(TEMPLATES.keys()),
        min_n_total=args.min_n_total,
        min_lift=args.min_lift,
        min_pass_rate=args.min_pass_rate,
        top_k=args.top_k,
        include_rare=args.include_rare,
        include_ar_binding_overlay=not args.no_ar_binding_overlay,
        meta_db_path=args.meta_db,
    )

    metadata = {
        "promoted_at": time.time(),
        "report_source": str(report_path),
        "include_rare": bool(args.include_rare),
        "ar_binding_overlay_included": not args.no_ar_binding_overlay,
        "meta_db": str(args.meta_db),
        "thresholds": {
            "min_n_total": args.min_n_total,
            "min_lift": args.min_lift,
            "min_pass_rate": args.min_pass_rate,
        },
        "existing_template_count": len(TEMPLATES),
    }
    out = write_promotion_registry(candidates, args.output, metadata=metadata)
    print(f"wrote {len(candidates)} promotion candidates to {out}")
    for c in candidates[:10]:
        chain = " → ".join(c["chain"])
        print(
            f"  {c['proposed_template_name']:<48s}  "
            f"chain={chain:<60s}  "
            f"n={c['n_total']:<4d}  lift={c['lift_vs_cohort']:.2f}  "
            f"score={c['promotion_score']:.2f}"
        )


if __name__ == "__main__":
    main()
