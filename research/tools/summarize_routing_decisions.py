"""CLI: summarize routing-knob decisions × outcomes.

Reads ``program_results`` rows whose graph_json contains routing_decisions
(emitted by Move #2's ``record_routing_decision``), joins with outcome
columns (stage1_passed, ar_gate_score, binding_intermediate_auc, etc.),
and writes a per-(template, decision_key, value) summary.

This is the input a learned routing policy (Thompson, UCB, surrogate)
needs to replace rng.choice in the routing templates.

Usage:
    python -m research.tools.summarize_routing_decisions
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from research.defaults import RUNS_DB
from research.meta_analysis.routing_decision_analytics import (
    iter_routing_decision_outcomes,
    summarize_routing_decisions,
)


_DEFAULT_OUTPUT = "research/notes/routing_decisions_summary.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-db", default=RUNS_DB)
    parser.add_argument("--template", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=_DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = list(
        iter_routing_decision_outcomes(
            args.runs_db,
            template_filter=args.template,
            limit=args.limit,
        )
    )
    summary = summarize_routing_decisions(rows)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "summarized_at": time.time(),
            "runs_db": args.runs_db,
            "template_filter": args.template,
            "n_decision_rows": len(rows),
            "n_buckets": len(summary),
        },
        "buckets": summary,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {len(summary)} buckets from {len(rows)} decision rows to {out_path}")
    for record in summary[:15]:
        line = (
            f"  {record['template_name']}.{record['decision_key']}={record['value']!r}  "
            f"n={record['n']:<4d}  pass_rate={record['pass_rate']:.3f}"
        )
        ar = record.get("mean_ar_gate_score")
        if ar is not None:
            line += f"  ar_gate={ar:.3f}"
        print(line)


if __name__ == "__main__":
    main()
