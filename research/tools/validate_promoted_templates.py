"""CLI: validate promoted template candidates against the synthesis pipeline.

Reads ``research/data/synthesis_candidates/promoted_template_candidates.json`` (output of
``promote_mined_templates.py``), runs each candidate's chain through the
build → validate → compile pipeline, and writes a structured registry of
validation outcomes. Compile-passing candidates are marked ready for
human/agent registration; failing ones carry the failure mode for
follow-up triage.

Usage:
    python -m research.tools.validate_promoted_templates
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from research.meta_analysis.template_validator import (
    annotate_candidates_with_validation,
    filter_to_passing,
)


_DEFAULT_INPUT = "research/data/synthesis_candidates/promoted_template_candidates.json"
_DEFAULT_OUTPUT = (
    "research/data/synthesis_candidates/validated_template_candidates.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=_DEFAULT_INPUT)
    parser.add_argument("--output", default=_DEFAULT_OUTPUT)
    parser.add_argument("--model-dim", type=int, default=64)
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"no input at {in_path}; run promote_mined_templates first")
        return

    payload = json.loads(in_path.read_text(encoding="utf-8"))
    candidates = payload.get("candidates", [])
    if not candidates:
        print(f"no candidates in {in_path}")
        return

    annotated = annotate_candidates_with_validation(
        list(candidates), model_dim=args.model_dim
    )
    passing = filter_to_passing(annotated)

    failure_counts: Counter[str] = Counter()
    for c in annotated:
        v = c.get("validation") or {}
        if not v.get("compile_passed"):
            failure_counts[str(v.get("failure_mode") or "unknown")] += 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_payload = {
        "metadata": {
            "validated_at": time.time(),
            "input_source": str(in_path),
            "n_total": len(annotated),
            "n_compile_passing": len(passing),
            "failures_by_mode": dict(failure_counts),
            "model_dim": args.model_dim,
        },
        "candidates": annotated,
        "ready_for_registration": passing,
    }
    out_path.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    print(
        f"validated {len(annotated)} candidates ({len(passing)} compile-passing) → {out_path}"
    )
    if failure_counts:
        print("failure breakdown:")
        for mode, n in failure_counts.most_common():
            print(f"  {mode}: {n}")
    for c in passing[:10]:
        print(
            f"  READY  {c.get('proposed_template_name', '?'):<40s} "
            f"chain={'->'.join(c.get('chain', []))}"
        )


if __name__ == "__main__":
    main()
