"""CLI: derive the do-not-compose triplet blocklist from measured profiler data.

Reads ``meta_analysis.db::op_triplet_profile_catalog``, filters to triplets
the profiler ran but found unstable (NaN, vanishing/exploding gradient,
collapsed output), and writes a structured blocklist for the grammar /
failure_signatures layer to consume.

Usage:
    python -m research.tools.derive_triplet_blocklist
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from research.meta_analysis.triplet_blocklist import derive_triplet_blocklist


_DEFAULT_META_DB = "research/meta_analysis.db"
_DEFAULT_OUTPUT = "research/data/synthesis_candidates/triplet_blocklist.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-db", default=_DEFAULT_META_DB)
    parser.add_argument("--output", default=_DEFAULT_OUTPUT)
    parser.add_argument(
        "--include-non-emergent",
        action="store_true",
        help="include unstable triplets even when pair predictions disagreed",
    )
    args = parser.parse_args()

    blocklist = derive_triplet_blocklist(
        args.meta_db,
        require_pair_predicted_stable=not args.include_non_emergent,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "count": len(blocklist),
        "emergent_only": not args.include_non_emergent,
        "by_reason": dict(Counter(row["reason"] for row in blocklist)),
        "blocklist": blocklist,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {len(blocklist)} blocked triplets to {out_path}")
    for row in blocklist[:10]:
        print(
            f"  {row['signature']:<50s}  reason={row['reason']:<18s}  "
            f"emergent={row['emergent']}"
        )


if __name__ == "__main__":
    main()
