"""CLI: build the AR/binding holdout queue from mined proposal registries."""

from __future__ import annotations

import argparse

from research.meta_analysis.holdout_queue import (
    DEFAULT_OUTPUT,
    DEFAULT_PAIR_PROPOSALS,
    DEFAULT_PROMOTED_TEMPLATES,
    DEFAULT_VALIDATED_TEMPLATES,
    build_holdout_queue,
    write_holdout_queue,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promoted-templates", default=str(DEFAULT_PROMOTED_TEMPLATES))
    parser.add_argument(
        "--validated-templates", default=str(DEFAULT_VALIDATED_TEMPLATES)
    )
    parser.add_argument("--pair-proposals", default=str(DEFAULT_PAIR_PROPOSALS))
    parser.add_argument("--max-pairs", type=int, default=50)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    payload = build_holdout_queue(
        promoted_templates_path=args.promoted_templates,
        validated_templates_path=args.validated_templates,
        pair_proposals_path=args.pair_proposals,
        max_pairs=args.max_pairs,
    )
    out = write_holdout_queue(payload, args.output)
    counts = payload["metadata"]["status_counts"]
    print(f"wrote {len(payload['items'])} holdout queue items to {out}")
    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")


if __name__ == "__main__":
    main()
