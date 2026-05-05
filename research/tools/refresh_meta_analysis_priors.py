#!/usr/bin/env python
"""Build a compact grammar prior from the separate meta-analysis database."""

from __future__ import annotations

import argparse
import json

from research.meta_analysis.metadata_db import DEFAULT_META_ANALYSIS_DB
from research.meta_analysis.priors import (
    DEFAULT_PRIOR_DIR,
    VALID_TARGETS,
    build_meta_analysis_prior,
    write_meta_analysis_prior,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--meta-db",
        default=DEFAULT_META_ANALYSIS_DB,
        help=f"Standalone meta-analysis DB path (default: {DEFAULT_META_ANALYSIS_DB}).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_PRIOR_DIR),
        help=f"Prior artifact directory (default: {DEFAULT_PRIOR_DIR}).",
    )
    parser.add_argument(
        "--target",
        choices=sorted(VALID_TARGETS),
        default="balanced",
        help="Optimization target for the grammar prior.",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=100,
        help="Minimum observation count before learned per-op/category weights are emitted.",
    )
    parser.add_argument(
        "--probe-queue-limit",
        type=int,
        default=32,
        help="Number of empirical-probe queue entries to include in the artifact.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prior = build_meta_analysis_prior(
        meta_db_path=args.meta_db,
        target=args.target,
        min_support=args.min_support,
        probe_queue_limit=args.probe_queue_limit,
    )
    path = write_meta_analysis_prior(prior, output_dir=args.output_dir)
    summary = {
        "path": str(path),
        "target": prior["target"],
        "version": prior["version"],
        "n_category_weights": len(prior["category_weights"]),
        "n_op_weights": len(prior["op_weights"]),
        "n_template_weights": len(prior["template_weights"]),
        "n_probe_queue": len(prior["probe_queue"]),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
