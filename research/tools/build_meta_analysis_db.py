#!/usr/bin/env python
"""Build the separate template/slot meta-analysis SQLite database."""

from __future__ import annotations

import argparse
import json

from research.defaults import RUNS_DB
from research.meta_analysis.metadata_db import (
    DEFAULT_META_ANALYSIS_DB,
    DEFAULT_PROFILING_DB,
    build_meta_analysis_db,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-db",
        default=RUNS_DB,
        help=f"Read-only runs DB path (default: {RUNS_DB}).",
    )
    parser.add_argument(
        "--output-db",
        default=DEFAULT_META_ANALYSIS_DB,
        help=f"Standalone meta-analysis DB path (default: {DEFAULT_META_ANALYSIS_DB}).",
    )
    parser.add_argument(
        "--profiling-db",
        default=DEFAULT_PROFILING_DB,
        help=f"Read-only component profiling DB path (default: {DEFAULT_PROFILING_DB}).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Keep existing DB file and refresh materialized tables in place.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_meta_analysis_db(
        source_db=args.source_db,
        output_db=args.output_db,
        profiling_db=args.profiling_db,
        replace=not args.append,
    )
    print(json.dumps(summary.as_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
