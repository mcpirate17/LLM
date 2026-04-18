#!/usr/bin/env python3
"""Legacy compatibility wrapper for the unified backfill runner."""

from __future__ import annotations

import argparse

from research.tools._legacy_backfill_cli import (
    add_common_backfill_args,
    run_legacy_backfill,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill HellaSwag via research.tools.backfill"
    )
    add_common_backfill_args(
        parser,
        default_top=20,
        default_tier="validation,investigation",
    )
    args = parser.parse_args()
    run_legacy_backfill(
        probes=("hellaswag",),
        tier_csv=str(args.tier),
        top_per_tier=int(args.top),
        device=str(args.device),
        force=bool(args.force),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    main()
