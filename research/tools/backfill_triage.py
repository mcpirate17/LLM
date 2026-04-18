#!/usr/bin/env python3
"""Legacy compatibility wrapper for the unified backfill runner."""

from __future__ import annotations

import argparse

from research.tools._legacy_backfill_cli import run_legacy_backfill


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill triage metrics via research.tools.backfill"
    )
    parser.add_argument("--limit", type=int, default=0, help="0 means default top=50")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_legacy_backfill(
        probes=("triage",),
        tier_csv="screening,investigation,validation,breakthrough",
        top_per_tier=int(args.limit) if int(args.limit) > 0 else 50,
        device="auto",
        force=bool(args.force),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    main()
