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
        description="Backfill fingerprint/CKA via research.tools.backfill"
    )
    add_common_backfill_args(
        parser,
        default_top=50,
        default_tier="screening,investigation,validation,breakthrough",
        allow_device_auto=True,
        include_timeout=True,
    )
    parser.add_argument("--limit", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    top = int(args.limit) if int(args.limit) > 0 else int(args.top)
    run_legacy_backfill(
        probes=("fingerprint",),
        tier_csv=str(args.tier),
        top_per_tier=top,
        device=str(args.device),
        force=bool(args.force),
        dry_run=bool(args.dry_run),
        fp_timeout=int(args.timeout),
    )


if __name__ == "__main__":
    main()
