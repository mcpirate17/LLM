#!/usr/bin/env python3
"""Legacy compatibility wrapper for the generic concurrent probe backfill runner."""

from __future__ import annotations

import argparse

from research.tools import run_probe_backfill


def main() -> int:
    parser = argparse.ArgumentParser(description="Concurrent HellaSwag backfill")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--concurrency", type=int, default=14)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-seconds", type=float, default=2.0)
    args = parser.parse_args()
    return run_probe_backfill.main_with_args(
        run_dir=args.run_dir,
        probe="hellaswag",
        concurrency=int(args.concurrency),
        device=str(args.device),
    )


if __name__ == "__main__":
    raise SystemExit(main())
