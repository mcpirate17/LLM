#!/usr/bin/env python3
"""Ensure canary execution paths avoid *_legacy_compile* when native is enabled."""

from __future__ import annotations

import sys


def _has_legacy_compile(paths: dict) -> bool:
    for key in paths.keys():
        if "legacy_compile" in str(key):
            return True
    return False


def main() -> int:
    from research.scientist.native_runner_canary import (
        run_selective_canary_latency_benchmark,
    )

    result = run_selective_canary_latency_benchmark(iterations=6, seed=1337)
    probe_paths = result.probe_execution_paths or {}
    selective_paths = result.selective_execution_paths or {}

    if _has_legacy_compile(probe_paths) or _has_legacy_compile(selective_paths):
        print(
            "[no-legacy-exec-paths] ERROR: legacy compile execution path seen in canary.",
            file=sys.stderr,
        )
        print(f"probe_execution_paths={probe_paths}", file=sys.stderr)
        print(f"selective_execution_paths={selective_paths}", file=sys.stderr)
        return 2

    print("[no-legacy-exec-paths] OK: no legacy compile execution paths detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
