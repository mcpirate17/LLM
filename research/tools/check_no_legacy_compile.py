#!/usr/bin/env python3
"""Fail if native-enabled path falls back to legacy compile."""

from __future__ import annotations

import os
import sys
from typing import List
from unittest.mock import patch

from research.scientist.native_runner_adapter import _env_flag


def main() -> int:
    from research.scientist.native_runner import (
        compile_model_native_first,
        native_runner_capability_report,
        reset_native_runner_telemetry,
    )
    from research.synthesis.graph import ComputationGraph

    expected = {
        "NATIVE_RUNNER_ENABLED": "1",
        "NATIVE_RUNNER_DISABLE_LEGACY_COMPILE_NATIVE_ENABLED": "1",
    }
    missing: List[str] = []
    for key, val in expected.items():
        if os.environ.get(key) != val:
            missing.append(f"{key}={val}")
    if missing:
        print(
            "[no-legacy-compile] WARNING: expected env not fully set; "
            f"missing {', '.join(missing)}",
            file=sys.stderr,
        )

    class _Session:
        def execute_tokens(self, token_ids, batch=1):
            vocab = 8
            return [0.0 for _ in range(vocab)]

    abi_report = {
        "requested": True,
        "attempted": True,
        "succeeded": True,
        "reason": "ok",
        "model_handle": 1,
        "session": _Session(),
    }

    g = ComputationGraph(model_dim=8)
    i0 = g.add_input()
    relu = g.add_op("relu", [i0])
    g.set_output(relu)

    reset_native_runner_telemetry()

    with (
        patch(
            "research.scientist.native_runner._maybe_prepare_runner_abi_session",
            return_value=abi_report,
        ),
        patch(
            "research.scientist.native_runner._legacy_compile_model",
            side_effect=RuntimeError("legacy compile invoked"),
        ),
    ):
        model = compile_model_native_first([g], vocab_size=8, max_seq_len=4)

    report = getattr(model, "_native_runner_report", {}) or {}
    if report.get("legacy_compile_used") is True:
        print("[no-legacy-compile] ERROR: legacy compile was used.", file=sys.stderr)
        return 2

    capability = native_runner_capability_report()
    metrics = capability.get("fallback_metrics") or {}
    if int(metrics.get("legacy_compile_count") or 0) > 0:
        print(
            "[no-legacy-compile] ERROR: legacy compile count recorded.",
            file=sys.stderr,
        )
        return 2

    if not _env_flag("NATIVE_RUNNER_DISABLE_LEGACY_COMPILE_NATIVE_ENABLED", False):
        print(
            "[no-legacy-compile] ERROR: legacy disable gate not active.",
            file=sys.stderr,
        )
        return 2

    print("[no-legacy-compile] OK: native path avoided legacy compile.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
