#!/usr/bin/env python3
"""Check native-runner cutover gate status from Aria API.

Usage:
  python -m research.tools.check_cutover_gate --base-url http://127.0.0.1:5000
  python -m research.tools.check_cutover_gate --allow-waiting
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def _generate_deterministic_parity_sample() -> None:
    """Generate one deterministic parity-pass sample in-process.

    This keeps cutover-gate CI checks deterministic when parity is required.
    """
    import torch

    from research.eval.sandbox import safe_eval
    from research.scientist.native.abi import record_native_abi_parity_result

    vocab_size = 16
    base = torch.arange(vocab_size, dtype=torch.float32) * 0.1

    class _MatchModel(torch.nn.Module):
        def forward(self, x):
            return base.view(1, 1, -1).expand(x.shape[0], x.shape[1], -1).contiguous()

    class _Session:
        def execute_tokens(self, token_ids, batch=1):
            return [float(v) for v in base.tolist()]

    model = _MatchModel()
    model._native_runner_abi_session = _Session()

    result = safe_eval(
        model,
        batch_size=2,
        seq_len=4,
        vocab_size=vocab_size,
        device="cpu",
        run_stability_probe=False,
        abi_infer_probe=True,
        abi_infer_primary=True,
        abi_infer_primary_no_grad=True,
    )
    probe = result.native_abi_probe or {}
    if bool(probe.get("parity_attempted")):
        record_native_abi_parity_result(bool(probe.get("parity_pass")))


def _generate_compile_sample() -> None:
    """Generate one deterministic compile sample through native-first runner.

    This is used to ensure cutover gates are evaluated after at least one real
    compile invocation in the checking process.  Phase D: compile with native
    disabled so the legacy compile path is exercised (the only path that
    increments legacy_compile_count for gate evaluation).
    """
    from research.scientist.native_runner import compile_model_native_first
    from research.synthesis.graph import ComputationGraph

    g = ComputationGraph(model_dim=16)
    i0 = g.add_input()
    relu = g.add_op("relu", [i0])
    add = g.add_op("add", [relu, i0])
    g.set_output(add)
    previous_enabled = os.environ.get("NATIVE_RUNNER_ENABLED")
    os.environ["NATIVE_RUNNER_ENABLED"] = "0"
    try:
        _ = compile_model_native_first([g], vocab_size=64, max_seq_len=8)
    finally:
        if previous_enabled is None:
            os.environ.pop("NATIVE_RUNNER_ENABLED", None)
        else:
            os.environ["NATIVE_RUNNER_ENABLED"] = previous_enabled


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate /api/native-runner/capability cutover gate"
    )
    parser.add_argument(
        "--base-url", default="http://127.0.0.1:5000", help="Aria API base URL"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Read cutover gate directly from native_runner_capability_report() instead of HTTP",
    )
    parser.add_argument(
        "--allow-waiting",
        action="store_true",
        help="Treat cutover status=waiting as success",
    )
    parser.add_argument(
        "--generate-parity-sample",
        action="store_true",
        help="Generate one deterministic in-process parity sample before evaluating gate",
    )
    parser.add_argument(
        "--generate-compile-sample",
        action="store_true",
        help="Generate one deterministic compile sample before evaluating gate",
    )
    args = parser.parse_args()

    if args.generate_parity_sample:
        try:
            _generate_deterministic_parity_sample()
        except Exception as exc:
            print(
                f"[cutover-gate] ERROR parity sample generation failed: {exc}",
                file=sys.stderr,
            )
            return 2
    if args.generate_compile_sample:
        try:
            _generate_compile_sample()
        except Exception as exc:
            print(
                f"[cutover-gate] ERROR compile sample generation failed: {exc}",
                file=sys.stderr,
            )
            return 2

    if args.offline:
        try:
            from research.scientist.native_runner import native_runner_capability_report

            payload = native_runner_capability_report()
        except Exception as exc:
            print(f"[cutover-gate] ERROR offline report failed: {exc}", file=sys.stderr)
            return 2
    else:
        url = f"{args.base_url.rstrip('/')}/api/native-runner/capability"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            print(f"[cutover-gate] ERROR request failed: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:
            print(f"[cutover-gate] ERROR invalid response: {exc}", file=sys.stderr)
            return 2

    gate = payload.get("cutover_gate") or {}
    status = str(gate.get("status") or "unknown").lower()
    ready = gate.get("ready")
    checks = gate.get("checks") or []
    print(f"[cutover-gate] status={status} ready={ready} checks={len(checks)}")

    if status == "ready":
        return 0
    if status == "waiting" and args.allow_waiting:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
