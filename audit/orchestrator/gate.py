#!/usr/bin/env python3
"""Regression gate.

A green test suite does NOT prove the scoring numbers are unchanged on this NAS pipeline
(CLAUDE.md). So a fix is accepted only when BOTH hold:
  1. the smoke subset stays green, and
  2. a fixed-seed scoring run reproduces the committed baseline bit-for-bit.

Step 2 is the one that catches "tests pass but the champion got re-scored on different
code" — and, given the mission, a fix that quietly reconverges on a softmax-shaped path
would move those numbers and be rejected here.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def _sh(cmd: str, cwd: Path, timeout: int = 3600) -> int:
    try:
        return subprocess.run(
            cmd, cwd=cwd, shell=True, text=True, timeout=timeout
        ).returncode
    except subprocess.TimeoutExpired:
        print(f"  gate command timed out: {cmd}")
        return 1


def run_smoke(c: dict, repo: Path) -> bool:
    cmd = c["gate"]["smoke_cmd"]
    if not cmd:
        raise SystemExit(
            "gate.smoke_cmd is empty — refusing to accept fixes without a smoke gate"
        )
    ok = _sh(cmd, repo) == 0
    print("  smoke subset " + ("passed" if ok else "FAILED"))
    return ok


def capture_baseline(c: dict, repo: Path) -> None:
    """Run the fixed-seed scoring command and freeze its output as the baseline."""
    ref_cmd = c["gate"]["reference_cmd"]
    out = repo / c["gate"]["reference_out"]
    base = repo / c["gate"]["reference_baseline"]
    if not ref_cmd:
        raise SystemExit(
            "gate.reference_cmd is empty — cannot capture a scoring baseline"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    if _sh(ref_cmd, repo) != 0:
        raise SystemExit(
            f"reference run failed; cannot trust a baseline from it:\n  {ref_cmd}"
        )
    if not out.exists():
        raise SystemExit(
            f"reference run did not write {out} — check reference_cmd --out path"
        )
    base.write_bytes(out.read_bytes())
    print(f"  baseline frozen: {base}")


def run_reference(c: dict, repo: Path) -> bool:
    """Re-run the scoring command and diff bit-for-bit against the frozen baseline."""
    ref_cmd = c["gate"]["reference_cmd"]
    out = repo / c["gate"]["reference_out"]
    base = repo / c["gate"]["reference_baseline"]
    if not ref_cmd:
        raise SystemExit(
            "gate.reference_cmd is empty — user requested smoke+scoring; refusing to skip"
        )
    if not base.exists():
        raise SystemExit(
            f"no scoring baseline at {base}; run `orchestrate.py capture-baseline` first"
        )
    if _sh(ref_cmd, repo) != 0:
        print("  reference run errored")
        return False
    if not out.exists() or out.read_bytes() != base.read_bytes():
        print(
            f"  scoring output DIFFERS from baseline ({out} vs {base}) — fix REJECTED"
        )
        return False
    print("  scoring output matches baseline bit-for-bit")
    return True


def passes(c: dict, repo: Path) -> bool:
    """Full gate: smoke must pass, then scoring must reproduce the baseline."""
    return run_smoke(c, repo) and run_reference(c, repo)
