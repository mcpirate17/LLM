#!/usr/bin/env python3
"""Guardrail: prevent new direct synthesis compiler callsites.

This keeps native-runner cutover work centralized in
`research/scientist/native_runner.py` and avoids accidental regressions.
"""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_ROOT = REPO_ROOT / "research"

# Allowed direct imports/callsites that are intentionally retained.
ALLOWED_PATH_SUFFIXES = {
    "research/scientist/native_runner.py",
}

# Tests are allowed to import legacy compiler for parity assertions.
ALLOWED_PREFIXES = ("research/tests/",)

IMPORT_PATTERNS = (
    re.compile(r"from\s+research\.synthesis\.compiler\s+import\s+compile_model"),
    re.compile(r"from\s+\.\.synthesis\.compiler\s+import\s+compile_model"),
    re.compile(r"from\s+\.\.\.synthesis\.compiler\s+import\s+compile_model"),
)
CALL_PATTERN = re.compile(r"research\.synthesis\.compiler\.compile_model\s*\(")


def _is_allowed(path: Path) -> bool:
    rel = path.relative_to(REPO_ROOT).as_posix()
    if rel in ALLOWED_PATH_SUFFIXES:
        return True
    return any(rel.startswith(prefix) for prefix in ALLOWED_PREFIXES)


def _scan_file(path: Path) -> list[str]:
    findings: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        return [f"{path.relative_to(REPO_ROOT)}: read-error: {exc}"]

    if any(pattern.search(text) for pattern in IMPORT_PATTERNS) or CALL_PATTERN.search(
        text
    ):
        if not _is_allowed(path):
            findings.append(path.relative_to(REPO_ROOT).as_posix())
    return findings


def main() -> int:
    findings: list[str] = []
    for path in RESEARCH_ROOT.rglob("*.py"):
        findings.extend(_scan_file(path))

    if findings:
        print(
            "[native-compile-callsites] ERROR: disallowed direct compile_model usage found:"
        )
        for rel in sorted(set(findings)):
            print(f"  - {rel}")
        print(
            "Use research.scientist.native_runner.compile_model_native_first instead."
        )
        return 1

    print(
        "[native-compile-callsites] OK: no disallowed direct compile_model callsites found."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
