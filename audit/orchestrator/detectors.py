#!/usr/bin/env python3
"""Deterministic ROI oracle.

The autonomous loop must KNOW whether a fix round actually reduced bloat — it cannot
trust an LLM's self-report ("I split the god file"). This module computes a stable,
dependency-light violation vector from the same thresholds CLAUDE.md / GLOBAL_DEV_PROMPT.md
enforce, so the loop can compare before/after and stop when nothing improves.

Metrics (all whole-repo over configured targets):
  god_files      .py files  > 1250 lines
  god_functions  functions  > 100 lines (via ast)
  lint           ruff F-codes (unused imports/vars/redefinition)  [if ruff present]
  dead_code      vulture high-confidence findings                 [if vulture present]
  duplicates     conductor/check_duplicate_function_bodies.py count [if present]

`total` is the weighted scalar the loop watches. Counts come from the same tools the
existing conductor/full_repo_audit gates use — this is the loop's measurement, not a
replacement for those detailed reports.
"""

from __future__ import annotations

import ast
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

GOD_FILE_LINES = 1250
GOD_FUNC_LINES = 100

# Weights make the scalar reflect impact, not raw count: a god file dwarfs one lint hit.
WEIGHTS = {
    "god_files": 10,
    "god_functions": 3,
    "duplicates": 4,
    "dead_code": 2,
    "lint": 1,
}


@dataclass
class Metrics:
    god_files: int = 0
    god_functions: int = 0
    lint: int = 0
    dead_code: int = 0
    duplicates: int = 0
    detail: dict = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(WEIGHTS[k] * getattr(self, k) for k in WEIGHTS)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["total"] = self.total
        return d


def _iter_py(targets: list[Path], exclude: set[str]) -> list[Path]:
    out: list[Path] = []
    for root in targets:
        if not root.exists():
            continue
        for p in root.rglob("*.py"):
            if any(part in exclude for part in p.parts):
                continue
            out.append(p)
    return out


def _god_counts(files: list[Path]) -> tuple[int, int, list[str], list[str]]:
    big_files, big_funcs = [], []
    for p in files:
        try:
            src = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        nlines = src.count("\n") + 1
        if nlines > GOD_FILE_LINES:
            big_files.append(f"{p} ({nlines})")
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                span = (node.end_lineno or node.lineno) - node.lineno + 1
                if span > GOD_FUNC_LINES:
                    big_funcs.append(f"{p}:{node.lineno} {node.name} ({span})")
    return len(big_files), len(big_funcs), big_files, big_funcs


def _ruff_f_count(repo: Path, rel_targets: list[str]) -> int:
    # --frozen: never let `uv run` rewrite uv.lock (it would dirty a tracked file every
    # time we measure, poisoning the clean-tree guard and fix commits).
    cmd = [
        "uv",
        "run",
        "--frozen",
        "ruff",
        "check",
        "--select",
        "F",
        "--output-format",
        "json",
        *rel_targets,
    ]
    try:
        proc = subprocess.run(
            cmd, cwd=repo, capture_output=True, text=True, timeout=300
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 0
    try:
        return len(json.loads(proc.stdout or "[]"))
    except json.JSONDecodeError:
        return 0


def _vulture_count(repo: Path, rel_targets: list[str]) -> int:
    cmd = ["uv", "run", "--frozen", "vulture", "--min-confidence", "80", *rel_targets]
    try:
        proc = subprocess.run(
            cmd, cwd=repo, capture_output=True, text=True, timeout=300
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 0
    return sum(1 for ln in proc.stdout.splitlines() if ln.strip())


def _duplicate_count(repo: Path) -> int:
    script = repo / "conductor" / "check_duplicate_function_bodies.py"
    if not script.exists():
        return 0
    try:
        proc = subprocess.run(
            ["python", str(script)],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 0
    # The checker prints one line per duplicate cluster; non-zero exit on findings.
    return sum(1 for ln in proc.stdout.splitlines() if "duplicate" in ln.lower())


def measure(repo: Path, targets: list[str], exclude: set[str]) -> Metrics:
    target_paths = [repo / t for t in targets] if targets else [repo]
    files = _iter_py(target_paths, exclude)
    gf, gfn, gf_list, gfn_list = _god_counts(files)
    rel = [t for t in targets] or ["."]
    m = Metrics(
        god_files=gf,
        god_functions=gfn,
        lint=_ruff_f_count(repo, rel),
        dead_code=_vulture_count(repo, rel),
        duplicates=_duplicate_count(repo),
        detail={"god_files": gf_list[:50], "god_functions": gfn_list[:80]},
    )
    return m
