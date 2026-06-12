"""Full-repo bloat / dead-code / complexity / duplication sweep.

The pre-commit hooks (ruff, vulture, xenon, jscpd) only gate *staged* changes,
and vulture's hook excludes `component_fab/` and every `tests/` tree. This runner
sweeps the *entire* repo with the same tools so accumulated bloat in already-committed
code gets re-audited periodically. It also enforces the CLAUDE.md god-file (>1250 lines)
and god-function (>100 lines) limits, which no existing hook checks.

Report-only by default; pass --fail to exit non-zero when any gate is breached (CI use).
Outputs a timestamped JSON + Markdown pair into research/reports/ (auto-pruned 14d).

Usage:
    uv run python research/tools/full_repo_audit.py            # sweep default packages
    uv run python research/tools/full_repo_audit.py component_fab/generator  # narrow
    uv run python research/tools/full_repo_audit.py --fail     # CI gate mode
"""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "research" / "reports"
WHITELIST = REPO_ROOT / "research" / "tools" / "vulture_whitelist.py"
JSCPD_CONFIG = REPO_ROOT / ".jscpd.json"

DEFAULT_TARGETS: tuple[str, ...] = (
    "component_fab",
    "research",
    "aria_core",
    "aria_designer",
)
# CLAUDE.md hard limits.
GOD_FILE_LINES = 1250
GOD_FUNCTION_LINES = 100
# Directories never worth auditing for bloat.
SKIP_DIR_PARTS = frozenset(
    {".venv", "node_modules", "__pycache__", ".run", "migrations", "build", "dist"}
)


@dataclass
class ToolResult:
    """One tool's run: its findings count, exit code, and captured output."""

    name: str
    findings: int
    exit_code: int
    gated: bool  # True if this tool's findings should fail --fail mode
    output: str = ""
    error: str | None = None


@dataclass
class AuditReport:
    timestamp: str
    targets: list[str]
    results: list[ToolResult] = field(default_factory=list)

    @property
    def total_findings(self) -> int:
        return sum(r.findings for r in self.results)

    @property
    def gated_findings(self) -> int:
        return sum(r.findings for r in self.results if r.gated)


def _require(tool: str) -> str:
    """Resolve a CLI tool or fail loud — no silent skips (CLAUDE.md: fail fast)."""
    path = shutil.which(tool)
    if path is None:
        raise FileNotFoundError(
            f"required tool '{tool}' not on PATH — install it into the project venv"
        )
    return path


def _run(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(cmd),
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _iter_py_files(targets: Sequence[str], *, skip_tests: bool = False) -> list[Path]:
    """Yield .py files under targets. The god-file/function checks pass
    skip_tests=True since long test modules/fixtures aren't production bloat."""

    def _is_test(path: Path) -> bool:
        return "tests" in path.parts or path.name.startswith("test_")

    files: list[Path] = []
    for target in targets:
        root = (REPO_ROOT / target).resolve()
        if root.is_file() and root.suffix == ".py":
            if not (skip_tests and _is_test(root)):
                files.append(root)
            continue
        for path in root.rglob("*.py"):
            if SKIP_DIR_PARTS.isdisjoint(path.parts) and not (
                skip_tests and _is_test(path)
            ):
                files.append(path)
    return files


def audit_ruff(targets: Sequence[str]) -> ToolResult:
    """Unused imports (F401), redefinitions (F811), unused locals (F841)."""
    _require("ruff")
    proc = _run(
        [
            "ruff",
            "check",
            "--select",
            "F401,F811,F841",
            "--output-format",
            "concise",
            *targets,
        ]
    )
    lines = [ln for ln in proc.stdout.splitlines() if ":" in ln and ".py" in ln]
    return ToolResult(
        "ruff (unused imports/vars)", len(lines), proc.returncode, True, proc.stdout
    )


def audit_vulture(targets: Sequence[str], min_confidence: int) -> ToolResult:
    """Dead code: unused functions, classes, methods, attributes."""
    _require("vulture")
    cmd = [
        "vulture",
        *targets,
        str(WHITELIST.relative_to(REPO_ROOT)),
        "--min-confidence",
        str(min_confidence),
        "--exclude",
        "*/.venv/*,*/node_modules/*,*/__pycache__/*,*/.run/*,*/tests/*,*/migrations/*",
    ]
    proc = _run(cmd)
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    # vulture is advisory here (false positives on dynamic dispatch) — not a hard gate.
    return ToolResult(
        "vulture (dead code)", len(lines), proc.returncode, False, proc.stdout
    )


def audit_radon(targets: Sequence[str]) -> ToolResult:
    """Complexity hotspots ranked C or worse (overengineering signal)."""
    _require("radon")
    proc = _run(["radon", "cc", "-n", "C", "-s", *targets])
    blocks = [
        ln
        for ln in proc.stdout.splitlines()
        if " - " in ln and any(g in ln for g in "CDEF")
    ]
    return ToolResult(
        "radon (complexity >= C)", len(blocks), proc.returncode, False, proc.stdout
    )


def audit_jscpd(targets: Sequence[str]) -> ToolResult:
    """Copy/paste duplication across the tree (cross-language)."""
    if shutil.which("npx") is None:
        return ToolResult(
            "jscpd (duplication)", 0, 0, False, error="npx not available — skipped"
        )
    cmd = [
        "npx",
        "--no-install",
        "jscpd",
        "--config",
        str(JSCPD_CONFIG),
        "--noTips",
        *targets,
    ]
    proc = _run(cmd)
    out = proc.stdout + proc.stderr
    clones = [
        ln for ln in out.splitlines() if "Clone found" in ln or "duplicated lines" in ln
    ]
    return ToolResult("jscpd (duplication)", len(clones), proc.returncode, False, out)


def audit_god_files_and_functions(
    targets: Sequence[str],
) -> tuple[ToolResult, ToolResult]:
    """Enforce the CLAUDE.md god-file (>1250) and god-function (>100) line limits.

    Test modules are excluded — long test files/fixtures are not production bloat."""
    big_files: list[str] = []
    big_funcs: list[str] = []
    for path in _iter_py_files(targets, skip_tests=True):
        rel = path.relative_to(REPO_ROOT)
        try:
            source = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as exc:
            big_files.append(f"{rel}: UNREADABLE ({exc})")
            continue
        n_lines = source.count("\n") + 1
        if n_lines > GOD_FILE_LINES:
            big_files.append(f"{rel}: {n_lines} lines (> {GOD_FILE_LINES})")
        try:
            tree = ast.parse(source, filename=str(rel))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.end_lineno
            ):
                span = node.end_lineno - node.lineno + 1
                if span > GOD_FUNCTION_LINES:
                    big_funcs.append(
                        f"{rel}:{node.lineno} {node.name}() {span} lines (> {GOD_FUNCTION_LINES})"
                    )
    files_res = ToolResult(
        f"god files (> {GOD_FILE_LINES} lines)",
        len(big_files),
        0,
        True,
        "\n".join(sorted(big_files)),
    )
    funcs_res = ToolResult(
        f"god functions (> {GOD_FUNCTION_LINES} lines)",
        len(big_funcs),
        0,
        True,
        "\n".join(sorted(big_funcs)),
    )
    return files_res, funcs_res


def write_reports(report: AuditReport) -> tuple[Path, Path]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = report.timestamp.replace(":", "").replace("-", "").replace("T", "_")[:15]
    json_path = REPORTS_DIR / f"full_repo_audit_{stamp}.json"
    md_path = REPORTS_DIR / f"full_repo_audit_{stamp}.md"

    json_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")

    lines = [
        f"# Full-repo audit — {report.timestamp}",
        "",
        f"Targets: `{', '.join(report.targets)}`",
        f"Total findings: **{report.total_findings}** (gated: {report.gated_findings})",
        "",
        "| Tool | Findings | Gated | Exit |",
        "|---|---|---|---|",
    ]
    for r in report.results:
        mark = "🚩" if (r.gated and r.findings) else ""
        lines.append(
            f"| {r.name} | {r.findings} {mark} | {'yes' if r.gated else 'no'} | {r.exit_code} |"
        )
    for r in report.results:
        body = (r.output or r.error or "").strip()
        if body:
            lines += ["", f"## {r.name}", "", "```", body[:20000], "```"]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "targets", nargs="*", default=list(DEFAULT_TARGETS), help="paths to sweep"
    )
    parser.add_argument(
        "--min-confidence", type=int, default=80, help="vulture confidence floor"
    )
    parser.add_argument(
        "--fail", action="store_true", help="exit non-zero if any gated finding exists"
    )
    args = parser.parse_args(argv)

    targets = args.targets or list(DEFAULT_TARGETS)
    report = AuditReport(
        timestamp=datetime.now().isoformat(timespec="seconds"), targets=targets
    )

    files_res, funcs_res = audit_god_files_and_functions(targets)
    report.results.extend(
        [
            audit_ruff(targets),
            audit_vulture(targets, args.min_confidence),
            audit_radon(targets),
            audit_jscpd(targets),
            files_res,
            funcs_res,
        ]
    )

    json_path, md_path = write_reports(report)

    print(f"\nFull-repo audit — {report.timestamp}  targets: {', '.join(targets)}")
    for r in report.results:
        mark = " 🚩" if (r.gated and r.findings) else ""
        note = f"  ({r.error})" if r.error else ""
        print(f"  {r.findings:>5}  {r.name}{mark}{note}")
    print(
        f"\n  total: {report.total_findings} findings ({report.gated_findings} gated)"
    )
    print(f"  report: {md_path.relative_to(REPO_ROOT)}")
    print(f"          {json_path.relative_to(REPO_ROOT)}")

    if args.fail and report.gated_findings:
        print("\nFAIL: gated findings present.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
