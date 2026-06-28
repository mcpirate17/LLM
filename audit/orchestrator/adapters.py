#!/usr/bin/env python3
"""Model-agnostic CLI adapters.

One small class per coding-CLI. Every adapter knows how to build a read-only AUDIT
invocation and (for the two fixer models) a code-mutating FIX invocation, and how to
parse the CLI's output into a uniform {ok, text, cost} result. The orchestrator never
hard-codes a vendor flag — it asks an adapter.

Roles:
  claude, minimax  -> audit + fix (claude is also the triage brain)
  codex            -> audit + fix
  agy              -> audit only

Launch flags (verified against installed CLIs, 2026-06-28):
  claude   claude -p PROMPT --output-format json --dangerously-skip-permissions [--allowedTools ...]
  minimax  minimax --yolo  ... (wraps claude; --yolo => --dangerously-skip-permissions)
  codex    codex exec --sandbox read-only | --dangerously-bypass-approvals-and-sandbox -C DIR -o LAST PROMPT
  agy      agy -p PROMPT --dangerously-skip-permissions
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

# Tools an auditor may use — read + analysis ONLY, never Edit/Write/NotebookEdit and never
# bare Bash. Comma-separated (spaces inside Bash() patterns are preserved). Anything not
# listed is denied in headless -p mode, so the auditor cannot mutate the tree. The
# orchestrator still verifies tracked-file integrity after each audit (defense in depth).
AUDIT_TOOLS = (
    "Read,Grep,Glob,"
    "Bash(ls:*),Bash(cat:*),Bash(head:*),Bash(tail:*),Bash(wc:*),Bash(find:*),"
    "Bash(grep:*),Bash(rg:*),Bash(sort:*),Bash(uniq:*),Bash(awk:*),Bash(sed:*),"
    "Bash(git status:*),Bash(git log:*),Bash(git diff:*),Bash(git show:*),Bash(git ls-files:*),"
    "Bash(git grep:*),Bash(git blame:*),"
    "Bash(ruff:*),Bash(vulture:*),Bash(radon:*),Bash(xenon:*),Bash(jscpd:*),Bash(npx:*),"
    "Bash(uv run ruff:*),Bash(uv run vulture:*),Bash(uv run radon:*),"
    "Bash(python -X importtime:*),Bash(python -m cProfile:*),Bash(python -m py_compile:*),"
    "Bash(uv pip list:*),Bash(uv tree:*)"
)


@dataclass
class Result:
    ok: bool
    text: str
    cost: float = 0.0


class Adapter:
    """Base CLI adapter. name is the logical model; binary is the executable that runs
    it (e.g. haiku runs via the `claude` binary). can_fix gates the fix phase."""

    name: str = ""
    can_fix: bool = False

    @property
    def binary(self) -> str:
        return self.name

    def audit_cmd(self, prompt: str, cwd: Path) -> tuple[list[str], dict]:
        raise NotImplementedError

    def fix_cmd(self, prompt: str, cwd: Path) -> tuple[list[str], dict]:
        raise NotImplementedError

    # Prompts go via STDIN, not argv — triage prompts (many findings) exceed ARG_MAX.
    stdin_prompt: bool = True

    def run(self, prompt: str, cwd: Path, *, fix: bool, timeout: int) -> Result:
        builder = self.fix_cmd if fix else self.audit_cmd
        if fix and not self.can_fix:
            return Result(False, f"{self.name} is audit-only; cannot run fix pass", 0.0)
        cmd, ctx = builder(prompt, cwd)
        stdin = prompt if self.stdin_prompt else None
        try:
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                input=stdin,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return Result(False, f"{self.name} timed out after {timeout}s", 0.0)
        except FileNotFoundError:
            return Result(False, f"{self.name} binary not found on PATH", 0.0)
        return self.parse(proc, ctx)

    def parse(self, proc: subprocess.CompletedProcess, ctx: dict) -> Result:
        if proc.returncode != 0:
            return Result(
                False,
                (proc.stderr or proc.stdout or "non-zero exit").strip()[:4000],
                0.0,
            )
        return Result(True, proc.stdout.strip(), 0.0)


class _ClaudeStyle(Adapter):
    """claude and its minimax wrapper share the same -p / --output-format json contract."""

    prefix: list[str] = []  # e.g. ["minimax", "--yolo"]
    skip_perms: list[str] = ["--dangerously-skip-permissions"]
    model_flag: list[str] = []  # e.g. ["--model", "claude-haiku-4-5-20251001"]

    @property
    def binary(self) -> str:
        return self.prefix[0] if self.prefix else self.name

    def _base(self, skip: bool) -> list[str]:
        # No prompt in argv — `claude -p` (no positional) reads the prompt from stdin.
        cmd = [*self.prefix, "-p", "--output-format", "json", *self.model_flag]
        return cmd + (self.skip_perms if skip else [])

    def audit_cmd(self, prompt: str, cwd: Path) -> tuple[list[str], dict]:
        # READ-ONLY INTEGRITY: do NOT pass --dangerously-skip-permissions for audits.
        # In headless -p mode, any tool not in --allowedTools is denied (not executed),
        # so the auditor physically cannot Edit/Write or run an un-allowlisted Bash that
        # mutates the tree. The orchestrator additionally verifies the tree post-audit.
        return [*self._base(skip=False), "--allowedTools", AUDIT_TOOLS], {}

    def fix_cmd(self, prompt: str, cwd: Path) -> tuple[list[str], dict]:
        # Full tool access for edits; --dangerously-skip-permissions bypasses prompts.
        return self._base(skip=True), {}

    def parse(self, proc: subprocess.CompletedProcess, ctx: dict) -> Result:
        if proc.returncode != 0:
            return Result(
                False,
                (proc.stderr or proc.stdout or "non-zero exit").strip()[:4000],
                0.0,
            )
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return Result(True, proc.stdout.strip(), 0.0)
        return Result(
            True,
            str(data.get("result", proc.stdout.strip())),
            float(data.get("total_cost_usd") or 0.0),
        )


class ClaudeAdapter(_ClaudeStyle):
    name = "claude"
    can_fix = True
    prefix = ["claude"]


class HaikuAdapter(_ClaudeStyle):
    name = "haiku"
    can_fix = False  # cheap/fast mechanical audit voice; only codex+claude fix
    prefix = ["claude"]
    model_flag = ["--model", "claude-haiku-4-5-20251001"]


class MinimaxAdapter(_ClaudeStyle):
    name = "minimax"
    can_fix = False  # diverse audit voice; per spec only codex+claude fix
    prefix = ["minimax", "--yolo"]
    skip_perms: list[str] = []  # --yolo already maps to --dangerously-skip-permissions


class CodexAdapter(Adapter):
    name = "codex"
    can_fix = True

    def _common(self, prompt: str, sandbox_flags: list[str]) -> tuple[list[str], dict]:
        last = Path(tempfile.mkstemp(prefix="codex_last_", suffix=".txt")[1])
        # trailing "-" makes codex read the prompt from stdin (avoids ARG_MAX on big prompts).
        cmd = [
            "codex",
            "exec",
            *sandbox_flags,
            "--skip-git-repo-check",
            "-o",
            str(last),
            "-",
        ]
        return cmd, {"last": last}

    def audit_cmd(self, prompt: str, cwd: Path) -> tuple[list[str], dict]:
        return self._common(prompt, ["--sandbox", "read-only"])

    def fix_cmd(self, prompt: str, cwd: Path) -> tuple[list[str], dict]:
        return self._common(prompt, ["--dangerously-bypass-approvals-and-sandbox"])

    def parse(self, proc: subprocess.CompletedProcess, ctx: dict) -> Result:
        last: Path = ctx["last"]
        text = ""
        if last.exists():
            text = last.read_text(encoding="utf-8", errors="replace").strip()
            last.unlink(missing_ok=True)
        if proc.returncode != 0 and not text:
            return Result(
                False, (proc.stderr or "codex non-zero exit").strip()[:4000], 0.0
            )
        return Result(True, text or proc.stdout.strip(), 0.0)


class AgyAdapter(Adapter):
    name = "agy"
    can_fix = False  # Antigravity is an audit-only voice in this pool
    stdin_prompt = (
        False  # agy -p takes the prompt as an argv arg (audit prompts are small)
    )

    def audit_cmd(self, prompt: str, cwd: Path) -> tuple[list[str], dict]:
        # --sandbox = terminal restrictions (closest read-only mode agy offers); the
        # orchestrator's post-round integrity check reverts any mutation as backstop.
        return ["agy", "-p", prompt, "--dangerously-skip-permissions", "--sandbox"], {}


_REGISTRY: dict[str, type[Adapter]] = {
    "claude": ClaudeAdapter,
    "haiku": HaikuAdapter,
    "minimax": MinimaxAdapter,
    "codex": CodexAdapter,
    "agy": AgyAdapter,
}


def get(name: str) -> Adapter:
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise SystemExit(
            f"unknown model adapter: {name!r} (known: {', '.join(_REGISTRY)})"
        )


def available(name: str) -> bool:
    """True if the model's underlying CLI binary resolves on PATH (haiku -> claude)."""
    from shutil import which

    return which(get(name).binary) is not None
