#!/usr/bin/env bash
# PreToolUse/Bash: Block commands that destroy work or violate project policy.
# Design: tight patterns only. No false positives on safe commands.

set -euo pipefail

CMD=$(cat | python3 -c "import sys,json; print(json.load(sys.stdin).get('tool_input',{}).get('command',''))" 2>/dev/null || true)

[ -z "$CMD" ] && { echo '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"allow"}}'; exit 0; }

block() {
    local reason="$1"
    printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"%s"}}' "$reason"
    exit 0
}

# ── Destructive git operations ────────────────────────────────────────
# Block force-push (any form: --force, -f after push)
echo "$CMD" | grep -qP 'git\s+push\s+.*(-f|--force)\b' && block "BLOCKED: git push --force. Use --force-with-lease if you must, or ask the user."

# Block hard reset
echo "$CMD" | grep -qP 'git\s+reset\s+--hard\b' && block "BLOCKED: git reset --hard destroys uncommitted work. Stash or commit first."

# Block git clean -fd (deletes untracked files)
echo "$CMD" | grep -qP 'git\s+clean\s+-[fdxX]' && block "BLOCKED: git clean deletes untracked files permanently. Be specific about what to remove."

# ── Filesystem destruction ────────────────────────────────────────────
# Block rm -rf on root, home, or broad globs
echo "$CMD" | grep -qP 'rm\s+-r[f ]*\s+(/|~/|\.\./|/home)\b' && block "BLOCKED: Dangerous recursive delete target."

# ── Package manager policy ────────────────────────────────────────────
# Block direct pip install (must use uv)
echo "$CMD" | grep -qP '^\s*pip\s+install\b' && block "BLOCKED: Use 'uv pip install' instead of raw pip."
echo "$CMD" | grep -qP '^\s*python.*-m\s+pip\s+install\b' && block "BLOCKED: Use 'uv pip install' instead of python -m pip."

# ── Impact analyzer for borderline-destructive commands ───────────────
# Soft-allow but force the impact (file count / size / row count) into the
# conversation, so the agent must surface it to the user before/after running.
# Hands stdin a re-synthesized payload since the original was consumed above.
HOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -x "$HOOK_DIR/_bash_impact.py" ]; then
    python3 "$HOOK_DIR/_bash_impact.py" <<EOF || echo '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"allow"}}'
{"tool_input":{"command":$(printf '%s' "$CMD" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')}}
EOF
    exit 0
fi

# ── Allow everything else ─────────────────────────────────────────────
echo '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"allow"}}'
