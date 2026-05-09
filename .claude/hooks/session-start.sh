#!/usr/bin/env bash
# SessionStart: surface graph stats AND inject a binding rule that any Edit/Write
# this session must be preceded by a code-review-graph tool call. This is the
# enforcement layer for CLAUDE.md's "use code review graph to understand all
# 20K files". Prior incidents (codex 2026-04-29) duplicated existing functions
# and mutated god files because the agent skipped the graph entirely.

set -euo pipefail

# Auto-prune: delete root-level files in research/reports/ older than 14d.
# Subdirs (nano_ar_inv_no_go/, nano_bind_backfill/) are intentionally preserved.
# Output goes to stderr so the JSON payload below stays clean.
REPO_ROOT="$(dirname "$(dirname "$(dirname "$(readlink -f "$0")")")")"
REPORTS_DIR="$REPO_ROOT/research/reports"
if [[ -d "$REPORTS_DIR" ]]; then
  PRUNED=$(find "$REPORTS_DIR" -maxdepth 1 -type f -mtime +14 -print -delete 2>/dev/null | wc -l || true)
  if [[ "${PRUNED:-0}" -gt 0 ]]; then
    echo "[session-start] pruned $PRUNED file(s) from research/reports older than 14d" >&2
  fi
fi

STATS=$(code-review-graph status 2>/dev/null | head -20 || echo "graph stats unavailable")

# Drop a per-session sentinel so the pre-edit hook can tell whether CRG was used yet.
mkdir -p /tmp/claude-crg-gate 2>/dev/null || true
SENTINEL="/tmp/claude-crg-gate/$(date +%s)-$$"
: > "$SENTINEL.pending"

cat <<JSON
{
  "hookSpecificOutput": {
    "hookEventName": "SessionStart",
    "additionalContext": "PROJECT-WIDE GRAPH GATE: Before ANY Edit/Write this session, you MUST first call one of: mcp__code-review-graph__semantic_search_nodes_tool, mcp__code-review-graph__query_graph_tool, mcp__code-review-graph__detect_changes_tool, or mcp__code-review-graph__get_review_context_tool. The graph is faster, cheaper, and shows callers/dependents/duplicates that file-by-file reads miss. Skipping this caused the 2026-04-29 incident (duplicated functionality, partial-data S1 writes). This rule applies to every chat session on this repo regardless of agent (Claude Code, codex, etc.). Graph snapshot:\n${STATS//$'\n'/\\n}"
  }
}
JSON
