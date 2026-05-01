#!/usr/bin/env bash
# SessionStart: surface graph stats AND inject a binding rule that any Edit/Write
# this session must be preceded by a code-review-graph tool call. This is the
# enforcement layer for CLAUDE.md's "use code review graph to understand all
# 20K files". Prior incidents (codex 2026-04-29) duplicated existing functions
# and mutated god files because the agent skipped the graph entirely.

set -euo pipefail

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
