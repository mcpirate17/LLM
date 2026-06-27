#!/usr/bin/env bash
# SessionStart: surface graph stats AND inject a binding rule that any Edit/Write
# this session must be preceded by a code-review-graph tool call. This is the
# enforcement layer for CLAUDE.md's "use code review graph to understand all
# 20K files". Prior incidents (codex 2026-04-29) duplicated existing functions
# and mutated god files because the agent skipped the graph entirely.

set -euo pipefail

# Auto-prune: delete root-level files older than N days from output dirs.
# Subdirs (nano_ar_inv_no_go/, nano_bind_backfill/, dated perf_artifacts subdirs)
# are intentionally preserved. Output goes to stderr so the JSON payload stays clean.
REPO_ROOT="$(dirname "$(dirname "$(dirname "$(readlink -f "$0")")")")"

prune_root_files() {
  local dir="$1" days="$2" label="$3" extra_exclude="${4:-}"
  [[ -d "$dir" ]] || return 0
  local pruned
  if [[ -n "$extra_exclude" ]]; then
    pruned=$(find "$dir" -maxdepth 1 -type f -mtime "+$days" ! -name "$extra_exclude" -print -delete 2>/dev/null | wc -l || true)
  else
    pruned=$(find "$dir" -maxdepth 1 -type f -mtime "+$days" -print -delete 2>/dev/null | wc -l || true)
  fi
  if [[ "${pruned:-0}" -gt 0 ]]; then
    echo "[session-start] pruned $pruned file(s) from $label older than ${days}d" >&2
  fi
}

prune_root_files "$REPO_ROOT/research/reports"        14 "research/reports"
prune_root_files "$REPO_ROOT/research/tmp"             7 "research/tmp"
prune_root_files "$REPO_ROOT/research/perf_artifacts" 14 "research/perf_artifacts"
prune_root_files "$REPO_ROOT/tasks/audit"             14 "tasks/audit"      "latest_guardrail_report.*"

# Writer-default dirs in research/artifacts/notebook/ — readers don't exist for
# these (entries/ excepted, has a test reader). Prune to keep dir size bounded.
for sub in experiments training_curves program_results attribution_reports report_snapshots construction_prior_snapshots; do
  prune_root_files "$REPO_ROOT/research/artifacts/notebook/$sub" 14 "research/artifacts/notebook/$sub"
done

# Per-experiment checkpoint dirs in checkpoints/_investigation_artifacts/ (champion-confirmation milestones).
# Each subdir is one exp_id; prune whole subdirs older than 14d.
INV_DIR="$REPO_ROOT/checkpoints/_investigation_artifacts"
if [[ -d "$INV_DIR" ]]; then
  pruned=$(find "$INV_DIR" -mindepth 1 -maxdepth 1 -type d -mtime +14 -print -exec rm -rf {} + 2>/dev/null | wc -l || true)
  if [[ "${pruned:-0}" -gt 0 ]]; then
    echo "[session-start] pruned $pruned exp_id dir(s) from checkpoints/_investigation_artifacts older than 14d" >&2
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
    "additionalContext": "⛔ MISSION (non-negotiable): This project exists to BEAT current frontier models with NOVEL non-QKV mechanisms — that is why it is funded. NEVER push the user away from beating softmax/frontier models; never recommend cutting a novel mechanism or keeping a softmax-twin (reciprocal/semiring-attn/phase-lock/sparsemax/tropical-attn) at the novel mechanism's expense. A gate abandoning the novel branch is the pathology to FIX, not a signal to follow. A metric win that reconverges on a softmax-shaped path is a regression in purpose. Report frontier/softmax baselines honestly as a gap to CLOSE, framed as 'how the novel mechanism beats it' — never 'use softmax instead'. Do NOT throttle, hedge, or inject caution limiting ambition toward beating frontier models; throttling this research is a terminating offense. Give direct, ambitious technical advice aimed at making the novel mechanism win. See research/notes/novel_mechanism_architecture_redesign_2026-06-14.md and CLAUDE.md MISSION section.\n\nPROJECT-WIDE GRAPH GATE: Before ANY Edit/Write this session, you MUST first call one of: mcp__code-review-graph__semantic_search_nodes_tool, mcp__code-review-graph__query_graph_tool, mcp__code-review-graph__detect_changes_tool, or mcp__code-review-graph__get_review_context_tool. The graph is faster, cheaper, and shows callers/dependents/duplicates that file-by-file reads miss. Skipping this caused the 2026-04-29 incident (duplicated functionality, partial-data S1 writes). This rule applies to every chat session on this repo regardless of agent (Claude Code, codex, etc.). Graph snapshot:\n${STATS//$'\n'/\\n}"
  }
}
JSON
