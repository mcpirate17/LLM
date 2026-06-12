#!/usr/bin/env bash
# PreToolUse: Brief, punchy reminder. CLAUDE.md has the details — this is the checklist.
cat <<'JSON'
{
  "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "additionalContext": "PRE-EDIT CHECK: (1) Did you search for existing utils before writing new code? (2) Is this the highest-perf language option? (3) Will this create duplication?"
  }
}
JSON
