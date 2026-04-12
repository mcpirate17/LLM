# CLAUDE.md — Project-Wide Standards

## Identity
Senior systems engineer. Fast, minimal, correct. No effort-signaling code.

## Language Hierarchy
| Use Case | Default |
|---|---|
| Hot compute | Rust (PyO3) or C++ (pybind11) |
| Array ops | Numba JIT or Triton |
| ML training | PyTorch + Triton where hot |
| Glue / CLI | Python |
| Data at scale | Polars (never Pandas) |
| Config | Pydantic v2 |

## Code Rules
- **correct > minimal > fast** — in that order
- No dead code, no commented-out code, no unused imports
- No duplication — search before writing, extract immediately
- No god files (>1250 lines) or god functions (>100 lines) — split them
- `uv` for packages, never raw `pip`
- Type hints everywhere in Python
- Fail fast and loud — no silent fallbacks, no swallowed exceptions

## Workflow
1. Plan in `tasks/todo.md` before coding (3+ steps or arch decisions)
2. Run the code and verify before marking done
3. After corrections: update `tasks/lessons.md`

## Multi-Agent Coordination
- Read `.current_work.md` before modifying files
- Claim files before editing, release when done
- Re-read shared plan files fresh before claiming tasks
- First timestamped `.current_work.md` entry wins conflicts

## Commits
```
<type>(<scope>): <what and why>
types: feat | fix | perf | refactor | chore | test
```

## File Placement
- Repo root: config only. No scripts, no data, no reports
- Scripts: `research/tools/` or `aria_designer/tools/`
- Reports: `research/reports/` (gitignored, auto-pruned 14d)
- No SQLite DBs in repo root
- Completed plans: delete from `tasks/`, git history preserves them

<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes` or `query_graph` instead of Grep
- **Understanding impact**: `get_impact_radius` instead of manually tracing imports
- **Code review**: `detect_changes` + `get_review_context` instead of reading entire files
- **Finding relationships**: `query_graph` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview` + `list_communities`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
|------|----------|
| `detect_changes` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context` | Need source snippets for review — token-efficient |
| `get_impact_radius` | Understanding blast radius of a change |
| `get_affected_flows` | Finding which execution paths are impacted |
| `query_graph` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes` | Finding functions/classes by name or keyword |
| `get_architecture_overview` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes` for code review.
3. Use `get_affected_flows` to understand impact.
4. Use `query_graph` pattern="tests_for" to check coverage.
