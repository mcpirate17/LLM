# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Multi-Agent Coordination

**IMPORTANT**: Before starting any file modifications, read `.current_work.md` in the project root.
- If another agent has claimed a file, do not modify it
- When you start working on a file, add it to the "Claimed" section
- When you finish, move it to "Recently Completed" with the commit hash
- Add any non-obvious context to "Notes" so the other agent doesn't break your work

## Shared Observations Workflow

To keep observations synchronized across agents:

- **Single source of observations:** `copilot.md`
- Do **not** create separate observation files per agent.
- Add updates in-place under a clearly labeled section with agent name and date
	(e.g., `## Claude Code Observations (YYYY-MM-DD)`).
- If updating observations, also add a short note in `.current_work.md` under "Recently Completed"
	so the other agent knows the shared observations file changed.

## Current Work Status (GitHub Copilot)

Updated: 2026-02-15

- Active focus: pipeline safety and interoperability checks (CLI entrypoint validity, dashboard API contract tests, dead-code audit tooling, evolution search reliability).
- Recent files touched: `__main__.py`, `search/evolution.py`, `tests/test_integration.py`, `tools/dead_code_audit.py`, `mathspaces/__init__.py`, `scientist/llm/__init__.py`.
- Detailed claim/completion tracking remains in `.current_work.md` (authoritative for collision avoidance).

## Environment Setup

```bash
# Python venv (required for all Python commands)
source /home/tim/venvs/llm/bin/activate

# Run tests
/home/tim/venvs/llm/bin/pytest tests/ -x

# Compile check
python -m py_compile <file>
```

## Project Structure

AI scientist system for autonomous architecture discovery via grammar-based program synthesis.

- `scientist/` -- orchestration layer (runner, persona, analytics, notebook, API)
- `synthesis/` -- graph generation (grammar, primitives, graph DAG, compiler, serializer)
- `eval/` -- evaluation pipeline (sandbox, metrics, flops, fingerprint, baseline)
- `training/` -- micro-training for evaluation stages
- `search/` -- search strategies (evolution, novelty search)
- `morphological_box.py` -- architecture spec dimensions and constraint checking
- `database.py` -- SQLite storage for morphological box experiments
- `evaluator.py` -- stage 0/1 evaluation entry points
- `explorer.py` -- CLI for rolling and evaluating random architectures
- `dashboard/` -- React dashboard for monitoring

## Key Conventions

- `scientist/notebook.py` `LabNotebook` is the central data store (SQLite). Dashboard summary keys: `stage1_survivors`, `total_programs_evaluated`, `total_experiments`, etc.
- `synthesis/graph.py` `ComputationGraph` caches computed properties (`_cache` dict). Always mutate via `add_op`/`set_output`, never assign to `.nodes` directly on a live graph.
- `database.py` `ExperimentDB` supports `with db.batch():` for batched commits.
- `scientist/analytics.py` imports `get_primitive` and `GrammarConfig` at module level.
- Grammar weights use multiplicative contrast amplification: `default * (s1_rate/mean)^2 * (1+novelty)`, clamped [0.5, 8.0]. Floor of 0.5 prevents category starvation.

## Testing

92+ integration tests in `tests/test_integration.py`. Always run after changes:
```bash
/home/tim/venvs/llm/bin/pytest tests/ -x --tb=short
```
