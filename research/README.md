# Research AI Scientist

Autonomous architecture-discovery system for neural network layer research.

## What this project does

- Synthesizes candidate computation graphs from a grammar
- Runs staged evaluation and lightweight training loops
- Tracks results in a lab notebook (SQLite)
- Serves a dashboard + strategy briefing API for monitoring and iteration

## Repository layout

- `scientist/` — orchestration (runner, analytics, notebook, API)
- `synthesis/` — graph generation/compiler/grammar/primitives
- `eval/` — metrics, sandbox, FLOPs, CKA/fingerprint helpers
- `training/` — training recipes and optimizers
- `search/` — evolution/novelty strategies
- `dashboard/` — React dashboard
- `tools/` — utility scripts (CKA export/integrity checks, audits)

## Environment setup

```bash
source /home/tim/venvs/llm/bin/activate
```

## Important launch rule

Run package entrypoints from the parent directory (`/home/tim/Projects/LLM`), not from inside `research/`.

### Correct

```bash
cd /home/tim/Projects/LLM
/home/tim/venvs/llm/bin/python -m research --mode=synthesize --n 10 --device cpu
```

### Incorrect

```bash
cd /home/tim/Projects/LLM/research
/home/tim/venvs/llm/bin/python -m research ...
# Fails with: No module named research
```

## Common run modes

```bash
# Synthesize programs
/home/tim/venvs/llm/bin/python -m research --mode=synthesize --n 20 --device cpu

# Continuous scientist loop
/home/tim/venvs/llm/bin/python -m research --mode=continuous --n 10 --device cpu

# Evolutionary search
/home/tim/venvs/llm/bin/python -m research --mode=evolve --n 20 --device cpu

# Dashboard API/UI backend
/home/tim/venvs/llm/bin/python -m research --mode=dashboard --port 5000
```

## Tests

```bash
/home/tim/venvs/llm/bin/pytest tests/ -x
```

## Notes and troubleshooting

- Optional `hydra.data` import may show as unresolved in static analysis when HYDRA is not installed/configured; standard synth runs still work.
- CKA reference warnings can fall back to heuristic mode if manifest fields are missing; run quality checks with:

```bash
/home/tim/venvs/llm/bin/python tools/cka_artifact_integrity.py --artifact-dir artifacts/cka_references/v1 --scaffold-if-missing --version v1 --code-version local --strict
```
