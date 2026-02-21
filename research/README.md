# Aria (Research)

Aria is the AI Scientist runtime: synthesis, evaluation, experiment orchestration, and dashboard APIs.

## Environment

```bash
source /home/tim/venvs/llm/bin/activate
```

Run from the parent directory (important):

```bash
cd /home/tim/Projects/LLM
```

## Run Modes

```bash
# Dashboard (recommended entrypoint)
python -m research --mode=dashboard --port 5000

# Synthesis experiment
python -m research --mode=synthesize --n 20 --device cpu

# Continuous scientist loop
python -m research --mode=continuous --n 10 --device cpu

# Evolutionary search
python -m research --mode=evolve --n 20 --device cpu
```

## Dashboard + Designer Integration

Aria Dashboard is served by `python -m research --mode=dashboard`.

Designer integration is automatic from dashboard entrypoints:
- Aria calls `/api/designer/ensure-running` before loading embedded Designer.
- Idle lifecycle is managed by Aria with `/api/designer/touch` keepalive and auto-stop policy.

Lifecycle endpoints:
- `GET /api/designer/lifecycle`
- `POST /api/designer/ensure-running`
- `POST /api/designer/touch`
- `POST /api/designer/stop`

Lineage endpoints:
- `POST /api/designer/lineage/sync`
- `GET /api/designer/lineage`
- `GET /api/designer/lineage/<run_id>`

## Repo Layout

- `scientist/` — runner, API, notebook, analytics, persona
- `synthesis/` — graph/primitives/compiler/grammar
- `eval/` — sandbox, metrics, fingerprinting, pruning/perf helpers
- `dashboard/` — React app (used by `--mode=dashboard`)
- `search/`, `training/`, `tools/` — search/training/utilities

## Frontend Dev (optional)

If you want standalone frontend dev hot reload:

```bash
cd /home/tim/Projects/LLM/research/dashboard
npm install
npm start
```

## Tests

```bash
cd /home/tim/Projects/LLM/research
python -m pytest tests/ -x
```

## Troubleshooting

- `No module named research`: you ran from inside `research/`; move to `/home/tim/Projects/LLM`.
- Designer fails to embed: verify `aria-designer/tools/dev_up.sh` and `dev_down.sh` exist and are executable.
