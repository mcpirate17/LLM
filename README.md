# Aria Research + Aria Designer

Unified workspace for:
- `research/` — Aria (AI Scientist) backend + dashboard
- `aria-designer/` — visual architecture designer and runtime bridge

## Quick Start (Recommended)

From project root:

```bash
source /home/tim/venvs/llm/bin/activate
cd /home/tim/Projects/LLM
python -m research --mode=dashboard --port 5000
```

Open `http://localhost:5000`.

When you click a Designer entry point in the dashboard, Aria auto-starts Aria Designer services (`api:8091`, `ui:5174`) as needed.

## Repos and Docs

- Aria runtime/docs: `research/README.md`
- Designer runtime/docs: `aria-designer/README.md`
- Designer workflow usage: `aria-designer/WORKFLOW_GUIDE.md`

## Common Commands

```bash
# Aria synth run
python -m research --mode=synthesize --n 20 --device cpu

# Aria continuous loop
python -m research --mode=continuous --n 10 --device cpu

# Aria evolutionary mode
python -m research --mode=evolve --n 20 --device cpu

# Aria dashboard
python -m research --mode=dashboard --port 5000
```

```bash
# Aria Designer local dev (manual)
cd aria-designer
make setup
make dev
make dev-stop
```

## Integration Notes

- Dashboard-to-designer lifecycle endpoints are under `research/scientist/api.py`:
  - `GET /api/designer/lifecycle`
  - `POST /api/designer/ensure-running`
  - `POST /api/designer/touch`
  - `POST /api/designer/stop`
- Designer lineage sync endpoints in Aria:
  - `POST /api/designer/lineage/sync`
  - `GET /api/designer/lineage`
  - `GET /api/designer/lineage/<run_id>`

## Troubleshooting

- If `python -m research` fails, confirm you are in `/home/tim/Projects/LLM` (parent of `research/`).
- If dashboard opens but Designer does not, check:
  - Aria logs for `/api/designer/ensure-running`
  - `aria-designer/.run/research_designer_boot.log`
- If ports are stuck:
  - `cd aria-designer && make dev-stop`
