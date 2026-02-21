# Aria Designer

Visual architecture editor for Aria workflows, with compile/evaluate bridge into `research/`.

## Quick Start

```bash
cd /home/tim/Projects/LLM/aria-designer
make setup
make dev
```

- API: `http://127.0.0.1:8091`
- UI: `http://localhost:5174`

Stop both:

```bash
cd /home/tim/Projects/LLM/aria-designer
make dev-stop
```

## Preferred Usage Path

Start Aria dashboard:

```bash
source /home/tim/venvs/llm/bin/activate
cd /home/tim/Projects/LLM
python -m research --mode=dashboard --port 5000
```

Then open Designer from dashboard buttons (Discoveries / Leaderboard / Program details / Report rankings).  
Aria will auto-start Designer services via lifecycle endpoints.

## Core Features

- Component palette + drag/drop graph editing
- Validate / Compile / Test / Run flow in toolbar
- Rich inspector property editing + config validation
- Ask Aria patch proposals
- Import workflows from research survivors
- Bridge evaluation via research pipeline
- Lineage sync and lifecycle auto-management

## Key API Endpoints

### Components
- `GET /api/v1/components`
- `GET /api/v1/components/{id}`
- `GET /api/v1/components/{id}/properties`
- `POST /api/v1/components/{id}/validate-config`
- `GET /api/v1/components/{id}/execution-capability`
- `GET /api/v1/integration/bridge-gap-report`

### Workflows
- `POST /api/v1/workflows/validate`
- `POST /api/v1/workflows/compile`
- `POST /api/v1/workflows/run`
- `POST /api/v1/workflows/evaluate`
- `POST /api/v1/workflows/evaluate/stream`
- `PUT /api/v1/workflows/{id}`
- `GET /api/v1/workflows/{id}`
- `GET /api/v1/workflows`

### Aria Assist
- `POST /api/v1/aria/suggest-components`
- `POST /api/v1/aria/propose-patch`
- `POST /api/v1/aria/generate-patch-from-prompt`
- `POST /api/v1/aria/apply-patch`

## Testing

```bash
cd /home/tim/Projects/LLM/aria-designer
python -m pytest tests/test_api.py tests/test_bridge.py -q
python tools/validate_manifests.py
python tools/audit_aria_integration.py
```

Optional:

```bash
make test
make test-runtime
make test-native
```

## Troubleshooting

- `npm run dev` from repo root fails: run from `aria-designer/ui` or use `make dev` from `aria-designer`.
- Port collisions: run `make dev-stop` then restart.
- If designer launched by Aria but blank, inspect:
  - `aria-designer/.run/research_designer_boot.log`
  - browser devtools network for `/api/designer/ensure-running`.

## Collaboration

- Active plan: `.current_plan.md` (single source of truth)
- Completed history: `.current_plan.archive.md`
