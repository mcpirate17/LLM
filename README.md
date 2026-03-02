# Aria Research + Aria Designer

Unified workspace for:
- `research/` — Aria (AI Scientist) backend + dashboard
- `aria_designer/` — visual architecture designer and runtime bridge

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
- Designer runtime/docs: `aria_designer/README.md`
- Designer workflow usage: `aria_designer/WORKFLOW_GUIDE.md`

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
cd aria_designer
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
- Native runner capability/cutover endpoint:
  - `GET /api/native-runner/capability` (includes `fallback_metrics` + `cutover_gate`)

Native cutover verification (strict no-legacy lane):
- env baseline:
  - `NATIVE_RUNNER_ENABLED=1`
  - `NATIVE_RUNNER_DISABLE_LEGACY_COMPILE_NATIVE_ENABLED=1`
  - `NATIVE_RUNNER_ABI_MODEL_ONLY=1`
- checks:
  - `python -m research.tools.check_no_legacy_compile`
  - `python -m research.tools.check_no_legacy_execution_paths`

Deprecation note:
- `NATIVE_RUNNER_LEGACY_ONLY` and `fallback_metrics.legacy_compile_invocations` are compatibility surfaces pending Phase-D removal; prefer ABI-model-only + no-legacy gates and `fallback_metrics.legacy_compile_count`.

Cutover gate readiness check (gate lane):
- `python -m research.tools.check_cutover_gate --offline --generate-compile-sample --generate-parity-sample`

## Troubleshooting

- If `python -m research` fails, confirm you are in `/home/tim/Projects/LLM` (parent of `research/`).
- If dashboard opens but Designer does not, check:
  - Aria logs for `/api/designer/ensure-running`
  - `aria_designer/.run/research_designer_boot.log`
- If ports are stuck:
  - `cd aria_designer && make dev-stop`
