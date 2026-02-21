# Aria Designer Workflow Guide

Practical flow for designing, validating, and evaluating architectures.

## 1) Start Services

Standalone:

```bash
cd /home/tim/Projects/LLM/aria-designer
make setup
make dev
```

Integrated (recommended): start Aria dashboard and open Designer from there.

```bash
source /home/tim/venvs/llm/bin/activate
cd /home/tim/Projects/LLM
python -m research --mode=dashboard --port 5000
```

## 2) Build a Graph

- Drag components from the left palette.
- Connect compatible ports.
- Select nodes to edit parameters in Inspector.
- Use component help (`?`) and parameter help for usage guidance.

## 3) Use the Step Toolbar

- `Step 1: Validate` checks graph structure/config.
- `Step 2: Compile` verifies compile path.
- `Step 3: Test` runs fast execution checks.
- `Step 4: Run` executes runtime path.

Status and timing are shown in UI state/response payloads.

## 4) Ask Aria (Guided Refinement)

Use **Ask Aria** for patch proposals.

Preset quick intents are available:
- `Refine Fingerprint`
- `Refine Recommended`
- `Refine Compression`
- `Refine Sparsity`
- `Investigate`

You can still write custom prompts directly.

## 5) Import/Export

- `Import JSON` / `Export` for workflow files.
- `Export Py` for generated Python module output.
- `Import Research` to load workflows from AI Scientist survivors.

## 6) Run Through Aria Bridge

For deep evaluation use:
- `POST /api/v1/workflows/evaluate`
- `POST /api/v1/workflows/evaluate/stream`

Responses include:
- bridge run metadata
- lineage sync status
- semantic mapping warnings (`semantic_warnings`) when approximations are used

## 7) Designer-in-Aria UX

From Aria dashboard, Designer opens in drawer mode:
- Discoveries
- Leaderboard
- Program details
- Report rankings

Aria manages lifecycle:
- ensure running on open
- keepalive while open
- idle auto-stop policy when unused

## 8) Stop Services

```bash
cd /home/tim/Projects/LLM/aria-designer
make dev-stop
```
