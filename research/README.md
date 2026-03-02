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

Native runner telemetry endpoint:
- `GET /api/native-runner/capability`
  - includes `fallback_metrics` and backend-computed `cutover_gate` (`status`, `ready`, `checks`)
  - useful for rollout decisions and dashboard parity/legacy/fallback health
  - example response shape:

```json
{
  "enabled": true,
  "strict": false,
  "designer_runtime_available": true,
  "status": "ready",
  "fallback_metrics": {
    "total_compiles": 42,
    "native_enabled_compiles": 42,
    "fallback_compiles": 3,
    "fallback_rate": 0.0714,
    "legacy_compile_count": 3,
    "legacy_compile_invocations": 3,
    "max_allowed_fallback_rate": "0.10",
    "max_allowed_legacy_compile_count": "5",
    "max_allowed_legacy_compile_invocations": "5"
  },
  "cutover_gate": {
    "ready": true,
    "status": "ready",
    "checks": [
      { "name": "fallback_rate", "active": true, "pass": true, "actual": 0.0714, "limit": 0.1 },
      { "name": "legacy_compile_invocations", "active": true, "pass": true, "actual": 3, "limit": 5 }
    ]
  }
}
```

Native runner cutover env controls:
- `NATIVE_RUNNER_MAX_FALLBACK_RATE` (0..1): fail when fallback ratio exceeds threshold
- `NATIVE_RUNNER_MAX_LEGACY_COMPILE_INVOCATIONS` (int): fail when legacy compile usage exceeds threshold
- `NATIVE_RUNNER_REQUIRE_PARITY_PASS` (`1|0`): include sampled ABI parity as required cutover gate
- `NATIVE_RUNNER_DISABLE_LEGACY_COMPILE` (`1|0`): hard gate that rejects compile when legacy compile path would be used (use for final cutover canaries)
- `NATIVE_RUNNER_DISABLE_LEGACY_COMPILE_NATIVE_ENABLED` (`1|0`): rejects legacy compile whenever native-runner mode is enabled (`NATIVE_RUNNER_ENABLED=1`), while leaving disabled-mode fallback behavior unchanged
- `NATIVE_RUNNER_ABI_MODEL_ONLY` (`1|0`): when native runner is enabled, build an inference-only model backed by runner ABI session instead of legacy compiler; requires successful ABI session preparation

Deprecated (compatibility window before Phase-D removal):
- `NATIVE_RUNNER_LEGACY_ONLY` (`1|0`): deprecated emergency bypass for native logic.
- `fallback_metrics.legacy_compile_invocations`: deprecated alias of `fallback_metrics.legacy_compile_count`.

Strict native no-legacy canary lane (CI/verification):
- `NATIVE_RUNNER_ENABLED=1`
- `NATIVE_RUNNER_DISABLE_LEGACY_COMPILE_NATIVE_ENABLED=1`
- `NATIVE_RUNNER_ABI_MODEL_ONLY=1`
- verify with:
  - `python -m research.tools.check_no_legacy_compile`
  - `python -m research.tools.check_no_legacy_execution_paths`

Execution mode classification (capability payload):
- `native_abi_model_only`
- `native_legacy_fallback`
- `legacy_only`

Legacy removal rollout plan:
- `research/CUTOVER_REMOVAL_PLAN.md`

Cutover gate helper:
- `python -m research.tools.check_cutover_gate --base-url http://127.0.0.1:5000`
- `python -m research.tools.check_cutover_gate --offline` (no API server required; reads local native-runner report)
- `python -m research.tools.check_cutover_gate --offline --generate-parity-sample` (deterministically seeds one parity-pass sample for strict parity-gated canaries)
- `python -m research.tools.check_cutover_gate --offline --generate-compile-sample` (runs one deterministic compile sample so legacy/fallback gates evaluate against real compile telemetry)
- add `--allow-waiting` during observe phase

## Architecture

### Synthesis Pipeline

Programs are generated from a probabilistic grammar over 75 primitive ops across 11 categories:

- **elementwise** (unary/binary), **reduction**, **linear_algebra**, **structural** — basic tensor ops
- **parameterized** — learnable ops (linear_proj, conv1d, selective_scan, swiglu_mlp, rwkv_channel, moe_topk, etc.)
- **mixing** — sequence mixing (softmax_attention, linear_attention, graph_attention, state_space, conv_only)
- **functional** — operator-learning (basis_expansion, integral_kernel, fixed_point_iter)
- **frequency**, **math_space**, **sequence** — domain-specific ops

Grammar weights are learned from op success rates via multiplicative contrast amplification with EMA smoothing.

### Evaluation Stages

- **S0**: Compile + forward pass (shape/CUDA sanity)
- **S0.5**: Stability check (grad norms, NaN/Inf detection)
- **S1**: Micro-training (128-step loss trajectory)
- **Scale-up**: Full training with behavioral fingerprinting and novelty scoring

Only S1 survivors and S1 failures with learning signal are stored in the database. S0 failures are tracked inline for op statistics but not persisted (saves ~80% storage).

### Novelty Scoring

S1 survivors get behavioral fingerprints computed while the model is still alive (in the S1 worker). Novelty is scored as a blend of structural novelty (graph topology) and behavioral novelty (fingerprint distance). Programs without fingerprints fall back to structural-only scoring at reduced confidence.

### Aria Chat Actions

Aria (the AI scientist persona) can take these actions via chat:

| Action | Purpose |
|---|---|
| `adjust_config` | Modify experiment parameters (n_programs, max_depth, excluded_ops, etc.) |
| `adjust_grammar` | Adjust grammar category weights |
| `start_experiment` | Launch synthesis, evolution, novelty, or refinement experiments |
| `edit_file` | Search-and-replace edits to .py/.js files (syntax-checked, auto-backup) |
| `spawn_agent` | Spawn local Ollama agent for multi-file investigation/fixes |
| `maintain_database` | DB housekeeping: purge junk, reset op stats, clear toxic signatures, vacuum |

### Toxic Pattern Detection

Op-pair bigrams are tracked in `failure_signatures`. Combinations that fail >85% of the time (min 5 observations) are blocklisted. Graphs with >50% toxic bigrams are skipped before evaluation.

## Repo Layout

- `scientist/` — runner, API, notebook, analytics, persona
- `synthesis/` — graph/primitives/compiler/grammar (75 ops, 11 categories)
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

## CI / Branch Protection

Dashboard hook-order safety check is enforced by GitHub Actions workflow:
- `.github/workflows/dashboard-hook-order.yml`
- Job name: `hook-order-check`

Recommended branch protection on your default branch:
- Require status checks to pass before merging.
- Add required check: `hook-order-check`.

Local pre-push equivalent:

```bash
cd /home/tim/Projects/LLM/research/dashboard
npm run check:hook-order
```

## Troubleshooting

- `No module named research`: you ran from inside `research/`; move to `/home/tim/Projects/LLM`.
- Designer fails to embed: verify `aria_designer/tools/dev_up.sh` and `dev_down.sh` exist and are executable.
