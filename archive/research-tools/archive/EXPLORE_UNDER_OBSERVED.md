# explore_under_observed

Tests components through the full pipeline: compile → forward → rapid screening → S1 micro-train.

## How It Works

The tool generates neural architecture graphs that contain your target ops, compiles
them into PyTorch models, and runs them through the eval pipeline to see if they can
actually learn.

### Graph anatomy

A graph is built by stacking **template blocks**. Each block is a structural pattern
(like `residual_block`, `transformer_block`, `spiking_moe_block`) that expands into
several **ops** (the actual components — `rmsnorm`, `linear_proj`, `tropical_gate`, etc.).

```
Graph = [block 1] → [block 2] → ... → [block N]

Example with --n-blocks=2:
  Block 1: rmsnorm → state_space → linear_proj → add     (4 ops)
  Block 2: layernorm → softmax_attention → linear_proj → add  (4 ops)
  Total: 8 ops
```

The key parameters:

| What you control | Flag | Default | What it means |
|------------------|------|---------|---------------|
| Ops per graph | `--max-ops` | 18 | Hard cap on total components. Lower = simpler graphs |
| Graph depth | `--max-depth` | 12 | Longest path from input to output. Lower = shallower |
| Template blocks | `--n-blocks` | 2 | How many structural blocks are stacked. More = more complex |
| Model dimension | `--model-dim` | 256 | Hidden size. Smaller = faster, less capacity |
| Model layers | `--n-layers` | 4 | How many times the graph is repeated as layers. More = deeper model |

### Relationship between the parameters

- `--n-blocks=1 --max-ops=8` → simple single-block graphs (~5-8 ops)
- `--n-blocks=2 --max-ops=18` → default: two blocks, moderate complexity
- `--n-blocks=3 --max-ops=24` → complex multi-block architectures
- `--max-depth` limits how deep any single path can go (prevents extreme serial chains)
- `--model-dim=128 --n-layers=2` → fast iteration (small model, quick training)
- `--model-dim=512 --n-layers=6` → capacity testing (bigger model, slower)

### Evaluation pipeline

Each generated graph goes through:

1. **Compile** — `compile_model()` builds a PyTorch model (graph repeated `--n-layers` times)
2. **Forward (S0)** — `safe_eval()` checks it runs without NaN/Inf (30s timeout)
3. **Rapid screening** — `--rapid-steps` gradient steps to check loss decreases
4. **S1 micro-train** — fresh model, `--s1-steps` steps, passes if `loss_ratio < 0.95`

The script uses vocab_size=32000 (not 100K) because small exploration models can't
reach `ln(100277)=11.52` baseline in 500 steps. With 32K, `ln(32000)=10.37` is reachable.

## Two Modes

**forced** (default): Generates dedicated graphs per target op using template boosting.
The grammar biases heavily toward templates containing your target ops. Total graphs =
`(number of target ops) × --graphs-per-op`.

**weighted**: Generates a batch of `--n-graphs` graphs with under-observed ops boosted
via inverse-observation weighting. Falls back to forced mode for any ops that still
got zero coverage.

## Quick Start

```bash
# Test specific ops on GPU, save results to DB
python -m research.tools.explore_under_observed \
  --mode=forced --device=cuda --record \
  --ops state_space tropical_center gated_delta

# 20 graphs per op for statistical significance
python -m research.tools.explore_under_observed \
  --mode=forced --graphs-per-op=20 --device=cuda --record \
  --ops sliding_window_mask conv_only

# Auto-discover all ops with <20 observations
python -m research.tools.explore_under_observed \
  --mode=forced --device=cuda --threshold=20 --record

# Dry run — generate graphs only, no training, no DB writes
python -m research.tools.explore_under_observed --dry-run \
  --ops lif_neuron sparse_threshold
```

## All Arguments

### Target selection

| Flag | Default | Description |
|------|---------|-------------|
| `--ops OP [OP ...]` | - | Explore these specific ops (ignores `--threshold`) |
| `--threshold` | `20` | Auto-target ops with fewer than N observations |
| `--mode` | `forced` | `forced` = dedicated graph per op; `weighted` = boosted batch |

### Graph generation

| Flag | Default | Description |
|------|---------|-------------|
| `--max-ops` | `18` | Max ops (components) per graph |
| `--max-depth` | `12` | Max graph depth (longest path input→output) |
| `--n-blocks` | `2` | Template blocks stacked per graph (each = ~3-8 ops) |
| `--model-dim` | `256` | Hidden dimension of the model |
| `--boost-factor` | `50.0` | How aggressively target ops are boosted during generation |
| `--graphs-per-op` | `1` | Distinct graphs to generate per target op (forced mode) |
| `--n-graphs` | `50` | Total graphs in weighted batch (weighted mode) |
| `--max-retries` | `100` | Seed attempts per graph before giving up (forced mode) |

### Evaluation

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | `cpu` | `cpu` or `cuda` |
| `--rapid-steps` | `150` | Rapid screening gradient steps |
| `--s1-steps` | `500` | S1 micro-training steps |
| `--no-s1` | - | Skip S1 (faster, only tests compile + forward + rapid) |
| `--n-layers` | `4` | Layers in compiled model |

### Output

| Flag | Default | Description |
|------|---------|-------------|
| `--record` | - | Write results to lab_notebook.db in real-time |
| `--dry-run` | - | Only generate graphs, skip evaluation |
| `--db` | `research/lab_notebook.db` | Path to notebook database |
| `--output-dir` | `research/reports` | Directory for reports |
| `--seed` | `42` | Random seed |
| `-v, --verbose` | - | Debug-level logging |

## Recipes

```bash
# Fast smoke test — small model, 1 graph, no S1
python -m research.tools.explore_under_observed \
  --mode=forced --device=cuda --no-s1 \
  --model-dim=128 --n-layers=2 \
  --ops chebyshev_spectral_mix

# Deep single-block architectures (isolate one template)
python -m research.tools.explore_under_observed \
  --mode=forced --graphs-per-op=10 --device=cuda --record \
  --n-blocks=1 --max-ops=10 --max-depth=8 \
  --ops state_space tropical_center

# Complex multi-block architectures
python -m research.tools.explore_under_observed \
  --mode=forced --graphs-per-op=10 --device=cuda --record \
  --n-blocks=3 --max-ops=24 --max-depth=16 \
  --ops gated_delta integral_kernel diff_attention

# Large campaign — all under-observed ops, 5 graphs each
python -m research.tools.explore_under_observed \
  --mode=forced --threshold=60 --graphs-per-op=5 \
  --device=cuda --record

# Weighted sweep — 200 graphs, auto-boost under-observed
python -m research.tools.explore_under_observed \
  --mode=weighted --threshold=50 --n-graphs=200 \
  --device=cuda --record
```

## What `--record` Does

1. Creates an experiment in `experiments` tagged `forced_exploration`
2. Writes every graph to `program_results` (S0/S1 pass, loss_ratio, graph_json, etc.)
3. Updates `op_success_rates` (`n_used`, `n_stage0_passed`, `n_stage1_passed`)
4. Results tagged `model_source="forced_exploration"` to distinguish from organic runs
5. Flushed to DB after each evaluation — safe to kill mid-run

## Tuning Tips

| Problem | Fix |
|---------|-----|
| 0% S1 pass rate | `--s1-steps=1000` or `--graphs-per-op=5` (more attempts) |
| Too many rapid failures | `--rapid-steps=300` or `--rapid-steps=500` |
| Slow runs | `--no-s1` or `--model-dim=128 --n-layers=2` |
| GPU OOM | `--model-dim=128 --n-layers=2` |
| Graphs too simple | `--n-blocks=3 --max-ops=24` |
| Graphs too complex | `--n-blocks=1 --max-ops=8` |
| Op never appears in graph | `--boost-factor=100` (more aggressive placement) |

## Output

Reports written to `--output-dir`:
- `exploration_YYYYMMDD_HHMMSS.md` — per-op coverage table
- `exploration_YYYYMMDD_HHMMSS.json` — full structured data

Exit code: `0` if all target ops covered, `1` if any couldn't be placed.
