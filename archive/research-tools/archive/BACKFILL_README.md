# Backfill Runner

Unified backfill tool for all scoring probes. Single entry point replaces the individual `backfill_*.py` scripts.

```
python -m research.tools.backfill --probe <probe> [options]
```

## Probes

| Probe | What it measures | GPU? | Micro-trains? | ~Time/entry |
|-------|-----------------|------|--------------|-------------|
| `binding` | AR + induction head + copy-at-distance | Yes | Yes (500 steps) | ~60s |
| `hellaswag` | 4-choice commonsense reasoning | Yes | Yes (500 steps) | ~5s |
| `blimp` | 67 linguistic minimal pair subtasks | Yes | Yes (500 steps) | ~50s |
| `triage` | Routing/compression/sparsity/params | Yes | No | ~2s |
| `fingerprint` | CKA similarity + novelty (subprocess-isolated) | Yes | No | ~15s |
| `rescore` | Recompute composite scores (no GPU) | No | No | <1s |

## Common Recipes

```bash
# Full evaluation of all dashboard-visible entries (3-pass averaged)
python -m research.tools.backfill --probe all --top 100 --tier breakthrough,validation,investigation --passes 3 --force

# Top 15 screening candidates (next to escalate), 3-pass averaged
python -m research.tools.backfill --probe all --top 15 --tier screening --passes 3

# Fast structural pass on screening (no training needed, instant)
python -m research.tools.backfill --probe triage --top 50 --tier screening

# Binding probes only on investigation+ entries
python -m research.tools.backfill --probe binding --top 50 --tier investigation,validation

# BLiMP + HellaSwag together (single model load per entry)
python -m research.tools.backfill --probe blimp,hellaswag --top 20

# Rescore all leaderboard entries after a scoring formula change
python -m research.tools.backfill --probe rescore

# Preview what would be evaluated
python -m research.tools.backfill --probe all --top 100 --tier breakthrough,validation,investigation --dry-run

# Force re-evaluation with 3-pass averaging (overwrites existing data)
python -m research.tools.backfill --probe blimp --force --passes 3 --top 10

# CPU-only (slow but works without GPU)
python -m research.tools.backfill --probe hellaswag --device cpu
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--probe` | (required) | Comma-separated probe names, `all`, or `rescore` |
| `--top` | 50 | Max entries per tier |
| `--tier` | `validation,investigation,breakthrough,screening` | Comma-separated tiers |
| `--device` | `cuda` (if available) | `cuda` or `cpu` |
| `--train-steps` | 500 | Micro-training steps before probes that need it |
| `--passes` | 1 | Run N passes and average results (reduces init variance) |
| `--fp-timeout` | 30 | Fingerprint subprocess timeout in seconds |
| `--force` | off | Re-evaluate entries that already have data |
| `--dry-run` | off | Show candidates without running probes |

## Multi-pass averaging

Models are reconstructed from `graph_json` with fresh random weights each time. Probes like BLiMP and HellaSwag evaluate learned behavior, so results vary across random initializations. Use `--passes N` to:

1. Reconstruct + micro-train the model N times independently
2. Run all requested probes on each trained model
3. Average numeric results (accuracy, AUC, sparsity scores)
4. Write the averaged values once

For dashboard-quality scores, use `--passes 3`. For quick exploration, `--passes 1` is fine.

## How it works

1. Queries leaderboard entries filtered by tier and missing data
2. For each entry, checks which probes still need data (skips already-populated probes unless `--force`)
3. Reconstructs model from stored `graph_json`
4. Micro-trains if any active probe requires it (binding, blimp, hellaswag)
5. Runs all requested probes, merging results
6. If `--passes N > 1`, repeats steps 3-5 and averages
7. Writes results to both `program_results` and `leaderboard` tables
8. Recomputes `composite_score` from scratch with all available data

Scores are computed from absolute values, not deltas. Re-running does not subtract or accumulate — it recomputes from the full metric set each time.

## Scoring impact

Each probe contributes to `compute_composite_v7()`:

| Probe | Max points | Condition |
|-------|-----------|-----------|
| Binding (AR + induction + binding_auc) | 120pt | S-curved composite, + 20% penalty if all signals ~0 |
| BLiMP | 40pt | Above 50% chance, S-curved at 60% frontier |
| HellaSwag | 0pt | Gates disabled at nano scale (data collected for analysis) |
| Triage | 30pt sparsity + 50pt routing + 30pt compression | Populates structural metrics |
| Fingerprint | 40pt novelty + 15pt NCD | CKA-based novelty scoring |

## Legacy scripts

The individual `backfill_*.py` scripts still exist for backward compatibility. This runner supersedes them:

| Old script | Equivalent |
|------------|-----------|
| `backfill_binding.py` | `backfill.py --probe binding` |
| `backfill_hellaswag.py` | `backfill.py --probe hellaswag` |
| `backfill_cka_novelty.py` | `backfill.py --probe fingerprint` |
| `backfill_triage.py` | `backfill.py --probe triage` |
| `rescore_all_v7.py` | `backfill.py --probe rescore` |
