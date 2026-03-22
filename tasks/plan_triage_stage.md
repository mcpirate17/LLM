# Plan: Post-Screening Triage Stage

## Problem

455 entries pass S1 screening. 10 get investigated. 5 get validated. 1 gets scaling measured. The scoring system has 901 possible points but most entries only get scored on ~130 points (loss + novelty) because the deep evals never run. The pipeline funnel is too narrow.

Key gaps across all 455 S1 passers:

| Eval | Entries scored | Coverage | Score budget |
|---|---|---|---|
| Loss ratio | 455 | 100% | 100 pts |
| WikiText | 118 | 26% | 55 pts |
| Scaling efficiency | 5 | 1% | 125 pts |
| Routing savings | 189 | 42% | 50 pts |
| Compression | 115 | 25% | 30 pts |
| Long-context | 4 | 1% | 70 pts |
| Robustness | 11 | 2% | 55 pts |
| NCD | 10 | 2% | 15 pts |
| Recursion/depth | 0 | 0% | 25 pts |

## Solution: Triage stage after S1 screening

Run cheap evals on **every S1 passer** immediately after screening completes, before the investigation gate decides whether to promote. Cost: ~3-5 seconds per entry on GPU (dominated by WikiText eval). This fills in the NULL scoring dimensions without the expensive multi-seed investigation/validation stages.

## What triage measures (cost per entry)

### Free (0ms) — already available from the graph/model

1. **Param efficiency estimate**: `reference_params_at_d768 / (candidate_params * (768/model_dim)^2)`. Published GPT-2 scaling curve gives reference params for any loss level. No training needed — just arithmetic on the candidate's param count and loss ratio. Fills `scaling_param_efficiency` (~125 pts budget).

2. **Routing op census**: Count routing/sparse/MoE ops in the graph. Extract routing telemetry from the screening run (expert counts, drop rates, confidence means, fast fractions). Already collected by `_record_routing_telemetry` during forward passes. Fills `n_routing_ops`, `n_sparse_ops`, `n_moe_ops`, `routing_expert_count`, `routing_confidence_mean`, `routing_drop_rate`, `routing_savings_ratio` (~65 pts budget).

3. **Activation sparsity**: Measure fraction of zero/near-zero activations in the final screening forward pass. Fills `activation_sparsity_score` (~10 pts budget).

4. **Compression ratio estimate**: Weight entropy → estimated bits-per-parameter. Fills `compression_ratio` (~30 pts budget).

### Cheap (~3s on GPU) — requires a forward pass on real data

5. **WikiText eval**: `screening_wikitext_eval()` already exists and runs in 2-5s. Currently only runs when `skip_screening_wikitext=False` (sometimes skipped). Make it always run for S1 passers. Fills `wikitext_score`, `wikitext_perplexity` (~55 pts budget).

### Not in triage (too expensive)

- **Long-context passkey/multi-hop**: Requires training on long sequences (~minutes). Stays in investigation stage.
- **Multi-seed robustness**: Requires 3+ training runs (~minutes). Stays in validation stage.
- **Full scaling comparison**: Requires retraining at d=512 (~minutes). Stays in validation stage.
- **NCD**: Requires compression algorithm run. Could be cheap enough but lower priority.

## Architecture: where triage plugs in

### Current flow
```
generate_graph → compile → rapid_screening (150 steps, 7s)
    → S0 pass? → S1 training (500-2500 steps)
        → S1 pass? → record_program_result → upsert_leaderboard
            → auto_escalate_screening (picks top 1 for investigation)
```

### New flow
```
generate_graph → compile → rapid_screening (150 steps, 7s)
    → S0 pass? → S1 training (500-2500 steps)
        → S1 pass? → **TRIAGE (3-5s)** → record_program_result → upsert_leaderboard
            → auto_escalate_screening (picks top N based on FULL composite score)
```

Triage runs **before** `record_program_result`, so the leaderboard entry is created with all triage fields populated. No separate backfill job needed.

### Implementation location

File: `research/scientist/runner/execution_training.py`

The S1 training result is built around line 1100-1170 (where `screening_wikitext_eval` already runs). Triage evals go right after the WikiText eval, in the same block where the trained model is still in memory.

```python
# After line ~1165 (end of WikiText eval block):

# ── Triage: cheap evals for composite score dimensions ──
if result.get("stage1_passed"):
    triage = _run_triage(model, graph, result, config, dev)
    result.update(triage)
```

### New function: `_run_triage`

File: `research/scientist/runner/execution_training.py` or new file `research/scientist/runner/execution_triage.py`

```python
def _run_triage(model, graph, result, config, device):
    """Cheap post-S1 evals to fill composite score dimensions. ~3-5s total."""
    triage = {}

    # 1. Param efficiency estimate (free)
    candidate_params = sum(p.numel() for p in model.parameters())
    loss = result.get("loss_ratio", 1.0)
    if loss < 0.95 and candidate_params > 0:
        # Scale to d=768 equivalent
        scale_factor = (768 / config.model_dim) ** 2
        scaled_params = candidate_params * scale_factor
        # GPT-2 reference: ~124M params achieves loss_ratio ~0.04 on random tokens
        # Interpolate: at our loss level, how many GPT-2 params would be needed?
        # Use published Kaplan scaling: L(N) = (N_c/N)^α, α≈0.076, N_c≈8.8e13
        # Invert: N(L) = N_c * L^(-1/α)
        # But simpler: compare against stored reference curve
        from research.eval.scaling_reference import ScalingCurveDB
        try:
            curve_db = ScalingCurveDB()
            ref_params = curve_db.params_for_loss("gpt2", loss * 2.3)  # convert ratio to nats
            if ref_params and ref_params > 0:
                triage["scaling_param_efficiency"] = ref_params / scaled_params
                triage["scaling_best_family"] = "gpt2"
                triage["scaling_confidence"] = "triage_estimate"
        except Exception:
            pass

    # 2. Routing census (free — from graph structure + telemetry)
    op_names = {n.op_name for n in graph.nodes.values() if not n.is_input}
    routing_ops = {"early_exit", "cascade", "route_lanes", "route_recursion",
                   "adaptive_recursion", "adaptive_lane_mixer", "n_way_sparse_router",
                   "routing_conditioned_compression", "token_merge"}
    sparse_ops = {"nm_sparse_linear", "semi_structured_2_4_linear", "block_sparse_linear",
                  "route_topk", "sparse_threshold"}
    moe_ops = {"moe_topk", "moe_2expert", "tropical_moe", "relu_gate_routing",
               "n_way_sparse_router", "compression_mixture_experts"}

    triage["n_routing_ops"] = len(op_names & routing_ops)
    triage["n_sparse_ops"] = len(op_names & sparse_ops)
    triage["n_moe_ops"] = len(op_names & moe_ops)

    # Extract routing telemetry from model (populated during screening forward pass)
    _extract_routing_telemetry(model, triage)

    # 3. Activation sparsity (from last forward pass)
    _estimate_activation_sparsity(model, triage)

    # 4. Compression ratio estimate (weight entropy)
    _estimate_compression(model, triage)

    return triage
```

### Changes to auto-escalation

File: `research/scientist/runner/results_auto_escalate_phase7.py`

Currently `auto_investigate_top_n = 1` — promote only the single best candidate. With triage filling in composite score dimensions, the auto-escalation decision is better informed. Raise to `auto_investigate_top_n = 3` so more entries flow through.

Also: the scoring function needs triage fields mapped to leaderboard columns. Check that `upsert_leaderboard` in `notebook_leaderboard.py` handles the new triage fields.

### Changes to leaderboard upsert

File: `research/scientist/notebook/notebook_leaderboard.py`

Ensure `upsert_leaderboard` maps triage result keys to the correct leaderboard columns:
- `scaling_param_efficiency` → `scaling_param_efficiency`
- `n_routing_ops` → already handled
- `n_sparse_ops` → already handled
- `n_moe_ops` → already handled
- `routing_savings_ratio` → from telemetry extraction
- `compression_ratio` → from weight entropy estimate
- `activation_sparsity_score` → from activation analysis

Most of these columns already exist in the leaderboard schema. The triage function just needs to produce the right keys.

## Backfill

For the 455 existing S1 entries with NULL dimensions, create a one-time backfill script:

File: `research/tools/backfill_triage.py`

```python
"""Backfill triage evals for existing S1-passing leaderboard entries.

Recompiles each graph, runs a single forward pass, and fills in:
- scaling_param_efficiency (from param count + published scaling curves)
- n_routing_ops, n_sparse_ops, n_moe_ops (from graph structure)
- compression_ratio (from weight entropy)
- activation_sparsity_score (from forward pass)

Does NOT retrain or modify loss metrics. Read-only on program_results,
write-only on leaderboard columns that are currently NULL.
"""
```

For entries where the model can't be recompiled (graph schema changes), fall back to graph-only metrics (routing census, param efficiency estimate from param_formula).

## What this unlocks

With triage running on every S1 passer:

1. **Composite scores jump** for entries with routing/sparse/MoE ops — currently scoring 0 on ~200 pts of budget because the fields are NULL.
2. **Auto-escalation becomes smarter** — promoting based on full composite score instead of just loss ratio means architecturally interesting entries (high routing savings, good compression) get investigated even if their loss is slightly worse.
3. **The "5x GPT" question becomes answerable** — with `scaling_param_efficiency` populated for all entries, we can immediately see which architectures beat GPT-2 on efficiency and by how much.
4. **Dashboard becomes useful** — the Component Health Grid and leaderboard show real data instead of NULLs for most scoring dimensions.

## Cost

- Per S1 passer: ~3-5 seconds additional GPU time (dominated by WikiText eval)
- For 455 existing entries backfill: ~30 minutes one-time GPU run
- Ongoing: negligible — triage adds <5s to an evaluation cycle that takes 30-60s

## Implementation order

1. Write `_run_triage()` and helper functions
2. Integrate into `execution_training.py` after WikiText eval
3. Verify leaderboard upsert handles triage fields
4. Write backfill script for existing entries
5. Raise `auto_investigate_top_n` from 1 to 3
6. Run backfill
7. Rebuild dashboard to see results
