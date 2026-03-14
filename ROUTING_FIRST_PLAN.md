# Phase 2: Routing-First Architecture Synthesis

## Thesis

A model that routes easy tokens through cheap paths and hard tokens through expensive
paths is fundamentally more efficient than a dense model. This isn't a feature to add
later — routing IS the architecture. The grammar must produce routing-aware architectures
by design, not hope routing emerges from random op composition.

## What We Already Have

The infrastructure is surprisingly mature:

### Routing Primitives (17 ops registered)
- **Route signals**: `entropy_router`, `token_type_classifier`, `route_topk`, `route_lanes`, `route_recursion`
- **Routing control**: `mod_topk`, `early_exit`, `cascade`, `speculative`, `adaptive_recursion`, `token_merging`
- **Adaptive experts**: `adaptive_lane_mixer` (3-way fast/medium/hard), `mixed_recursion_gate`, `routing_conditioned_compression`
- **MoE**: `moe_topk`, `moe_2expert`, `topk_gate`

### Token Difficulty Estimation
- C kernel: `aria_core.difficulty_scorer_f32` — 2-layer MLP, (B,S,D) → (B,S,1) sigmoid
- Python fallback in `aria_designer/components/routing/difficulty_scorer/`
- `entropy_router` computes softmax entropy as difficulty proxy

### C Kernels for Adaptive Routing (aria_core)
- `conditional_dispatch_f32`: extract tokens for a specific lane
- `conditional_gather_f32`: reconstruct sequence from lane outputs
- `lane_router_threshold_f32`: assign tokens to lanes by learned thresholds
- `load_balance_loss_f32`: auxiliary loss for balanced lane utilization

### Templates (2 routing-aware, but not routing-mandatory)
- `tpl_conditional_compute`: norm → entropy_router → sparse_core → gate → residual
- `tpl_token_merge_block`: norm → token_merge → mixer → sparse_ffn → residual

### Evaluation
- `eval/routing_heatmap.py`: Gini coefficient, routing entropy, collapse detection
- Routing telemetry: tokens_total, tokens_processed, expert_counts, entropy_sum

### Problem
The grammar CAN produce routing architectures, but doesn't MANDATE them. Out of 1377
programs in the latest novelty search, 0 used math-space ops and routing was incidental.
The `conditional_compute` template exists but competes equally with 12 other templates.

---

## Phase 2 Plan

### P2.0: Routing-Mandatory Grammar Mode (Priority: Critical)
**Files**: `synthesis/grammar.py`, `synthesis/templates.py`

Add a `GrammarConfig.routing_first()` class method that:
1. **Mandates a difficulty scorer** in every generated graph — either `entropy_router`
   or `token_type_classifier` must appear as the first non-input op
2. **Mandates lane routing** — every graph must have at least 2 parallel paths with
   different compute costs (fast path = identity/linear, slow path = attention/MLP/exotic)
3. **Mandates a merge point** — lane outputs must be recombined via learned gating

This is a structural constraint, not a weight bias. The grammar CANNOT produce a
graph without these three components.

Implementation:
```
difficulty_scorer → lane_router → [fast_path, slow_path] → weighted_merge → output
```

The fast_path and slow_path are slots that the grammar fills with any compatible ops.
This preserves search diversity while guaranteeing the routing structure.

### P2.1: Routing-First Templates (Priority: Critical)
**File**: `synthesis/templates.py`

Create 4 new mandatory-routing templates:

#### `tpl_difficulty_routed_block`
```
input → difficulty_scorer → lane_router(2 lanes)
  ├─ fast_lane: rmsnorm → linear_proj (or identity)
  └─ slow_lane: rmsnorm → [SLOT: attention/ssm/conv/exotic] → ffn
→ conditional_gather(weighted by difficulty) → residual
```

#### `tpl_three_lane_adaptive`
```
input → difficulty_scorer → lane_router(3 lanes)
  ├─ skip_lane:   identity (easiest tokens, ~40%)
  ├─ medium_lane: linear_proj_down → gelu → linear_proj_up (~35%)
  └─ hard_lane:   [SLOT: full attention block] → ffn (~25%)
→ conditional_gather → residual
```

#### `tpl_recursive_depth_router`
```
input → difficulty_scorer → route_recursion(max_depth=4)
→ for depth in 1..max_depth:
    tokens_at_depth = select(difficulty >= threshold[depth])
    tokens_at_depth = [SLOT: transform_block](tokens_at_depth)
→ merge all depths → residual
```

#### `tpl_early_exit_cascade`
```
input → layer_1 → difficulty_scorer
  ├─ easy (exit early): project to output
  └─ hard (continue):   layer_2 → difficulty_scorer
      ├─ medium (exit):  project to output
      └─ hardest:        layer_3 → layer_4 → project to output
→ merge exits → output
```

### P2.2: Difficulty-Aware Training (Priority: High)
**Files**: `scientist/runner/execution_training.py`, `eval/engine.py`

The training loop must:
1. **Collect routing statistics** during training — what fraction of tokens take each path
2. **Add load-balance auxiliary loss** — prevent all tokens routing to one lane
   (use existing `load_balance_loss_f32` C kernel)
3. **Report routing efficiency** — `tokens_skipped / tokens_total` as a first-class metric
   alongside loss_ratio

New metrics in `program_results`:
- `routing_fast_fraction`: fraction of tokens that took the cheapest path
- `routing_balance_score`: how evenly distributed across lanes (0=collapsed, 1=uniform)
- `routing_effective_flops`: actual FLOPs consumed accounting for routing savings

### P2.3: Efficiency-Aware Scoring (Priority: High)
**Files**: `scientist/leaderboard_scoring.py`

Change the composite score to reward routing efficiency:
- Current: `efficiency_multiple` is geometric mean of 6 ratios vs GPT-2
- New: Add `routing_efficiency_bonus` — models that achieve same loss with fewer
  effective FLOPs (because easy tokens skip compute) get a multiplicative bonus
- Formula: `effective_efficiency = efficiency_multiple * (1 + routing_savings_ratio)`
  where `routing_savings_ratio = 1 - (effective_flops / dense_flops)`

A model with 0.38 loss_ratio and 40% routing savings beats a model with 0.35 loss_ratio
and 0% routing savings, because at scale the routing model uses 60% of the compute.

### P2.4: Difficulty Scorer Pretraining (Priority: Medium)
**File**: `training/difficulty_pretraining.py` (new)

The difficulty scorer needs signal to learn from. Options:
1. **Entropy-based**: Use the model's own output entropy as ground truth —
   tokens where the model is uncertain (high entropy) are "hard"
2. **Loss-based**: Per-token cross-entropy loss from a reference model —
   tokens with high loss are hard
3. **Self-supervised**: Train difficulty scorer end-to-end with the routing
   architecture, using the load-balance loss to prevent collapse

Option 3 is cleanest (no external reference needed) but risks collapse.
Option 1 is practical and can bootstrap from any trained model.

Recommendation: Start with option 1 (entropy-based) for initial experiments,
add option 3 (end-to-end) once routing architectures are surviving training.

### P2.5: Grammar Integration (Priority: Medium)
**File**: `synthesis/grammar.py`

Add to `GrammarConfig`:
```python
routing_mandatory: bool = False       # Force routing structure in every graph
routing_min_lanes: int = 2            # Minimum number of routing lanes
routing_fast_path_ops: List[str]      # Allowed ops for fast lane
routing_slow_path_ops: List[str]      # Allowed ops for slow lane
difficulty_scorer_type: str = "entropy"  # "entropy" or "learned"
```

When `routing_mandatory=True`:
- Template selection draws ONLY from routing templates (P2.1)
- The grammar fills slots within the routing structure
- Validator rejects graphs without a difficulty scorer + lane router

### P2.6: Routing-Aware Fingerprinting (Priority: Low)
**File**: `eval/fingerprint.py`

Add routing-specific fingerprint dimensions:
- `routing_selectivity`: How much does difficulty score vary across tokens?
  (std of difficulty scores — higher = more selective routing)
- `routing_compute_ratio`: Ratio of slow-path FLOPs to fast-path FLOPs
- `routing_lane_correlation`: Does routing correlate with token position,
  content, or context length?

These help the novelty search distinguish between architectures that route
differently, not just architectures with different dense compute.

---

## Execution Order

| Step | Task | Depends On | Effort |
|------|------|-----------|--------|
| 1 | P2.1: Routing-first templates | — | 1 day |
| 2 | P2.0: routing_mandatory grammar mode | P2.1 | 1 day |
| 3 | P2.2: Difficulty-aware training loop | P2.0 | 1 day |
| 4 | P2.5: Grammar integration + validator | P2.0 | 0.5 day |
| 5 | P2.3: Efficiency-aware scoring | P2.2 | 0.5 day |
| 6 | Run routing-mandatory novelty search | P2.0-P2.3 | — |
| 7 | P2.4: Difficulty scorer pretraining | P2.2 results | 1 day |
| 8 | P2.6: Routing-aware fingerprinting | P2.2 results | 0.5 day |

Steps 1-5 unblock the first routing-mandatory search. Steps 6-8 refine based on results.

---

## Success Criteria

1. **Routing architectures survive training**: >15% of routing-mandatory programs
   pass stage 1 (vs ~10% for current random synthesis)
2. **Routing is non-trivial**: The difficulty scorer learns to differentiate tokens —
   routing_selectivity > 0.1 (not all tokens going to same lane)
3. **Efficiency wins**: At least one routing architecture achieves comparable loss_ratio
   to the best dense architecture (<1.2x) while using <70% of the FLOPs
4. **Novelty**: Routing architectures occupy a distinct region of fingerprint space
   (CKA distance from dense architectures > 0.3)

## Key Insight

The grammar currently treats routing ops as "risky" and weights them down. Phase 2
inverts this — routing is the skeleton, and dense compute fills in the slots. This is
how modern efficient models (Mixtral, DBRX, Jamba) actually work: the routing structure
is fixed by design, and training optimizes the router + experts jointly.

We're not trying to discover that routing is useful (we know it is). We're trying to
discover WHICH routing patterns + WHICH expert compositions produce the best
efficiency/quality tradeoff. That's what the grammar search should explore.
