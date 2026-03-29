# Fix Plan: Op Weight Disconnect — Grammar Ignores Loss-Correlated Ops

**Date**: 2026-03-27
**Status**: Planned
**Priority**: P0 — this is the single biggest bottleneck in search quality

---

## Problem Statement

The grammar generates architectures dominated by filler ops (`add`, `rmsnorm`, `layernorm`, `token_type_classifier`, `entropy_score`, `ternary_projection`) while under-using ops that the loss distribution shows correlate with low loss (`conv1d_seq`, `swiglu_mlp`, `block_sparse_linear`, `gelu`, `silu`, `speculative`, `adaptive_recursion`, `selective_scan`).

The system has the data to make better decisions. It's not using it.

---

## Root Causes

### 1. Category weights are the wrong granularity

Almost all good ops AND bad ops are in the same category: `PARAMETERIZED`.

| Op | Category | S1% | avg_loss_ratio |
|----|----------|-----|----------------|
| swiglu_mlp | PARAMETERIZED | 15% | 0.617 |
| conv1d_seq | PARAMETERIZED | 11% | 0.570 |
| block_sparse_linear | PARAMETERIZED | 25% | 0.525 |
| layernorm | PARAMETERIZED | 21% | 0.752 |
| rmsnorm | PARAMETERIZED | 22% | 0.745 |
| token_type_classifier | PARAMETERIZED | 20% | 0.725 |

Boosting or suppressing the `parameterized` category moves them all together. The category weight system (including the wiring I did today) cannot distinguish `swiglu_mlp` from `layernorm`.

### 2. Op-level weight computation uses S1 rate, not loss quality

`compute_op_weights()` and the contrast amplification formula (`(s1_rate / mean)^2`) use S1 pass rate as the primary signal. But S1 rates between good and bad ops differ by only 3-8 percentage points:

- gelu: 28% S1, avg_lr 0.558
- entropy_score: 23% S1, avg_lr 0.699
- ternary_projection: 24% S1, avg_lr 0.626

The S1 rate gap is too small to generate meaningful weight differences. The **real** signal is in `avg_loss_ratio` — good ops produce 0.52-0.62 avg loss vs 0.72-0.75 for filler ops. That's a 20-30% difference that the weight system ignores.

### 3. Static preset boosts drown out learned weights

`_build_grammar_config()` hardcodes baseline boosts:

```python
_routing_defaults = {
    "entropy_score": 2.5,
    "route_topk": 2.0,
    "moe_topk": 2.0,
    "moe_2expert": 2.0,
    ...
}
```

These static values override analytics. `entropy_score` gets 2.5x boost regardless of its actual performance. The routing preset configs (`GrammarConfig.routing_first()`, `.exotic()`) apply even larger static boosts.

### 4. Selective_scan has a fixable compilation problem

`selective_scan` has 35% S0 pass rate (crashes 65% at compile time) but when it works, avg_lr is 0.64 — genuinely good. The system sees "5% S1 rate" and effectively bans it, when the real issue is compilation reliability, not architecture quality.

### 5. Analytics category weights are backwards

The category weight system (`compute_grammar_weights()`) currently produces:
- `math_space: 2.50` — BOOSTED despite weak S1 rates
- `mixing: 0.51` — SUPPRESSED despite containing valuable mixing ops
- `linear_algebra: 0.75` — SUPPRESSED despite containing top sparse linear ops

---

## Evidence

### Loss distribution data (from dashboard)

Ops associated with lower loss (user-observed):
`routing_conditioned_compression`, `conv1d_seq`, `selective_scan`, `swiglu_mlp`, `adaptive_recursion`, `mixed_recursion_gate`, `silu`, `gelu`, `speculative`, `tanh`, `block_sparse_linear`

### Recent generation (last 5 experiments, 1172 ops)

| Op | Count | % | S1 rate | avg_lr |
|----|-------|---|---------|--------|
| add | 260 | 22.2% | 17% | 0.726 |
| rmsnorm | 83 | 7.1% | 22% | 0.745 |
| layernorm | 77 | 6.6% | 21% | 0.752 |
| token_type_classifier | 64 | 5.5% | 20% | 0.725 |
| entropy_score | 55 | 4.7% | 23% | 0.699 |
| ternary_projection | 54 | 4.6% | 24% | 0.626 |

Missing from recent generation: `conv1d_seq`, `swiglu_mlp`, `speculative` — barely appear despite being in leaderboard winners.

### Leaderboard #1 (c9c7075e74, composite=384.8, validation tier)

Ops: `nm_sparse_linear`, `conv1d_seq`, `swiglu_mlp`, `token_merge`, `linear_proj_down/up`, `rmsnorm`, `gelu`, `add`

This winning formula is not being replicated in recent generation.

---

## Fix Plan

### Phase 1: Fix op-level weight signal (highest impact)

**File**: `research/scientist/analytics/analytics_grammar.py` and/or `analytics_experiments.py`

**Change**: Modify `compute_op_weights()` to rank ops by **avg_loss_ratio of S1 survivors** (lower = better), not raw S1 pass rate. Specifically:

1. For each op, compute `mean_loss_ratio_when_s1_passed` — the average loss_ratio across programs where that op was present AND the program passed S1
2. Weight formula: `softmax(-mean_lr / temperature)` with temperature ~0.1, producing higher weights for ops that produce lower loss when they work
3. Minimum floor of 0.3 to prevent starvation
4. Require at least 3 S1 passes to compute a meaningful loss signal (cold-start fallback to 1.0)

**Why**: S1 rate differences are 3-8% (noise). Loss ratio differences are 20-30% (signal). The weight system should use the signal.

### Phase 2: Remove or reduce static preset boosts

**File**: `research/scientist/runner/execution_candidates.py`

**Change**:
1. Remove the `_routing_defaults` hardcoded boosts in `_build_grammar_config()`
2. Replace with: if analytics provides op weights, use them. If not (cold start), use modest defaults (1.5x, not 2.5x)
3. In preset configs (routing_first, exotic, efficient), reduce static op_weights to serve as priors that get overridden by analytics

**Why**: Static boosts of 2.5x for `entropy_score` drown out the learned signal. The analytics system should drive op selection, not hand-tuned constants.

### Phase 3: Fix selective_scan compilation

**File**: `research/synthesis/context_rules.py` or compilation path

**Change**:
1. Add a context rule for `selective_scan` requiring compatible predecessors (it crashes when fed wrong-shaped inputs)
2. Alternatively, add a shape-compatibility check in the compiler for selective_scan
3. Goal: raise S0 pass rate from 35% to >80%

**Why**: selective_scan produces good results when it compiles (avg_lr 0.64). Fixing compilation unlocks a proven op that's currently effectively banned.

### Phase 4: Add loss-ratio-based op boost to grammar generation

**File**: `research/synthesis/grammar.py` (or motifs.py `resolve_step`)

**Change**: When `op_weights` are provided via GrammarConfig, use them in `_pick_op()` or equivalent op selection. The weights should directly influence which ops get placed in graph slots.

Currently `op_weights` flows through to motif selection (via `resolve_step` in motifs.py) but the raw op selection path may not use them. Verify and fix.

### Phase 5: Condition on co-occurrence, not individual ops

**File**: New or extend `analytics_grammar.py`

**Change**: Track op *pair* and *triple* success rates, not just individual ops. The leaderboard winner uses `nm_sparse_linear + conv1d_seq + swiglu_mlp` — the combination matters more than any single op.

Implementation:
1. For each S1-passing program, extract the set of unique ops
2. Track pair co-occurrence: `{(op_a, op_b): {n: int, mean_lr: float}}`
3. Feed top co-occurring pairs into motif weights or template selection
4. Lower priority than phases 1-3

---

## Validation

After each phase, measure:
1. **Op diversity in generation**: Are `conv1d_seq`, `swiglu_mlp`, `block_sparse_linear` appearing more?
2. **S1 pass rate**: Should stay stable or improve (not regress)
3. **Best loss_ratio per experiment**: Should trend downward
4. **Leaderboard composite score**: New entries should approach or exceed current top

Run 200+ programs after each phase change before measuring.

---

## Files Involved

| File | Phase | Change |
|------|-------|--------|
| `research/scientist/analytics/analytics_grammar.py` | 1 | Loss-ratio-based op weight computation |
| `research/scientist/analytics/analytics_experiments.py` | 1 | `compute_op_weights()` uses loss signal |
| `research/scientist/runner/execution_candidates.py` | 2 | Remove static preset boosts |
| `research/synthesis/context_rules.py` | 3 | selective_scan context rule |
| `research/synthesis/grammar.py` | 4 | Verify op_weights flow to op selection |
| `research/synthesis/motifs.py` | 4 | Verify resolve_step uses weights |
| `research/scientist/analytics/analytics_grammar.py` | 5 | Co-occurrence tracking |

---

## Non-Goals

- Do NOT change the scoring system (composite_v7) — it's working
- Do NOT change the promotion/escalation pipeline — just fixed in prior session
- Do NOT restructure the category system — fix the signal, not the taxonomy
- Do NOT add new ops — use the existing 111 ops more effectively
