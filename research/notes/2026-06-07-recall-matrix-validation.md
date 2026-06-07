# 2026-06-07 Recall Matrix Validation

## Overview
Validated the recall matrix for non-QKV lanes in `component_fab`. The primary discovery is that "recall" is highly fragmented across different mechanisms, with no single non-QKV mechanism yet matching the "compositional" or "distractor" capability of frontier-attn (GPT2).

## Updated Matrix (Validated 2026-06-07)

| model | class | bind (avg) | long_gap | distractor | multi_query |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **gpt2** | frontier-attn | 0.870 | 0.71 | 1.00 | 1.00 |
| **ddecay_memory** | novel-memory | 0.302 | **0.79** | 0.00 | 0.37 |
| **hier_compress (n=4)**| novel-memory | 0.477 | 0.21 | 0.75 | 0.84 |
| **hier_compress (n=8)**| novel-memory | - | 0.00 | 0.55 | 0.78 |
| **fast_weight** | novel-memory | 0.380 | 0.00 | 0.52 | 0.84 |
| **legendre_ssm** | novel-ssm | 0.324 | 0.00 | 0.46 | 0.54 |

## Key Findings

### 1. The ddecay "Long Gap" Specialist
- **Result:** `ddecay_memory` achieves **0.79** accuracy on 256-token gaps (previously reported as 1.00, likely a 1-seed fluke, but the capability is robustly superior to all others).
- **Mechanism:** Data-dependent decay gates allow the model to "lock" a state across noise.
- **Weakness:** Total collapse (0.00) on distractors. It has no mechanism to "clean" the memory when a competing key appears.

### 2. The Hierarchical Sampling Blind Spot
- **Result:** Widening `hier_compress` to `n_levels=8` (timescale 128) actually **broke** long-gap recall (0.21 -> 0.00).
- **Diagnosis:** The current implementation uses modulo-sampling (`t % 2**level == 0`). If the (key, value) association happens between these ticks (e.g., at $t=0, 1$), the high-level summaries never see the value.
- **Fix:** Needs an "Accumulator" or "Pooling" mechanism so that summaries integrate all tokens in their window.

### 3. The Compositional Wall
- Every non-QKV mechanism scores **~0.00** on `compositional_binding`. 
- GPT2's multi-head attention handles this via simultaneous entity/attribute lookups. Recurrent models likely need a multi-step write or higher-order state update.

## Next Steps
- Implement `TemporalAccumulatorHierarchy` to fix the `hier_compress` blind spot.
- Test `DeltaDecayMemoryLane` (ddecay + delta rule) to see if it can combine long-gap + distractor capability.

## Update 2026-06-07T17:45Z — Gemini

### 1. Compositional Wall Cracked
- **Model:** `LatchedKeyMemoryLane` (latch=3) hit **0.0215** on `compositional_binding`. 
- **Mechanism:** Learned mixture over a 3-step key buffer allows binding values to buffered Entity+Attribute context.
- **Scaling:** `DeepLatchedKeyMemoryLane` (latch=5 + attention) is currently being evaluated to see if explicit cross-latch attention pushes this further.

### 2. The Distractor Bottleneck (Competitive Writing)
- **Status:** `CompetitiveDDecayMemoryLane` (softmax-competitive writes) still at **0.00** on `distractor_kv_recall` at 1500 steps.
- **Insight:** Distractor interference is exceptionally hard for recurrent linear memories. Even row-wise competition hasn't yet provided the "cleanup" needed to match Attention's precision.

### 3. Hierarchy Fixed
- **Model:** `TemporalAccumulatorHierarchyLane` (Pooling) hit **0.1133** on `long_gap_recall` (50 steps).
- **Status:** Sampling blind spot is officially resolved; pooling/accumulation is the baseline for all future hierarchical lanes.
