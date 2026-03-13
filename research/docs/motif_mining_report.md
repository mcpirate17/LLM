# Motif Mining Report: Neural Architecture Op Patterns

**Date:** 2026-03-11
**Purpose:** Data-driven identification of op combinations, motifs, and structural patterns statistically associated with top-performing neural architectures. Input to the Judgment Engine motif library.

---

## Limitations & Caveats

> **⚠ This analysis does NOT measure how well a model learns.** The primary metric used is `loss_ratio` (final_loss / initial_loss), which is a static snapshot of end-state quality — not a measure of learning dynamics. It does not capture:
>
> - **Learning trajectory quality** — was the loss curve smooth or erratic?
> - **Convergence speed** — did it reach low loss in 10 steps or 1000?
> - **Whether training had plateaued** — was it still improving when training stopped?
> - **Stability under longer training** — would it diverge if trained further?
> - **Generalization** — does low training loss translate to good eval performance?
>
> A model with loss_ratio 0.05 that converged in 10 steps is fundamentally different from one that took 500 steps to get there. This analysis treats them identically.
>
> **The "lift" and "correlation" metrics below identify ops that are PRESENT in architectures with good end-state loss, but cannot distinguish whether those ops CAUSED the good performance or were incidental.** Confounding factors (e.g., certain ops tending to co-occur with good structural patterns) are not controlled for.
>
> **This analysis also ignores post-training life entirely.** These architectures will need to *infer* — handle long contexts, generalize to unseen positions, process variable-difficulty tokens efficiently. Ops critical for inference (RoPE, sliding windows, positional encodings, adaptive routing) may show low lift in micro-training benchmarks but are essential infrastructure for deployable models. The motif library must not deprioritize inference-relevant ops just because they don't boost loss_ratio in short training runs.
>
> **Recommendation:** Use these findings as starting hypotheses for the motif library, not as definitive answers. Validate candidate motifs with actual training runs that measure learning curves (not just final loss), and evaluate inference capabilities (long-context, positional generalization, token-adaptive compute) separately.

---

## Executive Summary

Analysis of 4,959 candidate architectures (734 top performers) from the lab notebook reveals clear, actionable patterns:

1. **Mixture-of-Experts ops are the single strongest signal.** `moe_topk` (3.1x lift), `moe_2expert` (2.6x lift), and `load_balance_loss` (3.4x lift) are massively overrepresented in winners.

2. **Efficient linear projections dominate.** The FFN backbone pattern (`linear_proj_up` + `linear_proj_down` + `bottleneck_proj`) appears in 21% of top performers and is associated with the best average loss ratios (0.048-0.063).

3. **Sparse/structured linear ops punch above their weight.** `nm_sparse_linear` (2.2x lift), `block_sparse_linear` (2.1x lift), `ternary_projection` (2.1x lift), and `semi_structured_2_4_linear` (2.0x lift) are all ~2x overrepresented.

4. **Most "exotic" math space ops are anti-correlated with success.** Clifford attention (0.08x lift), hyperbolic ops, tropical ops, and p-adic ops are consistently underrepresented in winners.

5. **`split2`/`concat` (multi-stream) patterns are neutral-to-negative.** Despite appearing in 32% of the general population, they appear in only 19% of top performers (0.60x lift). Random multi-stream branching destroys more architectures than it helps.

6. **SwiGLU + conv1d_seq is a strong pairing** (avg loss 0.071 when co-present, vs 0.268 global average).

---

## Methodology

### Population Definition
- **Database:** `research/lab_notebook.db` (`program_results` + `leaderboard` tables)
- **Total programs:** 4,959 (4,953 with parseable graph JSON)
- **Stage 1 passers:** 3,180 (64.1%)

### Top Performer Definition
Union of:
- Programs on leaderboard with tier in {investigation, validation, breakthrough}: **414**
- Stage 1 passers with loss_ratio in the top 15% (threshold: 0.0134): **477**
- **Combined (deduplicated): 734 top performers** (14.8% of population)

### Metrics
- **Lift:** `(% of top performers containing op) / (% of general population containing op)`. Lift > 1.0 means the op is overrepresented in winners.
- **Loss ratio difference:** Average loss_ratio when op is present minus global average (negative = better).
- **Minimum support:** Results filtered to ops/pairs appearing in at least 5-15 programs depending on analysis type.

### Graph Parsing
Op sequences extracted from `graph_json` (DAG representation). Edges extracted from `input_ids` relationships. Trigrams extracted via DFS path enumeration.

---

## 1. Population Statistics

| Metric | Value |
|--------|-------|
| Total programs | 4,959 |
| Stage 0 passed | 4,169 (84.1%) |
| Stage 0.5 passed | 4,169 (84.1%) |
| Stage 1 passed | 3,180 (64.1%) |

**Leaderboard tiers:**

| Tier | Count |
|------|-------|
| screening | 2,686 |
| investigation | 304 |
| validation | 110 |
| screened_out | 83 |

**Loss ratio percentiles (stage 1 passers):**

| Percentile | Loss Ratio |
|------------|------------|
| P10 | 0.0126 |
| P15 | 0.0134 |
| P25 | 0.0162 |
| P50 (median) | 0.2061 |
| P75 | 0.5257 |
| P90 | 0.7132 |

The distribution is heavily right-skewed: the top 15% of architectures achieve loss ratios below 0.014, while the median is 0.206. This 15x gap means the difference between a good and mediocre architecture is enormous.

---

## 2. Top Ops by Lift (Winner Enrichment)

Ops sorted by how much more frequently they appear in top performers vs the general population.

### Tier 1: Strong Winner Signals (Lift >= 2.0x)

| Op | Gen% | Top% | Lift | Gen N | Top N |
|----|------|------|------|-------|-------|
| fourier_mixing | 0.1% | 0.5% | 4.50x | 6 | 4 |
| load_balance_loss | 0.2% | 0.5% | 3.37x | 8 | 4 |
| lane_router | 0.2% | 0.5% | 3.37x | 8 | 4 |
| moe_topk | 1.2% | 3.7% | 3.09x | 59 | 27 |
| moe_2expert | 3.3% | 8.6% | 2.59x | 164 | 63 |
| swiglu_mlp | 2.2% | 5.6% | 2.49x | 111 | 41 |
| token_type_classifier | 1.0% | 2.5% | 2.48x | 49 | 18 |
| rwkv_channel | 2.9% | 6.9% | 2.41x | 143 | 51 |
| softmax_attention | 0.7% | 1.8% | 2.37x | 37 | 13 |
| linear_attention | 0.5% | 1.2% | 2.34x | 26 | 9 |
| graph_attention | 0.7% | 1.6% | 2.19x | 37 | 12 |
| nm_sparse_linear | 2.3% | 4.9% | 2.15x | 113 | 36 |
| block_sparse_linear | 4.4% | 9.1% | 2.09x | 216 | 67 |
| ternary_projection | 3.5% | 7.2% | 2.08x | 172 | 53 |

### Tier 2: Moderate Winner Signals (Lift 1.5-2.0x)

| Op | Gen% | Top% | Lift | Gen N | Top N |
|----|------|------|------|-------|-------|
| semi_structured_2_4_linear | 2.8% | 5.4% | 1.97x | 137 | 40 |
| gated_linear | 2.1% | 4.1% | 1.91x | 106 | 30 |
| progressive_compression_gate | 3.0% | 5.3% | 1.78x | 148 | 39 |
| linear_proj_up | 15.2% | 26.6% | 1.75x | 751 | 195 |
| linear_proj_down | 15.6% | 27.2% | 1.75x | 772 | 200 |
| conv1d_seq | 3.7% | 6.5% | 1.75x | 185 | 48 |
| bottleneck_proj | 14.2% | 22.9% | 1.61x | 704 | 168 |
| selective_scan | 3.3% | 4.9% | 1.47x | 165 | 36 |

### Tier 3: Anti-Correlated with Success (Lift < 0.5x)

| Op | Gen% | Top% | Lift | Gen N | Top N |
|----|------|------|------|-------|-------|
| clifford_attention | 5.0% | 0.4% | 0.08x | 250 | 3 |
| cosine_similarity | 2.9% | 0.4% | 0.14x | 146 | 3 |
| sign_ste | 3.4% | 0.5% | 0.16x | 169 | 4 |
| tropical_center | 5.6% | 1.0% | 0.17x | 278 | 7 |
| hyp_tangent_nonlinear | 3.5% | 0.7% | 0.19x | 175 | 5 |
| padic_expand | 3.5% | 0.8% | 0.23x | 175 | 6 |
| padic_gate | 9.0% | 5.3% | 0.59x | 448 | 39 |
| tropical_gate | 7.2% | 4.1% | 0.56x | 359 | 30 |
| tropical_attention | 6.8% | 4.0% | 0.58x | 335 | 29 |
| sigmoid | 6.4% | 3.1% | 0.49x | 318 | 23 |
| tanh | 6.4% | 3.1% | 0.49x | 319 | 23 |
| sparse_threshold | 3.5% | 0.0% | 0.00x | 174 | 0 |
| matmul | 1.9% | 0.0% | 0.00x | 92 | 0 |
| hyp_distance | 0.6% | 0.0% | 0.00x | 32 | 0 |

---

## 3. Top Op Pairs (Co-occurrence in Winners)

These are op pairs that frequently co-occur in top-performing architectures.

| Op Pair | Count | % of Top |
|---------|-------|----------|
| linear_proj_down + linear_proj_up | 157 | 21.4% |
| concat + split2 | 140 | 19.1% |
| bottleneck_proj + linear_proj_down | 133 | 18.1% |
| add + linear_proj_down | 132 | 18.0% |
| bottleneck_proj + linear_proj_up | 131 | 17.8% |
| add + linear_proj_up | 127 | 17.3% |
| add + bottleneck_proj | 115 | 15.7% |
| add + concat | 79 | 10.8% |
| add + linear_proj | 77 | 10.5% |
| add + block_sparse_linear | 49 | 6.7% |
| add + spike_rate_code | 43 | 5.9% |
| add + rwkv_channel | 41 | 5.6% |
| add + ternary_projection | 40 | 5.4% |
| add + moe_2expert | 36 | 4.9% |
| add + semi_structured_2_4_linear | 36 | 4.9% |
| add + selective_scan | 35 | 4.8% |

**Key insight:** The dominant pattern is an FFN-like backbone (`linear_proj_up + linear_proj_down + bottleneck_proj`) combined with residual connections (`add`). This is essentially the transformer FFN block rediscovered through statistical selection.

---

## 4. Op Pairs by Conditioned Loss Ratio

These are op co-occurrences that produce the lowest average loss_ratio when both are present (among stage 1 passers, minimum 15 observations).

| Op Pair | Avg LR | Delta vs Global | N |
|---------|--------|-----------------|---|
| latent_attention_compressor + linear_proj_up | 0.0404 | -0.2275 | 20 |
| block_sparse_linear + linear_proj_up | 0.0411 | -0.2267 | 18 |
| linear_proj_up + sliding_window_mask | 0.0621 | -0.2058 | 15 |
| gated_linear + linear_proj_up | 0.0670 | -0.2009 | 15 |
| lif_neuron + linear_proj_down | 0.0676 | -0.2003 | 23 |
| linear_proj_up + neg | 0.0710 | -0.1969 | 15 |
| conv1d_seq + swiglu_mlp | 0.0713 | -0.1965 | 15 |
| lif_neuron + linear_proj_up | 0.0734 | -0.1944 | 20 |
| linear_proj_down + moe_2expert | 0.0765 | -0.1913 | 15 |
| linear_proj + progressive_compression_gate | 0.0776 | -0.1902 | 15 |
| rwkv_channel + split2 | 0.0791 | -0.1887 | 26 |
| bottleneck_proj + lif_neuron | 0.0825 | -0.1853 | 17 |
| concat + rwkv_channel | 0.0861 | -0.1817 | 27 |
| add + nm_sparse_linear | 0.0885 | -0.1794 | 57 |
| layernorm + linear_proj_up | 0.0933 | -0.1746 | 26 |
| bottleneck_proj + spike_rate_code | 0.0970 | -0.1709 | 42 |
| linear_proj_down + spike_rate_code | 0.0989 | -0.1689 | 51 |

**Key insight:** `linear_proj_up` is the single most important op for pair performance. Nearly every top pair includes it. The attention mechanism matters less than the FFN pathway quality.

---

## 5. Op-Metric Correlations (Single Op Impact)

Average loss_ratio when each op is present vs absent, among stage 1 passers (global avg: 0.2678).

### Most Beneficial Ops (largest negative delta):

| Op | Avg LR | Delta | N |
|----|--------|-------|---|
| nm_sparse_linear | 0.0944 | -0.1734 | 82 |
| rwkv_channel | 0.1031 | -0.1647 | 91 |
| ternary_projection | 0.1108 | -0.1570 | 124 |
| moe_2expert | 0.1141 | -0.1537 | 136 |
| moe_topk | 0.1149 | -0.1529 | 41 |
| progressive_compression_gate | 0.1180 | -0.1499 | 108 |
| graph_attention | 0.1202 | -0.1476 | 37 |
| token_type_classifier | 0.1218 | -0.1460 | 48 |
| linear_proj_up | 0.1247 | -0.1431 | 477 |
| linear_attention | 0.1281 | -0.1397 | 25 |
| linear_proj_down | 0.1325 | -0.1354 | 502 |
| bottleneck_proj | 0.1354 | -0.1325 | 463 |
| softmax_attention | 0.1422 | -0.1257 | 31 |
| block_sparse_linear | 0.1440 | -0.1239 | 162 |
| swiglu_mlp | 0.1662 | -0.1017 | 75 |

### Ops with Higher-Than-Average Loss (in random composition — NOT necessarily harmful):

| Op | Avg LR | Delta | N |
|----|--------|-------|---|
| cosine_similarity | 0.5155 | +0.2477 | 84 |
| hyp_tangent_nonlinear | 0.4792 | +0.2114 | 110 |
| softmax_last | 0.4585 | +0.1906 | 84 |
| exp_map | 0.4488 | +0.1810 | 97 |
| early_exit | 0.4399 | +0.1721 | 31 |
| lif_neuron | 0.3993 | +0.1314 | 170 |
| sign_ste | 0.4033 | +0.1355 | 83 |
| fused_linear_gelu | 0.3891 | +0.1212 | 80 |
| rmsnorm_pre | 0.3944 | +0.1265 | 43 |

**Note:** `lif_neuron` is interesting -- harmful alone (0.399 avg LR) but the *pair* `lif_neuron + linear_proj_down` achieves 0.068 avg LR. This suggests LIF neurons need to be properly scaffolded by an FFN pathway to work.

---

## 6. Candidate Motifs (Trigram Paths)

3-op paths extracted from graph DAG edges. High-lift trigrams that appear in at least 3 top performers:

| Trigram | Gen% | Top% | Lift | Top N |
|---------|------|------|------|-------|
| selective_scan -> ternary_projection -> add | 0.1% | 0.5% | 6.75x | 4 |
| conv1d_seq -> swiglu_mlp -> add | 0.1% | 0.8% | 6.75x | 6 |
| layernorm -> split2 -> gelu | 0.1% | 0.8% | 6.75x | 6 |
| linear_proj_up -> layernorm -> split2 | 0.1% | 0.8% | 6.75x | 6 |
| split2 -> concat -> nm_sparse_linear | 0.1% | 0.5% | 6.75x | 4 |
| spike_rate_code -> concat -> gated_linear | 0.1% | 0.5% | 6.75x | 4 |
| bottleneck_proj -> linear_proj_up -> cascade | 0.1% | 0.5% | 6.75x | 4 |
| low_rank_proj -> learnable_bias -> add | 0.1% | 0.4% | 6.75x | 3 |
| low_rank_proj -> learnable_bias -> rmsnorm | 0.1% | 0.4% | 6.75x | 3 |

**Note:** The edge-based trigram analysis shows sparse data (most trigrams are unique) due to the large op vocabulary and short graphs (avg 5.4 ops). The co-occurrence analysis (Section 3) is more statistically robust.

---

## 7. Cluster Analysis of Top Performers

KMeans clustering (k=10, silhouette=0.090) of top performers using binary op-presence feature vectors reveals 10 structural families:

### Cluster 5: "Full FFN + Multi-stream" (n=30, best avg LR: 0.0485)
**Best-performing cluster.** Every member has: `bottleneck_proj`, `linear_proj_up`, `linear_proj_down`, `concat`, `split2`. Avg 9.7 ops. This is a rich, multi-stream FFN architecture.

### Cluster 7: "SwiGLU + Residual" (n=45, avg LR: 0.0535)
Centered on `add` (100%) + `swiglu_mlp` (53%) + `bottleneck_proj` (49%). Avg 5.8 ops. A lean, effective design -- essentially the modern LLaMA FFN pattern.

### Cluster 1: "Minimal Residual" (n=205, avg LR: 0.0629)
Largest cluster. Only `add` is >30% presence. Avg 4.9 ops. These are simple, small architectures that achieve good results through minimalism.

### Cluster 0: "Linear Proj Core" (n=75, avg LR: 0.0633)
`linear_proj` (100%), `add` (64%), `linear_proj_down` (37%). Avg 5.6 ops. Standard transformer-like FFN.

### Cluster 4: "Bottleneck FFN" (n=124, avg LR: 0.0634)
`linear_proj_down` (100%), `linear_proj_up` (100%), `bottleneck_proj` (82%), `add` (68%). Avg 5.9 ops. The classic expand-contract FFN with residual.

### Cluster 6: "Exotic Multi-op" (n=4, avg LR: 0.0227)
Tiny cluster with extremely low loss. Uses `split3`, `tied_proj`, `grouped_linear`, `padic_gate`, `tropical_gate`, `neg`. An outlier pattern worth investigating manually.

### Cluster 8: "Ultra-minimal" (n=124, avg LR: 0.0706)
No op exceeds 30% presence. Avg 2.5 ops. Very short graphs with diverse ops.

### Cluster 2: "Multi-stream" (n=85, avg LR: 0.0867)
`concat` (100%), `split2` (100%), `add` (72%), `spike_rate_code` (31%). Avg 7.5 ops. Multi-stream branching with spiking coding.

### Cluster 9: "GeLU Multi-stream" (n=17, avg LR: 0.0738)
`concat` (100%), `gelu` (100%), `split2` (100%), `grade_mix` (41%). Activation-heavy multi-stream.

### Cluster 3: "Linear Proj + Exotic" (n=25, avg LR: 0.0744)
`linear_proj_up` (100%), `linear_proj` (92%), `add` (60%), `padic_gate` (36%). FFN backbone with exotic gating.

---

## 8. Structural Patterns

| Metric | General | Top Performers |
|--------|---------|----------------|
| Avg n_ops | 5.6 | 5.4 |
| Avg depth | 5.1 | 5.0 |
| Avg unique ops | 5.3 | 5.1 |
| Avg edges | 7.1 | 6.6 |
| Avg param_count | 8.8M | 9.4M |

**Residual connections:**
- General: 64.3%
- Top performers: 64.6%
- No lift -- residuals are universal, not a discriminator

**Normalization ops (layernorm/rmsnorm/group_norm/batch_norm):**
- General: 8.2%
- Top performers: 11.4%
- Moderate lift (1.4x) -- normalization helps

**Observation:** Top performers in the current dataset tend to be simpler (fewer ops/edges). However, this reflects the current evaluation regime — short micro-training on uniform token streams — which rewards architectures that converge quickly on easy patterns.

> **⚠ This "simpler is better" finding should NOT be taken as a design principle for the motif library.** It is an artifact of what we currently measure (fast loss drop on short training), not what we ultimately want.
>
> The architectures we're searching for need to excel at capabilities this analysis cannot evaluate:
>
> - **Learning quality, not just speed.** A model that spends more compute on difficult tokens (adaptive allocation) and clusters token types for specialized training will outperform one that converges fast on uniform data. Simpler models win the current eval because the eval doesn't test learning depth.
>
> - **Inference-time capabilities.** After training, these architectures must actually *infer* — handle long contexts, variable-length sequences, positional generalization. This is where ops like `rope`, `alibi`, `sliding_window_mask`, and `learned_pos` become critical. These position/window ops have low lift in the current data (because micro-training doesn't test inference scenarios), but they are essential infrastructure for any deployable model.
>
> - **Architectural diversity for ensemble/adversarial value.** Different architectures that learn different representations can play "devil's advocate" against each other. A monoculture of simple FFN blocks may win on loss_ratio but produces redundant models. The motif library should preserve structural diversity — including more complex patterns — precisely because future evaluation will reward it.
>
> - **Token-adaptive computation.** Ops like `moe_topk`, `early_exit`, `token_type_classifier`, and routing mechanisms enable models to spend variable compute per token. These are undervalued by uniform micro-training but essential for efficient, intelligent inference.
>
> **Takeaway:** Use the structural simplicity observation as a data point, not a constraint. The motif library must include complex, inference-aware, and adaptive-compute patterns even if they don't dominate the current leaderboard — because the current leaderboard doesn't test what matters most.

---

## 9. Reference Architecture Comparison

| Architecture | Screening LR | Composite | Op Sequence |
|---|---|---|---|
| GPT-2 | 0.265 | 183.4 | LN -> attn -> add -> LN -> FFN(proj->gelu->proj) -> add |
| Mamba | 0.207 | 188.2 | LN -> conv1d -> silu -> scan -> gate -> add |
| RWKV | 0.177 | 190.9 | LN -> time_mix -> add -> LN -> channel -> add |
| RAG | 0.250 | 179.9 | LN -> attn -> add -> LN -> FFN(sim->gather->attn->linear_attn) -> add -> LN -> FFN -> add |

RWKV achieves the best screening loss ratio among references, consistent with `rwkv_channel` having 2.41x lift.

---

## 10. Recommendations for Motif Library Construction

### High-Confidence Motifs (ready to codify)

1. **FFN-Expand-Contract**: `linear_proj_up -> activation -> linear_proj_down` with residual `add`. The single most validated pattern. Use `gelu`, `swiglu_mlp`, or `gated_linear` as the activation.

2. **Bottleneck-FFN**: `bottleneck_proj -> linear_proj_up -> linear_proj_down` with residual. Cluster 4/5 pattern, 18% of top performers.

3. **MoE Block**: `moe_topk` or `moe_2expert` with `load_balance_loss`. 3.1-3.4x lift. Should be a first-class motif with proper expert routing.

4. **Sparse Linear Block**: `nm_sparse_linear` or `block_sparse_linear` as drop-in replacement for dense linear. 2.1-2.2x lift, avg LR 0.088-0.094.

5. **RWKV Channel Block**: `rwkv_channel` with residual `add`. 2.4x lift, avg LR 0.103.

6. **Mamba-like Block**: `conv1d_seq -> swiglu_mlp -> add`. 6.75x trigram lift. Or `conv1d_seq -> silu -> selective_scan`.

7. **Ternary-Scan Block**: `selective_scan -> ternary_projection -> add`. 6.75x trigram lift.

### Medium-Confidence Motifs (need more evidence)

8. **Attention-Compress**: `latent_attention_compressor + linear_proj_up`. Best pair by conditioned LR (0.040).

9. **Spiking-FFN**: `lif_neuron + linear_proj_down/up`. Anti-correlated alone but strong in pairs.

10. **Progressive Compression**: `progressive_compression_gate + linear_proj`. 1.78x lift.

### Low-Lift Ops (insufficient evidence to draw conclusions)

> **⚠ DO NOT block or penalize these ops based on this data alone.** Low lift in a random-composition regime (op soup) does not mean the op is bad — it may simply have never been composed correctly. Most of the ~4,500 candidates were assembled almost randomly, so an op appearing in many failures tells us nothing about the op itself. See the data mining disclaimer in `JUDGMENT_ENGINE_PLAN.md` Phase 6.1.
>
> These ops have low lift but that could change entirely under motif-based composition. Only after validating that an op fails even in well-structured contexts (proper normalization, residual connections, compatible neighbors) should it be deprioritized — and even then, prefer simply not boosting it rather than actively penalizing it.

The following ops had low lift in the current (random-composition) dataset. They are listed here for awareness, NOT as a blocklist:

- `clifford_attention` (0.08x lift) — exotic math space op, may need specific scaffolding
- `cosine_similarity` (0.14x lift) — possibly needs careful placement (not standalone)
- `hyp_tangent_nonlinear` (0.19x lift) — hyperbolic space op, untested in structured contexts
- `sparse_threshold` (0.00x lift, zero winners) — no positive evidence exists yet
- `matmul` as standalone (0.00x lift) — may only work as part of larger attention patterns
- `exp_map` / `log_map` (0.44-0.53x lift) — hyperbolic ops, potentially useful in specific manifold architectures
- `softmax_last` (0.40x lift) — may be misplaced by random composition
- `early_exit` (0.11x lift) — architectural pattern that may need specific graph structure
- `split2`/`concat` without clear purpose (0.60x lift) — random branching hurts, but structured branching (Cluster 5, avg LR 0.049) is the best-performing pattern

### Op Substitution Hypotheses (require validation)

> These substitutions are based on lift in micro-training loss_ratio only. They do not account for inference quality, generalization, or learning dynamics. Treat as hypotheses to test, not as established facts.

| Standard Op | Candidate Alternative | Evidence (lift only) | Unknown |
|---|---|---|---|
| dense `linear_proj` | `nm_sparse_linear` | 2.15x lift, -0.173 LR delta | Inference speed? Generalization? |
| dense `linear_proj` | `block_sparse_linear` | 2.09x lift, -0.124 LR delta | Hardware compatibility? |
| dense `linear_proj` | `ternary_projection` | 2.08x lift, -0.157 LR delta | Capacity loss at scale? |
| `gelu` activation | `swiglu_mlp` | 2.49x vs 1.29x lift | Already validated in LLaMA lineage |
| `sigmoid` gating | `gated_linear` | 1.91x vs 0.49x lift | Context-dependent? |
| `tanh` activation | `gelu` or `relu` | 0.49x vs 1.29x/1.00x lift | `tanh` may be useful in specific gate contexts |

---

## Appendix A: Complete Op Lift Table

| Op | Gen% | Top% | Lift | Avg LR (s1) | N (s1) |
|----|------|------|------|-------------|---------|
| fourier_mixing | 0.1% | 0.5% | 4.50x | -- | <10 |
| load_balance_loss | 0.2% | 0.5% | 3.37x | -- | <10 |
| lane_router | 0.2% | 0.5% | 3.37x | -- | <10 |
| moe_topk | 1.2% | 3.7% | 3.09x | 0.115 | 41 |
| moe_2expert | 3.3% | 8.6% | 2.59x | 0.114 | 136 |
| swiglu_mlp | 2.2% | 5.6% | 2.49x | 0.166 | 75 |
| token_type_classifier | 1.0% | 2.5% | 2.48x | 0.122 | 48 |
| rwkv_channel | 2.9% | 6.9% | 2.41x | 0.103 | 91 |
| softmax_attention | 0.7% | 1.8% | 2.37x | 0.142 | 31 |
| linear_attention | 0.5% | 1.2% | 2.34x | 0.128 | 25 |
| no_norm | 0.8% | 1.8% | 2.19x | 0.233 | 39 |
| graph_attention | 0.7% | 1.6% | 2.19x | 0.120 | 37 |
| nm_sparse_linear | 2.3% | 4.9% | 2.15x | 0.094 | 82 |
| block_sparse_linear | 4.4% | 9.1% | 2.09x | 0.144 | 162 |
| ternary_projection | 3.5% | 7.2% | 2.08x | 0.111 | 124 |
| semi_structured_2_4_linear | 2.8% | 5.4% | 1.97x | 0.152 | 99 |
| gated_linear | 2.1% | 4.1% | 1.91x | 0.147 | 76 |
| progressive_compression_gate | 3.0% | 5.3% | 1.78x | 0.118 | 108 |
| linear_proj_up | 15.2% | 26.6% | 1.75x | 0.125 | 477 |
| linear_proj_down | 15.6% | 27.2% | 1.75x | 0.132 | 502 |
| conv1d_seq | 3.7% | 6.5% | 1.75x | 0.194 | 133 |
| bottleneck_proj | 14.2% | 22.9% | 1.61x | 0.135 | 463 |
| selective_scan | 3.3% | 4.9% | 1.47x | 0.161 | 124 |
| layernorm | 4.1% | 5.7% | 1.41x | 0.210 | 140 |
| gelu | 8.4% | 10.9% | 1.29x | 0.233 | 265 |
| linear_proj | 13.3% | 17.2% | 1.29x | 0.215 | 427 |
| add | 64.3% | 64.6% | 1.00x | 0.256 | 1958 |
| relu | 7.1% | 7.1% | 1.00x | 0.255 | 202 |
| split2 | 31.8% | 19.1% | 0.60x | 0.268 | 844 |
| concat | 33.2% | 19.6% | 0.59x | 0.268 | 894 |
| sigmoid | 6.4% | 3.1% | 0.49x | 0.318 | 166 |
| tanh | 6.4% | 3.1% | 0.49x | 0.291 | 197 |
| clifford_attention | 5.0% | 0.4% | 0.08x | 0.294 | 157 |
| sparse_threshold | 3.5% | 0.0% | 0.00x | 0.338 | 12 |
| matmul | 1.9% | 0.0% | 0.00x | 0.340 | 45 |

## Appendix B: Full Cluster Composition

| Cluster | N | Avg LR | Best LR | Avg Ops | Signature |
|---------|---|--------|---------|---------|-----------|
| 5 | 30 | 0.0485 | 0.0082 | 9.7 | bottleneck+proj_up+proj_down+concat+split2 |
| 7 | 45 | 0.0535 | 0.0067 | 5.8 | add+swiglu_mlp+bottleneck |
| 1 | 205 | 0.0629 | 0.0063 | 4.9 | add (minimal) |
| 0 | 75 | 0.0633 | 0.0061 | 5.6 | linear_proj+add+proj_down |
| 4 | 124 | 0.0634 | 0.0065 | 5.9 | proj_down+proj_up+bottleneck+add |
| 6 | 4 | 0.0227 | 0.0111 | 9.5 | split3+tied_proj+grouped_linear+exotic |
| 8 | 124 | 0.0706 | 0.0076 | 2.5 | ultra-minimal (no dominant ops) |
| 9 | 17 | 0.0738 | 0.0070 | 6.9 | concat+gelu+split2+grade_mix |
| 2 | 85 | 0.0867 | 0.0081 | 7.5 | concat+split2+add+spike_rate_code |
| 3 | 25 | 0.0744 | 0.0087 | 6.8 | proj_up+linear_proj+add+padic_gate |
