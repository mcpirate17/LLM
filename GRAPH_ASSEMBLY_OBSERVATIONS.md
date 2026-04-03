---
status: active
created: 2026-04-01
author: claude-opus
---

# Aria Graph Assembly Observations & Improvement Opportunities

## Current Architecture

Aria assembles computation graphs via a **three-stage compositional approach**:

1. **Template Selection** (1-3 templates per graph) — 70 registered templates with static weights
2. **Motif Filling** (2-4 ops per motif slot) — 113 validated motifs mined from top 734 performers
3. **Assembly & Validation** — shape/gradient/budget checks, rollback on failure

## Key Metrics (2026-04-01 Session)

- **Grammar rejection rate: 71-74%** — 247-280 failures per 349 attempts to produce 100 graphs
- **Deduplication rate: 32%** — many architectures are minor variants
- **S1 survival: ~16/73 S0 candidates (22%)** — decent once generated, poor generation efficiency
- **Loss ratios: 0.45-0.61** (median ~0.55)

## Critical Issues

### 1. Grammar Rejection Rate (71-74%) — Primary Bottleneck

**Root causes:**
- Template application failures (~60-70% of rejections): shape mismatches, algebraic space incompatibilities, context rule violations during `_instantiate_motif()`
- Budget rollbacks (~15-20%): graphs hit depth/ops limits mid-composition
- Wildcard slot fallback too conservative (prob=0.15)

**Fix:** Two-tier motif fallback — search adjacent motif classes before falling back to identity. Increase `wildcard_slot_prob` to 0.25-0.30 for high-rejection templates.

### 2. Static Template Weights — No Learning Loop

- `DEFAULT_TEMPLATE_WEIGHTS` are hardcoded; templates that fail repeatedly stay at fixed weight
- Only 17 of 70 templates see consistent use (residual_block, sequential, transformer_block)
- Routing templates (21 total, weight 5.0-6.0) rarely deploy in default configs
- Exotic templates (20+) fail validation frequently but never get downweighted

**Fix:** Extend `compute_grammar_weights()` to also compute adaptive template weights using the same contrast amplification formula: `(template_s1_rate / mean)^2`, clamped [0.5, 8.0].

### 3. Motif-Slot Class Mismatch

- Templates define slot requirements (`_FFN_CLASSES`, `_MIXER_CLASSES`), but when current graph shape doesn't match, `_pick_compatible_motif()` returns None
- Falls back to identity/passthrough, wasting template-allocated capacity
- Result: structurally empty graphs that look like templates but act like identity chains

### 4. Context Rules Checked Too Late

- Context rules (forbidden predecessor/successor chains) are checked post-composition
- Many failures are predictable at motif selection time
- Shifting to early rejection saves 2-3 seconds per failed graph generation

### 5. No Exploration/Stability Regime Separation

- Single grammar config tries to balance exploration and stability
- High exploration_boost (4.0) causes more rejections
- Alternating between stability mode (80% success, proven templates) and exploration mode (29% success, wild ops) would improve throughput

## Prioritized Improvement Roadmap

| Priority | Improvement | Expected Gain |
|----------|-------------|---------------|
| 1 | Relaxed slot matching / two-tier motif fallback | 20% fewer generation attempts |
| 2 | Adaptive template weighting from DB success rates | 10-15% better S1 rate |
| 3 | Context rule early rejection at pick time | 10-15% faster validation |
| 4 | Exploration/stability regime alternation | 5-10% overall throughput |
| 5 | Dedicated motif chains for investigation ops | 5-10% better under-observed coverage |

## Component Diversity Problem (6.5M program results analyzed)

### Op Convergence to ~10 Core Ops
148 unique ops exist but the system converges to the same handful:
- add (20,283 uses), rmsnorm (14,440), linear_proj (12,065), layernorm (8,478)
- These core ops have mediocre S1 rates (~0.15) but dominate by volume
- Best S1 performers are underused: spectral_filter (0.329, 1496 uses), linear_proj_up (0.301, 3328)

### Attention Is Broken
Only **7 of 89 templates (7.9%)** explicitly use attention operations.

| Attention Op | Uses | S1 Rate | Verdict |
|---|---|---|---|
| latent_attention_compressor | 552 | 0.302 | Good |
| local_window_attn | 186 | 0.275 | Good |
| stdp_attention | 151 | 0.244 | OK |
| diff_attention | 203 | 0.212 | OK |
| linear_attention | 272 | 0.113 | Poor |
| softmax_attention | 497 | **0.029** | Broken |
| rope_rotate | 285 | **0.036** | Broken |

- Classical softmax_attention: **2.9% S1 rate** despite 497 uses — essentially never learns
- RoPE positional encoding: **3.6% S1 rate** — broken integration
- Modern variants (latent compress, local window, diff) perform 7-10x better
- `rope_attention_block` template: 739 attempts, only 4.9% S1 — needs to be replaced or fixed

**Action needed:** At least 50% of templates should have attention slots. Replace softmax_attention with latent/local/diff variants. Kill rope_attention_block template or rebuild it.

### Template Stink List (S1 < 10%, high sample count)

| Template | Samples | S1 Rate | Problem |
|---|---|---|---|
| routing_conditioned_moe | 107 | **1.9%** | Routing collapse |
| arch_router_block | 271 | **2.2%** | Routing collapse |
| hetero_moe_block | 266 | **3.4%** | Load balancing |
| compute_budget_block | 357 | **3.6%** | Budget routing |
| rope_attention_block | 739 | **4.9%** | RoPE + softmax broken |
| depth_gated_block | 301 | **7.0%** | Gating mechanism |
| parallel_split | 435 | **7.1%** | Split structure |
| residual_block | 1,848 | **7.3%** | Most-used, worst ROI |
| moe | 541 | **8.1%** | MoE routing failure |

**residual_block** is the most damning: 1,848 samples at 7.3% S1. It's the system's default workhorse and it's terrible. Every cycle it gets selected because of static weights, generating hundreds of candidates that mostly fail.

### Template Winners (S1 > 50%)

| Template | Samples | S1 Rate |
|---|---|---|
| three_way_split | 936 | **86.4%** |
| hyperbolic_bridge_block | 141 | **60.3%** |
| routed_bottleneck | 530 | **58.9%** |
| rwkv_block | 228 | **53.5%** |
| conv_residual_block | 134 | **52.2%** |
| spiking_moe_block | 434 | **52.1%** |
| latent_compress_rwkv | 212 | **51.9%** |
| fused_gelu_ffn | 159 | **50.9%** |

### Routing Ops Are Universally Failing
All conditional routing mechanisms have <3.6% S1:
- routing_conditioned_moe (1.9%), arch_router_block (2.2%), compute_budget_block (3.6%)
- Root cause: complex branching prevents gradients from flowing to routing decisions
- MoE in general struggles except when simplified (spiking_moe_block at 52.1% is the exception)

### What Actually Works
1. Simple splits (three_way_split 86.4%) — avoid complexity
2. Compression before mixing (latent_compress_rwkv 51.9%, routed_bottleneck 58.9%)
3. Channel-mixing alternatives to attention (rwkv, conv, spiking)
4. Fused operations (fused_gelu_ffn 50.9%)

## Fundamental Problem: No Intelligence in Template Construction

### Templates Are Deterministic Scaffolds with Random Slot-Filling

The entire template system is: **fixed structural pattern → random motif stuffing → hope it trains**.

**`three_way_split` (86.4% S1) dissected** (`_templates_core.py:236-311`):
- Lane 0: random pick from `_MIXER_CLASSES` (attention, SSM, conv, channel, math_space)
- Lane 1: random pick from `_FFN_CLASSES` (FFN, gate, sparse, MOE)
- Lane 2: random pick from `_GATE_CLASSES` (MOE, gate, guarded_act)
- Routing: `gated_lane_blend` — learns per-token lane weights, but has NO semantic signal about what each lane should specialize in
- Merge: simple `concat` — all lanes treated equally, no learned merge

**Why tokens are split:** They're not. `gated_lane_blend` learns a soft blend during training. The template just says "3 lanes, random contents." There's no intent like "local context / global context / routing."

**What happens in each lane:** Random motif from a class bucket. No semantic reasoning. No learning objectives per lane.

**How they merge:** `concat` + skip connection. No learned gating at merge. No attention-over-lanes.

### What This Means
- High S1 rate on `three_way_split` is misleading — it's structurally permissive, not intelligent
- Mean loss (0.646) is worse than `routed_bottleneck` (0.530) or even `residual_block` (0.371)
- The system is optimizing for "doesn't crash" not "learns well"
- No template has semantic metadata about lane purpose, slot objectives, or learning targets

### Missing: Forced Continuous Learning Templates

What we need:
1. **Templates with explicit learning objectives per slot** — "this lane does local attention (window=64), this lane does global context (full sequence), this lane routes based on token difficulty"
2. **Learned merge operations** — not concat, but attention-over-lanes or learned gating at merge time
3. **Forced attention integration** — at least 50% of templates must have an attention slot, using modern variants (latent compress, local window, diff) not broken softmax
4. **Continuous learning targets** — templates that force the model to demonstrate specific capabilities (retrieval, composition, long-range dependency) not just minimize perplexity

### Motif Distribution Problem (113 motifs)

| Class | Count | % | Issue |
|---|---|---|---|
| math_space | 17 | 15% | Over-represented |
| guarded_act | 14 | 12% | Over-represented |
| gate_core | 13 | 12% | |
| moe_core | 12 | 11% | Mostly failing |
| **attention_core** | **11** | **10%** | Under-represented |
| channel_core | 8 | 7% | |
| ssm_core | 7 | 6% | |
| sparse_core | 7 | 6% | |
| efficient_proj | 7 | 6% | |
| **ffn_core** | **5** | **4%** | Under-represented |
| conv_core | 5 | 4% | |
| reduce_core | 5 | 4% | |

Attention at 10% and FFN at 4% — the two most proven components in language models are the least represented in the motif library.

### Attention Motifs: 4 Broken, 4 Good, 3 Untested

| Motif | Lift | Support | Core Op | S1 Rate | Status |
|---|---|---|---|---|---|
| attn_latent_compress | 1.80 | 20 | latent_attention_compressor | 0.302 | Best |
| attn_local_window | 3.00 | 15 | local_window_attn | 0.275 | Good |
| attn_graph | 3.50 | 12 | graph_attention | 0.089 | Poor |
| attn_softmax | 2.37 | 13 | softmax_attention | 0.029 | **Broken** |
| attn_linear | 2.34 | 9 | linear_attention | 0.113 | Poor |
| attn_rope | 1.50 | 0 | rope_rotate+softmax | 0.036 | **Broken** |
| attn_causal_mask | 0.05 | 0 | softmax+mask | — | Untested |
| attn_diff | 1.00 | 0 | diff_attention | 0.212 | Good (few tests) |
| attn_gated_delta | 2.50 | 0 | gated_delta | — | Untested |
| attn_sliding_window | 3.00 | 0 | sliding_window+linear | — | Untested |
| tropical_attention_gate | 2.10 | 3 | tropical_attention | 0.110 | Poor |

## Action Items: Next Session

### 1. More Attention Types (Priority)
We need significantly more attention variants. Current coverage (11 motifs, 10%) is inadequate for a language model search system. Candidates to add:
- Multi-head attention (standard transformer) — missing entirely
- Grouped query attention (GQA) — proven in Llama/Mistral
- Multi-query attention (MQA) — proven in PaLM
- Flash attention / memory-efficient attention
- Cross-attention variants
- Sparse attention (fixed/learnable patterns)
- Mixture-of-attention-heads
- Neighborhood attention (NAT)

The broken variants (softmax_attention 2.9% S1, rope_rotate 3.6%) need to be diagnosed and fixed, not just ignored — they represent fundamental attention patterns that should work.

### 2. FFN Should Be Assumed/Default
Every template should assume FFN is present — it's not optional in a language model. Currently only 5 FFN motifs (4% of library). FFN should be:
- A default slot in every template (not a class to randomly sample)
- Pre-wired as norm→attention→norm→FFN (the proven transformer pattern)
- Variants: standard, gated (SwiGLU), sparse, MoE-routed FFN

### 3. Intelligent Template Redesign (see "Fundamental Problem" section above)
- Templates need explicit learning objectives per slot
- Learned merge operations instead of concat
- Forced continuous learning targets
- Separate "structural robustness" metric from "learning quality"

## Non-Critical Warnings Observed

- LLM report narrative fails (no Anthropic API key configured) — falls back to rule-based
- Auto-report dict comparison bug (fixed 2026-04-01: `persona_rules.py` line 408)
- `nb.db` → `nb.conn` typo in mode selection (fixed 2026-04-01: `continuous_loop.py` line 545)
- `self.notebook` → `self._make_notebook()` (fixed 2026-04-01: `continuous_loop.py` line 288)
- FK constraint in `upsert_leaderboard` due to async writes (fixed 2026-04-01: `dashboard.py` flush_writes)
