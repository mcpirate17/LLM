# Baseline Scoring Reference — DO NOT DELETE

Established 2026-03-23. This document records the empirical measurements
and scoring design decisions for the Aria composite score v7.

---

## 1. Frontier Reference Perplexity

### 1.1 Reference Architectures

Four reference architectures serve as the scoring anchor. All measurements
use **WikiText-103-raw-v1** (546MB train, 1.1MB val) with tiktoken cl100k_base
tokenizer, d_model=256, 4 layers, seq_len=128, batch_size=4, lr=3e-4.

**Full training trajectory:**

| Model | Params | @100 | @500 | @1000 | @1500 | @2500 | @5000 | @10000 |
|---|---|---|---|---|---|---|---|---|
| RWKV | 28.6M | 11.6 | 8.4 | 7.2 | 6.0 | 5.4 | 4.7 | 4.2 |
| Mamba-SSM | 26.7M | 11.1 | 8.7 | 7.5 | 6.7 | 6.0 | 5.1 | 4.6 |
| GPT-2 | 29.1M | 15.5 | 13.2 | 12.6 | 12.0 | 11.9 | 10.2 | 7.1 |
| RAG | 29.9M | 18.9 | 14.0 | 12.8 | 12.3 | 11.1 | 8.7 | 6.6 |

**Frontier averages (scoring anchors):**

| Stage | Steps | Frontier Avg PPL |
|---|---|---|
| Screening (short) | 1,000 | **10.0** |
| Investigation (medium) | 2,500 | **8.6** |
| Validation (long) | 10,000 | **5.6** |

**Early convergence anchor:** avg ppl@100 / ppl@500 = **1.30**

### 1.2 Why WikiText-103

WikiText-2 (11MB train) causes overfitting after ~500 steps — perplexity
degrades monotonically with more training. WikiText-103 shows clean
monotonic improvement through 10,000 steps for all models.

### 1.3 Dataset Configuration

- Variant: `wikitext-103-raw-v1`
- max_chars_train: 200,000,000 (effectively unlimited)
- max_chars_val: 200,000
- n_eval_batches: 16
- LR warmup: 10x for first 100 steps (weight-tied lm_head init)

### 1.4 Graph Compilation

Reference models compiled via `compile_model([graph]*4, vocab_size=100277, max_seq_len=256)`.
GPT-2 graph: `layernorm → softmax_attention → add(residual) → layernorm → swiglu_mlp → add(residual)`.
Other references use their registered graph from `synthesis/reference_architectures.py`.

---

## 2. Composite Score v7 — Component Table

14 components. Tier-gated: some only active at investigation or validation.

| # | Component | Type | Formula | Max | Active at |
|---|---|---|---|---|---|
| 1 | Performance (short) | S-curve | `S(10.0 / model_ppl_1000) * 50` | **50** | all tiers |
| 2 | Performance (medium) | S-curve | `S(8.6 / model_ppl_2500) * 75` | **75** | investigation+ |
| 3 | Performance (long) | S-curve | `S(5.6 / model_ppl_10000) * 100` | **100** | validation |
| 4 | Parameter efficiency | S-curve | `S(model_eff / 1.09) * 50` | **50** | all tiers |
| 5 | Learning efficiency | S-curve | `S(model_conv / 1.13) * 25` | **25** | all tiers |
| 6 | Routing savings | Additive | `50 * routing_savings` | **50** | all tiers |
| 7 | Compression | Additive | `20 * max(0, 1-ratio) + quant` | **30** | all tiers |
| 8 | Activation sparsity | Additive | structural + activation | **30** | all tiers |
| 9 | Adaptive computation | Additive | depth + recursion savings | **25** | all tiers |
| 10 | Novelty | Additive | `score * confidence` (refs=0) | **40** | all tiers |
| 11 | NCD | Additive | `15 * max(0, 1 - ncd)` | **15** | all tiers |
| 12 | Robustness | Additive | spectral + noise + quant | **40** | investigation+ |
| 13 | Long context | S-curve | `S(model_lc / 0.375) * 25` | **25** | validation |
| 14 | Early convergence | S-curve | `S((ppl@100/ppl@500) / 1.30) * 10` | **10** | all tiers |

### Score Budget by Tier

| Tier | Available Components | Max Points |
|---|---|---|
| Screening | #1, #4-11, #14 | **325** |
| Investigation | #1-2, #4-12, #14 | **440** |
| Validation | #1-14 (all) | **565** |

### Tier-Gating Rules

- **Robustness** (#12): investigation+ only. Noise/quant/spectral measured during investigation.
- **Long context** (#13): validation only. Expensive eval, only justified at final stage.
- **Performance medium/long** (#2, #3): only when ppl at those step counts exists.
- **References score 0 on novelty** (#10). They ARE the baseline.

---

## 3. Escalation Thresholds

Reference architecture average scores at each tier determine escalation gates.

### Screening → Investigation

| Model | Screening Score |
|---|---|
| Mamba | 96.2 |
| RWKV | 97.7 |
| GPT-2 | 42.1 |
| RAG | 42.9 |
| **Average** | **69.7** |
| **Threshold (90%)** | **62.7** |

Components active at screening: perf_short, param_efficiency,
learning_efficiency, early_convergence + additive (routing, compression,
sparsity, adaptive, novelty, NCD).

### Investigation → Validation

| Model | Investigation Score |
|---|---|
| Mamba | 187.3 |
| RWKV | 196.2 |
| GPT-2 | 60.7 |
| RAG | 94.5 |
| **Average** | **134.7** |
| **Threshold (90%)** | **121.2** |

Components added at investigation: perf_medium, robustness.

Note: GPT-2 scores low because robustness data was never measured (0pts),
not because it lacks robustness. This is a data gap.

---

## 4. Pre-GPU Hard Gates

Five structural gates applied before any GPU time, plus one cheap-GPU gate.

| # | Gate | Rule | Cost | Kill Rate |
|---|---|---|---|---|
| 1 | Min ops | `n_ops() <= 5` → kill | free | 13.1% |
| 2 | Gradient path | `has_gradient_path() == False` → kill | free | 0% (insurance) |
| 3 | Residual path | `has_residual_path() == False` → kill | free | 4.4% |
| 4 | Parameterized ops | zero parameterized ops → kill | free | 0% (insurance) |
| 5 | Efficiency ops | no routing/MoE/sparse/compression op → kill | free | 14.7% |
| 6 | Flatline check | ppl@100/ppl@500 ratio < 1.10 → kill | ~8s GPU | ~45% of survivors |

**Gates 1-5 combined: 32.1% filtered before GPU.**
Gate 6 applied last (after compilation), not yet implemented as hard gate.

### Flatline Detection (validated, not gated yet)

Models flat from @100→@500 don't recover:
- `d1185b31`: 13.1→12.6→12.4→12.4→**12.4** (plateaued at 500)
- `31074db0`: 14.1→12.9→12.6→12.4→**12.5** (plateaued at 500)

---

## 5. Design Decisions Log

- **wikitext-2 rejected**: overfits after 500 steps
- **wiki103-20MB vs full**: identical at these step counts
- **Performance split into 3 tiers**: screening=cheap/noisy, validation=expensive/definitive
- **Zero-baseline components kept additive**: refs have no routing/compression/sparsity
- **Novelty zeroed for references**: they ARE the baseline, not novel
- **Robustness gated to investigation+**: only measured during investigation runs
- **Long context gated to validation**: expensive eval, final stage only
- **Early convergence added at 10pts**: ppl@100/ppl@500 ratio, soft flatline signal
- **Hard gates**: 5 free structural + 1 cheap GPU, 32.1% pre-GPU filter

---

## Changelog

- 2026-03-23: Initial perplexity measurements for 4 frontier refs at 500-1500 steps
- 2026-03-23: Established wiki103 @1000 steps as screening benchmark, avg=10.0
- 2026-03-23: Extended to 2500/5000/10000 steps for investigation/validation anchors
- 2026-03-23: Frontier avg @2500=8.6, @10000=5.6
- 2026-03-23: LOCKED 14-component scoring table
- 2026-03-23: Added 5 pre-GPU hard gates (32.1% filter)
- 2026-03-23: Novelty zeroed for references
- 2026-03-23: Robustness gated to investigation+, long context gated to validation
- 2026-03-23: Early convergence added (10pts, ppl@100/ppl@500, anchor=1.30)
- 2026-03-23: Escalation thresholds: screening→investigation ≥63, investigation→validation ≥121
