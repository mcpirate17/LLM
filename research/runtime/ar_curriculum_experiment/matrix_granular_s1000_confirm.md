# AR Curriculum factor-matrix — granular_s1000_confirm

archs=['gpt2', 'mamba', 'rwkv', 'retrieval_augmented'] seeds=[0, 1, 2] (3 per cell)

## Conditions

| name | stages | d_model | n_layers | steps |
|---|---|---:|---:|---:|
| baseline | default (6) | 256 | 6 | 250 |
| granular | fine (9) | 256 | 6 | 250 |
| bump | default (6) | 384 | 8 | 250 |
| s500 | default (6) | 256 | 6 | 500 |
| s750 | default (6) | 256 | 6 | 750 |
| s1000 | default (6) | 256 | 6 | 1000 |
| granular_s1000 | fine (9) | 256 | 6 | 1000 |

## AUC pair (final, mean ± std across seeds) — by condition × arch

| condition | GPT-2 | Mamba | RWKV | Retrieval-Augmented | wall(s) |
|---|---:|---:|---:|---:|---:|
| baseline | — | — | — | — | 0 |
| granular | — | — | — | — | 0 |
| bump | — | — | — | — | 0 |
| s500 | — | — | — | — | 0 |
| s750 | — | — | — | — | 0 |
| s1000 | — | — | — | — | 0 |
| granular_s1000 | 0.196±0.033 | 0.366±0.016 | 0.294±0.025 | 0.153±0.045 | 595 |

## AUC pair (peak, mean across seeds)

| condition | GPT-2 | Mamba | RWKV | Retrieval-Augmented | — |
|---|---:|---:|---:|---:|---:|
| baseline | — | — | — | — | — |
| granular | — | — | — | — | — |
| bump | — | — | — | — | — |
| s500 | — | — | — | — | — |
| s750 | — | — | — | — | — |
| s1000 | — | — | — | — | — |
| granular_s1000 | 0.387 | 0.409 | 0.408 | 0.341 | — |

## Retention (final / peak) — closer to 1.0 = no forgetting

| condition | GPT-2 | Mamba | RWKV | Retrieval-Augmented | — |
|---|---:|---:|---:|---:|---:|
| baseline | — | — | — | — | — |
| granular | — | — | — | — | — |
| bump | — | — | — | — | — |
| s500 | — | — | — | — | — |
| s750 | — | — | — | — | — |
| s1000 | — | — | — | — | — |
| granular_s1000 | 0.51 | 0.90 | 0.72 | 0.45 | — |

## Spread (max - min AUC final across archs, per condition)

Higher spread = better discrimination among the 4 archs.

| condition | spread | best arch | worst arch |
|---|---:|---|---|
| granular_s1000 | 0.213 | Mamba (0.366) | Retrieval-Augmented (0.153) |
