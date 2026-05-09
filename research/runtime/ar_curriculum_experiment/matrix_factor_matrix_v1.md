# AR Curriculum factor-matrix — factor_matrix_v1

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

## AUC pair (final, mean ± std across seeds) — by condition × arch

| condition | GPT-2 | Mamba | RWKV | Retrieval-Augmented | wall(s) |
|---|---:|---:|---:|---:|---:|
| baseline | 0.150±0.025 | 0.245±0.008 | 0.263±0.019 | 0.143±0.041 | 102 |
| granular | 0.203±0.044 | 0.311±0.054 | 0.305±0.013 | 0.179±0.059 | 154 |
| bump | 0.131±0.042 | 0.246±0.007 | 0.238±0.018 | 0.099±0.034 | 137 |
| s500 | 0.222±0.055 | 0.271±0.046 | 0.304±0.010 | 0.175±0.073 | 201 |
| s750 | 0.201±0.022 | 0.279±0.016 | 0.284±0.029 | 0.146±0.112 | 300 |
| s1000 | 0.169±0.025 | 0.289±0.015 | 0.320±0.016 | 0.155±0.041 | 396 |

## AUC pair (peak, mean across seeds)

| condition | GPT-2 | Mamba | RWKV | Retrieval-Augmented | — |
|---|---:|---:|---:|---:|---:|
| baseline | 0.279 | 0.281 | 0.307 | 0.263 | — |
| granular | 0.350 | 0.385 | 0.400 | 0.340 | — |
| bump | 0.257 | 0.295 | 0.306 | 0.196 | — |
| s500 | 0.318 | 0.299 | 0.329 | 0.280 | — |
| s750 | 0.305 | 0.298 | 0.317 | 0.271 | — |
| s1000 | 0.298 | 0.303 | 0.337 | 0.297 | — |

## Retention (final / peak) — closer to 1.0 = no forgetting

| condition | GPT-2 | Mamba | RWKV | Retrieval-Augmented | — |
|---|---:|---:|---:|---:|---:|
| baseline | 0.54 | 0.87 | 0.86 | 0.54 | — |
| granular | 0.58 | 0.81 | 0.76 | 0.52 | — |
| bump | 0.50 | 0.83 | 0.78 | 0.50 | — |
| s500 | 0.70 | 0.90 | 0.92 | 0.62 | — |
| s750 | 0.66 | 0.94 | 0.90 | 0.51 | — |
| s1000 | 0.57 | 0.95 | 0.95 | 0.52 | — |

## Spread (max - min AUC final across archs, per condition)

Higher spread = better discrimination among the 4 archs.

| condition | spread | best arch | worst arch |
|---|---:|---|---|
| baseline | 0.120 | RWKV (0.263) | Retrieval-Augmented (0.143) |
| granular | 0.132 | Mamba (0.311) | Retrieval-Augmented (0.179) |
| bump | 0.147 | Mamba (0.246) | Retrieval-Augmented (0.099) |
| s500 | 0.129 | RWKV (0.304) | Retrieval-Augmented (0.175) |
| s750 | 0.138 | RWKV (0.284) | Retrieval-Augmented (0.146) |
| s1000 | 0.165 | RWKV (0.320) | Retrieval-Augmented (0.155) |
