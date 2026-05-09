# AR curriculum trends — n400

Analyzed n=400 archs with ar_curriculum data.

## Distribution summary

| metric | n | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|---:|
| AUC pair final | 400 | 0.314 | 0.291 | 0.158 | 0.107 | 0.841 |
| S0 retention | 400 | 0.481 | 0.291 | 0.303 | 0.000 | 1.000 |
| Max passing stage | 400 | 2.498 | 5.000 | 2.896 | -1.000 | 5.000 |

## Quadrant distribution (auc≥0.3, retention≥0.5)

| quadrant | description | n | % |
|---|---|---:|---:|
| Q1_learns_retains | high AUC + high retention (Mamba-class) | 138 | 34.5% |
| Q2_learns_forgets | high AUC + low retention (RWKV/dense-attn-class) | 56 | 14.0% |
| Q3_no_learn_retains | low AUC + high retention (under-trained or capacity-bound) | 12 | 3.0% |
| Q4_no_learn_no_retain | low AUC + low retention (broken) | 194 | 48.5% |

## Top archs by AUC (top 30)

| rank | fp | tier | composite | AUC | retention | max_pass | quadrant | motifs | template |
|---:|---|---|---:|---:|---:|---:|---|---|---|
| 1 | 13021b4ebe7a | validation | 173.4 | 0.841 | 1.000 | 5 | Q1 learns retains | — | residual_block |
| 2 | bb0b8d5856da | validation | 284.9 | 0.797 | 1.000 | 5 | Q1 learns retains | — | local_attn_ssm_hybrid |
| 3 | 0fd897894fd7 | validation | 306.5 | 0.742 | 1.000 | 5 | Q1 learns retains | — | attn_state_space_hybrid |
| 4 | a70ccfcb6e8b | validation | 215.0 | 0.729 | 0.240 | 5 | Q2 learns forgets | — | windowed_attention |
| 5 | c26893ddbd22 | validation | 416.6 | 0.712 | 1.000 | 5 | Q1 learns retains | — | local_attn_ssm_hybrid |
| 6 | a266fe39a62c | validation | 376.8 | 0.706 | 1.000 | 5 | Q1 learns retains | — | attn_state_space_hybrid |
| 7 | abbfd3b41436 | validation | 348.8 | 0.601 | 1.000 | 5 | Q1 learns retains | — | recursive_depth_router |
| 8 | bed21f3b2620 | validation | 190.0 | 0.591 | 0.291 | 5 | Q2 learns forgets | — | attn_multi_head_mix |
| 9 | 350981143549 | validation | 421.2 | 0.574 | 1.000 | 5 | Q1 learns retains | — | latent_attn_conv_hybrid |
| 10 | efe2a7ffef27 | validation | 378.9 | 0.573 | 1.000 | 5 | Q1 learns retains | — | three_lane_adaptive |
| 11 | 32f4ffc22315 | validation | 401.1 | 0.564 | 1.000 | 5 | Q1 learns retains | — | latent_attn_conv_hybrid |
| 12 | 5d7c2086c4a4 | validation | 405.6 | 0.564 | 1.000 | 5 | Q1 learns retains | — | local_attn_ssm_hybrid |
| 13 | 8376b3f7114c | validation | 419.1 | 0.558 | 0.766 | 5 | Q1 learns retains | — | local_attn_ssm_hybrid |
| 14 | 2a49423162be | validation | 418.5 | 0.554 | 1.000 | 5 | Q1 learns retains | — | latent_attn_conv_hybrid |
| 15 | 1748bb43888b | validation | 408.6 | 0.550 | 1.000 | 5 | Q1 learns retains | — | latent_attn_ssm_hybrid |
| 16 | 1b6ee885e190 | validation | 406.0 | 0.550 | 1.000 | 5 | Q1 learns retains | — | local_attn_ssm_hybrid |
| 17 | f325b3fc5b23 | validation | 392.2 | 0.549 | 1.000 | 5 | Q1 learns retains | — | local_attn_ssm_hybrid |
| 18 | ac25efaf73ff | validation | 438.5 | 0.546 | 1.000 | 5 | Q1 learns retains | — | latent_attn_conv_hybrid |
| 19 | c79d10bcb03a | validation | 420.0 | 0.544 | 1.000 | 5 | Q1 learns retains | — | latent_attn_conv_hybrid |
| 20 | 0ce65cff746d | validation | 411.2 | 0.543 | 0.709 | 5 | Q1 learns retains | — | latent_attn_ssm_hybrid |
| 21 | cf0f26ca1637 | validation | 384.2 | 0.539 | 0.291 | 5 | Q2 learns forgets | — | attn_multi_head_mix |
| 22 | f4854b5ea22f | validation | 138.7 | 0.539 | 1.000 | 5 | Q1 learns retains | — | latent_attn_ssm_hybrid |
| 23 | 17276feaf2ff | validation | 391.4 | 0.531 | 1.000 | 5 | Q1 learns retains | — | latent_attn_conv_hybrid |
| 24 | 20f66027a8b9 | validation | 116.3 | 0.531 | 1.000 | 5 | Q1 learns retains | — | local_attn_ssm_hybrid |
| 25 | 41cd140aa822 | validation | 430.6 | 0.527 | 1.000 | 5 | Q1 learns retains | — | latent_attn_conv_hybrid |
| 26 | 6360afaeec2d | validation | 399.2 | 0.527 | 1.000 | 5 | Q1 learns retains | — | local_attn_ssm_hybrid |
| 27 | 492faf1799b1 | validation | 285.8 | 0.525 | 1.000 | 5 | Q1 learns retains | — | recursive_depth_router |
| 28 | 32138eb24339 | validation | 419.8 | 0.525 | 1.000 | 5 | Q1 learns retains | — | latent_attn_conv_hybrid |
| 29 | 057dcc78f66b | validation | 317.8 | 0.524 | 1.000 | 5 | Q1 learns retains | — | recursive_depth_router |
| 30 | 8737668ad680 | validation | 403.4 | 0.524 | 1.000 | 5 | Q1 learns retains | — | latent_attn_conv_hybrid |

## Per-template breakdown (n≥3 archs/template)

| template | n | mean AUC | median AUC | std |
|---|---:|---:|---:|---:|
| local_attention_block | 3 | 0.523 | 0.481 | 0.189 |
| attn_state_space_hybrid | 9 | 0.510 | 0.454 | 0.129 |
| rwkv_sparse_chain | 4 | 0.476 | 0.544 | 0.254 |
| ultrametric_attention_block | 4 | 0.449 | 0.448 | 0.052 |
| latent_attn_ssm_hybrid | 49 | 0.424 | 0.428 | 0.108 |
| recursive_depth_router | 17 | 0.418 | 0.421 | 0.155 |
| residual_block | 13 | 0.417 | 0.427 | 0.261 |
| latent_attn_conv_hybrid | 76 | 0.404 | 0.491 | 0.167 |
| causal_mix_block | 6 | 0.403 | 0.400 | 0.079 |
| linear_attn_sparse_ffn | 25 | 0.395 | 0.399 | 0.037 |
| depth_gated_block | 29 | 0.381 | 0.412 | 0.112 |
| spiking_moe_block | 3 | 0.374 | 0.362 | 0.040 |
| windowed_attention | 6 | 0.365 | 0.315 | 0.207 |
| local_attn_moe | 25 | 0.331 | 0.358 | 0.091 |
| topk_retrieval | 4 | 0.327 | 0.300 | 0.126 |
| local_attn_ssm_hybrid | 106 | 0.322 | 0.360 | 0.193 |
| rwkv_block | 10 | 0.321 | 0.319 | 0.107 |
| diff_attn_conv_hybrid | 17 | 0.317 | 0.284 | 0.080 |
| attn_multi_head_mix | 36 | 0.301 | 0.266 | 0.148 |
| mixed_recursion | 13 | 0.292 | 0.289 | 0.106 |
| spiking_residual_block | 4 | 0.290 | 0.260 | 0.113 |
| three_way_split | 9 | 0.273 | 0.265 | 0.115 |
| latent_attn_sparse_ffn | 3 | 0.243 | 0.122 | 0.215 |
| residual_difference | 6 | 0.239 | 0.222 | 0.133 |
| gated_product | 7 | 0.238 | 0.223 | 0.100 |
| attn_softmax_normalized_matmul_compact_ffn | 4 | 0.238 | 0.249 | 0.029 |
| conditional_compute | 29 | 0.232 | 0.165 | 0.144 |
| three_lane_adaptive | 9 | 0.227 | 0.176 | 0.145 |
| sparse_moe_block | 8 | 0.227 | 0.197 | 0.124 |
| difficulty_routed_block | 53 | 0.225 | 0.202 | 0.098 |

## Upstream-feature correlations

Predictive value: high |spearman_vs_auc| means we can predict ar_curriculum from this cheap upstream signal.

| feature | n | spearman vs AUC | pearson vs AUC | spearman vs retention |
|---|---:|---:|---:|---:|
| ar_validation_held_pair_acc | 213 | +0.311 | +0.362 | +0.374 |
| fp_jacobian_erf_density | 378 | +0.190 | +0.098 | +0.002 |
| fp_jacobian_erf_decay_slope | 378 | -0.185 | -0.273 | -0.283 |
| blimp_overall_accuracy | 400 | +0.126 | +0.117 | +0.047 |
| binding_screening_auc | 380 | -0.124 | -0.157 | -0.149 |
| ar_legacy_auc | 395 | -0.123 | -0.131 | -0.116 |
| induction_screening_auc | 400 | -0.118 | -0.143 | -0.135 |
| hellaswag_acc | 400 | +0.086 | +0.094 | +0.045 |
| validation_loss_ratio | 400 | +0.069 | +0.108 | +0.043 |
| wikitext_perplexity | 400 | +0.039 | +0.170 | -0.052 |
