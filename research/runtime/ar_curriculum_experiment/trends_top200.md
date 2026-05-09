# AR curriculum trends — top200

Analyzed n=200 archs with ar_curriculum data.

## Distribution summary

| metric | n | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|---:|
| AUC pair final | 200 | 0.331 | 0.331 | 0.167 | 0.107 | 0.712 |
| S0 retention | 200 | 0.517 | 0.291 | 0.328 | 0.000 | 1.000 |
| Max passing stage | 200 | 2.745 | 5.000 | 2.864 | -1.000 | 5.000 |

## Quadrant distribution (auc≥0.3, retention≥0.5)

| quadrant | description | n | % |
|---|---|---:|---:|
| Q1_learns_retains | high AUC + high retention (Mamba-class) | 77 | 38.5% |
| Q2_learns_forgets | high AUC + low retention (RWKV/dense-attn-class) | 31 | 15.5% |
| Q3_no_learn_retains | low AUC + high retention (under-trained or capacity-bound) | 1 | 0.5% |
| Q4_no_learn_no_retain | low AUC + low retention (broken) | 91 | 45.5% |

## Top archs by AUC (top 30)

| rank | fp | tier | composite | AUC | retention | max_pass | quadrant | motifs | template |
|---:|---|---|---:|---:|---:|---:|---|---|---|
| 1 | c26893ddbd22 | validation | 416.6 | 0.712 | 1.000 | 5 | Q1 learns retains | — | local_attn_ssm_hybrid |
| 2 | a266fe39a62c | validation | 376.8 | 0.706 | 1.000 | 5 | Q1 learns retains | — | attn_state_space_hybrid |
| 3 | abbfd3b41436 | validation | 348.8 | 0.601 | 1.000 | 5 | Q1 learns retains | — | recursive_depth_router |
| 4 | 350981143549 | validation | 421.2 | 0.574 | 1.000 | 5 | Q1 learns retains | — | latent_attn_conv_hybrid |
| 5 | efe2a7ffef27 | validation | 378.9 | 0.573 | 1.000 | 5 | Q1 learns retains | — | three_lane_adaptive |
| 6 | 32f4ffc22315 | validation | 401.1 | 0.564 | 1.000 | 5 | Q1 learns retains | — | latent_attn_conv_hybrid |
| 7 | 5d7c2086c4a4 | validation | 405.6 | 0.564 | 1.000 | 5 | Q1 learns retains | — | local_attn_ssm_hybrid |
| 8 | 8376b3f7114c | validation | 419.1 | 0.558 | 0.766 | 5 | Q1 learns retains | — | local_attn_ssm_hybrid |
| 9 | 2a49423162be | validation | 418.5 | 0.554 | 1.000 | 5 | Q1 learns retains | — | latent_attn_conv_hybrid |
| 10 | 1748bb43888b | validation | 408.6 | 0.550 | 1.000 | 5 | Q1 learns retains | — | latent_attn_ssm_hybrid |
| 11 | 1b6ee885e190 | validation | 406.0 | 0.550 | 1.000 | 5 | Q1 learns retains | — | local_attn_ssm_hybrid |
| 12 | f325b3fc5b23 | validation | 392.2 | 0.549 | 1.000 | 5 | Q1 learns retains | — | local_attn_ssm_hybrid |
| 13 | ac25efaf73ff | validation | 438.5 | 0.546 | 1.000 | 5 | Q1 learns retains | — | latent_attn_conv_hybrid |
| 14 | c79d10bcb03a | validation | 420.0 | 0.544 | 1.000 | 5 | Q1 learns retains | — | latent_attn_conv_hybrid |
| 15 | 0ce65cff746d | validation | 411.2 | 0.543 | 0.709 | 5 | Q1 learns retains | — | latent_attn_ssm_hybrid |
| 16 | cf0f26ca1637 | validation | 384.2 | 0.539 | 0.291 | 5 | Q2 learns forgets | — | attn_multi_head_mix |
| 17 | 17276feaf2ff | validation | 391.4 | 0.531 | 1.000 | 5 | Q1 learns retains | — | latent_attn_conv_hybrid |
| 18 | 41cd140aa822 | validation | 430.6 | 0.527 | 1.000 | 5 | Q1 learns retains | — | latent_attn_conv_hybrid |
| 19 | 6360afaeec2d | validation | 399.2 | 0.527 | 1.000 | 5 | Q1 learns retains | — | local_attn_ssm_hybrid |
| 20 | 32138eb24339 | validation | 419.8 | 0.525 | 1.000 | 5 | Q1 learns retains | — | latent_attn_conv_hybrid |
| 21 | 8737668ad680 | validation | 403.4 | 0.524 | 1.000 | 5 | Q1 learns retains | — | latent_attn_conv_hybrid |
| 22 | 61754c8e5ed7 | validation | 424.3 | 0.523 | 1.000 | 5 | Q1 learns retains | — | residual_block |
| 23 | 272af5a32bbf | validation | 471.2 | 0.520 | 1.000 | 5 | Q1 learns retains | — | latent_attn_ssm_hybrid |
| 24 | 1c3cd4948e96 | validation | 382.9 | 0.520 | 1.000 | 5 | Q1 learns retains | — | latent_attn_conv_hybrid |
| 25 | 7e88106fb031 | validation | 399.4 | 0.518 | 1.000 | 5 | Q1 learns retains | — | latent_attn_conv_hybrid |
| 26 | 6f738a917c7a | validation | 377.9 | 0.518 | 1.000 | 5 | Q1 learns retains | — | latent_attn_ssm_hybrid |
| 27 | f6f837cdcb1f | validation | 376.5 | 0.517 | 1.000 | 5 | Q1 learns retains | — | latent_attn_ssm_hybrid |
| 28 | bb7c0555c8b4 | validation | 404.1 | 0.517 | 1.000 | 5 | Q1 learns retains | — | latent_attn_conv_hybrid |
| 29 | 496d079ec3e8 | validation | 369.7 | 0.516 | 1.000 | 5 | Q1 learns retains | — | conditional_compute |
| 30 | 32e2634a79c6 | validation | 404.6 | 0.513 | 1.000 | 5 | Q1 learns retains | — | latent_attn_conv_hybrid |

## Per-template breakdown (n≥3 archs/template)

| template | n | mean AUC | median AUC | std |
|---|---:|---:|---:|---:|
| attn_state_space_hybrid | 5 | 0.477 | 0.454 | 0.132 |
| rwkv_sparse_chain | 4 | 0.476 | 0.544 | 0.254 |
| attn_multi_head_mix | 3 | 0.455 | 0.539 | 0.145 |
| latent_attn_ssm_hybrid | 25 | 0.431 | 0.481 | 0.119 |
| local_attn_ssm_hybrid | 51 | 0.430 | 0.475 | 0.144 |
| recursive_depth_router | 3 | 0.406 | 0.480 | 0.242 |
| latent_attn_conv_hybrid | 71 | 0.400 | 0.492 | 0.172 |
| linear_attn_sparse_ffn | 8 | 0.372 | 0.379 | 0.032 |
| topk_retrieval | 3 | 0.347 | 0.335 | 0.146 |
| local_attn_moe | 7 | 0.343 | 0.315 | 0.121 |
| residual_block | 5 | 0.300 | 0.268 | 0.187 |
| rwkv_block | 6 | 0.300 | 0.305 | 0.104 |
| mixed_recursion | 3 | 0.299 | 0.278 | 0.199 |
| residual_difference | 3 | 0.292 | 0.291 | 0.168 |
| difficulty_routed_block | 13 | 0.281 | 0.268 | 0.167 |
| three_way_split | 6 | 0.281 | 0.284 | 0.115 |
| diff_attn_conv_hybrid | 6 | 0.269 | 0.272 | 0.020 |
| depth_gated_block | 5 | 0.265 | 0.359 | 0.129 |
| routed_bottleneck | 4 | 0.261 | 0.213 | 0.183 |
| attn_softmax_normalized_matmul_compact_ffn | 3 | 0.252 | 0.249 | 0.006 |
| sparse_moe_block | 7 | 0.243 | 0.207 | 0.125 |
| latent_attn_sparse_ffn | 3 | 0.243 | 0.122 | 0.215 |
| graph_attn_moe | 10 | 0.243 | 0.232 | 0.050 |
| diff_attn_moe | 4 | 0.233 | 0.233 | 0.007 |
| three_lane_adaptive | 7 | 0.231 | 0.176 | 0.158 |
| conditional_compute | 22 | 0.225 | 0.133 | 0.149 |
| normalized_matmul | 15 | 0.177 | 0.176 | 0.053 |
| token_merge_block | 13 | 0.165 | 0.122 | 0.108 |
| rope_attention_block | 4 | 0.150 | 0.150 | 0.023 |
| cascaded_attn_ffn | 3 | 0.120 | 0.117 | 0.005 |

## Upstream-feature correlations

Predictive value: high |spearman_vs_auc| means we can predict ar_curriculum from this cheap upstream signal.

| feature | n | spearman vs AUC | pearson vs AUC | spearman vs retention |
|---|---:|---:|---:|---:|
| fp_jacobian_erf_decay_slope | 200 | -0.436 | -0.560 | -0.500 |
| ar_validation_held_pair_acc | 195 | +0.323 | +0.351 | +0.390 |
| fp_jacobian_erf_density | 200 | +0.253 | +0.095 | +0.031 |
| binding_screening_auc | 183 | -0.181 | -0.236 | -0.248 |
| induction_screening_auc | 200 | -0.164 | +0.023 | -0.230 |
| validation_loss_ratio | 200 | +0.154 | +0.213 | +0.175 |
| hellaswag_acc | 200 | +0.073 | +0.071 | +0.034 |
| blimp_overall_accuracy | 200 | +0.066 | +0.066 | +0.022 |
| wikitext_perplexity | 200 | +0.016 | +0.212 | -0.068 |
| ar_legacy_auc | 197 | -0.003 | -0.033 | -0.056 |
