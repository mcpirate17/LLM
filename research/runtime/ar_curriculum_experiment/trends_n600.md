# AR curriculum trends — n600

Analyzed n=600 archs with ar_curriculum data.

## Distribution summary

| metric | n | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|---:|
| AUC pair final | 600 | 0.296 | 0.271 | 0.149 | 0.076 | 0.841 |
| S0 retention | 600 | 0.446 | 0.291 | 0.283 | 0.000 | 1.000 |
| Max passing stage | 600 | 2.260 | 5.000 | 2.932 | -1.000 | 5.000 |

## Quadrant distribution (auc≥0.3, retention≥0.5)

| quadrant | description | n | % |
|---|---|---:|---:|
| Q1_learns_retains | high AUC + high retention (Mamba-class) | 173 | 28.8% |
| Q2_learns_forgets | high AUC + low retention (RWKV/dense-attn-class) | 94 | 15.7% |
| Q3_no_learn_retains | low AUC + high retention (under-trained or capacity-bound) | 23 | 3.8% |
| Q4_no_learn_no_retain | low AUC + low retention (broken) | 310 | 51.7% |

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
| 21 | 3c42550f4966 | validation | 265.9 | 0.542 | 1.000 | 5 | Q1 learns retains | — | recursive_depth_router |
| 22 | cf0f26ca1637 | validation | 384.2 | 0.539 | 0.291 | 5 | Q2 learns forgets | — | attn_multi_head_mix |
| 23 | f4854b5ea22f | validation | 138.7 | 0.539 | 1.000 | 5 | Q1 learns retains | — | latent_attn_ssm_hybrid |
| 24 | 17276feaf2ff | validation | 391.4 | 0.531 | 1.000 | 5 | Q1 learns retains | — | latent_attn_conv_hybrid |
| 25 | 20f66027a8b9 | validation | 116.3 | 0.531 | 1.000 | 5 | Q1 learns retains | — | local_attn_ssm_hybrid |
| 26 | 41cd140aa822 | validation | 430.6 | 0.527 | 1.000 | 5 | Q1 learns retains | — | latent_attn_conv_hybrid |
| 27 | 6360afaeec2d | validation | 399.2 | 0.527 | 1.000 | 5 | Q1 learns retains | — | local_attn_ssm_hybrid |
| 28 | 492faf1799b1 | validation | 285.8 | 0.525 | 1.000 | 5 | Q1 learns retains | — | recursive_depth_router |
| 29 | 32138eb24339 | validation | 419.8 | 0.525 | 1.000 | 5 | Q1 learns retains | — | latent_attn_conv_hybrid |
| 30 | 057dcc78f66b | validation | 317.8 | 0.524 | 1.000 | 5 | Q1 learns retains | — | recursive_depth_router |

## Per-template breakdown (n≥3 archs/template)

| template | n | mean AUC | median AUC | std |
|---|---:|---:|---:|---:|
| attn_state_space_hybrid | 9 | 0.510 | 0.454 | 0.129 |
| rwkv_sparse_chain | 4 | 0.476 | 0.544 | 0.254 |
| decay_sequence | 3 | 0.445 | 0.421 | 0.042 |
| sparse_ffn | 4 | 0.426 | 0.341 | 0.287 |
| residual_block | 15 | 0.405 | 0.427 | 0.249 |
| recursive_depth_router | 26 | 0.398 | 0.414 | 0.156 |
| latent_attn_ssm_hybrid | 63 | 0.398 | 0.419 | 0.125 |
| linear_attn_sparse_ffn | 25 | 0.395 | 0.399 | 0.037 |
| latent_attn_conv_hybrid | 84 | 0.381 | 0.471 | 0.178 |
| causal_mix_block | 18 | 0.374 | 0.385 | 0.111 |
| depth_gated_block | 45 | 0.365 | 0.378 | 0.112 |
| ultrametric_attention_block | 16 | 0.365 | 0.409 | 0.115 |
| cascaded_early_exit | 3 | 0.364 | 0.416 | 0.113 |
| adaptive_ssm_chain | 6 | 0.363 | 0.384 | 0.066 |
| latent_compress_rwkv | 4 | 0.332 | 0.372 | 0.165 |
| topk_retrieval | 4 | 0.327 | 0.300 | 0.126 |
| local_attention_block | 12 | 0.325 | 0.339 | 0.192 |
| multi_head_mix_block | 3 | 0.325 | 0.239 | 0.161 |
| transformer_block | 3 | 0.325 | 0.339 | 0.097 |
| local_attn_ssm_hybrid | 106 | 0.322 | 0.360 | 0.193 |
| spiking_moe_block | 10 | 0.320 | 0.341 | 0.102 |
| local_attn_moe | 31 | 0.318 | 0.315 | 0.087 |
| windowed_attention | 11 | 0.317 | 0.282 | 0.173 |
| signal_routed_compression | 7 | 0.306 | 0.211 | 0.157 |
| diff_attn_conv_hybrid | 26 | 0.306 | 0.284 | 0.074 |
| recursive_attn_ssm_hybrid | 3 | 0.304 | 0.321 | 0.066 |
| attn_multi_head_mix | 36 | 0.301 | 0.266 | 0.148 |
| rwkv_block | 16 | 0.291 | 0.308 | 0.113 |
| integral_kernel_block | 9 | 0.291 | 0.230 | 0.145 |
| attn_cross_dim | 3 | 0.272 | 0.220 | 0.187 |

## Upstream-feature correlations

Predictive value: high |spearman_vs_auc| means we can predict ar_curriculum from this cheap upstream signal.

| feature | n | spearman vs AUC | pearson vs AUC | spearman vs retention |
|---|---:|---:|---:|---:|
| ar_validation_held_pair_acc | 214 | +0.310 | +0.361 | +0.375 |
| fp_jacobian_erf_decay_slope | 577 | -0.193 | -0.300 | -0.245 |
| blimp_overall_accuracy | 600 | +0.095 | +0.079 | +0.043 |
| fp_jacobian_erf_density | 577 | +0.086 | +0.012 | -0.057 |
| hellaswag_acc | 600 | +0.062 | +0.074 | +0.034 |
| induction_screening_auc | 600 | -0.058 | -0.091 | -0.076 |
| ar_legacy_auc | 593 | -0.057 | -0.073 | -0.063 |
| binding_screening_auc | 533 | -0.040 | -0.119 | -0.052 |
| validation_loss_ratio | 600 | +0.016 | +0.014 | -0.003 |
| wikitext_perplexity | 600 | -0.015 | -0.007 | -0.065 |
