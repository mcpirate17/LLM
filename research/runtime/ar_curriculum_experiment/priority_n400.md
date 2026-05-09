# AR curriculum priority list — n400

n_remaining = 839

## Used correlations (>|0.05|)

| feature | spearman vs AUC |
|---|---:|
| induction_screening_auc | -0.118 |
| binding_screening_auc | -0.124 |
| ar_legacy_auc | -0.123 |
| hellaswag_acc | +0.086 |
| blimp_overall_accuracy | +0.126 |
| fp_jacobian_erf_density | +0.190 |
| fp_jacobian_erf_decay_slope | -0.185 |
| validation_loss_ratio | +0.069 |
| ar_validation_held_pair_acc | +0.311 |

## Top 30 by priority

| rank | fp | tier | composite | priority | pred_auc | tpl_bonus | susp_bonus | motifs |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | 977cc6860707 | validation | 186 | 1.065 | 0.755 | 0.449 | 0.002 | — |
| 2 | 94f92815f3bf | validation | 229 | 1.059 | 0.721 | 0.523 | 0.012 | — |
| 3 | 586a514db734 | validation | 210 | 1.052 | 0.757 | 0.449 | 0.010 | — |
| 4 | 5dffd90cf456 | validation | 200 | 1.051 | 0.747 | 0.449 | 0.005 | — |
| 5 | d4733bb7e94f | validation | 187 | 1.046 | 0.754 | 0.449 | 0.002 | — |
| 6 | ce9a5f44f052 | validation | 186 | 1.045 | 0.749 | 0.449 | 0.002 | — |
| 7 | 80a9dc25dbcc | validation | 274 | 1.044 | 0.753 | 0.449 | 0.009 | — |
| 8 | 8127f0c77cf1 | validation | 185 | 1.043 | 0.764 | 0.403 | 0.002 | — |
| 9 | 272c97e68c64 | validation | 281 | 1.038 | 0.748 | 0.403 | 0.007 | — |
| 10 | fdcac8035e32 | validation | 317 | 1.038 | 0.759 | 0.381 | 0.003 | — |
| 11 | 2fb05c0f145b | validation | 334 | 1.036 | 0.680 | 0.523 | 0.002 | — |
| 12 | a0ff92fd4ab1 | validation | 281 | 1.035 | 0.727 | 0.449 | 0.006 | — |
| 13 | 3ace85e62de0 | validation | 325 | 1.035 | 0.767 | 0.374 | 0.006 | — |
| 14 | fd18768dcd93 | validation | 266 | 1.034 | 0.748 | 0.417 | 0.004 | — |
| 15 | 871a3a4c5143 | validation | 269 | 1.029 | 0.753 | 0.449 | 0.006 | — |
| 16 | 4bb50d028a40 | validation | 270 | 1.025 | 0.757 | 0.381 | 0.008 | — |
| 17 | 216581abaee6 | validation | 327 | 1.024 | 0.769 | 0.374 | 0.006 | — |
| 18 | bbc99e183faf | validation | 256 | 1.020 | 0.761 | 0.381 | 0.004 | — |
| 19 | af4bb02cc1eb | validation | 290 | 1.017 | 0.757 | 0.374 | 0.006 | — |
| 20 | 94e5eab296ed | validation | 305 | 1.011 | 0.755 | 0.389 | 0.004 | — |
| 21 | f218cfed87e5 | validation | 303 | 1.000 | 0.755 | 0.384 | 0.029 | — |
| 22 | 63f4a857c86a | validation | 262 | 0.988 | 0.758 | 0.381 | 0.004 | — |
| 23 | 478f411d483d | validation | 150 | 0.987 | 0.755 | 0.318 | 0.008 | — |
| 24 | af7f470574d1 | validation | 340 | 0.986 | 0.758 | 0.377 | 0.002 | — |
| 25 | ea65c6469154 | validation | 202 | 0.985 | 0.734 | 0.374 | 0.000 | — |
| 26 | ed72be530a94 | validation | 198 | 0.973 | 0.616 | 0.523 | 0.002 | — |
| 27 | 8fb2bde9815a | validation | 283 | 0.961 | 0.682 | 0.424 | 0.002 | — |
| 28 | 16411cdd1873 | validation | 331 | 0.957 | 0.688 | 0.381 | 0.002 | — |
| 29 | 657bf4bb89d2 | validation | 165 | 0.955 | 0.753 | 0.290 | 0.004 | — |
| 30 | 2cb179a39904 | validation | 217 | 0.955 | 0.759 | 0.251 | 0.004 | — |
