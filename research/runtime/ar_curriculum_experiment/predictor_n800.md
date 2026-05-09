# AR curriculum predictor — n800

n_rows_with_curriculum = 798

Multi-model 5-fold CV. Models: gbm (default GBM), gbm_deep (GBM with deeper trees), rf (random forest), tree (single decision tree depth=5).
Decision threshold: best_model Spearman ρ ≥ 0.7 → predictor reliable enough to triage; < 0.7 → run more backfill.

## Per-target best model summary

| target | n | best model | best Spearman test | y mean ± std | verdict |
|---|---:|---|---|---|---|
| ar_curriculum_auc_pair_final | 798 | rf | +0.520 ± 0.039 | 0.286 ± 0.144 | MORE DATA NEEDED |
| ar_curriculum_s0_retention | 798 | rf | +0.337 ± 0.048 | 0.427 ± 0.270 | MORE DATA NEEDED |

## ar_curriculum_auc_pair_final — model comparison (5-fold CV)

| model | Spearman train (mean ± std) | Spearman test (mean ± std) | per-fold test | MAE test |
|---|---|---|---|---:|
| gbm | +0.838 ± 0.009 | +0.511 ± 0.051 | +0.58, +0.47, +0.56, +0.49, +0.45 | 0.093 |
| gbm_deep | +0.957 ± 0.002 | +0.519 ± 0.060 | +0.62, +0.47, +0.56, +0.48, +0.47 | 0.094 |
| rf | +0.831 ± 0.008 | +0.520 ± 0.039 | +0.56, +0.47, +0.57, +0.50, +0.50 | 0.092 |
| tree | +0.575 ± 0.024 | +0.486 ± 0.042 | +0.54, +0.45, +0.48, +0.53, +0.43 | 0.092 |

### Feature importances (rf)

| feature | importance |
|---|---:|
| fp_jacobian_erf_decay_slope | 0.291 |
| fp_jacobian_erf_density | 0.217 |
| wikitext_perplexity | 0.120 |
| validation_loss_ratio | 0.081 |
| blimp_overall_accuracy | 0.067 |
| ar_validation_held_pair_acc | 0.058 |
| induction_screening_auc | 0.052 |
| binding_screening_auc | 0.041 |
| ar_legacy_auc | 0.039 |
| hellaswag_acc | 0.035 |

## ar_curriculum_s0_retention — model comparison (5-fold CV)

| model | Spearman train (mean ± std) | Spearman test (mean ± std) | per-fold test | MAE test |
|---|---|---|---|---:|
| gbm | +0.699 ± 0.014 | +0.309 ± 0.092 | +0.43, +0.35, +0.34, +0.26, +0.16 | 0.184 |
| gbm_deep | +0.845 ± 0.006 | +0.299 ± 0.083 | +0.42, +0.30, +0.35, +0.23, +0.19 | 0.186 |
| rf | +0.700 ± 0.009 | +0.337 ± 0.048 | +0.39, +0.37, +0.37, +0.28, +0.28 | 0.185 |
| tree | +0.417 ± 0.020 | +0.283 ± 0.049 | +0.36, +0.25, +0.33, +0.24, +0.25 | 0.186 |

### Feature importances (rf)

| feature | importance |
|---|---:|
| fp_jacobian_erf_decay_slope | 0.294 |
| wikitext_perplexity | 0.122 |
| fp_jacobian_erf_density | 0.119 |
| ar_validation_held_pair_acc | 0.116 |
| validation_loss_ratio | 0.086 |
| blimp_overall_accuracy | 0.082 |
| induction_screening_auc | 0.050 |
| ar_legacy_auc | 0.047 |
| hellaswag_acc | 0.044 |
| binding_screening_auc | 0.040 |
