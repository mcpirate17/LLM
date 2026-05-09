# AR curriculum predictor — n700

n_rows_with_curriculum = 700

Multi-model 5-fold CV. Models: gbm (default GBM), gbm_deep (GBM with deeper trees), rf (random forest), tree (single decision tree depth=5).
Decision threshold: best_model Spearman ρ ≥ 0.7 → predictor reliable enough to triage; < 0.7 → run more backfill.

## Per-target best model summary

| target | n | best model | best Spearman test | y mean ± std | verdict |
|---|---:|---|---|---|---|
| ar_curriculum_auc_pair_final | 700 | rf | +0.577 ± 0.058 | 0.288 ± 0.147 | MORE DATA NEEDED |
| ar_curriculum_s0_retention | 700 | rf | +0.374 ± 0.052 | 0.433 ± 0.276 | MORE DATA NEEDED |

## ar_curriculum_auc_pair_final — model comparison (5-fold CV)

| model | Spearman train (mean ± std) | Spearman test (mean ± std) | per-fold test | MAE test |
|---|---|---|---|---:|
| gbm | +0.864 ± 0.005 | +0.569 ± 0.065 | +0.49, +0.60, +0.66, +0.59, +0.50 | 0.092 |
| gbm_deep | +0.965 ± 0.002 | +0.570 ± 0.058 | +0.50, +0.60, +0.64, +0.60, +0.50 | 0.092 |
| rf | +0.847 ± 0.003 | +0.577 ± 0.058 | +0.54, +0.62, +0.66, +0.57, +0.49 | 0.092 |
| tree | +0.625 ± 0.012 | +0.493 ± 0.061 | +0.45, +0.59, +0.47, +0.53, +0.42 | 0.095 |

### Feature importances (rf)

| feature | importance |
|---|---:|
| fp_jacobian_erf_decay_slope | 0.300 |
| fp_jacobian_erf_density | 0.230 |
| wikitext_perplexity | 0.103 |
| validation_loss_ratio | 0.074 |
| blimp_overall_accuracy | 0.067 |
| ar_validation_held_pair_acc | 0.064 |
| induction_screening_auc | 0.051 |
| binding_screening_auc | 0.041 |
| ar_legacy_auc | 0.036 |
| hellaswag_acc | 0.035 |

## ar_curriculum_s0_retention — model comparison (5-fold CV)

| model | Spearman train (mean ± std) | Spearman test (mean ± std) | per-fold test | MAE test |
|---|---|---|---|---:|
| gbm | +0.722 ± 0.007 | +0.366 ± 0.051 | +0.33, +0.33, +0.44, +0.42, +0.32 | 0.183 |
| gbm_deep | +0.855 ± 0.006 | +0.349 ± 0.048 | +0.36, +0.29, +0.38, +0.42, +0.30 | 0.187 |
| rf | +0.710 ± 0.009 | +0.374 ± 0.052 | +0.37, +0.40, +0.36, +0.45, +0.29 | 0.185 |
| tree | +0.473 ± 0.017 | +0.341 ± 0.078 | +0.24, +0.42, +0.33, +0.44, +0.27 | 0.189 |

### Feature importances (rf)

| feature | importance |
|---|---:|
| fp_jacobian_erf_decay_slope | 0.300 |
| fp_jacobian_erf_density | 0.127 |
| ar_validation_held_pair_acc | 0.118 |
| wikitext_perplexity | 0.108 |
| blimp_overall_accuracy | 0.081 |
| validation_loss_ratio | 0.079 |
| ar_legacy_auc | 0.051 |
| induction_screening_auc | 0.045 |
| hellaswag_acc | 0.045 |
| binding_screening_auc | 0.045 |
