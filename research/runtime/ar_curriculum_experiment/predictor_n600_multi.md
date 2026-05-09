# AR curriculum predictor — n600_multi

n_rows_with_curriculum = 600

Multi-model 5-fold CV. Models: gbm (default GBM), gbm_deep (GBM with deeper trees), rf (random forest), tree (single decision tree depth=5).
Decision threshold: best_model Spearman ρ ≥ 0.7 → predictor reliable enough to triage; < 0.7 → run more backfill.

## Per-target best model summary

| target | n | best model | best Spearman test | y mean ± std | verdict |
|---|---:|---|---|---|---|
| ar_curriculum_auc_pair_final | 600 | rf | +0.605 ± 0.056 | 0.296 ± 0.149 | MORE DATA NEEDED |
| ar_curriculum_s0_retention | 600 | rf | +0.414 ± 0.069 | 0.446 ± 0.283 | MORE DATA NEEDED |

## ar_curriculum_auc_pair_final — model comparison (5-fold CV)

| model | Spearman train (mean ± std) | Spearman test (mean ± std) | per-fold test | MAE test |
|---|---|---|---|---:|
| gbm | +0.877 ± 0.007 | +0.589 ± 0.072 | +0.55, +0.69, +0.56, +0.65, +0.49 | 0.091 |
| gbm_deep | +0.970 ± 0.004 | +0.584 ± 0.062 | +0.54, +0.66, +0.54, +0.66, +0.52 | 0.092 |
| rf | +0.869 ± 0.007 | +0.605 ± 0.056 | +0.57, +0.69, +0.59, +0.64, +0.53 | 0.091 |
| tree | +0.645 ± 0.013 | +0.550 ± 0.082 | +0.47, +0.66, +0.47, +0.64, +0.53 | 0.092 |

### Feature importances (rf)

| feature | importance |
|---|---:|
| fp_jacobian_erf_decay_slope | 0.289 |
| fp_jacobian_erf_density | 0.256 |
| wikitext_perplexity | 0.111 |
| blimp_overall_accuracy | 0.068 |
| validation_loss_ratio | 0.063 |
| ar_validation_held_pair_acc | 0.062 |
| induction_screening_auc | 0.048 |
| binding_screening_auc | 0.035 |
| hellaswag_acc | 0.034 |
| ar_legacy_auc | 0.034 |

## ar_curriculum_s0_retention — model comparison (5-fold CV)

| model | Spearman train (mean ± std) | Spearman test (mean ± std) | per-fold test | MAE test |
|---|---|---|---|---:|
| gbm | +0.738 ± 0.002 | +0.408 ± 0.040 | +0.37, +0.48, +0.39, +0.43, +0.38 | 0.189 |
| gbm_deep | +0.868 ± 0.005 | +0.370 ± 0.029 | +0.35, +0.41, +0.34, +0.39, +0.35 | 0.194 |
| rf | +0.725 ± 0.006 | +0.414 ± 0.069 | +0.35, +0.54, +0.41, +0.42, +0.35 | 0.191 |
| tree | +0.505 ± 0.013 | +0.368 ± 0.068 | +0.40, +0.42, +0.42, +0.35, +0.24 | 0.185 |

### Feature importances (rf)

| feature | importance |
|---|---:|
| fp_jacobian_erf_decay_slope | 0.299 |
| fp_jacobian_erf_density | 0.134 |
| wikitext_perplexity | 0.123 |
| ar_validation_held_pair_acc | 0.117 |
| validation_loss_ratio | 0.080 |
| blimp_overall_accuracy | 0.077 |
| ar_legacy_auc | 0.048 |
| induction_screening_auc | 0.043 |
| hellaswag_acc | 0.040 |
| binding_screening_auc | 0.039 |
