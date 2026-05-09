# AR curriculum predictor — n600

n_rows_with_curriculum = 600

Lightweight GradientBoostingRegressor (sklearn) on upstream cheap features.
Decision threshold: Spearman ρ ≥ 0.7 on test → predictor reliable enough to triage; < 0.7 → run more backfill.

## Per-target results (5-fold CV)

| target | n | folds | Spearman train (mean ± std) | Spearman test (mean ± std) | per-fold test | MAE test | y mean ± std | verdict |
|---|---:|---:|---|---|---|---:|---|---|
| ar_curriculum_auc_pair_final | 600 | 5 | +0.877 ± 0.007 | +0.589 ± 0.072 | +0.55, +0.69, +0.56, +0.65, +0.49 | 0.091 | 0.296 ± 0.149 | MORE DATA NEEDED |
| ar_curriculum_s0_retention | 600 | 5 | +0.738 ± 0.002 | +0.408 ± 0.040 | +0.37, +0.48, +0.39, +0.43, +0.38 | 0.189 | 0.446 ± 0.283 | MORE DATA NEEDED |

### Feature importances — predicting ar_curriculum_auc_pair_final

| feature | importance |
|---|---:|
| fp_jacobian_erf_decay_slope | 0.356 |
| fp_jacobian_erf_density | 0.229 |
| wikitext_perplexity | 0.128 |
| blimp_overall_accuracy | 0.067 |
| validation_loss_ratio | 0.064 |
| induction_screening_auc | 0.049 |
| binding_screening_auc | 0.040 |
| hellaswag_acc | 0.029 |
| ar_legacy_auc | 0.022 |
| ar_validation_held_pair_acc | 0.018 |

### Feature importances — predicting ar_curriculum_s0_retention

| feature | importance |
|---|---:|
| fp_jacobian_erf_decay_slope | 0.367 |
| wikitext_perplexity | 0.132 |
| fp_jacobian_erf_density | 0.112 |
| validation_loss_ratio | 0.096 |
| blimp_overall_accuracy | 0.079 |
| ar_validation_held_pair_acc | 0.070 |
| binding_screening_auc | 0.048 |
| induction_screening_auc | 0.035 |
| ar_legacy_auc | 0.033 |
| hellaswag_acc | 0.029 |
