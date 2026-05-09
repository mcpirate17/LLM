# AR curriculum predictor — n400_cv

n_rows_with_curriculum = 405

Lightweight GradientBoostingRegressor (sklearn) on upstream cheap features.
Decision threshold: Spearman ρ ≥ 0.7 on test → predictor reliable enough to triage; < 0.7 → run more backfill.

## Per-target results (5-fold CV)

| target | n | folds | Spearman train (mean ± std) | Spearman test (mean ± std) | per-fold test | MAE test | y mean ± std | verdict |
|---|---:|---:|---|---|---|---:|---|---|
| ar_curriculum_auc_pair_final | 405 | 5 | +0.923 ± 0.006 | +0.656 ± 0.077 | +0.58, +0.56, +0.65, +0.75, +0.74 | 0.089 | 0.315 ± 0.158 | MORE DATA NEEDED |
| ar_curriculum_s0_retention | 405 | 5 | +0.830 ± 0.011 | +0.475 ± 0.087 | +0.41, +0.43, +0.42, +0.65, +0.46 | 0.198 | 0.481 ± 0.302 | MORE DATA NEEDED |

### Feature importances — predicting ar_curriculum_auc_pair_final

| feature | importance |
|---|---:|
| fp_jacobian_erf_decay_slope | 0.310 |
| fp_jacobian_erf_density | 0.284 |
| wikitext_perplexity | 0.111 |
| induction_screening_auc | 0.070 |
| validation_loss_ratio | 0.059 |
| blimp_overall_accuracy | 0.056 |
| binding_screening_auc | 0.042 |
| ar_legacy_auc | 0.030 |
| hellaswag_acc | 0.022 |
| ar_validation_held_pair_acc | 0.016 |

### Feature importances — predicting ar_curriculum_s0_retention

| feature | importance |
|---|---:|
| fp_jacobian_erf_decay_slope | 0.374 |
| wikitext_perplexity | 0.122 |
| fp_jacobian_erf_density | 0.121 |
| validation_loss_ratio | 0.088 |
| blimp_overall_accuracy | 0.069 |
| induction_screening_auc | 0.054 |
| binding_screening_auc | 0.050 |
| ar_validation_held_pair_acc | 0.047 |
| ar_legacy_auc | 0.039 |
| hellaswag_acc | 0.036 |
