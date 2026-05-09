# AR curriculum predictor — n400

n_rows_with_curriculum = 400

Lightweight GradientBoostingRegressor (sklearn) on upstream cheap features.
Decision threshold: Spearman ρ ≥ 0.7 on test → predictor reliable enough to triage; < 0.7 → run more backfill.

## Per-target results

| target | n total | n train | n test | Spearman train | Spearman test | MAE test | y_test mean ± std | verdict |
|---|---:|---:|---:|---:|---:|---:|---|---|
| ar_curriculum_auc_pair_final | 400 | 320 | 80 | +0.938 | +0.625 | 0.095 | 0.306 ± 0.157 | MORE DATA NEEDED |
| ar_curriculum_s0_retention | 400 | 320 | 80 | +0.839 | +0.448 | 0.193 | 0.487 ± 0.300 | MORE DATA NEEDED |

### Feature importances — predicting ar_curriculum_auc_pair_final

| feature | importance |
|---|---:|
| fp_jacobian_erf_decay_slope | 0.352 |
| fp_jacobian_erf_density | 0.294 |
| induction_screening_auc | 0.083 |
| wikitext_perplexity | 0.073 |
| blimp_overall_accuracy | 0.056 |
| validation_loss_ratio | 0.050 |
| ar_legacy_auc | 0.028 |
| hellaswag_acc | 0.026 |
| binding_screening_auc | 0.026 |
| ar_validation_held_pair_acc | 0.011 |

### Feature importances — predicting ar_curriculum_s0_retention

| feature | importance |
|---|---:|
| fp_jacobian_erf_decay_slope | 0.439 |
| fp_jacobian_erf_density | 0.118 |
| wikitext_perplexity | 0.097 |
| validation_loss_ratio | 0.094 |
| blimp_overall_accuracy | 0.074 |
| induction_screening_auc | 0.048 |
| ar_validation_held_pair_acc | 0.047 |
| hellaswag_acc | 0.041 |
| ar_legacy_auc | 0.025 |
| binding_screening_auc | 0.019 |
