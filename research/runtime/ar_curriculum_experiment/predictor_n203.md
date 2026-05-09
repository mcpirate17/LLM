# AR curriculum predictor — n203

n_rows_with_curriculum = 205

Lightweight GradientBoostingRegressor (sklearn) on upstream cheap features.
Decision threshold: Spearman ρ ≥ 0.7 on test → predictor reliable enough to triage; < 0.7 → run more backfill.

## Per-target results

| target | n total | n train | n test | Spearman train | Spearman test | MAE test | y_test mean ± std | verdict |
|---|---:|---:|---:|---:|---:|---:|---|---|
| ar_curriculum_auc_pair_final | 205 | 164 | 41 | +0.957 | +0.735 | 0.076 | 0.347 ± 0.170 | RELIABLE |
| ar_curriculum_s0_retention | 205 | 164 | 41 | +0.884 | +0.582 | 0.198 | 0.570 ± 0.339 | MORE DATA NEEDED |

### Feature importances — predicting ar_curriculum_auc_pair_final

| feature | importance |
|---|---:|
| fp_jacobian_erf_decay_slope | 0.449 |
| fp_jacobian_erf_density | 0.285 |
| induction_screening_auc | 0.069 |
| wikitext_perplexity | 0.043 |
| validation_loss_ratio | 0.043 |
| blimp_overall_accuracy | 0.040 |
| binding_screening_auc | 0.027 |
| ar_validation_held_pair_acc | 0.017 |
| ar_legacy_auc | 0.016 |
| hellaswag_acc | 0.012 |

### Feature importances — predicting ar_curriculum_s0_retention

| feature | importance |
|---|---:|
| fp_jacobian_erf_decay_slope | 0.486 |
| fp_jacobian_erf_density | 0.138 |
| wikitext_perplexity | 0.074 |
| induction_screening_auc | 0.069 |
| blimp_overall_accuracy | 0.057 |
| validation_loss_ratio | 0.044 |
| ar_legacy_auc | 0.041 |
| binding_screening_auc | 0.038 |
| ar_validation_held_pair_acc | 0.032 |
| hellaswag_acc | 0.021 |
