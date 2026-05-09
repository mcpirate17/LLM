# AR curriculum predictor — n200

n_rows_with_curriculum = 202

Lightweight GradientBoostingRegressor (sklearn) on upstream cheap features.
Decision threshold: Spearman ρ ≥ 0.7 on test → predictor reliable enough to triage; < 0.7 → run more backfill.

## Per-target results

| target | n total | n train | n test | Spearman train | Spearman test | MAE test | y_test mean ± std | verdict |
|---|---:|---:|---:|---:|---:|---:|---|---|
| ar_curriculum_auc_pair_final | — | — | — | — | — | — | — | too_few_samples |
| ar_curriculum_s0_retention | — | — | — | — | — | — | — | too_few_samples |
