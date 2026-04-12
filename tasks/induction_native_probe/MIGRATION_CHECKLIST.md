# Induction Migration Checklist

## Objective

Make `native_pool_64` at `500` train steps the canonical induction metric for:

- future training/investigation/backfill runs
- notebook persistence
- leaderboard rescoring
- ML/export analysis

## Canonical Metric

- metric version: `native_pool_64_v1`
- speed mode: `native_pool_64`
- train steps: `500`
- eval examples: `100`
- batch size: `16`
- pool size: `64`
- gaps: `4,8,16,32,64`

## Checklist

1. Add canonical storage.
- add `induction_metrics_v2`
- add `induction_metrics_archive`
- add induction metadata columns to `program_results`

2. Archive old induction state.
- snapshot existing `program_results` induction fields
- snapshot existing `leaderboard.induction_auc`

3. Load canonical backfill.
- import `tasks/induction_native_probe/induction_auc_results.csv`
- choose one canonical row per fingerprint
- prefer:
  - `stage1_passed_all500`
  - `binding_only_s1_all500`
  - `public_reference_pool64`
  - legacy cohorts

4. Overwrite compatibility fields.
- update `program_results.induction_*`
- update `leaderboard.induction_auc`

5. Cut over runtime code.
- screening path
- investigation path
- validation path
- all binding/induction backfill scripts

6. Bulk rescore.
- archive old scores through existing leaderboard fields
- rerun centralized composite scoring

7. Export ML dataset.
- one row per fingerprint
- include graph json, graph features, canonical induction labels, per-gap metrics

8. Run trend analysis.
- learners vs non-learners
- template/template-type enrichment
- slot/op/component enrichment
- predictive models for induction capability
