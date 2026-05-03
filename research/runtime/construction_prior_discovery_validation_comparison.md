# Construction Prior Discovery Validation

## Summary
- prior_off: exp=d8e49434-370 generated=11 rows=4 S0=4 S0.5=4 S1=0 missing_s1_metrics=0 best_loss=None
- prior_on: exp=bc1265db-2b2 generated=11 rows=10 S0=10 S0.5=10 S1=2 missing_s1_metrics=0 best_loss=0.5104817686215752
- prior_on_small: exp=4c67d799-f57 generated=6 rows=2 S0=2 S0.5=2 S1=0 missing_s1_metrics=0 best_loss=None

## Funnel Counts
- prior_off: raw=16 post_dedup=11 considered=11 structural_drops=7 stage1_queued=4 stage1_survived=0
- prior_on: raw=16 post_dedup=11 considered=11 structural_drops=1 stage1_queued=10 stage1_survived=2

## S1 Survivors
- a9a5d678-535: loss_ratio=0.5105 ppl=152.6 hellaswag=0.260 blimp=0.523 induction=0.002 binding=0.050 ar=0.003
- c1a2a729-1a8: loss_ratio=0.5825 ppl=1223.6 hellaswag=0.280 blimp=0.514 induction=0.000 binding=0.004 ar=0.006

## Interpretation
- The prior wiring is active: generated persisted graphs in the prior-on runs carry slot-motif multiplier metadata, while the prior-off run does not.
- On this small matched run, prior-on improved the screening funnel: more persisted S0/S0.5 rows and 2 full-metric S1 survivors versus 0 with construction priors disabled.
- This is promising but not conclusive. The sample is still small and should be treated as a canary, not proof of global uplift.
- No S1-passed row in these validation runs is missing required S1 metrics.
