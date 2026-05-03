# Ablation Intelligence Summary: Next 20 Leaderboard Graphs

## Data set
- Valid deletion evidence: 254
- Valid smart replacement evidence: 176
- Quarantined invalid evidence: 5
- Smart replacement child links: 295 + 286
- Missing required S1 metrics in accepted S1 ablation children: 0 in all four run audits

## Parent-level signal
- 9f583a0c-33b: strong_loose_or_inefficient | deletion improves/hurts/near=18/0/0 avg=-0.1460; replacement/original/near=10/0/0 avg=-0.1507; metric wins/losses=29/39
- dc0d8d48-12b: strong_loose_or_inefficient | deletion improves/hurts/near=15/0/0 avg=-0.1752; replacement/original/near=11/0/0 avg=-0.1852; metric wins/losses=101/71
- 32ab1458-5d4: strong_loose_or_inefficient | deletion improves/hurts/near=14/0/0 avg=-0.0869; replacement/original/near=9/0/0 avg=-0.0886; metric wins/losses=91/63
- e18d9bb4-457: strong_loose_or_inefficient | deletion improves/hurts/near=12/0/0 avg=-0.0748; replacement/original/near=11/0/0 avg=-0.0987; metric wins/losses=78/77
- 8e7b7eb9-913: strong_loose_or_inefficient | deletion improves/hurts/near=14/1/0 avg=-0.0893; replacement/original/near=8/0/0 avg=-0.1379; metric wins/losses=32/19
- 8d087a16-692: strong_loose_or_inefficient | deletion improves/hurts/near=12/0/4 avg=-0.0519; replacement/original/near=9/0/0 avg=-0.0700; metric wins/losses=107/63
- a311fc5d-12c: strong_loose_or_inefficient | deletion improves/hurts/near=13/1/1 avg=-0.1021; replacement/original/near=8/0/0 avg=-0.1590; metric wins/losses=26/29
- f15c755a-7b1: strong_loose_or_inefficient | deletion improves/hurts/near=12/1/1 avg=-0.0463; replacement/original/near=9/0/0 avg=-0.0624; metric wins/losses=87/70
- ad096d0b-286: strong_packed_or_protective | deletion improves/hurts/near=1/13/1 avg=0.0937; replacement/original/near=0/5/2 avg=0.0325; metric wins/losses=13/36
- 296cb201-3b3: strong_packed_or_protective | deletion improves/hurts/near=0/4/0 avg=0.1468; replacement/original/near=0/3/0 avg=0.1549; metric wins/losses=7/13
- 6397b342-0a0: strong_packed_or_protective | deletion improves/hurts/near=0/11/0 avg=0.1251; replacement/original/near=0/8/0 avg=0.1286; metric wins/losses=58/71
- d81a6f53-591: strong_packed_or_protective | deletion improves/hurts/near=0/12/0 avg=0.1816; replacement/original/near=0/8/0 avg=0.1486; metric wins/losses=54/85
- ebdd9a5c-8a3: strong_packed_or_protective | deletion improves/hurts/near=0/15/0 avg=0.1585; replacement/original/near=0/8/0 avg=0.1321; metric wins/losses=7/44
- adf5da8e-f2d: weak_loose_or_local_cleanup | deletion improves/hurts/near=9/0/4 avg=-0.0442; replacement/original/near=8/0/0 avg=-0.0495; metric wins/losses=65/78
- d49297a5-3bd: weak_loose_or_local_cleanup | deletion improves/hurts/near=5/3/5 avg=0.0145; replacement/original/near=9/2/2 avg=-0.0425; metric wins/losses=6/83
- 903157e5-219: weak_loose_or_local_cleanup | deletion improves/hurts/near=5/2/3 avg=-0.0318; replacement/original/near=8/0/2 avg=-0.0622; metric wins/losses=19/49
- a3b532c7-4d8: weak_loose_or_local_cleanup | deletion improves/hurts/near=4/1/1 avg=-0.0262; replacement/original/near=9/0/0 avg=-0.1261; metric wins/losses=30/71
- 8b9b42ed-c58: weak_loose_or_local_cleanup | deletion improves/hurts/near=5/1/5 avg=-0.0096; replacement/original/near=6/2/0 avg=-0.0172; metric wins/losses=9/45
- 0aabe135-e1a: mixed_context | deletion improves/hurts/near=2/4/6 avg=0.0100; replacement/original/near=7/1/3 avg=-0.0188; metric wins/losses=102/53
- ce318e1c-43d: mixed_context | deletion improves/hurts/near=3/7/3 avg=0.0021; replacement/original/near=4/1/3 avg=-0.0232; metric wins/losses=28/28

## Smart replacement classes
- projection: n=47 replacement_better=33 original_better=11 near=3 avg_effect=-0.0474 metric_wins=139 metric_losses=177
- sequence_mixer: n=38 replacement_better=27 original_better=10 near=1 avg_effect=-0.0249 metric_wins=111 metric_losses=149
- activation: n=29 replacement_better=22 original_better=5 near=2 avg_effect=-0.0669 metric_wins=85 metric_losses=111
- normalization: n=31 replacement_better=22 original_better=6 near=3 avg_effect=-0.0417 metric_wins=86 metric_losses=121
- binary_merge: n=27 replacement_better=19 original_better=5 near=3 avg_effect=-0.0620 metric_wins=68 metric_losses=113
- routing_signal: n=4 replacement_better=3 original_better=1 near=0 avg_effect=-0.0458 metric_wins=8 metric_losses=19

## Deletion op aggregates
- rmsnorm: n=50 deletion_improves_child=30 deletion_hurts=13 near=7 avg_effect=-0.0329
- add: n=55 deletion_improves_child=30 deletion_hurts=17 near=8 avg_effect=-0.0010
- latent_attention_compressor: n=6 deletion_improves_child=5 deletion_hurts=1 near=0 avg_effect=-0.0358
- gelu: n=7 deletion_improves_child=5 deletion_hurts=1 near=1 avg_effect=-0.0633
- nm_sparse_linear: n=6 deletion_improves_child=5 deletion_hurts=1 near=0 avg_effect=-0.1217
- adjacent_token_merge: n=8 deletion_improves_child=5 deletion_hurts=2 near=1 avg_effect=-0.0007
- selective_scan: n=6 deletion_improves_child=4 deletion_hurts=1 near=1 avg_effect=-0.0276
- ternary_projection: n=8 deletion_improves_child=5 deletion_hurts=2 near=1 avg_effect=-0.0425
- linear_proj: n=18 deletion_improves_child=9 deletion_hurts=7 near=2 avg_effect=-0.0076
- silu: n=7 deletion_improves_child=3 deletion_hurts=1 near=3 avg_effect=-0.0339
- mul: n=5 deletion_improves_child=3 deletion_hurts=1 near=1 avg_effect=-0.0609
- sin: n=2 deletion_improves_child=2 deletion_hurts=0 near=0 avg_effect=-0.0619
- token_entropy: n=4 deletion_improves_child=3 deletion_hurts=1 near=0 avg_effect=-0.0838
- spectral_filter: n=2 deletion_improves_child=2 deletion_hurts=0 near=0 avg_effect=-0.1011
- token_class_proj: n=4 deletion_improves_child=3 deletion_hurts=1 near=0 avg_effect=-0.1143
- neg: n=2 deletion_improves_child=2 deletion_hurts=0 near=0 avg_effect=-0.1566
- layernorm: n=17 deletion_improves_child=7 deletion_hurts=6 near=4 avg_effect=0.0202
- swiglu_mlp: n=7 deletion_improves_child=4 deletion_hurts=3 near=0 avg_effect=-0.0251
- sigmoid: n=1 deletion_improves_child=1 deletion_hurts=0 near=0 avg_effect=-0.0272
- feature_sparsity: n=1 deletion_improves_child=1 deletion_hurts=0 near=0 avg_effect=-0.0277

## Conclusions
- This is enough evidence to start using ablation-derived priors, but not as global op weights. The same ops and component classes flip sign across parents, so graph discovery should use parent-shape, slot, predecessor/successor, and metric-context features.
- Strong loose/inefficient parents by both deletion and replacement evidence: 9f583a0c, dc0d8d48, 32ab1458, e18d9bb4, 8e7b7eb9, 8d087a16, a311fc5d, f15c755a. Strong protective parents: ad096d0b, 296cb201, 6397b342, d81a6f53, ebdd9a5c. Weak or mixed parents should not be used for hard rules: adf5da8e, d49297a5, 903157e5, a3b532c7, 8b9b42ed, 0aabe135, ce318e1c.
- Loss deltas and full metric deltas often disagree. Discovery should store/use the full seven-metric delta vector and penalize candidates that improve loss while damaging most task probes.
- The immediate platform upgrade should be an ablation-evidence aggregator that emits local edit priors for node slots/templates, plus a generator hook that checks proposed edits against those priors before scheduling expensive runs.
