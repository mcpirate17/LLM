# Smart Component Replacement Ablation: Second Next-10 Leaderboard Window

Run window: 1777675359.170 to 1777679634.590; completed suites: 93; planned children: 300; pre-run rejected variants: 145.

## Data quality
- ablation_s1_missing_required_total: 0
- component_replace_evidence_current: 93
- component_replace_child_links_current: 286
- duplicate_evidence_child_links_current: 0
- duplicate_child_fingerprints_current: 0
- valid_rows_null_effect_current: 0
- failed_child_links_current: 0
- non_s1_child_links_current: 0
- current_s1_child_rows_missing_required: 0
- valid_evidence_with_failed_or_non_s1_child: 0

## Parent summary
- e18d9bb4-457: n=11 replacement_better=11 original_better=0 near_tie=0 avg_effect=-0.0987 metric_wins=34 metric_losses=39
- dc0d8d48-12b: n=11 replacement_better=11 original_better=0 near_tie=0 avg_effect=-0.1852 metric_wins=42 metric_losses=29
- f15c755a-7b1: n=9 replacement_better=9 original_better=0 near_tie=0 avg_effect=-0.0624 metric_wins=33 metric_losses=29
- 8d087a16-692: n=9 replacement_better=9 original_better=0 near_tie=0 avg_effect=-0.0700 metric_wins=35 metric_losses=25
- 32ab1458-5d4: n=9 replacement_better=9 original_better=0 near_tie=0 avg_effect=-0.0886 metric_wins=38 metric_losses=22
- a3b532c7-4d8: n=9 replacement_better=9 original_better=0 near_tie=0 avg_effect=-0.1261 metric_wins=18 metric_losses=42
- adf5da8e-f2d: n=8 replacement_better=8 original_better=0 near_tie=0 avg_effect=-0.0495 metric_wins=25 metric_losses=30
- 0aabe135-e1a: n=11 replacement_better=7 original_better=1 near_tie=3 avg_effect=-0.0188 metric_wins=53 metric_losses=21
- 6397b342-0a0: n=8 replacement_better=0 original_better=8 near_tie=0 avg_effect=0.1286 metric_wins=23 metric_losses=32
- d81a6f53-591: n=8 replacement_better=0 original_better=8 near_tie=0 avg_effect=0.1486 metric_wins=20 metric_losses=36

## Component class summary
- sequence_mixer: n=24 parents=10 replacement_better=21 original_better=2 near_tie=1 avg_effect=-0.0574 metric_wins=87 metric_losses=77
- binary_merge: n=18 parents=7 replacement_better=14 original_better=2 near_tie=2 avg_effect=-0.0817 metric_wins=53 metric_losses=66
- activation: n=14 parents=10 replacement_better=12 original_better=2 near_tie=0 avg_effect=-0.0585 metric_wins=52 metric_losses=44
- projection: n=19 parents=8 replacement_better=13 original_better=6 near_tie=0 avg_effect=-0.0219 metric_wins=67 metric_losses=61
- normalization: n=17 parents=8 replacement_better=12 original_better=5 near_tie=0 avg_effect=-0.0300 metric_wins=60 metric_losses=53
- routing_signal: n=1 parents=1 replacement_better=1 original_better=0 near_tie=0 avg_effect=-0.0381 metric_wins=2 metric_losses=4

## Strongest replacement wins
- dc0d8d48-12b projection:4:ternary_projection effect=-0.2433 best_child=6c16a4e3-2c7 metric_wins=5 metric_losses=2
- dc0d8d48-12b binary_merge:14:add effect=-0.2343 best_child=4216d210-740 metric_wins=3 metric_losses=3
- dc0d8d48-12b projection:2:token_class_proj effect=-0.2309 best_child=f3481695-d08 metric_wins=5 metric_losses=2
- dc0d8d48-12b binary_merge:10:add effect=-0.2154 best_child=8966503e-364 metric_wins=4 metric_losses=2
- dc0d8d48-12b activation:13:gelu effect=-0.2134 best_child=82112537-497 metric_wins=5 metric_losses=2
- a3b532c7-4d8 binary_merge:7:add effect=-0.2066 best_child=6f4b1f36-6d0 metric_wins=2 metric_losses=5
- dc0d8d48-12b normalization:8:rmsnorm effect=-0.2032 best_child=c19a0b49-5b6 metric_wins=5 metric_losses=1
- a3b532c7-4d8 binary_merge:3:add effect=-0.1757 best_child=07f200c7-ebd metric_wins=3 metric_losses=4
- dc0d8d48-12b binary_merge:7:add effect=-0.1738 best_child=55a1a7e1-1b4 metric_wins=3 metric_losses=3
- dc0d8d48-12b activation:12:swiglu_mlp effect=-0.1726 best_child=2c012076-ae2 metric_wins=2 metric_losses=4
- dc0d8d48-12b normalization:1:layernorm effect=-0.1678 best_child=13ad9da7-327 metric_wins=4 metric_losses=2
- a3b532c7-4d8 activation:6:gelu effect=-0.1669 best_child=3061a9bd-be4 metric_wins=1 metric_losses=6

## Strongest original-op wins
- d81a6f53-591 binary_merge:5:add effect=0.1960 best_child=59f0440a-4a2 metric_wins=2 metric_losses=5
- d81a6f53-591 normalization:1:layernorm effect=0.1732 best_child=5abdd32b-d8f metric_wins=3 metric_losses=4
- 6397b342-0a0 normalization:1:layernorm effect=0.1693 best_child=8ecf1569-cb7 metric_wins=1 metric_losses=6
- d81a6f53-591 activation:14:swiglu_mlp effect=0.1692 best_child=97cd0ad2-a0c metric_wins=4 metric_losses=3
- d81a6f53-591 projection:9:linear_proj effect=0.1612 best_child=973bc526-d46 metric_wins=2 metric_losses=5
- 6397b342-0a0 binary_merge:5:add effect=0.1383 best_child=8b440a62-b60 metric_wins=2 metric_losses=5
- 6397b342-0a0 activation:14:swiglu_mlp effect=0.1382 best_child=0cde5059-660 metric_wins=4 metric_losses=3
- d81a6f53-591 projection:4:linear_proj effect=0.1345 best_child=155d16e2-8f8 metric_wins=2 metric_losses=5
- 6397b342-0a0 projection:9:linear_proj effect=0.1279 best_child=6beefe28-8cf metric_wins=4 metric_losses=3
- d81a6f53-591 sequence_mixer:2:softmax_attention effect=0.1263 best_child=106feff8-6c3 metric_wins=3 metric_losses=4
- d81a6f53-591 projection:8:linear_proj effect=0.1252 best_child=1da3d00b-78f metric_wins=2 metric_losses=5
- 6397b342-0a0 sequence_mixer:2:softmax_attention effect=0.1161 best_child=040e4a45-e64 metric_wins=4 metric_losses=3

## Interpretation
- The DB audit is clean for this run: no missing required S1 metrics, no duplicate evidence-child links, no duplicate child fingerprints inside a rule, no failed/non-S1 child links, and no valid evidence with null effect.
- The second window confirms the deletion signal split: loose parents such as dc0d8d48, 32ab1458, 8d087a16, e18d9bb4, and f15c755a accept many replacements; protective parents such as d81a6f53 and 6397b342 mostly reject replacements.
- Component classes remain context-dependent. Projection, normalization, activation, merge, and sequence-mixer edits all flip sign by parent, so the evidence supports parent/slot-local edit priors rather than global op bans or boosts.
- Several loss-improving replacements still lose on a majority of non-loss metrics, which means graph discovery should rank candidates with multi-metric evidence vectors and not a single loss delta.
