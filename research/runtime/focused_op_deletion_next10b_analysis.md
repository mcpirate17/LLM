# Focused Op Deletion Ablation: Second Next-10 Leaderboard Window

Initial pass planned 128 children and recorded 125 results; targeted rerun added 6 rule-level records after patching historical non-S1 filtering; 4 contaminated historical evidence rows were quarantined.

## Data quality
- ablation_s1_missing_required_total: 0
- valid_node_delete_evidence_window: 125
- invalid_node_delete_evidence_window: 4
- valid_child_links_window_all_obs: 125
- valid_evidence_with_failed_or_non_s1_child: 0
- valid_s1_child_rows_missing_required: 0
- valid_rows_null_effect: 0
- duplicate_valid_evidence_child_fingerprint_groups: 0

## Parent summary
- 8d087a16-692: n=16 deletion_hurts=0 deletion_improves_child=12 near_tie=4 avg_effect=-0.0519 metric_wins=72 metric_losses=38
- dc0d8d48-12b: n=15 deletion_hurts=0 deletion_improves_child=15 near_tie=0 avg_effect=-0.1752 metric_wins=59 metric_losses=42
- f15c755a-7b1: n=14 deletion_hurts=1 deletion_improves_child=12 near_tie=1 avg_effect=-0.0463 metric_wins=54 metric_losses=41
- 32ab1458-5d4: n=14 deletion_hurts=0 deletion_improves_child=14 near_tie=0 avg_effect=-0.0869 metric_wins=53 metric_losses=41
- adf5da8e-f2d: n=13 deletion_hurts=0 deletion_improves_child=9 near_tie=4 avg_effect=-0.0442 metric_wins=40 metric_losses=48
- e18d9bb4-457: n=12 deletion_hurts=0 deletion_improves_child=12 near_tie=0 avg_effect=-0.0748 metric_wins=44 metric_losses=38
- d81a6f53-591: n=12 deletion_hurts=12 deletion_improves_child=0 near_tie=0 avg_effect=0.1816 metric_wins=34 metric_losses=49
- 0aabe135-e1a: n=12 deletion_hurts=4 deletion_improves_child=2 near_tie=6 avg_effect=0.0100 metric_wins=49 metric_losses=32
- 6397b342-0a0: n=11 deletion_hurts=11 deletion_improves_child=0 near_tie=0 avg_effect=0.1251 metric_wins=35 metric_losses=39
- a3b532c7-4d8: n=6 deletion_hurts=1 deletion_improves_child=4 near_tie=1 avg_effect=-0.0262 metric_wins=12 metric_losses=29

## Op summary
- rmsnorm: n=28 parents=10 deletion_hurts=6 deletion_improves_child=18 near_tie=4 avg_effect=-0.0230
- add: n=26 parents=10 deletion_hurts=6 deletion_improves_child=17 near_tie=3 avg_effect=-0.0329
- linear_proj: n=11 parents=7 deletion_hurts=4 deletion_improves_child=7 near_tie=0 avg_effect=-0.0066
- conv1d_seq: n=8 parents=7 deletion_hurts=2 deletion_improves_child=5 near_tie=1 avg_effect=-0.0139
- layernorm: n=7 parents=5 deletion_hurts=3 deletion_improves_child=3 near_tie=1 avg_effect=0.0400
- silu: n=6 parents=6 deletion_hurts=0 deletion_improves_child=3 near_tie=3 avg_effect=-0.0430
- selective_scan: n=6 parents=6 deletion_hurts=1 deletion_improves_child=4 near_tie=1 avg_effect=-0.0276
- latent_attention_compressor: n=4 parents=4 deletion_hurts=0 deletion_improves_child=4 near_tie=0 avg_effect=-0.0534
- gelu: n=4 parents=4 deletion_hurts=0 deletion_improves_child=3 near_tie=1 avg_effect=-0.0681
- swiglu_mlp: n=3 parents=3 deletion_hurts=2 deletion_improves_child=1 near_tie=0 avg_effect=0.0759
- ternary_projection: n=2 parents=2 deletion_hurts=0 deletion_improves_child=2 near_tie=0 avg_effect=-0.1705
- softmax_attention: n=2 parents=2 deletion_hurts=2 deletion_improves_child=0 near_tie=0 avg_effect=0.1312
- neg: n=2 parents=2 deletion_hurts=0 deletion_improves_child=2 near_tie=0 avg_effect=-0.1566
- matmul: n=2 parents=2 deletion_hurts=2 deletion_improves_child=0 near_tie=0 avg_effect=0.1613
- local_window_attn: n=2 parents=2 deletion_hurts=1 deletion_improves_child=1 near_tie=0 avg_effect=-0.0226
- adjacent_token_merge: n=2 parents=2 deletion_hurts=0 deletion_improves_child=2 near_tie=0 avg_effect=-0.0796
- token_entropy: n=1 parents=1 deletion_hurts=0 deletion_improves_child=1 near_tie=0 avg_effect=-0.1772
- token_class_proj: n=1 parents=1 deletion_hurts=0 deletion_improves_child=1 near_tie=0 avg_effect=-0.2237
- tanh: n=1 parents=1 deletion_hurts=0 deletion_improves_child=1 near_tie=0 avg_effect=-0.0627
- sin: n=1 parents=1 deletion_hurts=0 deletion_improves_child=1 near_tie=0 avg_effect=-0.0739

## Strongest deletion-improves-child signals
- dc0d8d48-12b 4:ternary_projection effect=-0.3043 child=4c01f5e7-097 metric_wins=5 metric_losses=2
- dc0d8d48-12b 10:add effect=-0.2262 child=a0e9f1a9-75d metric_wins=6 metric_losses=1
- dc0d8d48-12b 2:token_class_proj effect=-0.2237 child=b288138f-05f metric_wins=3 metric_losses=3
- dc0d8d48-12b 11:rmsnorm effect=-0.2188 child=d8d01f6c-2db metric_wins=4 metric_losses=3
- dc0d8d48-12b 5:neg effect=-0.2160 child=907117fa-dd8 metric_wins=5 metric_losses=1
- dc0d8d48-12b 8:rmsnorm effect=-0.2037 child=4232d3b3-120 metric_wins=5 metric_losses=2
- dc0d8d48-12b 6:mul effect=-0.1968 child=af6b5986-66a metric_wins=3 metric_losses=3
- dc0d8d48-12b 16:rmsnorm effect=-0.1905 child=b8c907b6-0b3 metric_wins=4 metric_losses=3
- dc0d8d48-12b 13:gelu effect=-0.1847 child=1151089b-f42 metric_wins=4 metric_losses=2
- 32ab1458-5d4 9:add effect=-0.1792 child=a54ec977-e13 metric_wins=3 metric_losses=4
- dc0d8d48-12b 3:token_entropy effect=-0.1772 child=1d719594-bf1 metric_wins=5 metric_losses=2
- 32ab1458-5d4 3:linear_proj effect=-0.1653 child=ee53c1d4-911 metric_wins=4 metric_losses=3

## Strongest deletion-hurts signals
- d81a6f53-591 7:rmsnorm effect=0.2319 child=345d9bad-830 metric_wins=2 metric_losses=5
- d81a6f53-591 4:linear_proj effect=0.2135 child=e0242994-736 metric_wins=3 metric_losses=3
- d81a6f53-591 6:layernorm effect=0.2067 child=37f3fb1c-c25 metric_wins=2 metric_losses=5
- d81a6f53-591 13:rmsnorm effect=0.2025 child=f71523a8-0b3 metric_wins=3 metric_losses=4
- d81a6f53-591 10:matmul effect=0.1830 child=b062aafe-2ca metric_wins=2 metric_losses=5
- d81a6f53-591 1:layernorm effect=0.1793 child=eac8d3d3-128 metric_wins=3 metric_losses=4
- 6397b342-0a0 12:add effect=0.1770 child=69ea56f3-eeb metric_wins=1 metric_losses=5
- d81a6f53-591 14:swiglu_mlp effect=0.1724 child=3d057bcd-7b2 metric_wins=2 metric_losses=5
- d81a6f53-591 11:linear_proj effect=0.1708 child=4bade62b-e26 metric_wins=2 metric_losses=5
- d81a6f53-591 2:softmax_attention effect=0.1560 child=26ea8fd1-778 metric_wins=5 metric_losses=2
- d81a6f53-591 12:add effect=0.1560 child=e656ef3f-3bc metric_wins=3 metric_losses=4
- d81a6f53-591 16:rmsnorm effect=0.1542 child=7268353f-b2a metric_wins=3 metric_losses=4

## Interpretation
- This second window is highly polarized: parents 8d087a16, dc0d8d48, 32ab1458, e18d9bb4, and f15c755a have many deletion-improves-child signals, while d81a6f53 and 6397b342 are strongly deletion-sensitive/protective.
- Because both protective and dead-weight behavior appears inside the same common ops, these data support parent/slot-local priors more than global op weights.
- The historical-reuse bug would have overstated some add-node effects; those rows are quarantined and the rerun supplies fresh S1-complete evidence.
