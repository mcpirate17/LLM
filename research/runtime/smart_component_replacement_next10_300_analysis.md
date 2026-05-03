# Smart Component Replacement Ablation: Next 10 Leaderboard Graphs

Run window: 1777668510.680 to 1777672863.531; completed suites: 83; planned children: 300; pre-run rejected variants: 91.

## Data quality
- ablation_s1_missing_required_total: 0
- component_replace_evidence_current: 83
- component_replace_child_links_current: 295
- duplicate_evidence_child_links_current: 0
- duplicate_child_fingerprints_current: 0
- valid_rows_null_effect_current: 0
- failed_child_links_current: 0
- non_s1_child_links_current: 0
- current_s1_child_rows_missing_required: 0

## Parent summary
- d49297a5-3bd: n=13 replacement_better=9 original_better=2 near_tie=2 avg_effect=-0.0425 metric_wins=6 metric_losses=83
- 9f583a0c-33b: n=10 replacement_better=10 original_better=0 near_tie=0 avg_effect=-0.1507 metric_wins=29 metric_losses=39
- 903157e5-219: n=10 replacement_better=8 original_better=0 near_tie=2 avg_effect=-0.0622 metric_wins=19 metric_losses=49
- ebdd9a5c-8a3: n=8 replacement_better=0 original_better=8 near_tie=0 avg_effect=0.1321 metric_wins=7 metric_losses=44
- ce318e1c-43d: n=8 replacement_better=4 original_better=1 near_tie=3 avg_effect=-0.0232 metric_wins=28 metric_losses=28
- a311fc5d-12c: n=8 replacement_better=8 original_better=0 near_tie=0 avg_effect=-0.1590 metric_wins=26 metric_losses=29
- 8e7b7eb9-913: n=8 replacement_better=8 original_better=0 near_tie=0 avg_effect=-0.1379 metric_wins=32 metric_losses=19
- 8b9b42ed-c58: n=8 replacement_better=6 original_better=2 near_tie=0 avg_effect=-0.0172 metric_wins=9 metric_losses=45
- ad096d0b-286: n=7 replacement_better=0 original_better=5 near_tie=2 avg_effect=0.0325 metric_wins=13 metric_losses=36
- 296cb201-3b3: n=3 replacement_better=0 original_better=3 near_tie=0 avg_effect=0.1549 metric_wins=7 metric_losses=13

## Component class summary
- projection: n=28 parents=9 replacement_better=20 original_better=5 near_tie=3 avg_effect=-0.0648 metric_wins=72 metric_losses=116
- activation: n=15 parents=8 replacement_better=10 original_better=3 near_tie=2 avg_effect=-0.0747 metric_wins=33 metric_losses=67
- sequence_mixer: n=14 parents=8 replacement_better=6 original_better=8 near_tie=0 avg_effect=0.0307 metric_wins=24 metric_losses=72
- normalization: n=14 parents=9 replacement_better=10 original_better=1 near_tie=3 avg_effect=-0.0558 metric_wins=26 metric_losses=68
- binary_merge: n=9 parents=4 replacement_better=5 original_better=3 near_tie=1 avg_effect=-0.0226 metric_wins=15 metric_losses=47
- routing_signal: n=3 parents=3 replacement_better=2 original_better=1 near_tie=0 avg_effect=-0.0483 metric_wins=6 metric_losses=15

## Strongest replacement wins
- a311fc5d-12c projection:2:token_class_proj effect=-0.2231 best_child=8b6e2be6-755 metric_wins=4 metric_losses=3
- a311fc5d-12c activation:5:gelu effect=-0.1979 best_child=e6cfcb6b-d9f metric_wins=4 metric_losses=3
- 9f583a0c-33b projection:3:linear_proj effect=-0.1955 best_child=6598fc28-e32 metric_wins=5 metric_losses=2
- a311fc5d-12c normalization:1:layernorm effect=-0.1925 best_child=912b6af2-b19 metric_wins=3 metric_losses=3
- 9f583a0c-33b routing_signal:6:adaptive_rank_gate effect=-0.1920 best_child=466a70e9-8a7 metric_wins=1 metric_losses=6
- 9f583a0c-33b projection:7:linear_proj effect=-0.1840 best_child=9d3c3741-324 metric_wins=2 metric_losses=4
- a311fc5d-12c projection:4:ternary_projection effect=-0.1780 best_child=5c19a58d-0b3 metric_wins=4 metric_losses=3
- 8e7b7eb9-913 activation:5:gelu effect=-0.1727 best_child=49a815ed-225 metric_wins=5 metric_losses=2
- 8e7b7eb9-913 projection:14:nm_sparse_linear effect=-0.1707 best_child=d70251c5-3d1 metric_wins=5 metric_losses=2
- 8e7b7eb9-913 projection:2:token_class_proj effect=-0.1693 best_child=d5b18387-5d6 metric_wins=4 metric_losses=2
- 9f583a0c-33b activation:14:gelu effect=-0.1692 best_child=0296b023-47c metric_wins=4 metric_losses=3
- 8e7b7eb9-913 activation:13:silu effect=-0.1621 best_child=eb327097-1b3 metric_wins=3 metric_losses=2

## Strongest original-op wins
- ebdd9a5c-8a3 sequence_mixer:10:adjacent_token_merge effect=0.2357 best_child=5b11e23b-588 metric_wins=1 metric_losses=5
- 296cb201-3b3 routing_signal:5:padic_gate effect=0.1606 best_child=d62894c1-e6a metric_wins=2 metric_losses=5
- 296cb201-3b3 sequence_mixer:7:conv1d_seq effect=0.1586 best_child=8142fc21-672 metric_wins=2 metric_losses=5
- ebdd9a5c-8a3 sequence_mixer:13:conv1d_seq effect=0.1493 best_child=c80ced90-c93 metric_wins=1 metric_losses=6
- 296cb201-3b3 binary_merge:12:add effect=0.1456 best_child=ddf28577-2e0 metric_wins=3 metric_losses=3
- ebdd9a5c-8a3 activation:14:gelu effect=0.1375 best_child=ca43b961-c33 metric_wins=1 metric_losses=6
- ebdd9a5c-8a3 projection:6:ternary_projection effect=0.1353 best_child=36b3fe9a-e47 metric_wins=1 metric_losses=5
- ebdd9a5c-8a3 activation:7:relu effect=0.1111 best_child=a419bd67-ac2 metric_wins=1 metric_losses=5
- ebdd9a5c-8a3 normalization:1:layernorm effect=0.1006 best_child=6dbc5aa7-54a metric_wins=0 metric_losses=6
- ebdd9a5c-8a3 projection:3:linear_proj effect=0.0985 best_child=0440a80b-17f metric_wins=1 metric_losses=5
- 8b9b42ed-c58 sequence_mixer:13:spectral_filter effect=0.0898 best_child=d9792284-207 metric_wins=2 metric_losses=5
- ebdd9a5c-8a3 sequence_mixer:2:latent_attention_compressor effect=0.0890 best_child=cfd5b507-bec metric_wins=1 metric_losses=6

## Interpretation
- The DB audit is clean for this run: no missing required S1 metrics, no duplicate evidence-child links, no duplicate child fingerprints inside a rule, no failed child links, and no valid evidence with null effect.
- Replacement behavior is strongly parent-dependent. Some parents that deletion marked as packed/protective also reject most replacements; others accept many replacements and likely contain inefficient local choices rather than globally bad component classes.
- Metric agreement is mixed enough that slot/template priors should not be boosted or downweighted globally from loss alone. Treat high-confidence per-parent rule keys as local edit priors and use class aggregates only as weak evidence.
