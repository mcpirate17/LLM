# Smart Component Replacement Ablation Analysis

Expanded run: 31/31 suites, 181 planned children.
Log: `research/runtime_events/smart_component_replacement_expanded_20260501_141318.log`

## Data Integrity
- Final metric audit: `{'574271ca-f37': {'global_s1_missing_core': 0, 'global_s1_missing_extended': 0, 'parent_s1_missing_core': 0, 'parent_s1_missing_extended': 0}, 'ec7025d7-338': {'global_s1_missing_core': 0, 'global_s1_missing_extended': 0, 'parent_s1_missing_core': 0, 'parent_s1_missing_extended': 0}}`
- Expanded unique child observations: 181
- Expanded S1 passed/non-S1: 175/6
- Cumulative component replacement children: 281

## Expanded By Class
- activation: children=1 s1=1 non_s1=0 avg_loss_delta=0.037506 avg_induction=0.586 avg_binding_composite=0.1779
- binary_merge: children=78 s1=73 non_s1=5 avg_loss_delta=-0.000818 avg_induction=0.39463 avg_binding_composite=0.127948
- normalization: children=11 s1=11 non_s1=0 avg_loss_delta=0.003111 avg_induction=0.374364 avg_binding_composite=0.125809
- projection: children=41 s1=41 non_s1=0 avg_loss_delta=-0.004002 avg_induction=0.496341 avg_binding_composite=0.162449
- routing_signal: children=3 s1=3 non_s1=0 avg_loss_delta=0.013078 avg_induction=0.839333 avg_binding_composite=0.2557
- scalar_signal: children=1 s1=1 non_s1=0 avg_loss_delta=0.00874 avg_induction=0.028 avg_binding_composite=0.0094
- sequence_mixer: children=46 s1=45 non_s1=1 avg_loss_delta=-0.003749 avg_induction=0.438533 avg_binding_composite=0.137622

## Top Expanded Loss Improvements (S1)
- 574271ca-f37 binary_merge:13:add -> sub: loss_delta=-0.159109 wiki_ppl=120.58 hs=0.22 blimp=0.5048 induction=0.002 binding_c=0.0469
- 574271ca-f37 binary_merge:11:matmul -> geometric_product: loss_delta=-0.113887 wiki_ppl=143.75 hs=0.18 blimp=0.5033 induction=0.006 binding_c=0.0349
- ec7025d7-338 binary_merge:20:add -> dual_compression_blend: loss_delta=-0.111317 wiki_ppl=844.69 hs=0.24 blimp=0.5164 induction=0.01 binding_c=0.0059
- ec7025d7-338 binary_merge:21:add -> difficulty_blend_3way: loss_delta=-0.095572 wiki_ppl=783.71 hs=0.22 blimp=0.5173 induction=0.0 binding_c=0.002
- 574271ca-f37 binary_merge:3:add -> minimum: loss_delta=-0.086974 wiki_ppl=192.11 hs=0.16 blimp=0.5113 induction=0.014 binding_c=0.043
- 574271ca-f37 sequence_mixer:2:adjacent_token_merge -> softmax_attention: loss_delta=-0.079763 wiki_ppl=894.21 hs=0.26 blimp=0.5182 induction=0.012 binding_c=0.006
- 574271ca-f37 binary_merge:13:add -> dual_compression_blend: loss_delta=-0.07798 wiki_ppl=686.41 hs=0.22 blimp=0.523 induction=0.01 binding_c=0.005
- 574271ca-f37 sequence_mixer:2:adjacent_token_merge -> state_space: loss_delta=-0.077339 wiki_ppl=647.6 hs=0.18 blimp=0.5412 induction=0.006 binding_c=0.0043
- ec7025d7-338 sequence_mixer:3:latent_attention_compressor -> gated_linear_attention: loss_delta=-0.077183 wiki_ppl=272.94 hs=0.26 blimp=0.5099 induction=0.016 binding_c=0.0329
- ec7025d7-338 sequence_mixer:19:spectral_filter -> gated_linear_attention: loss_delta=-0.076444 wiki_ppl=223.12 hs=0.34 blimp=0.5033 induction=0.658 binding_c=0.2344

## Top Expanded Binding Composite (S1)
- ec7025d7-338 routing_signal:9:token_type_classifier -> feature_sparsity: binding_c=0.3046 loss_delta=0.00532 induction=1.0 wiki_ppl=626.24
- ec7025d7-338 binary_merge:20:add -> tropical_matmul: binding_c=0.3034 loss_delta=0.047444 induction=1.0 wiki_ppl=829.83
- ec7025d7-338 binary_merge:7:add -> tropical_add: binding_c=0.302 loss_delta=0.050098 induction=0.992 wiki_ppl=644.8
- ec7025d7-338 sequence_mixer:13:softmax_attention -> adjacent_token_merge: binding_c=0.3019 loss_delta=0.039483 induction=1.0 wiki_ppl=798.74
- ec7025d7-338 sequence_mixer:13:softmax_attention -> graph_attention: binding_c=0.3014 loss_delta=-0.024427 induction=0.99 wiki_ppl=654.57
- ec7025d7-338 binary_merge:4:add -> maximum: binding_c=0.3008 loss_delta=0.024868 induction=0.992 wiki_ppl=603.72
- ec7025d7-338 sequence_mixer:19:spectral_filter -> rwkv_time_mixing: binding_c=0.3005 loss_delta=0.01861 induction=0.992 wiki_ppl=571.76
- ec7025d7-338 projection:5:semi_structured_2_4_linear -> bottleneck_proj: binding_c=0.2987 loss_delta=-0.02084 induction=0.982 wiki_ppl=666.17
- ec7025d7-338 binary_merge:4:add -> difficulty_blend_3way: binding_c=0.2986 loss_delta=0.007569 induction=0.984 wiki_ppl=671.51
- ec7025d7-338 normalization:1:layernorm -> hyperbolic_norm: binding_c=0.2982 loss_delta=0.022059 induction=0.98 wiki_ppl=643.68

## Non-S1 Expanded Children
- 574271ca-f37 sequence_mixer:2:adjacent_token_merge -> long_conv_hyena child=1209acc2-710
- 574271ca-f37 binary_merge:7:add -> tropical_matmul child=b7cf9a71-26f
- ec7025d7-338 binary_merge:7:add -> tropical_matmul child=e3762ab9-6f9
- ec7025d7-338 binary_merge:15:mul -> dual_compression_blend child=1f2d2ff4-7f2
- ec7025d7-338 binary_merge:15:mul -> tropical_add child=359d2ea3-b2d
- ec7025d7-338 binary_merge:15:mul -> geometric_product child=f4c704ec-5d1
