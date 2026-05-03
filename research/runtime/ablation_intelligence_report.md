# Ablation Intelligence Report

Evidence rows are first deduped by the write idempotency key, then the latest measured row per parent, phase, and rule key is used for analysis. Verdicts use `program_results` metrics, not raw evidence counts.

## Parent 574271ca-f37

- experiment: `5e26ddbe-e5f`
- duplicate evidence rows ignored: 52
- superseded older evidence rows ignored: 21
- verdict counts: `{"failed_knockout": 11, "metric_conflict_do_not_prune": 15}`

| phase | rule | pass | loss | wiki | hs | blimp | ind | bind | bc | ind_v2 | bind_v2 | verdict |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| investigation | `1:rmsnorm` | no | 0.9945 | n/a | n/a | n/a | n/a | n/a | n/a | 0.437 | 0.172 | failed_knockout |
| investigation | `2:adjacent_token_merge` | yes | 0.5210 | 519.2 | 0.190 | 0.520 | 0.006 | 0.004 | 0.004 | 0.004 | 0.064 | metric_conflict_do_not_prune |
| investigation | `3:add` | no | 0.9782 | n/a | n/a | n/a | n/a | n/a | n/a | 0.004 | 0.213 | failed_knockout |
| investigation | `4:rmsnorm` | no | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.494 | 0.177 | failed_knockout |
| investigation | `5:swiglu_mlp` | yes | 0.8332 | 78.1 | 0.220 | 0.500 | 0.416 | 0.218 | 0.191 | 0.392 | 0.233 | metric_conflict_do_not_prune |
| investigation | `6:gelu` | yes | 0.6082 | 42.1 | 0.180 | 0.506 | 0.014 | 0.198 | 0.065 | 0.318 | 0.179 | metric_conflict_do_not_prune |
| investigation | `7:add` | yes | 0.7429 | 75.7 | 0.190 | 0.512 | 0.352 | 0.220 | 0.172 | 0.296 | 0.179 | metric_conflict_do_not_prune |
| investigation | `8:layernorm` | no | 0.6003 | n/a | n/a | n/a | n/a | n/a | n/a | 0.365 | 0.179 | failed_knockout |
| investigation | `9:linear_proj` | no | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.006 | 0.160 | failed_knockout |
| investigation | `10:linear_proj` | no | 0.6226 | n/a | n/a | n/a | n/a | n/a | n/a | 0.189 | 0.167 | failed_knockout |
| investigation | `11:matmul` | no | 0.5752 | n/a | n/a | n/a | n/a | n/a | n/a | 0.003 | 0.152 | failed_knockout |
| investigation | `13:add` | yes | 0.4965 | 738.0 | 0.210 | 0.525 | 0.004 | 0.004 | 0.003 | 0.003 | 0.000 | metric_conflict_do_not_prune |
| investigation | `14:rmsnorm` | no | 0.9506 | n/a | n/a | n/a | n/a | n/a | n/a | 0.048 | 0.198 | failed_knockout |
| s1 | `1:rmsnorm` | yes | 0.5330 | 66.2 | 0.180 | 0.495 | 0.006 | 0.185 | 0.060 | 0.437 | 0.172 | metric_conflict_do_not_prune |
| s1 | `2:adjacent_token_merge` | yes | 0.4396 | 519.2 | 0.190 | 0.520 | 0.006 | 0.004 | 0.004 | 0.004 | 0.064 | metric_conflict_do_not_prune |
| s1 | `3:add` | yes | 0.3111 | 97.5 | 0.220 | 0.496 | 0.000 | 0.101 | 0.032 | 0.004 | 0.213 | metric_conflict_do_not_prune |
| s1 | `4:rmsnorm` | yes | 0.5798 | 69.4 | 0.200 | 0.521 | 0.002 | 0.144 | 0.045 | 0.494 | 0.177 | metric_conflict_do_not_prune |
| s1 | `5:swiglu_mlp` | no | 0.6776 | 78.1 | 0.220 | 0.500 | 0.416 | 0.218 | 0.191 | 0.392 | 0.233 | failed_knockout |
| s1 | `6:gelu` | yes | 0.5489 | 42.1 | 0.180 | 0.506 | 0.014 | 0.198 | 0.065 | 0.318 | 0.179 | metric_conflict_do_not_prune |
| s1 | `7:add` | no | 0.6570 | 75.7 | 0.190 | 0.512 | 0.352 | 0.220 | 0.172 | 0.296 | 0.179 | failed_knockout |
| s1 | `8:layernorm` | yes | 0.5668 | 69.6 | 0.220 | 0.524 | 0.002 | 0.202 | 0.061 | 0.365 | 0.179 | metric_conflict_do_not_prune |
| s1 | `9:linear_proj` | no | 0.6768 | n/a | n/a | n/a | n/a | n/a | n/a | 0.006 | 0.160 | failed_knockout |
| s1 | `10:linear_proj` | yes | 0.5671 | 102.4 | 0.140 | 0.514 | 0.264 | 0.120 | 0.116 | 0.189 | 0.167 | metric_conflict_do_not_prune |
| s1 | `11:matmul` | yes | 0.5925 | 71.6 | 0.220 | 0.502 | 0.000 | 0.198 | 0.060 | 0.003 | 0.152 | metric_conflict_do_not_prune |
| s1 | `13:add` | yes | 0.5993 | 738.0 | 0.210 | 0.525 | 0.004 | 0.004 | 0.003 | 0.003 | 0.000 | metric_conflict_do_not_prune |
| s1 | `14:rmsnorm` | yes | 0.4911 | 85.4 | 0.180 | 0.517 | 0.008 | 0.146 | 0.047 | 0.048 | 0.198 | metric_conflict_do_not_prune |

### Notes
- `investigation 1:rmsnorm`: failed_knockout - investigation_training_no_passes
- `investigation 2:adjacent_token_merge`: metric_conflict_do_not_prune - loss worsened +0.0329; WikiText worsened 14.2x; HellaSwag down -0.050; binding composite down -0.053; binding_auc collapsed; induction-v2 down -0.307
- `investigation 3:add`: failed_knockout - investigation_training_no_passes
- `investigation 4:rmsnorm`: failed_knockout - investigation_training_no_passes
- `investigation 5:swiglu_mlp`: metric_conflict_do_not_prune - loss worsened +0.3451; WikiText worsened 2.1x
- `investigation 6:gelu`: metric_conflict_do_not_prune - loss worsened +0.1201; HellaSwag down -0.060
- `investigation 7:add`: metric_conflict_do_not_prune - loss worsened +0.2548; WikiText worsened 2.1x; HellaSwag down -0.050
- `investigation 8:layernorm`: failed_knockout - investigation_training_no_passes
- `investigation 9:linear_proj`: failed_knockout - investigation_training_no_passes
- `investigation 10:linear_proj`: failed_knockout - investigation_training_no_passes
- `investigation 11:matmul`: failed_knockout - investigation_training_no_passes
- `investigation 13:add`: metric_conflict_do_not_prune - loss near-neutral +0.0084; WikiText worsened 20.1x; binding composite down -0.055; binding_auc collapsed; induction-v2 down -0.308; binding-v2 down -0.178
- `investigation 14:rmsnorm`: failed_knockout - investigation_training_no_passes
- `s1 1:rmsnorm`: metric_conflict_do_not_prune - loss worsened +0.0449; WikiText worsened 1.8x; HellaSwag down -0.060
- `s1 2:adjacent_token_merge`: metric_conflict_do_not_prune - loss improved -0.0485; WikiText worsened 14.2x; HellaSwag down -0.050; binding composite down -0.053; binding_auc collapsed; induction-v2 down -0.307
- `s1 3:add`: metric_conflict_do_not_prune - loss improved -0.1770; WikiText worsened 2.7x; induction-v2 down -0.307
- `s1 4:rmsnorm`: metric_conflict_do_not_prune - loss worsened +0.0917; WikiText worsened 1.9x; HellaSwag down -0.040
- `s1 5:swiglu_mlp`: failed_knockout - insufficient_learning
- `s1 6:gelu`: metric_conflict_do_not_prune - loss worsened +0.0608; HellaSwag down -0.060
- `s1 7:add`: failed_knockout - insufficient_learning
- `s1 8:layernorm`: metric_conflict_do_not_prune - loss worsened +0.0787; WikiText worsened 1.9x
- `s1 9:linear_proj`: failed_knockout - insufficient_learning
- `s1 10:linear_proj`: metric_conflict_do_not_prune - loss worsened +0.0790; WikiText worsened 2.8x; HellaSwag down -0.100
- `s1 11:matmul`: metric_conflict_do_not_prune - loss worsened +0.1044; WikiText worsened 2.0x; induction-v2 down -0.308
- `s1 13:add`: metric_conflict_do_not_prune - loss worsened +0.1112; WikiText worsened 20.1x; binding composite down -0.055; binding_auc collapsed; induction-v2 down -0.308; binding-v2 down -0.178
- `s1 14:rmsnorm`: metric_conflict_do_not_prune - loss near-neutral +0.0030; WikiText worsened 2.3x; HellaSwag down -0.060; induction-v2 down -0.263

## Parent ec7025d7-338

- experiment: `292fba84-563`
- duplicate evidence rows ignored: 0
- superseded older evidence rows ignored: 0
- verdict counts: `{"failed_knockout": 10, "metric_conflict_do_not_prune": 28}`

| phase | rule | pass | loss | wiki | hs | blimp | ind | bind | bc | ind_v2 | bind_v2 | verdict |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| investigation | `1:layernorm` | no | 0.9925 | n/a | n/a | n/a | n/a | n/a | n/a | 0.999 | 0.999 | failed_knockout |
| investigation | `2:linear_proj` | yes | 0.7243 | 412.6 | 0.220 | 0.531 | 1.000 | 0.005 | 0.303 | 0.985 | 1.000 | metric_conflict_do_not_prune |
| investigation | `3:latent_attention_compressor` | no | 1.0059 | n/a | n/a | n/a | n/a | n/a | n/a | 0.987 | 1.000 | failed_knockout |
| investigation | `4:add` | no | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.987 | 1.000 | failed_knockout |
| investigation | `5:semi_structured_2_4_linear` | yes | 0.7663 | 467.1 | 0.180 | 0.524 | 0.812 | 0.003 | 0.245 | 0.005 | 0.993 | metric_conflict_do_not_prune |
| investigation | `6:relu` | yes | 0.5884 | 416.6 | 0.220 | 0.532 | 0.984 | 0.002 | 0.298 | 1.000 | 0.998 | metric_conflict_do_not_prune |
| investigation | `7:add` | yes | 0.7506 | 424.8 | 0.160 | 0.532 | 0.990 | 0.004 | 0.300 | 0.923 | 0.995 | metric_conflict_do_not_prune |
| investigation | `8:layernorm` | no | 0.9565 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | failed_knockout |
| investigation | `9:token_type_classifier` | no | 0.9477 | n/a | n/a | n/a | n/a | n/a | n/a | 0.993 | 1.000 | failed_knockout |
| investigation | `10:entropy_score` | yes | 0.5797 | 416.6 | 0.190 | 0.518 | 0.122 | 0.005 | 0.040 | 0.926 | 0.996 | metric_conflict_do_not_prune |
| investigation | `11:linear_proj` | no | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.996 | 0.999 | failed_knockout |
| investigation | `12:rope_rotate` | no | 0.6169 | n/a | n/a | n/a | n/a | n/a | n/a | 0.045 | 0.255 | failed_knockout |
| investigation | `13:softmax_attention` | no | 0.5691 | n/a | n/a | n/a | n/a | n/a | n/a | 0.003 | 0.000 | failed_knockout |
| investigation | `14:linear_proj` | yes | 0.5407 | 408.6 | 0.220 | 0.541 | 1.000 | 0.006 | 0.303 | 1.000 | 1.000 | metric_conflict_do_not_prune |
| investigation | `15:mul` | no | 1.0169 | n/a | n/a | n/a | n/a | n/a | n/a | 1.000 | 1.000 | failed_knockout |
| investigation | `16:add` | yes | 0.7511 | 451.1 | 0.230 | 0.535 | 0.000 | 0.005 | 0.002 | 0.004 | 0.000 | metric_conflict_do_not_prune |
| investigation | `18:rmsnorm` | yes | 0.6565 | 401.7 | 0.210 | 0.570 | 0.972 | 0.005 | 0.295 | 0.998 | 1.000 | metric_conflict_do_not_prune |
| investigation | `19:spectral_filter` | yes | 0.1436 | 399.1 | 0.210 | 0.535 | 0.968 | 0.004 | 0.293 | 0.998 | 1.000 | metric_conflict_do_not_prune |
| investigation | `20:add` | no | 0.9006 | n/a | n/a | n/a | n/a | n/a | n/a | 0.998 | 1.000 | failed_knockout |
| s1 | `1:layernorm` | yes | 0.6026 | 640.5 | 0.220 | 0.505 | 0.148 | 0.007 | 0.047 | 0.999 | 0.999 | metric_conflict_do_not_prune |
| s1 | `2:linear_proj` | yes | 0.5497 | 412.6 | 0.220 | 0.531 | 1.000 | 0.005 | 0.303 | 0.985 | 1.000 | metric_conflict_do_not_prune |
| s1 | `3:latent_attention_compressor` | yes | 0.5721 | 651.7 | 0.200 | 0.520 | 0.938 | 0.006 | 0.285 | 0.987 | 1.000 | metric_conflict_do_not_prune |
| s1 | `4:add` | yes | 0.5647 | 719.2 | 0.240 | 0.523 | 0.844 | 0.006 | 0.256 | 0.987 | 1.000 | metric_conflict_do_not_prune |
| s1 | `5:semi_structured_2_4_linear` | yes | 0.5520 | 467.1 | 0.180 | 0.524 | 0.812 | 0.003 | 0.245 | 0.005 | 0.993 | metric_conflict_do_not_prune |
| s1 | `6:relu` | yes | 0.5650 | 416.6 | 0.220 | 0.532 | 0.984 | 0.002 | 0.298 | 1.000 | 0.998 | metric_conflict_do_not_prune |
| s1 | `7:add` | yes | 0.5133 | 424.8 | 0.160 | 0.532 | 0.990 | 0.004 | 0.300 | 0.923 | 0.995 | metric_conflict_do_not_prune |
| s1 | `8:layernorm` | yes | 0.5314 | 631.9 | 0.240 | 0.504 | 0.002 | 0.006 | 0.003 | n/a | n/a | metric_conflict_do_not_prune |
| s1 | `9:token_type_classifier` | yes | 0.5710 | 595.2 | 0.200 | 0.521 | 0.780 | 0.005 | 0.237 | 0.993 | 1.000 | metric_conflict_do_not_prune |
| s1 | `10:entropy_score` | yes | 0.5638 | 416.6 | 0.190 | 0.518 | 0.122 | 0.005 | 0.040 | 0.926 | 0.996 | metric_conflict_do_not_prune |
| s1 | `11:linear_proj` | yes | 0.5604 | 657.1 | 0.220 | 0.512 | 0.108 | 0.004 | 0.035 | 0.996 | 0.999 | metric_conflict_do_not_prune |
| s1 | `12:rope_rotate` | yes | 0.5136 | 611.1 | 0.200 | 0.517 | 0.006 | 0.007 | 0.005 | 0.045 | 0.255 | metric_conflict_do_not_prune |
| s1 | `13:softmax_attention` | yes | 0.5377 | 701.1 | 0.220 | 0.536 | 0.002 | 0.003 | 0.002 | 0.003 | 0.000 | metric_conflict_do_not_prune |
| s1 | `14:linear_proj` | yes | 0.5490 | 408.6 | 0.220 | 0.541 | 1.000 | 0.006 | 0.303 | 1.000 | 1.000 | metric_conflict_do_not_prune |
| s1 | `15:mul` | yes | 0.5789 | 660.0 | 0.240 | 0.506 | 0.458 | 0.004 | 0.139 | 1.000 | 1.000 | metric_conflict_do_not_prune |
| s1 | `16:add` | yes | 0.5836 | 451.1 | 0.230 | 0.535 | 0.000 | 0.005 | 0.002 | 0.004 | 0.000 | metric_conflict_do_not_prune |
| s1 | `18:rmsnorm` | yes | 0.5768 | 401.7 | 0.210 | 0.570 | 0.972 | 0.005 | 0.295 | 0.998 | 1.000 | metric_conflict_do_not_prune |
| s1 | `19:spectral_filter` | yes | 0.5169 | 399.1 | 0.210 | 0.535 | 0.968 | 0.004 | 0.293 | 0.998 | 1.000 | metric_conflict_do_not_prune |
| s1 | `20:add` | yes | 0.5545 | 688.4 | 0.260 | 0.515 | 0.766 | 0.006 | 0.232 | 0.998 | 1.000 | metric_conflict_do_not_prune |

### Notes
- `investigation 1:layernorm`: failed_knockout - stage1_not_passed
- `investigation 2:linear_proj`: metric_conflict_do_not_prune - loss worsened +0.1909; HellaSwag down -0.040; binding composite down -0.128; binding_auc collapsed
- `investigation 3:latent_attention_compressor`: failed_knockout - stage1_not_passed
- `investigation 4:add`: failed_knockout - stage1_not_passed
- `investigation 5:semi_structured_2_4_linear`: metric_conflict_do_not_prune - loss worsened +0.2329; HellaSwag down -0.080; binding composite down -0.186; binding_auc collapsed; induction-v2 down -0.995
- `investigation 6:relu`: metric_conflict_do_not_prune - loss worsened +0.0550; HellaSwag down -0.040; binding composite down -0.133; binding_auc collapsed
- `investigation 7:add`: metric_conflict_do_not_prune - loss worsened +0.2172; HellaSwag down -0.100; binding composite down -0.130; binding_auc collapsed
- `investigation 8:layernorm`: failed_knockout - stage1_not_passed
- `investigation 9:token_type_classifier`: failed_knockout - stage1_not_passed
- `investigation 10:entropy_score`: metric_conflict_do_not_prune - loss worsened +0.0463; HellaSwag down -0.070; binding composite down -0.391; binding_auc collapsed
- `investigation 11:linear_proj`: failed_knockout - stage1_not_passed
- `investigation 12:rope_rotate`: failed_knockout - stage1_not_passed
- `investigation 13:softmax_attention`: failed_knockout - stage1_not_passed
- `investigation 14:linear_proj`: metric_conflict_do_not_prune - loss near-neutral +0.0073; HellaSwag down -0.040; binding composite down -0.128; binding_auc collapsed
- `investigation 15:mul`: failed_knockout - stage1_not_passed
- `investigation 16:add`: metric_conflict_do_not_prune - loss worsened +0.2176; binding composite down -0.428; binding_auc collapsed; induction-v2 down -0.996; binding-v2 down -1.000
- `investigation 18:rmsnorm`: metric_conflict_do_not_prune - loss worsened +0.1231; HellaSwag down -0.050; binding composite down -0.135; binding_auc collapsed
- `investigation 19:spectral_filter`: metric_conflict_do_not_prune - loss improved -0.3898; HellaSwag down -0.050; binding composite down -0.138; binding_auc collapsed
- `investigation 20:add`: failed_knockout - stage1_not_passed
- `s1 1:layernorm`: metric_conflict_do_not_prune - loss worsened +0.0692; HellaSwag down -0.040; binding composite down -0.383; binding_auc collapsed
- `s1 2:linear_proj`: metric_conflict_do_not_prune - loss near-neutral +0.0163; HellaSwag down -0.040; binding composite down -0.128; binding_auc collapsed
- `s1 3:latent_attention_compressor`: metric_conflict_do_not_prune - loss worsened +0.0386; HellaSwag down -0.060; binding composite down -0.145; binding_auc collapsed
- `s1 4:add`: metric_conflict_do_not_prune - loss worsened +0.0312; binding composite down -0.174; binding_auc collapsed
- `s1 5:semi_structured_2_4_linear`: metric_conflict_do_not_prune - loss near-neutral +0.0186; HellaSwag down -0.080; binding composite down -0.186; binding_auc collapsed; induction-v2 down -0.995
- `s1 6:relu`: metric_conflict_do_not_prune - loss worsened +0.0316; HellaSwag down -0.040; binding composite down -0.133; binding_auc collapsed
- `s1 7:add`: metric_conflict_do_not_prune - loss improved -0.0202; HellaSwag down -0.100; binding composite down -0.130; binding_auc collapsed
- `s1 8:layernorm`: metric_conflict_do_not_prune - loss near-neutral -0.0021; binding composite down -0.428; binding_auc collapsed
- `s1 9:token_type_classifier`: metric_conflict_do_not_prune - loss worsened +0.0376; HellaSwag down -0.060; binding composite down -0.193; binding_auc collapsed
- `s1 10:entropy_score`: metric_conflict_do_not_prune - loss worsened +0.0303; HellaSwag down -0.070; binding composite down -0.391; binding_auc collapsed
- `s1 11:linear_proj`: metric_conflict_do_not_prune - loss worsened +0.0270; HellaSwag down -0.040; binding composite down -0.396; binding_auc collapsed
- `s1 12:rope_rotate`: metric_conflict_do_not_prune - loss near-neutral -0.0198; HellaSwag down -0.060; binding composite down -0.426; binding_auc collapsed; induction-v2 down -0.955; binding-v2 down -0.745
- `s1 13:softmax_attention`: metric_conflict_do_not_prune - loss near-neutral +0.0042; HellaSwag down -0.040; binding composite down -0.429; binding_auc collapsed; induction-v2 down -0.997; binding-v2 down -1.000
- `s1 14:linear_proj`: metric_conflict_do_not_prune - loss near-neutral +0.0155; HellaSwag down -0.040; binding composite down -0.128; binding_auc collapsed
- `s1 15:mul`: metric_conflict_do_not_prune - loss worsened +0.0455; binding composite down -0.292; binding_auc collapsed
- `s1 16:add`: metric_conflict_do_not_prune - loss worsened +0.0501; binding composite down -0.428; binding_auc collapsed; induction-v2 down -0.996; binding-v2 down -1.000
- `s1 18:rmsnorm`: metric_conflict_do_not_prune - loss worsened +0.0434; HellaSwag down -0.050; binding composite down -0.135; binding_auc collapsed
- `s1 19:spectral_filter`: metric_conflict_do_not_prune - loss near-neutral -0.0166; HellaSwag down -0.050; binding composite down -0.138; binding_auc collapsed
- `s1 20:add`: metric_conflict_do_not_prune - loss worsened +0.0211; binding composite down -0.199; binding_auc collapsed
