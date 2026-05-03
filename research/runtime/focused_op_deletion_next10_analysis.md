# Focused op deletion ablation: next 10 leaderboard graphs

Parents: 10; planned children: 130; pre-run rejected variants: 19.
Evidence rows: 130; valid performance rows: 129; invalid failed-child rows quarantined: 1.

## Data quality
- passed_ablation_children_missing_required_s1_metrics: 0
- duplicate_evidence_child_links: 0
- duplicate_child_fingerprint_links: 0
- valid_rows_with_null_effect: 0

## Parent summary
| parent | valid | deletion hurts | deletion improves child | inconclusive | avg effect | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| 903157e5-219 | 10 | 2 | 5 | 3 | -0.0318 | -0.1543 | 0.0256 |
| 296cb201-3b3 | 4 | 4 | 0 | 0 | 0.1468 | 0.1419 | 0.1561 |
| ebdd9a5c-8a3 | 15 | 15 | 0 | 0 | 0.1585 | 0.0840 | 0.2457 |
| ad096d0b-286 | 15 | 13 | 1 | 1 | 0.0937 | -0.0297 | 0.5112 |
| 8b9b42ed-c58 | 11 | 1 | 5 | 5 | -0.0096 | -0.0378 | 0.0904 |
| d49297a5-3bd | 13 | 3 | 5 | 5 | 0.0145 | -0.0500 | 0.2958 |
| ce318e1c-43d | 13 | 7 | 3 | 3 | 0.0021 | -0.0918 | 0.0477 |
| 8e7b7eb9-913 | 15 | 1 | 14 | 0 | -0.0893 | -0.1906 | 0.3182 |
| 9f583a0c-33b | 18 | 0 | 18 | 0 | -0.1460 | -0.2194 | -0.0590 |
| a311fc5d-12c | 15 | 1 | 13 | 1 | -0.1021 | -0.2077 | 0.1573 |

## Strongest dead-weight candidates
- 9f583a0c-33b 15:nm_sparse_linear effect=-0.2194 child=c42606e7-3f2
- 9f583a0c-33b 9:rmsnorm effect=-0.2133 child=dee66056-d79
- a311fc5d-12c 14:nm_sparse_linear effect=-0.2077 child=78e4a118-291
- 8e7b7eb9-913 14:nm_sparse_linear effect=-0.1906 child=4ccee16c-36f
- 8e7b7eb9-913 12:swiglu_mlp effect=-0.1847 child=ccb798cc-117
- 9f583a0c-33b 19:add effect=-0.1758 child=35186285-585
- a311fc5d-12c 10:add effect=-0.1741 child=0d476683-139
- 8e7b7eb9-913 11:rmsnorm effect=-0.1734 child=6a0a37f7-b79
- 9f583a0c-33b 3:linear_proj effect=-0.1717 child=52b80ca2-5a1
- 9f583a0c-33b 18:spectral_filter effect=-0.1702 child=1eaf6bfe-e5a
- 9f583a0c-33b 5:rmsnorm effect=-0.1688 child=43c53009-606
- 8e7b7eb9-913 2:token_class_proj effect=-0.1680 child=a7eacbad-e17

## Strongest protective candidates
- ad096d0b-286 15:add effect=0.5112 child=1d60ac5e-256
- 8e7b7eb9-913 7:add effect=0.3182 child=534e6f6b-d05
- d49297a5-3bd 13:add effect=0.2958 child=652eadf0-cee
- ebdd9a5c-8a3 10:adjacent_token_merge effect=0.2457 child=b3ae10e3-90f
- ebdd9a5c-8a3 1:layernorm effect=0.2338 child=3e8c3462-8ef
- ebdd9a5c-8a3 4:add effect=0.2074 child=4d6cd38a-b00
- ebdd9a5c-8a3 3:linear_proj effect=0.1823 child=f35d04aa-e6a
- ebdd9a5c-8a3 15:add effect=0.1783 child=dfe01569-9ba
- ebdd9a5c-8a3 13:conv1d_seq effect=0.1724 child=85b52ff0-3e6
- ebdd9a5c-8a3 6:ternary_projection effect=0.1696 child=63aaf13f-e13
- a311fc5d-12c 7:add effect=0.1573 child=3eafae2e-63c
- 296cb201-3b3 7:conv1d_seq effect=0.1561 child=b3e26f92-641

## Op aggregates
| op | n | parents | deletion hurts | improves child | inconclusive | avg effect |
|---|---:|---:|---:|---:|---:|---:|
| add | 29 | 9 | 11 | 13 | 5 | 0.0276 |
| layernorm | 10 | 9 | 3 | 4 | 3 | 0.0064 |
| rmsnorm | 22 | 8 | 7 | 12 | 3 | -0.0456 |
| ternary_projection | 6 | 6 | 2 | 3 | 1 | 0.0002 |
| adjacent_token_merge | 6 | 6 | 2 | 3 | 1 | 0.0256 |
| linear_proj | 7 | 5 | 3 | 2 | 2 | -0.0091 |
| nm_sparse_linear | 6 | 5 | 1 | 5 | 0 | -0.1217 |
| mul | 4 | 4 | 1 | 2 | 1 | -0.0270 |
| swiglu_mlp | 4 | 4 | 1 | 3 | 0 | -0.1008 |
| conv1d_seq | 3 | 3 | 3 | 0 | 0 | 0.1377 |
| gelu | 3 | 3 | 1 | 2 | 0 | -0.0569 |
| token_class_proj | 3 | 3 | 1 | 2 | 0 | -0.0778 |
| token_entropy | 3 | 3 | 1 | 2 | 0 | -0.0527 |
| matmul | 2 | 2 | 1 | 1 | 0 | -0.0065 |
| latent_attention_compressor | 2 | 2 | 1 | 1 | 0 | -0.0005 |
| relu | 2 | 2 | 1 | 1 | 0 | 0.0531 |
| spectral_filter | 2 | 2 | 0 | 2 | 0 | -0.1011 |
| sigmoid | 1 | 1 | 0 | 1 | 0 | -0.0272 |
| token_type_classifier | 1 | 1 | 0 | 0 | 1 | -0.0073 |
| entropy_score | 1 | 1 | 0 | 1 | 0 | -0.1137 |
| padic_gate | 1 | 1 | 1 | 0 | 0 | 0.1419 |
| rwkv_channel | 1 | 1 | 1 | 0 | 0 | 0.1438 |
| moe_2expert | 1 | 1 | 1 | 0 | 0 | 0.1455 |
| cos | 1 | 1 | 1 | 0 | 0 | 0.0865 |
| learnable_scale | 1 | 1 | 0 | 0 | 1 | -0.0191 |
| rwkv_time_mixing | 1 | 1 | 1 | 0 | 0 | 0.0904 |
| speculative | 1 | 1 | 0 | 1 | 0 | -0.0378 |
| silu | 1 | 1 | 1 | 0 | 0 | 0.0207 |
| feature_sparsity | 1 | 1 | 0 | 1 | 0 | -0.0277 |
| sin | 1 | 1 | 0 | 1 | 0 | -0.0500 |

## Invalid failed children
- 8b9b42ed-c58 1:linear_proj_down child=ae92aeeb-2b0 stage=stage0 error=    return a + b
           ~~^~~
RuntimeError: The size of tensor a (256) must match the size of tensor b (64) at non-singleton dimension 2
