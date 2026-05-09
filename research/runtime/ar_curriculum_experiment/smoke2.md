# AR Curriculum experiment — smoke2

stage_set=default vocab_lo=1000 vocab_size=3200 stages=6 steps_per_stage=10 d_model=64 n_layers=2 seeds=2

## Stage configuration

| stage | n_keys | n_values | pairs/ex | n_train | n_held |
|---:|---:|---:|---:|---:|---:|
| 0 | 32 | 8 | 2 | 8 | 4 |
| 1 | 64 | 12 | 2 | 16 | 8 |
| 2 | 128 | 16 | 3 | 32 | 16 |
| 3 | 256 | 24 | 4 | 64 | 24 |
| 4 | 512 | 32 | 6 | 96 | 32 |
| 5 | 1024 | 48 | 9 | 128 | 48 |

## Headline (mean ± std across seeds)

| arch | paradigm | seeds | AUC final | AUC peak | retention |
|---|---|---:|---:|---:|---:|
| GPT-2 | dense_attention_transformer | 2 | 0.073±0.044 | 0.135±0.044 | 0.51 |

## Per-stage held_pair_acc (mean across seeds)

| arch | S0 | S1 | S2 | S3 | S4 | S5 |
|---|---:|---:|---:|---:|---:|---:|
| GPT-2 | 0.12 | 0.06 | 0.06 | 0.00 | 0.12 | 0.06 |
