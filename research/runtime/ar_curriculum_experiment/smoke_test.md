# AR Curriculum experiment — smoke_test

vocab_lo=1000 vocab_size=3200 stages=6 steps_per_stage=20

## Stage configuration

| stage | n_keys | n_values | pairs/ex | n_train | n_held |
|---:|---:|---:|---:|---:|---:|
| 0 | 32 | 8 | 2 | 8 | 4 |
| 1 | 64 | 12 | 2 | 16 | 8 |
| 2 | 128 | 16 | 3 | 32 | 16 |
| 3 | 256 | 24 | 4 | 64 | 24 |
| 4 | 512 | 32 | 6 | 96 | 32 |
| 5 | 1024 | 48 | 9 | 128 | 48 |

## Headline (post all 6 stages)

| arch | paradigm | wall(s) | AUC pair | AUC class | max passing | status |
|---|---|---:|---:|---:|---:|---|
| GPT-2 | dense_attention_transformer | 0.9 | 0.125 | 0.333 | -1 | ok |

## Per-stage held_pair_acc (final)

| arch | S0 | S1 | S2 | S3 | S4 | S5 |
|---|---:|---:|---:|---:|---:|---:|
| GPT-2 | 0.12 | 0.12 | 0.12 | 0.12 | 0.25 | 0.00 |

## Forgetting matrix (rows=after-stage, cols=eval-stage, pair_acc)

### GPT-2

| after\eval | S0 | S1 | S2 | S3 | S4 | S5 |
|---|---:|---:|---:|---:|---:|---:|
| S0 | 0.12 | — | — | — | — | — |
| S1 | 0.12 | 0.12 | — | — | — | — |
| S2 | 0.12 | 0.12 | 0.00 | — | — | — |
| S3 | 0.12 | 0.12 | 0.12 | 0.00 | — | — |
| S4 | 0.12 | 0.12 | 0.12 | 0.12 | 0.00 | — |
| S5 | 0.12 | 0.12 | 0.12 | 0.12 | 0.25 | 0.00 |

