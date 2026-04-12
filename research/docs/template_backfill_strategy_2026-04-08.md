# Template Backfill Strategy

## Goal

Spend backfill budget on templates that are already structurally valid and
show the strongest combination of:

- repeated `S1` conversion
- low best loss
- reasonable exposure-normalized pass rate
- no recurring generation-time structural failures

Avoid spending harvest budget on templates that are merely unblocked but still
show repeated `0`-`S1` behavior.

## Current Tiers

### Harvest

Use larger stack/isolation backfills and keep collecting evidence unless the
template regresses structurally.

| Template | Eval | S1 | Best Loss | Notes |
| --- | ---: | ---: | ---: | --- |
| `latent_attn_ssm_hybrid` | 156 | 68 | 0.2932 | strongest current combined yield |
| `intelligent_multilane_router` | 91 | 39 | 0.5427 | high sustained S1 conversion |
| `local_attn_ssm_hybrid` | 71 | 28 | 0.3430 | strong stack conversion |
| `attn_softmax_normalized_matmul_compact_ffn` | 74 | 12 | 0.3778 | repaired attention-tail family with real S1 signal |
| `attn_spectral_filter` | 239 | 10 | 0.3753 | broad evidence base, still converting |
| `linear_attn_ffn_block` | 110 | 9 | 0.2385 | lower pass rate but excellent best loss |

### Rehab

Use capped coverage backfills to decide whether these are seed-sensitive winners
or just weak-but-valid graphs.

| Template | Eval | S1 | Best Loss | Notes |
| --- | ---: | ---: | ---: | --- |
| `depth_gated_block_matmul_norm` | 20 | 3 | 0.4359 | promising, underexposed |
| `attn_softmax_normalized_matmul_fixed_tail_norm` | 26 | 5 | 0.3896 | viable but still sparse |
| `attn_linear_no_matmul_ffn` | 130 | 6 | 0.4111 | valid and some signal, but weaker conversion |
| `attn_softmax_normalized_matmul` | 103 | 3 | 0.3009 | very good best loss, weak total S1 count |

### Freeze Or Constrain

Do not spend broader harvest budget here until there is a direct template or
placement fix beyond the generation unblocking already done.

| Template | Current State | Notes |
| --- | --- | --- |
| `attn_softmax_normalized_matmul_v2` | valid generation, repeated `0` S1 | governance bug fixed; quality still poor |
| `attn_linear_no_matmul_ffn_v2` | valid generation, repeated `0` S1 | governance bug fixed; quality still poor |
| `attn_linear_no_matmul_ffn_dense_tail` | stable, repeated `0` S1 | not structurally broken, not harvest-worthy |

## Execution Rules

1. Harvest wave:
   run `stack` backfills on Harvest templates with `scaffold_guided` weights,
   but keep learned candidate weights, learned grammar weights, screening
   signal weights, and GBM prescreening disabled.

2. Rehab wave:
   run smaller `isolation` backfills on Rehab templates.
   Promote to Harvest only after either:
   - `best_loss <= 0.45` and `S1 >= 8`, or
   - repeated `S1` conversion across multiple runs without structural stops.

3. Freeze wave:
   if a template is structurally blocked, fix it first.
   If it is valid but repeatedly produces `0` `S1` after enough exposure, stop
   broad backfill and prefer template repair or slot constraints instead.

## Active Broad Wave

The current broad harvest batch was launched with:

```bash
python -m research.tools.backfill_templates \
  --target 40 \
  --min-s1 8 \
  --batch-size 16 \
  --device cuda \
  --db research/lab_notebook.db \
  --policy auto \
  --weights scaffold_guided \
  --phase stack \
  --templates \
    latent_attn_ssm_hybrid \
    local_attn_ssm_hybrid \
    intelligent_multilane_router \
    depth_gated_block_matmul_norm \
    attn_softmax_normalized_matmul_compact_ffn \
    attn_softmax_normalized_matmul_fixed_tail_norm \
    attn_linear_no_matmul_ffn \
    linear_attn_ffn_block \
    attn_softmax_normalized_matmul \
    attn_spectral_filter
```

The active wave starts with `depth_gated_block_matmul_norm`, which is correct
under the current deficit logic because it is both underexposed and already
showing non-zero `S1` signal.
