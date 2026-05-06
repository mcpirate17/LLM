# Controlled-Lang Nano-Bind / Nano-BLiMP Funnel Snapshot

Snapshot date: 2026-05-06.

This note explains the controlled-language SA/NB metrics and records the current
bucket distributions for using them as a funnel before the more expensive
Nano-AR investigation probe.

## Metric Meaning

The current `controlled_lang_*` path trains on short symbolic association
sequences:

```text
[noun, query, answer]
```

Example mapping:

```text
cat + verb_query -> sat
cat + adj_query  -> small
dog + verb_query -> ran
dog + adj_query  -> loud
```

`SA` is `synthetic_association_score`. It asks whether the model can choose the
exact associated answer from same-class candidates:

```text
Prompt:     [cat, verb_query, PAD]
Candidates: [ran, sat, jumped, slept]
Correct:    sat
```

`NB` is `nano_blimp_score`. It is a three-part minimal-pair score:

```text
class:   good [cat, verb_query, sat] > bad [cat, verb_query, dog]
binding: good [cat, verb_query, sat] > bad [cat, verb_query, ran]
order:   good [cat, verb_query, sat] > bad [verb_query, cat, sat]
```

The stored NB score is:

```text
NB = (class_coherence + binding_fidelity + order_grammaticality) / 3
```

Important interpretation: high SA plus weak NB order means the model learned the
association mapping, but does not strongly prefer the well-formed token order.
Order should be treated as a structural diagnostic, not as direct evidence that
binding failed.

## Tier Configs

Current controlled-language tier configs in `research/tools/controlled_lang_backfill.py`:

| tier | active vocab | train steps | checkpoints |
| --- | ---: | ---: | --- |
| S0.5 | 120 | 40 | final only |
| S1.0 | 240 | 2000 | 500, 1000, 2000 |
| INV | 360 | 2000 | 500, 1000, 2000 |

## Distribution Buckets

Bucket definitions used below:

```text
<65    = score < 0.65
65-75  = 0.65 <= score < 0.76
76-85  = 0.76 <= score < 0.86
86-95  = 0.86 <= score < 0.96
>95    = score >= 0.96
```

Counts:

| tier | metric | n | <65 | 65-75 | 76-85 | 86-95 | >95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S0.5 | NB | 8066 | 362 | 1991 | 3541 | 1152 | 1020 |
| S0.5 | SA | 8066 | 4625 | 121 | 152 | 266 | 2902 |
| S1.0 | NB | 2717 | 10 | 398 | 865 | 645 | 799 |
| S1.0 | SA | 2717 | 466 | 21 | 18 | 28 | 2184 |
| INV | NB | 868 | 4 | 156 | 171 | 207 | 330 |
| INV | SA | 868 | 29 | 9 | 2 | 1 | 827 |

Percentages:

| tier | metric | <65 | 65-75 | 76-85 | 86-95 | >95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| S0.5 | NB | 4.5% | 24.7% | 43.9% | 14.3% | 12.6% |
| S0.5 | SA | 57.3% | 1.5% | 1.9% | 3.3% | 36.0% |
| S1.0 | NB | 0.4% | 14.6% | 31.8% | 23.7% | 29.4% |
| S1.0 | SA | 17.2% | 0.8% | 0.7% | 1.0% | 80.4% |
| INV | NB | 0.5% | 18.0% | 19.7% | 23.8% | 38.0% |
| INV | SA | 3.3% | 1.0% | 0.2% | 0.1% | 95.3% |

Order-only counts, included because order is the main NB spread source:

| tier | n | <65 | 65-75 | 76-85 | 86-95 | >95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S0.5 order | 8066 | 1890 | 725 | 1119 | 1739 | 2593 |
| S1.0 order | 2717 | 1021 | 270 | 232 | 305 | 889 |
| INV order | 868 | 372 | 76 | 62 | 82 | 276 |

## Current Gate Behavior

Existing controlled-language hard gates:

| gate | condition | current applied failures |
| --- | --- | ---: |
| S0.5 SA | `controlled_lang_s05_sa_score < 0.65` unless ERF/mixing escape | 3689 |
| S0.5 NB | `controlled_lang_s05_nb_score < 0.65` | 320 |
| S1.0 NB | `controlled_lang_s10_nb_score < 0.65` | 10 |
| S1.0 NB+SA | `controlled_lang_s10_nb_score < 0.80 AND controlled_lang_s10_sa_score < 0.65` | 29 |
| INV NB | `controlled_lang_inv_nb_score < 0.65` | 4 |

The live distributions show that NB total rarely fails below 0.65 after S1.0.
That is because class and binding are usually high. Many weak order rows remain,
but low order alone is not a reliable hard-fail signal.

## Nano-AR Overlap Check

Among rows with both INV controlled-language metrics and Nano-AR:

| n | avg Nano-AR | Nano-AR no-go | Nano-AR < 0.50 | avg INV SA | avg INV NB | avg INV order |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 97 | 0.742 | 7 | 21 | 0.983 | 0.838 | 0.530 |

INV NB buckets vs Nano-AR:

| INV NB bucket | n | avg Nano-AR | Nano-AR < 0.50 | Nano-AR no-go |
| --- | ---: | ---: | ---: | ---: |
| NB <65 | 4 | 0.460 | 4 | 0 |
| NB 65-75 | 28 | 0.730 | 5 | 1 |
| NB 76-85 | 20 | 0.764 | 3 | 1 |
| NB 86-95 | 18 | 0.820 | 2 | 0 |
| NB >95 | 27 | 0.727 | 7 | 5 |

INV SA buckets vs Nano-AR:

| INV SA bucket | n | avg Nano-AR | Nano-AR < 0.50 | Nano-AR no-go |
| --- | ---: | ---: | ---: | ---: |
| SA <65 | 4 | 0.460 | 4 | 0 |
| SA 86-95 | 1 | 0.460 | 1 | 0 |
| SA >95 | 92 | 0.757 | 16 | 7 |

S1.0 candidate gates vs Nano-AR, for rows with S1.0 and Nano-AR:

| candidate | n | avg Nano-AR | Nano-AR < 0.50 | Nano-AR no-go |
| --- | ---: | ---: | ---: | ---: |
| `s10_sa < 0.96` | 81 | 0.553 | 65 | 0 |
| `s10_nb < 0.76` | 35 | 0.681 | 12 | 1 |
| `s10_sa < 0.96 AND s10_nb < 0.86` | 64 | 0.476 | 62 | 0 |
| `s10_nb < 0.86` | 105 | 0.576 | 71 | 1 |

## Funnel Recommendation

Do not add a hard gate on NB order alone. Order is the main source of NB spread,
but low order includes good Nano-AR rows and high order includes Nano-AR no-go
rows. It is useful for architecture diagnosis, not as a no-go filter.

Conservative pre-Nano-AR filter:

```text
skip or defer Nano-AR if controlled_lang_inv_sa_score < 0.96
```

Rationale:

| set | rows missing Nano-AR | proposed skips |
| --- | ---: | ---: |
| all rows with INV metrics and no Nano-AR | 779 | 36 |
| same, excluding existing controlled-lang failures | 771 | 30 |

Overlap evidence is small but clean: in the current rows with both INV and
Nano-AR, `inv_sa < 0.96` catches 5 rows and all 5 have Nano-AR < 0.50.

I would initially implement this as a backfill priority/defer rule, not as a
permanent demotion gate:

```text
if controlled_lang_inv_sa_score < 0.96:
    do not spend Nano-AR budget in the normal queue
    mark/defer as controlled_lang_inv_sa_pre_ar_defer
```

For the screening cascade, a broader but riskier early filter is:

```text
controlled_lang_s10_sa_score < 0.96 AND controlled_lang_s10_nb_score < 0.86
```

Current overlap: 64 rows match, 62 have Nano-AR < 0.50. But there are observed
false positives with good Nano-AR, so this should be a priority/defer filter
unless an explicit escape hatch exists.

Practical funnel:

1. Run S0.5 controlled-lang.
2. Apply existing S0.5 hard gates.
3. Run S1.0 only for S0.5 pass/escape rows.
4. Apply existing S1.0 hard gates.
5. Run INV before Nano-AR for serious investigation candidates.
6. Defer Nano-AR when `controlled_lang_inv_sa_score < 0.96`.
7. Do not use NB order as a hard gate.

This gives a budget funnel without treating the order diagnostic as failed
binding.
