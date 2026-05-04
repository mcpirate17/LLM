# NanoBind-S0

NanoBind-S0 is a hard no-go screen for one narrow failure mode: persistent
exact-zero slot-ending accuracy on a small noun-to-adjective binding corpus.

It is not a language benchmark, not a leaderboard score, and not a positive
ranking signal. A nonzero result means only that the architecture remains
eligible for the normal evaluation pipeline.

## Decision Rule

Reject only when slot-ending accuracy is exactly `0.00` at every checkpoint in
the calibrated sweep.

```text
all checkpoint slot-ending accuracies == 0.00  -> no-go
any checkpoint slot-ending accuracy > 0.00     -> continue evaluation
```

Do not reject on strict learner diagnostics, low nonzero scores, held-out
weakness, low unique-token count, or poor qualitative predictions. Those fields
are audit evidence, not rejection criteria.

## Implementation

Primary implementation:

- `research/eval/nano_bind.py`

Corpus source:

- `research/tools/nano_corpus_v0.py`

Metric version:

- `nano_bind_v0`

Default sweep:

- checkpoints: `1000`, `2000`
- optimizer: AdamW
- learning rate: `1e-3`
- batch size: `32`
- timeout: `60s`
- training source: fresh model compiled from `graph_json`
- no random-token warmup

## Vocabulary

NanoBind uses a deliberately small controlled vocabulary. Words are expected to
tokenize as single tokens under the configured tokenizer.

Current NanoBind tokenizer:

- `cl100k_base`

Core default adjective subset:

```text
fat thin big small wet dry warm cold soft hard
```

Extended adjective list for sweeps:

```text
red blue green white black fast slow tall short fresh clean dirty sweet sharp
deep light heavy loud quiet bright sad happy rich poor strong weak old new dark
full
```

Nouns:

```text
cat dog bird mouse rabbit man boy woman girl child car ship lamp book chair
```

Held-out nouns in the default gate:

```text
cat book lamp child ship
```

Default test nouns:

```text
cat book lamp child ship dog bird man boy woman
```

Verb and frame words:

```text
ran jumped slept sat see
the a I was
```

## Corpus

The default training corpus has 280 sentences:

- Bucket A: `80` target-frame sentences
- Bucket B: `120` same-noun other-frame sentences
- Bucket C: `80` same-adjective other-construction sentences

### Bucket A

Target frame:

```text
the {noun} was {adj}
```

Held-out nouns are excluded from this bucket, so they never appear in the
training corpus in the exact target frame `the {noun} was {adj}`.

Strict selection is enabled by default:

- each in-distribution noun receives `3` adjectives
- adjective assignment is deterministic and overlapping
- this creates noun-specific adjective signatures without making each noun
exclusive to one adjective

Example shape:

```text
the dog was fat
the bird was thin
the woman was warm
```

### Bucket B

Same nouns, different frames:

```text
the {noun} ran
the {noun} jumped
the {noun} slept
the {noun} sat
I see the {noun}
```

This ensures held-out nouns are still present in training, just not in the
target `was {adj}` slot.

### Bucket C

Same adjectives, different constructions:

```text
the {adj} {noun} ran
the {adj} {noun} sat
I see a {adj} {noun}
```

This ensures adjectives are seen in non-target contexts.

## Evaluation Prompt

Evaluation prompts use the target frame without the final adjective:

```text
the {noun} was
```

The model is scored on whether its top-1 next-token prediction is one of the
adjective tokens in the active adjective vocabulary. It is not required to
predict the exact training adjective for a noun. The S0 gate only checks whether
the model ever predicts any valid adjective in the slot across the sweep.

Default prompt set includes both held-out and in-distribution nouns:

```text
the cat was
the book was
the lamp was
the child was
the ship was
the dog was
the bird was
the man was
the boy was
the woman was
```

## Result Fields

`NanoBindResult.to_dict()` emits the `nano_bind_*` namespace.

Core decision fields:

- `nano_bind_metric_version`
- `nano_bind_is_no_go`
- `nano_bind_scores`
- `nano_bind_status`
- `nano_bind_error`

Audit fields:

- `nano_bind_held_acc`
- `nano_bind_n_unique`
- `nano_bind_elapsed_ms`
- `nano_bind_top5_token_ids`
- `nano_bind_top5_tokens`
- `nano_bind_prompt_sentences`
- `nano_bind_sweep_metadata`

The audit payload should preserve raw top-k predictions and sweep metadata so
any rejected architecture can be reviewed after the fact. The rejection decision
must remain based only on persistent exact-zero slot-ending accuracy.

## Operational Use

Use NanoBind-S0 before expensive tier-1 evaluation to eliminate architectures
that collapse completely on this binding slot.

Recommended behavior:

- `nano_bind_is_no_go=True`: reject at S0, record `failure_op='nano_bind'`
- `nano_bind_is_no_go=False`: continue normal evaluation
- non-`ok` status: do not treat as a successful no-go unless the caller has an
  explicit infrastructure-failure policy

Do not use NanoBind scores as:

- a leaderboard component
- a promotion score
- an attention-vs-non-attention ranking signal
- evidence that a nonzero architecture is good

## Known Scope

NanoBind-S0 is intentionally a narrow degeneracy filter. It says nothing about
general language quality, HellaSwag, BLiMP, WikiText perplexity, or broad
compositional reasoning.

The safe interpretation is:

```text
persistent exact zero -> architecture is not worth tier-1 budget
nonzero              -> architecture has not failed this one hard no-go screen
```
