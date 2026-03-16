# Aria Audit Prompt 03 — Byte-Mode and Sequence-Structure Safety

You are auditing which Aria components are safe, unsafe, or conditional for byte-level or very low-level tokenization.

Assume Aria components act on post-embedding hidden states, not raw bytes, but evaluate whether their assumptions still break under byte-like representations.

## Focus areas
- reorder / sort ops
- merge / compression ops
- token dropping / masking ops
- routing-conditioned transforms
- difficulty / entropy / mean-based gating
- feature-dimension sparsification vs sequence-dimension destruction

## Main questions
1. Which components preserve sequence order and local compositional structure?
2. Which components destroy order, locality, or token identity?
3. Which components are unsafe for byte-mode even if they operate on embeddings?
4. Which components are safe only after several contextual layers?
5. Which components should be hard-banned in byte-mode?
6. Which components are safe enough for a strict byte-safe subset?

## Required method
- Inspect all components that reorder, merge, compress, drop, gate, or mask positions.
- For each, classify:
  - order-safe
  - conditionally safe after contextualization
  - information-destructive
  - likely nonsense for byte-mode
- Pay special attention to:
  - sort_seq / argsort_seq / differentiable_sort
  - token_merge / token_merging / merge_scan
  - mod_topk / early_exit / cascade / speculative
  - routing_conditioned_compression
  - compression_mixture_experts
  - route_topk

## Deliverables
1. Byte-safe subset
2. Hard-ban list for byte-mode
3. Components only allowed after minimum layer depth
4. Grammar guardrails for byte-mode
5. Sequence-destructive components that need residual bypass or removal
6. Recommended byte-mode search space

## Rules
- Do not assume embedding-space operations are automatically byte-safe.
- Do not approve sequence reordering or lossy merge ops for byte-mode unless there is a strong recovery path.
- Distinguish feature-dim sparsification from sequence-dim destruction.

## Output format
1. Executive summary
2. Component-by-component byte-safety table
3. Hard bans
4. Conditional ops and minimum depth rules
5. Byte-safe subset
6. Grammar and architecture guardrails
