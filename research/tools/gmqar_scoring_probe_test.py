"""Decisive CPU-only test of whether gMQAR SCORING is mis-specified.

No checkpoint, no GPU, does not touch the running job. Builds one gMQAR cell,
then feeds the probe a HAND-CRAFTED "oracle" logits tensor that ranks the correct
value token #1 *among the in-context candidate values* but puts a common
natural-language token (id 50) slightly higher overall. If the probe scores this
oracle 0.0, the metric is mis-specified (full-vocab argmax floors any model that
hasn't formed a literal copy/induction head), which explains softmax=0.206 and
memory-lanes=0.0 even when binding is present.
"""

import sys
import torch

sys.path.insert(0, "/home/tim/Projects/LLM")
from research.eval.gmqar import GMQARConfig, make_gmqar_batch  # noqa: E402

VOCAB = 100000
cfg = GMQARConfig(
    vocab_size=VOCAB,
    n_pairs=2,
    n_queries=2,
    distractor_tokens=0,
    batch_size=8,
    seed=0,
    token_pool=2048,
)
g = torch.Generator().manual_seed(0)
input_ids, target_ids, answer_mask = make_gmqar_batch(cfg, g, "cpu")
B, S = input_ids.shape
print(f"cell: 2pairs/0distract  batch={B} seq={S}  n_answers={int(answer_mask.sum())}")

# Build an ORACLE that has perfectly bound every pair: at each answer position it
# ranks the TRUE value highest *within the candidate value set*, but a frequent
# token (id 50) is globally a bit higher (mimicking a model whose top raw logit is
# a common natural-language token mid-gibberish — exactly the realistic case).
logits = torch.full((B, S, VOCAB), -10.0)
COMMON = 50
logits[:, :, COMMON] = 5.0  # common token globally dominant everywhere
for b in range(B):
    for t in range(S):
        if answer_mask[b, t]:
            v = int(target_ids[b, t])
            logits[b, t, v] = 4.0  # true value: high, but below COMMON's 5.0

preds = logits.argmax(-1)
full_vocab_acc = (preds[answer_mask] == target_ids[answer_mask]).float().mean().item()
print(f"[probe as-written] full-vocab argmax acc = {full_vocab_acc:.4f}")

# Now score the SAME oracle the 'fair' way: restrict argmax to the in-context
# candidate value set for each row (what an associative-recall readout should do).
fair_correct, fair_total = 0, 0
for b in range(B):
    # candidate values = the value tokens present in this row's KV block
    cand = torch.unique(target_ids[b][answer_mask[b]])
    # also include all values actually bound in the row (targets ARE the values)
    for t in range(S):
        if answer_mask[b, t]:
            row_logits = logits[b, t]
            # restrict to candidate set
            sub = row_logits[cand]
            pred_v = int(cand[int(sub.argmax())])
            fair_correct += int(pred_v == int(target_ids[b, t]))
            fair_total += 1
fair_acc = fair_correct / fair_total
print(f"[candidate-restricted] argmax-over-in-context-values acc = {fair_acc:.4f}")

# And rank-of-target (how far down the true value is, full vocab)
ranks = []
for b in range(B):
    for t in range(S):
        if answer_mask[b, t]:
            v = int(target_ids[b, t])
            r = int((logits[b, t] > logits[b, t, v]).sum())
            ranks.append(r)
print(
    f"true-value rank (full vocab): min={min(ranks)} max={max(ranks)} "
    f"(0 = top-1; >0 means a non-value token outranks it)"
)
print(
    "VERDICT: if full-vocab=0.0 but candidate-restricted=1.0, the probe SCORING "
    "is mis-specified for nano binding (floors anything lacking a literal copy head)."
)
